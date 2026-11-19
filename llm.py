"""Thin adapter around the language models of the pipeline.

Only this module knows provider APIs, so the rest of the pipeline is independent of the
models it uses. Two providers are supported:

  gemini   Google Gemini (Interactions API).
  openai   Any endpoint that speaks the OpenAI chat completions format. Besides OpenAI
           this covers local servers such as Ollama or LM Studio and most hosted
           gateways, so a local model needs no code change.

Each call is made in a role. The role "answer" writes the answers; the role "verify"
decomposes and checks them. Each role is configured by four settings, taken from the
environment with priority and otherwise from config.py:

  role answer:  LLM_PROVIDER     LLM_MODEL     LLM_URL     LLM_API_KEY
  role verify:  VERIFY_PROVIDER  VERIFY_MODEL  VERIFY_URL  VERIFY_API_KEY

A verify setting that is not set falls back to the answer setting of the same name. The
model, endpoint and key fall back only while both roles use the same provider, so the key
of one provider is never sent to another. Example of a local model:

  LLM_PROVIDER=openai LLM_URL=http://localhost:11434/v1/chat/completions LLM_MODEL=<name>
"""
import json
import os
import re
import urllib.error
import urllib.request

import config


class LLMError(RuntimeError):
    """The model could not be reached or returned no usable answer."""


def _gemini(system, prompt, model, key, temperature, as_json):
    # JSON is requested through the instruction only (see generate_json).
    headers = {"x-goog-api-key": key, "Api-Revision": "2026-05-20"}
    body = {"model": model, "system_instruction": system, "input": prompt}
    if temperature is not None:
        body["generation_config"] = {"temperature": temperature}

    def text(data):
        outputs = [s for s in data.get("steps", []) if s.get("type") == "model_output"]
        blocks = outputs[-1].get("content", []) if outputs else []
        return "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
    return headers, body, text


def _openai(system, prompt, model, key, temperature, as_json):
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    body = {"model": model, "messages": [{"role": "system", "content": system},
                                         {"role": "user", "content": prompt}]}
    if temperature is not None:
        body["temperature"] = temperature
    if as_json:
        # The endpoint requires the word "JSON" in the messages; generate_json adds it.
        body["response_format"] = {"type": "json_object"}

    def text(data):
        return data["choices"][0]["message"]["content"] or ""
    return headers, body, text


# provider -> (request builder, default endpoint, default model, key required)
PROVIDERS = {
    "gemini": (_gemini, "https://generativelanguage.googleapis.com/v1beta/interactions",
               "gemini-3.8-flash", True),
    "openai": (_openai, "https://api.openai.com/v1/chat/completions", None, False),
}

# role -> prefix of its settings; the first role is the one every other role falls back to
ROLES = {"answer": "LLM", "verify": "VERIFY"}


def setting(name):
    return os.environ.get(name) or getattr(config, name, None)


def resolve(role):
    """-> (provider, model, url, key) of a role, with fallbacks and defaults applied."""
    if role not in ROLES:
        raise LLMError(f"unknown role {role!r} (supported: {', '.join(ROLES)})")
    base, own = "LLM", ROLES[role]
    provider = setting(f"{own}_PROVIDER") or setting(f"{base}_PROVIDER")
    if provider not in PROVIDERS:
        raise LLMError(f"unknown {own}_PROVIDER {provider!r} "
                       f"(supported: {', '.join(PROVIDERS)})")
    inherit = own != base and provider == setting(f"{base}_PROVIDER")

    def value(name):
        return setting(f"{own}_{name}") or (setting(f"{base}_{name}") if inherit else None)

    _, default_url, default_model, needs_key = PROVIDERS[provider]
    model, url, key = value("MODEL") or default_model, value("URL") or default_url, \
        value("API_KEY")
    if not model:
        raise LLMError(f"{own}_MODEL is not set; provider {provider} has no default model")
    if needs_key and not key:
        raise LLMError(f"{own}_API_KEY is not set; provider {provider} requires a key")
    return provider, model, url, key


def generate(system, prompt, role="answer", temperature=None, timeout=120, as_json=False):
    """-> the model's answer to prompt under the system instruction, as text.

    temperature None leaves the provider default. as_json asks the provider for a JSON
    response where it supports that; use generate_json to also parse it.
    """
    provider, model, url, key = resolve(role)
    build = PROVIDERS[provider][0]
    headers, body, text_of = build(system, prompt, model, key, temperature, as_json)
    request = urllib.request.Request(url, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json", **headers})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.load(response)
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")
        try:
            detail = json.loads(detail)["error"]["message"]
        except (ValueError, KeyError, TypeError):
            pass
        raise LLMError(f"{provider} {model}: HTTP {e.code}: {str(detail)[:300]}") from None
    except (urllib.error.URLError, TimeoutError, ValueError) as e:
        raise LLMError(f"{provider} {model} at {url}: {e}") from None
    try:
        text = text_of(data).strip()
    except (KeyError, IndexError, TypeError, AttributeError):
        text = ""
    if not text:
        raise LLMError(f"{provider} {model}: response contains no text")
    return text


JSON_NOTE = "Antworte ausschließlich mit gültigem JSON, ohne weiteren Text."
FENCE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)


def parse_json(text):
    """-> the JSON value in a model response; tolerates a surrounding code fence."""
    match = FENCE.match(text.strip())
    try:
        return json.loads(match.group(1) if match else text)
    except ValueError:
        raise LLMError(f"response is not valid JSON: {text[:200]!r}") from None


def generate_json(system, prompt, role="verify", temperature=0, timeout=120):
    """-> the model's answer parsed as JSON. Deterministic by default (temperature 0).

    The instruction to answer in JSON is appended to the system instruction, because not
    every provider offers a JSON mode and some require the instruction even with one.
    """
    return parse_json(generate(f"{system}\n\n{JSON_NOTE}", prompt, role, temperature,
                               timeout, as_json=True))

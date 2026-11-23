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

  role answer:  LLM_PROVIDER     LLM_MODEL     LLM_URL     LLM_API_KEY     LLM_REASONING
  role verify:  VERIFY_PROVIDER  VERIFY_MODEL  VERIFY_URL  VERIFY_API_KEY  VERIFY_REASONING

REASONING sets how much a reasoning model thinks before it answers, e.g. "none", "low" or
"high"; unset leaves the provider default. Thinking dominates the response time of local
reasoning models, and models without reasoning reject the setting, so it is opt-in.

A verify setting that is not set falls back to the answer setting of the same name. The
model, endpoint, key and reasoning fall back only while both roles use the same provider, so the key
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


class FormatError(LLMError):
    """The model answered, but not in the requested form."""


def _gemini(system, prompt, model, key, temperature, as_json, reasoning):
    # JSON is requested through the instruction only (see generate_json).
    headers = {"x-goog-api-key": key, "Api-Revision": "2026-05-20"}
    body = {"model": model, "system_instruction": system, "input": prompt}
    config = {k: v for k, v in (("temperature", temperature), ("thinking_level", reasoning))
              if v is not None}
    if config:
        body["generation_config"] = config

    def text(data):
        outputs = [s for s in data.get("steps", []) if s.get("type") == "model_output"]
        blocks = outputs[-1].get("content", []) if outputs else []
        return "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
    return headers, body, text


def _openai(system, prompt, model, key, temperature, as_json, reasoning):
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    body = {"model": model, "messages": [{"role": "system", "content": system},
                                         {"role": "user", "content": prompt}]}
    if temperature is not None:
        body["temperature"] = temperature
    if reasoning is not None:
        body["reasoning_effort"] = reasoning
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
    """-> (provider, model, url, key, reasoning) of a role, fallbacks and defaults applied."""
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
    return provider, model, url, key, value("REASONING")


def _open(role, system, prompt, temperature, as_json, timeout, stream=False):
    """Send the request of a role. -> (open response, text extractor, provider, model, url)."""
    provider, model, url, key, reasoning = resolve(role)
    build = PROVIDERS[provider][0]
    headers, body, text_of = build(system, prompt, model, key, temperature, as_json, reasoning)
    if stream:
        body["stream"] = True
    request = urllib.request.Request(url, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json", **headers})
    try:
        return urllib.request.urlopen(request, timeout=timeout), text_of, provider, model, url
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")
        try:
            detail = json.loads(detail)["error"]["message"]
        except (ValueError, KeyError, TypeError):
            pass
        raise LLMError(f"{provider} {model}: HTTP {e.code}: {str(detail)[:300]}") from None
    except (urllib.error.URLError, TimeoutError, ValueError) as e:
        raise LLMError(f"{provider} {model} at {url}: {e}") from None


def generate(system, prompt, role="answer", temperature=None, timeout=120, as_json=False):
    """-> the model's answer to prompt under the system instruction, as text.

    temperature None leaves the provider default. as_json asks the provider for a JSON
    response where it supports that; use generate_json to also parse it.
    """
    response, text_of, provider, model, url = _open(role, system, prompt, temperature,
                                                    as_json, timeout)
    try:
        with response:
            data = json.load(response)
    except (TimeoutError, ValueError) as e:
        raise LLMError(f"{provider} {model} at {url}: {e}") from None
    try:
        text = text_of(data).strip()
    except (KeyError, IndexError, TypeError, AttributeError):
        text = ""
    if not text:
        raise LLMError(f"{provider} {model}: response contains no text")
    return text


def stream(system, prompt, role="answer", temperature=None, timeout=120):
    """Yield the model's answer in pieces as it is generated.

    Streams with the openai provider (server-sent events). The gemini provider yields the
    whole answer as one piece, since its stream format is not implemented here.
    """
    if resolve(role)[0] != "openai":
        yield generate(system, prompt, role, temperature, timeout)
        return
    response, _, provider, model, url = _open(role, system, prompt, temperature, False,
                                              timeout, stream=True)
    produced = False
    try:
        with response:
            for raw in response:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):        # blank lines and keep-alive comments
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                event = json.loads(data)
                if "error" in event:
                    raise LLMError(f"{provider} {model}: {str(event['error'])[:300]}")
                choices = event.get("choices") or []
                piece = (choices[0].get("delta") or {}).get("content") if choices else None
                if piece:
                    produced = True
                    yield piece
    except (TimeoutError, ValueError, urllib.error.URLError) as e:
        raise LLMError(f"{provider} {model} at {url}: {e}") from None
    if not produced:
        raise LLMError(f"{provider} {model}: response contains no text")


JSON_NOTE = "Antworte ausschließlich mit gültigem JSON, ohne weiteren Text."
FENCE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)


def parse_json(text):
    """-> the JSON value in a model response; tolerates a surrounding code fence."""
    match = FENCE.match(text.strip())
    try:
        return json.loads(match.group(1) if match else text)
    except ValueError:
        raise FormatError(f"response is not valid JSON: {text[:200]!r}") from None


def generate_json(system, prompt, role="verify", temperature=0, timeout=120):
    """-> the model's answer parsed as JSON. Deterministic by default (temperature 0).

    The instruction to answer in JSON is appended to the system instruction, because not
    every provider offers a JSON mode and some require the instruction even with one.
    """
    return parse_json(generate(f"{system}\n\n{JSON_NOTE}", prompt, role, temperature,
                               timeout, as_json=True))

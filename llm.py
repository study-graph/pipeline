"""Thin adapter around the generation LLM.

Only this module knows provider APIs, so the rest of the pipeline is independent of the
model that writes the answers. Two providers are supported:

  gemini   Google Gemini (Interactions API).
  openai   Any endpoint that speaks the OpenAI chat completions format. Besides OpenAI
           this covers local servers such as Ollama or LM Studio and most hosted
           gateways, so a local model needs no code change.

The provider, model, endpoint and key come from config.py or, with priority, from the
environment variables LLM_PROVIDER, LLM_MODEL, LLM_URL and LLM_API_KEY, e.g.

  LLM_PROVIDER=openai LLM_URL=http://localhost:11434/v1/chat/completions LLM_MODEL=<name>
"""
import json
import os
import urllib.error
import urllib.request

import config


class LLMError(RuntimeError):
    """The model could not be reached or returned no usable answer."""


def _gemini(system, prompt, model, key):
    headers = {"x-goog-api-key": key, "Api-Revision": "2026-05-20"}
    body = {"model": model, "system_instruction": system, "input": prompt}

    def text(data):
        outputs = [s for s in data.get("steps", []) if s.get("type") == "model_output"]
        blocks = outputs[-1].get("content", []) if outputs else []
        return "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
    return headers, body, text


def _openai(system, prompt, model, key):
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    body = {"model": model, "messages": [{"role": "system", "content": system},
                                         {"role": "user", "content": prompt}]}

    def text(data):
        return data["choices"][0]["message"]["content"] or ""
    return headers, body, text


# provider -> (request builder, default endpoint, default model, key required)
PROVIDERS = {
    "gemini": (_gemini, "https://generativelanguage.googleapis.com/v1beta/interactions",
               "gemini-3.8-flash", True),
    "openai": (_openai, "https://api.openai.com/v1/chat/completions", None, False),
}


def setting(name):
    return os.environ.get(name) or getattr(config, name, None)


def generate(system, prompt, timeout=120):
    """-> the model's answer to prompt under the system instruction, as text."""
    provider = setting("LLM_PROVIDER")
    if provider not in PROVIDERS:
        raise LLMError(f"unknown LLM_PROVIDER {provider!r} (supported: {', '.join(PROVIDERS)})")
    build, default_url, default_model, needs_key = PROVIDERS[provider]
    model, url, key = setting("LLM_MODEL") or default_model, setting("LLM_URL") or default_url, \
        os.environ.get("LLM_API_KEY")
    if not model:
        raise LLMError(f"LLM_MODEL is not set; provider {provider} has no default model")
    if needs_key and not key:
        raise LLMError(f"LLM_API_KEY is not set; provider {provider} requires a key")

    headers, body, text_of = build(system, prompt, model, key)
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

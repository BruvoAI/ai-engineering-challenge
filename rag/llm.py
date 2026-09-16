"""Thin LLM wrapper — edit freely.

Defaults to a local Ollama model (nothing leaves your laptop). Swap in a
cloud model any time by setting LLM_PROVIDER=openai or LLM_PROVIDER=anthropic
(plus the matching API key) in your .env — no other code changes needed.

Every call through call_llm() records token usage (see _usage below) so the
harness can report a token-frugality score alongside the accuracy score. If
you replace call_llm's internals wholesale, keep calling _record_usage(...)
or the frugality score will just read as 0 for you.
"""

import os

import requests
from dotenv import load_dotenv

load_dotenv()

PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_MODEL = {
    "ollama": os.getenv("OLLAMA_MODEL", "llama3.2:3b"),
    "openai": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    "anthropic": os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001"),
}[PROVIDER]

_usage = {"input_tokens": 0, "output_tokens": 0, "calls": 0}


def _record_usage(input_tokens: int, output_tokens: int) -> None:
    _usage["input_tokens"] += input_tokens
    _usage["output_tokens"] += output_tokens
    _usage["calls"] += 1


def reset_usage() -> None:
    """Zero the running token counter. Called by the harness before scoring."""
    _usage["input_tokens"] = 0
    _usage["output_tokens"] = 0
    _usage["calls"] = 0


def get_usage() -> dict:
    """Return cumulative token usage since the last reset_usage() call."""
    return dict(_usage)


def call_llm(prompt: str, system: str | None = None, model: str | None = None) -> str:
    """Send a single-turn prompt to the configured LLM and return the text response."""
    model = model or DEFAULT_MODEL

    if PROVIDER == "ollama":
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        try:
            resp = requests.post(
                f"{OLLAMA_HOST}/api/chat",
                json={"model": model, "messages": messages, "stream": False},
                timeout=120,
            )
            resp.raise_for_status()
        except requests.exceptions.ConnectionError as exc:
            raise RuntimeError(
                "Could not reach Ollama. Is it running? Start it with `ollama serve` "
                f"and make sure the model is pulled: `ollama pull {model}`."
            ) from exc
        data = resp.json()
        _record_usage(data.get("prompt_eval_count", 0), data.get("eval_count", 0))
        return data["message"]["content"]

    if PROVIDER == "openai":
        from openai import OpenAI

        client = OpenAI()
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        resp = client.chat.completions.create(model=model, messages=messages)
        _record_usage(resp.usage.prompt_tokens, resp.usage.completion_tokens)
        return resp.choices[0].message.content

    if PROVIDER == "anthropic":
        import anthropic

        client = anthropic.Anthropic()
        resp = client.messages.create(
            model=model,
            max_tokens=512,
            system=system or "",
            messages=[{"role": "user", "content": prompt}],
        )
        _record_usage(resp.usage.input_tokens, resp.usage.output_tokens)
        return resp.content[0].text

    raise ValueError(f"Unknown LLM_PROVIDER: {PROVIDER}")

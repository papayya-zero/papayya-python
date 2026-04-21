"""Shape-based usage extraction for LLM provider responses.

This module pulls token counts, model, and stop-reason from whatever
shape a provider SDK returns. It duck-types on attributes — no SDK
imports — so new providers are added by appending a branch to the
shape ladder, not by pinning a new dependency.

The return value always has the same fields; missing data surfaces as
``None`` rather than zero so callers can distinguish "unknown" from
"zero tokens". ``provider_shape`` names the branch that matched, which
is useful for debugging and for per-provider rollups on the dashboard.

Shape ladder order matters — specific shapes first, generic dict last:

1. ``response.usage.prompt_tokens`` + ``.completion_tokens`` → OpenAI /
   Azure OpenAI / OpenAI-compatible (Groq, Fireworks, Together, vLLM,
   Ollama /v1/chat/completions, Mistral /v1/*).
2. ``response.usage.input_tokens`` + ``.output_tokens`` → Anthropic.
3. ``response.usage_metadata.prompt_token_count`` → Gemini /
   google-generativeai.
4. Dict with ``["usage"]["prompt_tokens"]`` or ``["usage"]["input_tokens"]``
   → raw HTTP responses, OpenAI-compat services that return dicts.
5. Nothing matched → ``provider_shape="unknown"`` and all fields ``None``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class LlmUsage:
    """Uniform summary of an LLM call extracted from a provider response.

    All numeric fields are ``None`` when the provider did not report them
    (distinguishing "unknown" from "zero tokens"). ``provider_shape``
    names which branch of the extractor matched.
    """

    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    model: str | None
    stop_reason: str | None
    provider_shape: str


_UNKNOWN = LlmUsage(
    prompt_tokens=None,
    completion_tokens=None,
    total_tokens=None,
    model=None,
    stop_reason=None,
    provider_shape="unknown",
)


def _safe_int(value: Any) -> int | None:
    """Coerce to ``int`` when sensible; return ``None`` on any failure."""
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_str(value: Any) -> str | None:
    """Coerce to ``str`` when present; return ``None`` when absent."""
    if value is None:
        return None
    try:
        s = str(value)
    except Exception:
        return None
    return s or None


def _extract_openai(response: Any) -> LlmUsage | None:
    """OpenAI shape: ``response.usage.{prompt,completion,total}_tokens``."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    prompt = getattr(usage, "prompt_tokens", None)
    completion = getattr(usage, "completion_tokens", None)
    if prompt is None and completion is None:
        return None

    total = getattr(usage, "total_tokens", None)
    model = getattr(response, "model", None)

    stop_reason = None
    choices = getattr(response, "choices", None)
    if choices:
        first = choices[0] if hasattr(choices, "__getitem__") else next(iter(choices), None)
        if first is not None:
            stop_reason = getattr(first, "finish_reason", None)

    return LlmUsage(
        prompt_tokens=_safe_int(prompt),
        completion_tokens=_safe_int(completion),
        total_tokens=_safe_int(total),
        model=_safe_str(model),
        stop_reason=_safe_str(stop_reason),
        provider_shape="openai",
    )


def _extract_anthropic(response: Any) -> LlmUsage | None:
    """Anthropic shape: ``response.usage.{input,output}_tokens``."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    input_tokens = getattr(usage, "input_tokens", None)
    output_tokens = getattr(usage, "output_tokens", None)
    if input_tokens is None and output_tokens is None:
        return None

    model = getattr(response, "model", None)
    stop_reason = getattr(response, "stop_reason", None)

    prompt_int = _safe_int(input_tokens)
    completion_int = _safe_int(output_tokens)
    total = None
    if prompt_int is not None and completion_int is not None:
        total = prompt_int + completion_int

    return LlmUsage(
        prompt_tokens=prompt_int,
        completion_tokens=completion_int,
        total_tokens=total,
        model=_safe_str(model),
        stop_reason=_safe_str(stop_reason),
        provider_shape="anthropic",
    )


def _extract_gemini(response: Any) -> LlmUsage | None:
    """Gemini shape: ``response.usage_metadata.{prompt,candidates,total}_token_count``."""
    meta = getattr(response, "usage_metadata", None)
    if meta is None:
        return None
    prompt = getattr(meta, "prompt_token_count", None)
    candidates = getattr(meta, "candidates_token_count", None)
    if prompt is None and candidates is None:
        return None

    total = getattr(meta, "total_token_count", None)
    model = getattr(response, "model_version", None) or getattr(response, "model", None)

    stop_reason = None
    gemini_candidates = getattr(response, "candidates", None)
    if gemini_candidates:
        first = (
            gemini_candidates[0]
            if hasattr(gemini_candidates, "__getitem__")
            else next(iter(gemini_candidates), None)
        )
        if first is not None:
            stop_reason = getattr(first, "finish_reason", None)

    return LlmUsage(
        prompt_tokens=_safe_int(prompt),
        completion_tokens=_safe_int(candidates),
        total_tokens=_safe_int(total),
        model=_safe_str(model),
        stop_reason=_safe_str(stop_reason),
        provider_shape="gemini",
    )


def _extract_dict(response: Any) -> LlmUsage | None:
    """Dict / JSON shape — OpenAI-compat and raw HTTP responses.

    Accepts either OpenAI-style (``usage.prompt_tokens``) or
    Anthropic-style (``usage.input_tokens``) keys inside a dict.
    """
    if not isinstance(response, dict):
        return None
    usage = response.get("usage")
    if not isinstance(usage, dict):
        return None

    if "prompt_tokens" in usage or "completion_tokens" in usage:
        prompt = usage.get("prompt_tokens")
        completion = usage.get("completion_tokens")
        total = usage.get("total_tokens")
        shape = "dict-openai"
    elif "input_tokens" in usage or "output_tokens" in usage:
        prompt = usage.get("input_tokens")
        completion = usage.get("output_tokens")
        total = usage.get("total_tokens")
        shape = "dict-anthropic"
    else:
        return None

    model = response.get("model")

    stop_reason = None
    choices = response.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            stop_reason = first.get("finish_reason")
    if stop_reason is None:
        stop_reason = response.get("stop_reason")

    prompt_int = _safe_int(prompt)
    completion_int = _safe_int(completion)
    total_int = _safe_int(total)
    if total_int is None and prompt_int is not None and completion_int is not None:
        total_int = prompt_int + completion_int

    return LlmUsage(
        prompt_tokens=prompt_int,
        completion_tokens=completion_int,
        total_tokens=total_int,
        model=_safe_str(model),
        stop_reason=_safe_str(stop_reason),
        provider_shape=shape,
    )


_EXTRACTORS = (
    _extract_openai,
    _extract_anthropic,
    _extract_gemini,
    _extract_dict,
)


def extract_llm_usage(response: Any) -> LlmUsage:
    """Run the shape ladder and return a best-effort :class:`LlmUsage`.

    Falls through to ``LlmUsage(provider_shape="unknown", …None)`` when
    no branch matches — callers still get a durable record that the
    call happened, they just lose token granularity.
    """
    if response is None:
        return _UNKNOWN
    for extract in _EXTRACTORS:
        try:
            result = extract(response)
        except Exception:
            # Extractors must never raise; a malformed response shouldn't
            # fail the user's run. Fall through to the next shape.
            continue
        if result is not None:
            return result
    return _UNKNOWN

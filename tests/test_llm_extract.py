"""Tests for shape-based LLM response extraction.

Covers the full ladder (OpenAI → Anthropic → Gemini → dict → unknown)
without importing any LLM SDK. Fake response objects are assembled via
``SimpleNamespace``, which duck-types cleanly against the extractor's
``getattr``-based inspection.
"""

from __future__ import annotations

from types import SimpleNamespace

from papayya.llm_extract import LlmUsage, extract_llm_usage


# ---------------------------------------------------------------------------
# OpenAI shape — usage.prompt_tokens + usage.completion_tokens
# ---------------------------------------------------------------------------

def test_openai_shape_full_response():
    response = SimpleNamespace(
        model="gpt-4o-mini",
        usage=SimpleNamespace(
            prompt_tokens=120,
            completion_tokens=40,
            total_tokens=160,
        ),
        choices=[SimpleNamespace(finish_reason="stop")],
    )
    usage = extract_llm_usage(response)
    assert usage.provider_shape == "openai"
    assert usage.prompt_tokens == 120
    assert usage.completion_tokens == 40
    assert usage.total_tokens == 160
    assert usage.model == "gpt-4o-mini"
    assert usage.stop_reason == "stop"


def test_openai_shape_finish_reason_length_preserved():
    # The extractor must not flag or drop truncation — it just records it.
    response = SimpleNamespace(
        model="gpt-4o-mini",
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=4000, total_tokens=4010),
        choices=[SimpleNamespace(finish_reason="length")],
    )
    usage = extract_llm_usage(response)
    assert usage.provider_shape == "openai"
    assert usage.stop_reason == "length"


def test_openai_shape_missing_choices():
    response = SimpleNamespace(
        model="gpt-4o-mini",
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=20, total_tokens=30),
    )
    usage = extract_llm_usage(response)
    assert usage.provider_shape == "openai"
    assert usage.stop_reason is None


# ---------------------------------------------------------------------------
# Anthropic shape — usage.input_tokens + usage.output_tokens
# ---------------------------------------------------------------------------

def test_anthropic_shape_full_response():
    response = SimpleNamespace(
        model="claude-sonnet-4-6",
        usage=SimpleNamespace(input_tokens=500, output_tokens=150),
        stop_reason="end_turn",
    )
    usage = extract_llm_usage(response)
    assert usage.provider_shape == "anthropic"
    assert usage.prompt_tokens == 500
    assert usage.completion_tokens == 150
    # Anthropic doesn't report total; extractor computes it.
    assert usage.total_tokens == 650
    assert usage.model == "claude-sonnet-4-6"
    assert usage.stop_reason == "end_turn"


def test_anthropic_shape_max_tokens_preserved():
    response = SimpleNamespace(
        model="claude-sonnet-4-6",
        usage=SimpleNamespace(input_tokens=10, output_tokens=4096),
        stop_reason="max_tokens",
    )
    usage = extract_llm_usage(response)
    assert usage.provider_shape == "anthropic"
    assert usage.stop_reason == "max_tokens"


# ---------------------------------------------------------------------------
# Gemini shape — usage_metadata.*_token_count
# ---------------------------------------------------------------------------

def test_gemini_shape_full_response():
    response = SimpleNamespace(
        model_version="gemini-2.0-flash",
        usage_metadata=SimpleNamespace(
            prompt_token_count=80,
            candidates_token_count=25,
            total_token_count=105,
        ),
        candidates=[SimpleNamespace(finish_reason="STOP")],
    )
    usage = extract_llm_usage(response)
    assert usage.provider_shape == "gemini"
    assert usage.prompt_tokens == 80
    assert usage.completion_tokens == 25
    assert usage.total_tokens == 105
    assert usage.model == "gemini-2.0-flash"
    assert usage.stop_reason == "STOP"


def test_gemini_shape_falls_back_to_model_field():
    # Some google-genai variants expose ``model`` instead of ``model_version``.
    response = SimpleNamespace(
        model="gemini-1.5-pro",
        usage_metadata=SimpleNamespace(
            prompt_token_count=10,
            candidates_token_count=5,
            total_token_count=15,
        ),
    )
    usage = extract_llm_usage(response)
    assert usage.provider_shape == "gemini"
    assert usage.model == "gemini-1.5-pro"


# ---------------------------------------------------------------------------
# Dict fallback — OpenAI-compat and raw HTTP responses
# ---------------------------------------------------------------------------

def test_dict_openai_shape():
    response = {
        "model": "llama-3.1-70b-instruct",
        "usage": {"prompt_tokens": 50, "completion_tokens": 10, "total_tokens": 60},
        "choices": [{"finish_reason": "stop"}],
    }
    usage = extract_llm_usage(response)
    assert usage.provider_shape == "dict-openai"
    assert usage.prompt_tokens == 50
    assert usage.completion_tokens == 10
    assert usage.total_tokens == 60
    assert usage.model == "llama-3.1-70b-instruct"
    assert usage.stop_reason == "stop"


def test_dict_anthropic_shape():
    response = {
        "model": "custom-claude-clone",
        "usage": {"input_tokens": 100, "output_tokens": 20},
        "stop_reason": "end_turn",
    }
    usage = extract_llm_usage(response)
    assert usage.provider_shape == "dict-anthropic"
    assert usage.prompt_tokens == 100
    assert usage.completion_tokens == 20
    # Total is derived when absent.
    assert usage.total_tokens == 120
    assert usage.stop_reason == "end_turn"


# ---------------------------------------------------------------------------
# Unknown fallback + robustness
# ---------------------------------------------------------------------------

def test_unknown_shape_returns_none_fields():
    response = SimpleNamespace(some="random", shape="we", dont="know")
    usage = extract_llm_usage(response)
    assert usage.provider_shape == "unknown"
    assert usage.prompt_tokens is None
    assert usage.completion_tokens is None
    assert usage.total_tokens is None
    assert usage.model is None
    assert usage.stop_reason is None


def test_none_response_returns_unknown():
    usage = extract_llm_usage(None)
    assert usage.provider_shape == "unknown"


def test_empty_dict_returns_unknown():
    usage = extract_llm_usage({})
    assert usage.provider_shape == "unknown"


def test_non_numeric_token_fields_degrade_gracefully():
    # Some SDKs stub tokens as strings; extractor must not crash.
    response = SimpleNamespace(
        model="gpt-4o-mini",
        usage=SimpleNamespace(
            prompt_tokens="not-a-number",
            completion_tokens=5,
            total_tokens=None,
        ),
    )
    usage = extract_llm_usage(response)
    assert usage.provider_shape == "openai"
    assert usage.prompt_tokens is None
    assert usage.completion_tokens == 5


def test_extractor_never_raises_on_hostile_response():
    class Hostile:
        @property
        def usage(self):
            raise RuntimeError("boom")

    usage = extract_llm_usage(Hostile())
    # Extractor catches and falls through — call didn't crash the agent.
    assert usage.provider_shape == "unknown"


def test_returned_type_is_llm_usage():
    usage = extract_llm_usage({})
    assert isinstance(usage, LlmUsage)

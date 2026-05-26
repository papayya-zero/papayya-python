"""Unit tests for papayya.outcomes structural inspectors.

Plan 02 — pure-function tests with no I/O. Covers each inspector in
isolation plus the orchestrator's precedence order.
"""

from __future__ import annotations

from papayya.llm_extract import LlmUsage
from papayya.outcomes import (
    OK,
    inspect_degenerate_embedding,
    inspect_empty,
    inspect_llm_stop_reason,
    inspect_result,
)


def _verdict(v):
    return (v.status, v.reason)


# --- inspect_empty ------------------------------------------------------ #

def test_inspect_empty_none():
    assert _verdict(inspect_empty(None)) == ("degraded", "empty_none")


def test_inspect_empty_false():
    assert _verdict(inspect_empty(False)) == ("degraded", "empty_false")


def test_inspect_empty_string():
    assert _verdict(inspect_empty("")) == ("degraded", "empty_string")


def test_inspect_empty_bytes():
    assert _verdict(inspect_empty(b"")) == ("degraded", "empty_string")


def test_inspect_empty_list():
    assert _verdict(inspect_empty([])) == ("degraded", "empty_sequence")


def test_inspect_empty_tuple():
    assert _verdict(inspect_empty(())) == ("degraded", "empty_sequence")


def test_inspect_empty_dict():
    assert _verdict(inspect_empty({})) == ("degraded", "empty_dict")


def test_inspect_empty_zero_is_ok():
    # Numeric 0 / 0.0 are legitimate outputs; not degraded.
    assert inspect_empty(0) is OK
    assert inspect_empty(0.0) is OK


def test_inspect_empty_populated_dict_is_ok():
    assert inspect_empty({"foo": "bar"}) is OK


def test_inspect_empty_populated_list_is_ok():
    assert inspect_empty(["x"]) is OK


# --- inspect_degenerate_embedding -------------------------------------- #

def test_inspect_degenerate_embedding_zero_list():
    assert _verdict(inspect_degenerate_embedding([0.0, 0.0, 0.0])) == (
        "degraded",
        "degenerate_embedding",
    )


def test_inspect_degenerate_embedding_in_dict():
    assert _verdict(inspect_degenerate_embedding({"embedding": [0.0] * 1536})) == (
        "degraded",
        "degenerate_embedding",
    )


def test_inspect_degenerate_embedding_plural_key():
    assert _verdict(inspect_degenerate_embedding({"embeddings": [0.0, 0.0]})) == (
        "degraded",
        "degenerate_embedding",
    )


def test_inspect_degenerate_embedding_nonzero_is_ok():
    assert inspect_degenerate_embedding([0.1, 0.0, 0.2]) is OK


def test_inspect_degenerate_embedding_unknown_shape_is_ok():
    # No "embedding" key, not a bare list — skip silently.
    assert inspect_degenerate_embedding({"foo": "bar"}) is OK


# --- inspect_llm_stop_reason ------------------------------------------- #

def _usage(stop_reason: str | None) -> LlmUsage:
    return LlmUsage(
        prompt_tokens=10,
        completion_tokens=20,
        total_tokens=30,
        model="test-model",
        stop_reason=stop_reason,
        provider_shape="openai",
    )


def test_inspect_llm_stop_reason_length():
    assert _verdict(inspect_llm_stop_reason(_usage("length"))) == (
        "degraded",
        "llm_stop_reason:length",
    )


def test_inspect_llm_stop_reason_content_filter():
    assert _verdict(inspect_llm_stop_reason(_usage("content_filter"))) == (
        "degraded",
        "llm_stop_reason:content_filter",
    )


def test_inspect_llm_stop_reason_normal_stop_is_ok():
    assert inspect_llm_stop_reason(_usage("stop")) is OK


def test_inspect_llm_stop_reason_none_usage_is_ok():
    assert inspect_llm_stop_reason(None) is OK


# --- inspect_result orchestrator --------------------------------------- #

def test_inspect_result_empty_wins_over_missing_embedding():
    assert _verdict(inspect_result([])) == ("degraded", "empty_sequence")


def test_inspect_result_degenerate_embedding_in_dict():
    assert _verdict(inspect_result({"embedding": [0.0] * 8})) == (
        "degraded",
        "degenerate_embedding",
    )


def test_inspect_result_clean_dict_is_ok():
    assert inspect_result({"foo": "bar"}) is OK


def test_inspect_result_stop_reason_wins_over_response_shape():
    # An LLM call with stop_reason="length" but a populated dict body —
    # stop-reason verdict must win over the empty/embedding checks.
    verdict = inspect_result({"text": "hi"}, usage=_usage("length"))
    assert _verdict(verdict) == ("degraded", "llm_stop_reason:length")

"""Unit tests for papayya.outcomes structural inspectors.

Plan 02 — pure-function tests with no I/O. Covers each inspector in
isolation plus the orchestrator's precedence order.
"""

from __future__ import annotations

from papayya.llm_extract import LlmUsage
from papayya.outcomes import (
    OK,
    build_contract,
    inspect_conservation,
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


# --- build_contract ------------------------------------------------------ #

def test_build_contract_absent_by_default():
    assert build_contract() is None


def test_build_contract_bare_string_is_one_field():
    assert build_contract(expect_fields="email").fields == ("email",)


def test_build_contract_drops_malformed_declarations():
    # Lenient by construction: a bad declaration is dropped, never raised.
    assert build_contract(expect_count="500") is None
    assert build_contract(expect_coverage=42) is None
    assert build_contract(expect_fields=[]) is None


def test_build_contract_rejects_bool_count():
    # expect_count=True is always a typo, and bool is an int subclass.
    assert build_contract(expect_count=True) is None


# --- inspect_conservation: cardinality ----------------------------------- #

def test_conservation_absent_contract_is_ok():
    assert inspect_conservation([1, 2, 3], None) is OK


def test_conservation_count_shortfall():
    # The motivating case: 40 rows back from a step declaring 500.
    contract = build_contract(expect_count=500)
    verdict = inspect_conservation([{"id": i} for i in range(40)], contract)
    assert _verdict(verdict) == ("degraded", "conservation:count_short")


def test_conservation_count_exact_is_ok():
    contract = build_contract(expect_count=3)
    assert inspect_conservation([1, 2, 3], contract) is OK


def test_conservation_count_overage():
    contract = build_contract(expect_count=3)
    verdict = inspect_conservation([1, 2, 3, 4], contract)
    assert _verdict(verdict) == ("degraded", "conservation:count_over")


def test_conservation_count_callable_over_the_input():
    contract = build_contract(expect_count=lambda docs: len(docs))
    inputs = {"docs": ["a", "b", "c"]}
    assert _verdict(inspect_conservation(["x"], contract, inputs)) == (
        "degraded",
        "conservation:count_short",
    )


def test_conservation_count_callable_that_raises_is_a_pass():
    def boom(**_kwargs):
        raise RuntimeError("contract is broken")

    contract = build_contract(expect_count=boom)
    assert inspect_conservation([], contract, {"docs": []}) is OK


def test_conservation_count_on_uncountable_result_is_a_pass():
    contract = build_contract(expect_count=5)
    assert inspect_conservation(object(), contract) is OK


def test_conservation_count_ignores_string_length():
    # len("hello") == 5 must not satisfy "five records".
    contract = build_contract(expect_count=5)
    assert inspect_conservation("hello", contract) is OK


def test_conservation_count_counts_dict_keys():
    contract = build_contract(expect_count=3)
    verdict = inspect_conservation({"a": 1, "b": 2}, contract)
    assert _verdict(verdict) == ("degraded", "conservation:count_short")


# --- inspect_conservation: coverage -------------------------------------- #

def test_conservation_coverage_finds_the_missing_ids():
    contract = build_contract(expect_coverage="doc_id")
    inputs = {"docs": [{"doc_id": "a"}, {"doc_id": "b"}, {"doc_id": "c"}]}
    verdict = inspect_conservation([{"doc_id": "a"}, {"doc_id": "c"}], contract, inputs)
    assert _verdict(verdict) == ("degraded", "conservation:coverage")


def test_conservation_coverage_complete_is_ok():
    contract = build_contract(expect_coverage="doc_id")
    inputs = {"docs": [{"doc_id": "a"}, {"doc_id": "b"}]}
    result = [{"doc_id": "b"}, {"doc_id": "a"}]
    assert inspect_conservation(result, contract, inputs) is OK


def test_conservation_coverage_locates_the_input_by_key_name():
    # Two collections bound; only one carries the declared id key.
    contract = build_contract(expect_coverage="doc_id")
    inputs = {"opts": [{"flag": True}], "docs": [{"doc_id": "a"}, {"doc_id": "b"}]}
    verdict = inspect_conservation([{"doc_id": "a"}], contract, inputs)
    assert _verdict(verdict) == ("degraded", "conservation:coverage")


def test_conservation_coverage_no_matching_input_is_a_pass():
    contract = build_contract(expect_coverage="doc_id")
    assert inspect_conservation([{"doc_id": "a"}], contract, {"n": 3}) is OK


def test_conservation_coverage_reads_object_attributes():
    class Doc:
        def __init__(self, doc_id):
            self.doc_id = doc_id

    contract = build_contract(expect_coverage="doc_id")
    inputs = {"docs": [Doc("a"), Doc("b")]}
    assert _verdict(inspect_conservation([Doc("a")], contract, inputs)) == (
        "degraded",
        "conservation:coverage",
    )


# --- inspect_conservation: field-presence floor -------------------------- #

def test_conservation_fields_half_empty_dict_is_degraded():
    # Three of eight fields populated — passes inspect_empty, fails here.
    contract = build_contract(expect_fields=["name", "email", "phone"])
    result = {"name": "Ada", "email": None, "phone": ""}
    assert _verdict(inspect_conservation(result, contract)) == (
        "degraded",
        "conservation:field:email",
    )


def test_conservation_fields_names_first_declared_missing_field():
    # Reason must not depend on which record happens to come first.
    contract = build_contract(expect_fields=["name", "email"])
    result = [{"name": "Ada", "email": "a@b.c"}, {"name": "", "email": "d@e.f"}]
    assert _verdict(inspect_conservation(result, contract)) == (
        "degraded",
        "conservation:field:name",
    )


def test_conservation_fields_zero_is_populated():
    contract = build_contract(expect_fields=["count"])
    assert inspect_conservation([{"count": 0}], contract) is OK


def test_conservation_fields_missing_key_is_blank():
    contract = build_contract(expect_fields=["email"])
    assert _verdict(inspect_conservation([{"name": "Ada"}], contract)) == (
        "degraded",
        "conservation:field:email",
    )


def test_conservation_fields_on_non_record_result_is_a_pass():
    contract = build_contract(expect_fields=["email"])
    assert inspect_conservation("just a string", contract) is OK


# --- inspect_conservation: relation order -------------------------------- #

def test_conservation_evaluates_smallest_relation_first():
    contract = build_contract(expect_count=2, expect_fields=["email"])
    verdict = inspect_conservation([{"email": None}], contract)
    assert _verdict(verdict) == ("degraded", "conservation:count_short")


# --- inspect_result: conservation in the pipeline ------------------------ #

def test_inspect_result_conservation_outranks_empty():
    contract = build_contract(expect_count=500)
    assert _verdict(inspect_result([], contract=contract)) == (
        "degraded",
        "conservation:count_short",
    )


def test_inspect_result_stop_reason_outranks_conservation():
    contract = build_contract(expect_count=500)
    verdict = inspect_result([], usage=_usage("length"), contract=contract)
    assert _verdict(verdict) == ("degraded", "llm_stop_reason:length")


def test_inspect_result_without_a_contract_is_unchanged():
    # D4: absence of a contract means no check — a filtering step that
    # legitimately drops rows must be untouched.
    assert inspect_result([{"id": 1}]) is OK
    assert _verdict(inspect_result([])) == ("degraded", "empty_sequence")

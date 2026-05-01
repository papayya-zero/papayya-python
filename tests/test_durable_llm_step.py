"""Tests for ``run.llm_step(label, fn)`` — the explicit LLM wrapper.

Mirrors the ``test_durable_llm_kind.py`` matrix using the new explicit
method. Behavior must be identical to ``run.step(label, fn, kind='llm')``:
shape-based usage extraction, credit-shaped error promotion, runtime
reporter handoff, schema persistence.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from papayya import CreditExhausted
from papayya.durable.run import PapayyaRun
from papayya.durable.types import DurableRunConfig


class FakeErr(Exception):
    """Shape-preserving stand-in for provider SDK exceptions."""

    def __init__(
        self,
        *,
        message: str = "",
        status_code: int | None = None,
        body: dict | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.body = body


def _make_run() -> PapayyaRun:
    return PapayyaRun(DurableRunConfig(agent="test-agent"))


def test_llm_step_extracts_openai_usage() -> None:
    run = _make_run()

    def call():
        return SimpleNamespace(
            model="gpt-4o-mini",
            usage=SimpleNamespace(prompt_tokens=100, completion_tokens=30, total_tokens=130),
            choices=[SimpleNamespace(finish_reason="stop")],
        )

    wrapped = run.llm_step("call-openai", call)
    wrapped()

    entry = run._cache["call-openai"]
    assert entry.kind == "llm"
    assert entry.llm_provider_shape == "openai"
    assert entry.llm_prompt_tokens == 100
    assert entry.llm_completion_tokens == 30
    assert entry.llm_total_tokens == 130
    assert entry.llm_model == "gpt-4o-mini"
    assert entry.llm_stop_reason == "stop"


def test_llm_step_extracts_anthropic_usage() -> None:
    run = _make_run()

    def call():
        return SimpleNamespace(
            model="claude-sonnet-4-6",
            usage=SimpleNamespace(input_tokens=200, output_tokens=50),
            stop_reason="end_turn",
        )

    run.llm_step("call-anthropic", call)()
    entry = run._cache["call-anthropic"]
    assert entry.llm_provider_shape == "anthropic"
    assert entry.llm_prompt_tokens == 200
    assert entry.llm_completion_tokens == 50
    assert entry.llm_total_tokens == 250
    assert entry.llm_stop_reason == "end_turn"


def test_llm_step_promotes_credit_error_to_credit_exhausted() -> None:
    run = _make_run()

    def call():
        raise FakeErr(
            message="insufficient_quota",
            status_code=429,
            body={"error": {"code": "insufficient_quota"}},
        )

    with pytest.raises(CreditExhausted):
        run.llm_step("call-openai", call)()


def test_llm_step_unknown_shape_completes() -> None:
    run = _make_run()

    def call():
        return SimpleNamespace(weird_field="no usage block here")

    run.llm_step("call-mystery", call)()
    entry = run._cache["call-mystery"]
    assert entry.kind == "llm"
    # Unknown shape: no token granularity, but the row still landed.
    assert entry.llm_prompt_tokens is None
    assert entry.llm_total_tokens is None


def test_llm_step_passes_item_id_through() -> None:
    run = _make_run()

    def call():
        return SimpleNamespace(
            model="gpt-4o-mini",
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )

    run.llm_step("call", call, item_id="co_42")()
    entry = run._cache["call"]
    assert entry.item_id == "co_42"


class TestKindLlmDeprecation:
    def test_kind_llm_emits_deprecation_warning(self, recwarn) -> None:
        run = _make_run()

        def call():
            return SimpleNamespace(
                model="gpt-4o-mini",
                usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            )

        run.step("call", call, kind="llm")()

        deprecations = [
            w for w in recwarn.list if issubclass(w.category, DeprecationWarning)
        ]
        assert deprecations, "expected DeprecationWarning for kind='llm'"
        msg = str(deprecations[0].message)
        assert "llm_step" in msg

    def test_kind_llm_warning_dedupes_per_label(self, recwarn) -> None:
        run = _make_run()

        def call():
            return SimpleNamespace(
                model="gpt-4o-mini",
                usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            )

        # Two calls under the same label only emit one warning.
        run.step("call", call, kind="llm")()
        run.step("call", call, kind="llm")()

        kind_warnings = [
            w
            for w in recwarn.list
            if issubclass(w.category, DeprecationWarning) and "llm_step" in str(w.message)
        ]
        assert len(kind_warnings) == 1

    def test_llm_step_does_not_emit_warning(self, recwarn) -> None:
        run = _make_run()

        def call():
            return SimpleNamespace(
                model="gpt-4o-mini",
                usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            )

        run.llm_step("call", call)()

        deprecations = [
            w
            for w in recwarn.list
            if issubclass(w.category, DeprecationWarning) and "llm_step" in str(w.message)
        ]
        assert not deprecations, "llm_step should not warn about itself"

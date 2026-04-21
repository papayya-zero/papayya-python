"""Tests for the runtime-reporter contextvar + dedupe path.

Exercises ``run.step(kind="llm")`` with a fake :class:`LlmCallReporter`
to confirm:

* An unpatched-provider call (interceptor count unchanged) triggers a
  wrapper emission with the right payload.
* A patched-provider call (interceptor count bumped during the fn) is
  NOT emitted by the wrapper — interceptor retains sole ownership.
* A raising unpatched-provider call emits an error_category report AND
  still propagates (or promotes) the exception to the caller.
* The contextvar reset invariant holds — after the wrapper finishes,
  ``get_current_reporter`` returns to its prior value.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from papayya import CreditExhausted
from papayya.durable.run import PapayyaRun
from papayya.durable.types import DurableRunConfig
from papayya.llm_extract import LlmUsage
from papayya.runtime_context import (
    get_current_reporter,
    reset_current_reporter,
    set_current_reporter,
)


class FakeErr(Exception):
    def __init__(
        self, *, message: str = "", status_code: int | None = None, body: dict | None = None
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.body = body


class FakeReporter:
    """Minimal LlmCallReporter stand-in.

    * :meth:`intercepted_call_count` returns ``self.count``, which tests
      bump manually to simulate interceptor-observed calls.
    * :meth:`report_llm_call` appends the payload to ``self.emitted`` so
      tests can assert on it.
    """

    def __init__(self) -> None:
        self.count = 0
        self.emitted: list[dict] = []

    def intercepted_call_count(self) -> int:
        return self.count

    def report_llm_call(
        self,
        *,
        label: str,
        usage: LlmUsage,
        duration_ms: int,
        error_category: str | None = None,
    ) -> None:
        self.emitted.append(
            {
                "label": label,
                "usage": usage,
                "duration_ms": duration_ms,
                "error_category": error_category,
            }
        )


def _make_run() -> PapayyaRun:
    return PapayyaRun(DurableRunConfig(agent="test-agent"))


# ---------------------------------------------------------------------------
# Unpatched provider — wrapper emits
# ---------------------------------------------------------------------------

def test_unpatched_provider_emits_through_bridge():
    reporter = FakeReporter()
    token = set_current_reporter(reporter)
    try:
        run = _make_run()

        def gemini_call():
            return SimpleNamespace(
                model_version="gemini-2.0-flash",
                usage_metadata=SimpleNamespace(
                    prompt_token_count=40, candidates_token_count=10, total_token_count=50
                ),
            )

        run.step("gemini", gemini_call, kind="llm")()
    finally:
        reset_current_reporter(token)

    assert len(reporter.emitted) == 1
    emitted = reporter.emitted[0]
    assert emitted["label"] == "gemini"
    assert emitted["error_category"] is None
    assert emitted["usage"].provider_shape == "gemini"
    assert emitted["usage"].prompt_tokens == 40
    assert emitted["usage"].completion_tokens == 10


# ---------------------------------------------------------------------------
# Patched provider — interceptor bumped count; wrapper must NOT emit
# ---------------------------------------------------------------------------

def test_patched_provider_dedupes_against_interceptor():
    reporter = FakeReporter()
    token = set_current_reporter(reporter)
    try:
        run = _make_run()

        def openai_call():
            # Simulate the interceptor recording a step while the fn runs.
            reporter.count += 1
            return SimpleNamespace(
                model="gpt-4o-mini",
                usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            )

        run.step("openai", openai_call, kind="llm")()
    finally:
        reset_current_reporter(token)

    # Interceptor saw the call → wrapper must skip emission.
    assert reporter.emitted == []


# ---------------------------------------------------------------------------
# Error paths — emit error, still promote credit exceptions
# ---------------------------------------------------------------------------

def test_unpatched_credit_error_emits_and_promotes():
    reporter = FakeReporter()
    token = set_current_reporter(reporter)
    try:
        run = _make_run()

        def gemini_402():
            raise FakeErr(status_code=402, message="Payment required")

        with pytest.raises(CreditExhausted):
            run.step("gemini-402", gemini_402, kind="llm")()
    finally:
        reset_current_reporter(token)

    assert len(reporter.emitted) == 1
    assert reporter.emitted[0]["error_category"] == "credit"
    assert reporter.emitted[0]["label"] == "gemini-402"


def test_patched_credit_error_skips_wrapper_emit():
    # Simulate: interceptor caught the error and already emitted a failed
    # step (its count incremented). Wrapper sees the rethrown exception
    # but must NOT emit a duplicate.
    reporter = FakeReporter()
    token = set_current_reporter(reporter)
    try:
        run = _make_run()

        def openai_credit_error():
            reporter.count += 1
            raise CreditExhausted("already reported by interceptor")

        with pytest.raises(CreditExhausted):
            run.step("openai-credit", openai_credit_error, kind="llm")()
    finally:
        reset_current_reporter(token)

    assert reporter.emitted == []


def test_unpatched_transient_error_emits_then_propagates():
    reporter = FakeReporter()
    token = set_current_reporter(reporter)
    try:
        run = _make_run()

        def gemini_500():
            raise FakeErr(status_code=500, message="Server error")

        with pytest.raises(FakeErr):
            run.step("gemini-500", gemini_500, kind="llm")()
    finally:
        reset_current_reporter(token)

    assert len(reporter.emitted) == 1
    assert reporter.emitted[0]["error_category"] == "transient"


# ---------------------------------------------------------------------------
# Non-LLM steps untouched
# ---------------------------------------------------------------------------

def test_non_llm_step_does_not_consult_reporter():
    reporter = FakeReporter()
    token = set_current_reporter(reporter)
    try:
        run = _make_run()

        def plain():
            return {"not": "an llm call"}

        run.step("plain", plain)()
    finally:
        reset_current_reporter(token)

    # Non-LLM steps MUST NOT emit. This guards against accidental
    # coupling between the durable primitive and the runtime channel.
    assert reporter.emitted == []


# ---------------------------------------------------------------------------
# Context isolation
# ---------------------------------------------------------------------------

def test_contextvar_defaults_to_none():
    # Fresh check — in a clean context, the reporter is None.
    assert get_current_reporter() is None


def test_contextvar_resets_after_with_block():
    assert get_current_reporter() is None
    reporter = FakeReporter()
    token = set_current_reporter(reporter)
    try:
        assert get_current_reporter() is reporter
    finally:
        reset_current_reporter(token)
    assert get_current_reporter() is None


def test_wrapper_without_contextvar_is_noop_on_emission():
    # No reporter installed → wrapper still extracts and records to the
    # entry, but nothing emits anywhere. Local mode behavior.
    run = _make_run()

    def call():
        return SimpleNamespace(
            model="gpt-4o-mini",
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=2, total_tokens=3),
        )

    run.step("no-reporter", call, kind="llm")()
    entry = run._cache["no-reporter"]
    assert entry.kind == "llm"
    assert entry.llm_total_tokens == 3  # local capture still works

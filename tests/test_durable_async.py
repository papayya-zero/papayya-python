"""Tests for async ``run.step`` / ``run.llm_step`` and per-call token dedupe.

Covers:

* The async wrapper mirrors the sync wrapper through ``await``.
* Cache hit on replay returns a plain value (not a coroutine).
* Concurrent ``asyncio.gather`` fan-out emits one row per call with no
  cross-coroutine token leakage — the property the legacy global
  ``intercepted_call_count`` snapshot could not guarantee.
* Exception classification (CreditExhausted promotion) on the async path.
* Sync regression: passing a sync fn still yields a sync wrapper.
* ``functools.wraps``-decorated coroutines are correctly detected as async.
* ``asyncio.CancelledError`` doesn't leak the per-call token on the shim
  side.
"""

from __future__ import annotations

import asyncio
import functools
import inspect
from contextvars import ContextVar
from types import SimpleNamespace
from typing import Any

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


def _make_run() -> PapayyaRun:
    return PapayyaRun(DurableRunConfig(agent="async-test-agent"))


# ---------------------------------------------------------------------------
# Reporter stand-ins
# ---------------------------------------------------------------------------


class FakeErr(Exception):
    """Provider-shaped exception for credit/transient classification tests."""

    def __init__(
        self, *, message: str = "", status_code: int | None = None, body: dict | None = None
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.body = body


class _LegacyCounterReporter:
    """Old-shim shape — only exposes ``intercepted_call_count``.

    Lets us prove the SDK still works against pre-token shims by
    falling through the legacy pre/post snapshot path.
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
        self.emitted.append({"label": label, "error_category": error_category})


class _TokenReporter:
    """New-shim shape with per-call token dedupe.

    Mirrors the shim's ``ShimLlmCallReporter`` semantics in pure
    Python — uses a ``ContextVar`` to hold the active token so each
    ``asyncio.gather`` task sees its own copy. ``intercept(token)``
    is the test-only knob that simulates "the interceptor recorded
    this call" by marking the token as emitted.
    """

    def __init__(self) -> None:
        self.emitted: list[dict] = []
        self._tokens: dict[int, bool] = {}
        self._cv: ContextVar[object | None] = ContextVar(
            "test_active_token", default=None
        )
        self._cv_tokens: dict[int, Any] = {}

    def begin_call(self, label: str) -> object:
        token = object()
        self._tokens[id(token)] = False
        self._cv_tokens[id(token)] = self._cv.set(token)
        return token

    def was_emitted_for(self, token: object) -> bool:
        cv_token = self._cv_tokens.pop(id(token), None)
        if cv_token is not None:
            try:
                self._cv.reset(cv_token)
            except (ValueError, LookupError):
                pass
        return self._tokens.pop(id(token), False)

    def intercepted_call_count(self) -> int:
        # Legacy method left in place; new path doesn't touch it.
        return 0

    def report_llm_call(
        self,
        *,
        label: str,
        usage: LlmUsage,
        duration_ms: int,
        error_category: str | None = None,
    ) -> None:
        self.emitted.append({"label": label, "error_category": error_category})

    def intercept_active(self) -> None:
        """Test helper: mark the currently-active token as emitted.

        Mirrors what the real ``UsageTracker`` observer does after
        ``record()``. Called from inside the wrapped coroutine to
        simulate "the interceptor patched the provider this call uses".
        """
        active = self._cv.get()
        if active is not None and id(active) in self._tokens:
            self._tokens[id(active)] = True


# ---------------------------------------------------------------------------
# Phase 1 — async wrapper basics
# ---------------------------------------------------------------------------


async def test_async_step_basic_pipeline():
    run = _make_run()

    async def fetch():
        await asyncio.sleep(0)
        return {"got": "value"}

    wrapped = run.step("fetch", fetch)
    assert inspect.iscoroutinefunction(wrapped)

    result = await wrapped()
    assert result == {"got": "value"}
    assert "fetch" in run._cache
    assert run._cache["fetch"].result == {"got": "value"}


async def test_async_step_replay_returns_plain_value_not_coroutine():
    run = _make_run()

    calls: list[int] = []

    async def fetch():
        calls.append(1)
        return "first-call"

    wrapped = run.step("fetch", fetch)
    first = await wrapped()
    second = await wrapped()

    assert first == "first-call"
    assert second == "first-call"
    # The cache short-circuit must yield the plain value through the
    # coroutine — never a nested coroutine that the caller would have
    # to await twice.
    assert not inspect.isawaitable(second)
    assert calls == [1]


async def test_async_two_step_sequence_caches_on_replay():
    run = _make_run()
    counts = {"a": 0, "b": 0}

    async def step_a():
        counts["a"] += 1
        return "A"

    async def step_b():
        counts["b"] += 1
        return "B"

    wrap_a = run.step("a", step_a)
    wrap_b = run.step("b", step_b)
    assert await wrap_a() == "A"
    assert await wrap_b() == "B"
    assert await wrap_a() == "A"  # cached
    assert await wrap_b() == "B"  # cached
    assert counts == {"a": 1, "b": 1}


async def test_iscoroutinefunction_detection_handles_functools_wraps():
    run = _make_run()

    async def underlying():
        return "ok"

    @functools.wraps(underlying)
    async def decorated():
        return await underlying()

    wrapped = run.step("decorated", decorated)
    # ``inspect.iscoroutinefunction`` follows ``__wrapped__`` from
    # functools.wraps — ``asyncio.iscoroutinefunction`` does not. The
    # wrapper itself is defined with ``async def`` so this also works.
    assert inspect.iscoroutinefunction(wrapped)
    assert await wrapped() == "ok"


def test_sync_fn_still_returns_sync_wrapper():
    run = _make_run()

    def plain():
        return 42

    wrapped = run.step("plain", plain)
    assert not inspect.iscoroutinefunction(wrapped)
    assert wrapped() == 42


# ---------------------------------------------------------------------------
# Phase 3 — per-call token correctness under fan-out
# ---------------------------------------------------------------------------


async def test_async_gather_fan_out_emits_distinct_steps():
    """5 parallel ``run.llm_step`` calls under ``asyncio.gather``.

    No interceptor in play — every call should produce one wrapper
    emission, with the right label, no cross-coroutine bleed.
    """
    reporter = _TokenReporter()
    rtoken = set_current_reporter(reporter)
    try:

        async def make_call(label: str):
            run = PapayyaRun(DurableRunConfig(agent=f"agent-{label}"))

            async def inner():
                # Yield to the loop so the scheduler interleaves all
                # five coroutines — proves token isolation, not just
                # serial execution.
                await asyncio.sleep(0)
                return SimpleNamespace(
                    model="test-model",
                    usage=SimpleNamespace(
                        prompt_tokens=10,
                        completion_tokens=5,
                        total_tokens=15,
                    ),
                )

            wrapped = run.llm_step(label, inner)
            await wrapped()

        await asyncio.gather(*(make_call(f"call_{i}") for i in range(5)))
    finally:
        reset_current_reporter(rtoken)

    labels = sorted(e["label"] for e in reporter.emitted)
    assert labels == [f"call_{i}" for i in range(5)]
    assert all(e["error_category"] is None for e in reporter.emitted)


async def test_async_gather_with_partial_intercept():
    """Load-bearing test for token correctness.

    Five coroutines fan out under gather; three of them mark their
    own active token as "intercepted" mid-call (simulating the real
    interceptor patching that provider). The SDK must emit only the
    other two — and crucially must attribute "intercepted" to the
    RIGHT coroutine. The legacy global counter would race here and
    drop the wrong rows.
    """
    reporter = _TokenReporter()
    rtoken = set_current_reporter(reporter)
    intercepted_labels = {"call_1", "call_2", "call_4"}
    try:

        async def make_call(label: str):
            run = PapayyaRun(DurableRunConfig(agent=f"agent-{label}"))

            async def inner():
                # Stagger the awaits so that token A's "intercept"
                # observation happens while tokens B/C/D/E are also
                # in flight on this event loop.
                await asyncio.sleep(0)
                if label in intercepted_labels:
                    reporter.intercept_active()
                await asyncio.sleep(0)
                return SimpleNamespace(
                    model="test-model",
                    usage=SimpleNamespace(
                        prompt_tokens=1,
                        completion_tokens=1,
                        total_tokens=2,
                    ),
                )

            wrapped = run.llm_step(label, inner)
            await wrapped()

        await asyncio.gather(*(make_call(f"call_{i}") for i in range(5)))
    finally:
        reset_current_reporter(rtoken)

    emitted_labels = sorted(e["label"] for e in reporter.emitted)
    expected = sorted({f"call_{i}" for i in range(5)} - intercepted_labels)
    assert emitted_labels == expected


# ---------------------------------------------------------------------------
# Exception path on async wrapper
# ---------------------------------------------------------------------------


async def test_async_credit_error_promotes_to_credit_exhausted():
    reporter = _TokenReporter()
    rtoken = set_current_reporter(reporter)
    try:
        run = _make_run()

        async def call_402():
            await asyncio.sleep(0)
            raise FakeErr(status_code=402, message="Payment required")

        wrapped = run.llm_step("call-402", call_402)
        with pytest.raises(CreditExhausted):
            await wrapped()
    finally:
        reset_current_reporter(rtoken)

    assert len(reporter.emitted) == 1
    assert reporter.emitted[0]["error_category"] == "credit"
    assert reporter.emitted[0]["label"] == "call-402"


async def test_async_intercepted_credit_error_skips_wrapper_emit():
    """Interceptor saw and recorded the failure → wrapper must not
    emit a duplicate. CreditExhausted still propagates."""
    reporter = _TokenReporter()
    rtoken = set_current_reporter(reporter)
    try:
        run = _make_run()

        async def call_credit():
            reporter.intercept_active()
            raise CreditExhausted("already reported by interceptor")

        wrapped = run.llm_step("call-credit", call_credit)
        with pytest.raises(CreditExhausted):
            await wrapped()
    finally:
        reset_current_reporter(rtoken)

    assert reporter.emitted == []


async def test_async_non_llm_step_does_not_consult_reporter():
    reporter = _TokenReporter()
    rtoken = set_current_reporter(reporter)
    try:
        run = _make_run()

        async def plain():
            return {"ok": True}

        await run.step("plain", plain)()
    finally:
        reset_current_reporter(rtoken)

    assert reporter.emitted == []


# ---------------------------------------------------------------------------
# Cancellation safety
# ---------------------------------------------------------------------------


async def test_async_cancellation_cleans_up_token():
    """Cancelling a wrapped coroutine mid-await must reset the per-call
    token on the shim side, otherwise the contextvar / dict entries
    leak across calls.

    ``asyncio.CancelledError`` extends ``BaseException`` so the
    wrapper's ``except Exception`` doesn't catch it — the cleanup has
    to live in ``finally``.
    """
    reporter = _TokenReporter()
    rtoken = set_current_reporter(reporter)
    try:
        run = _make_run()

        async def slow():
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                # Re-raise so the wrapper's finally still runs the
                # ensure-cleanup path, but the SDK can't muffle the
                # cancellation.
                raise

        wrapped = run.llm_step("slow", slow)
        task = asyncio.create_task(wrapped())
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        reset_current_reporter(rtoken)

    # The reporter's bookkeeping should be empty after cancellation:
    # the wrapper's finally block called was_emitted_for(token), which
    # popped both internal dicts.
    assert reporter._tokens == {}
    assert reporter._cv_tokens == {}


# ---------------------------------------------------------------------------
# Backward-compat: legacy counter-only reporters still work
# ---------------------------------------------------------------------------


async def test_async_works_against_legacy_counter_reporter():
    """SDK falls back to ``intercepted_call_count`` when the reporter
    has no ``begin_call`` (simulates an old shim build)."""
    reporter = _LegacyCounterReporter()
    rtoken = set_current_reporter(reporter)
    try:
        run = _make_run()

        async def call():
            await asyncio.sleep(0)
            return SimpleNamespace(
                model="test-model",
                usage=SimpleNamespace(
                    prompt_tokens=1, completion_tokens=2, total_tokens=3
                ),
            )

        await run.llm_step("legacy", call)()
    finally:
        reset_current_reporter(rtoken)

    assert len(reporter.emitted) == 1
    assert reporter.emitted[0]["label"] == "legacy"


async def test_async_legacy_counter_intercepted_skips_wrapper_emit():
    reporter = _LegacyCounterReporter()
    rtoken = set_current_reporter(reporter)
    try:
        run = _make_run()

        async def call():
            await asyncio.sleep(0)
            reporter.count += 1  # simulate interceptor recording mid-call
            return SimpleNamespace(
                model="test-model",
                usage=SimpleNamespace(
                    prompt_tokens=1, completion_tokens=2, total_tokens=3
                ),
            )

        await run.llm_step("legacy-intercepted", call)()
    finally:
        reset_current_reporter(rtoken)

    assert reporter.emitted == []

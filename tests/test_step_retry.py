"""Plan 58 R — a step that raises gets retried.

Walk 57's DOC-2001 failed because an OCR sidecar returned 503 on page 17.
Temporal and Inngest would both have retried that step and completed the
document with no human involved. Papayya failed the whole run, parked it in
triage until an operator typed `papayya replay`, and then charged 40 page
executions to fix one page.

`run.step` called the customer's function exactly once and re-raised. There had
never been a retry policy — ADR 0002 deferred "retry policies" as post-launch
UX — while `classify_provider_error`'s docstring had been promising
"transient → retry with backoff" for several plans.
"""

from __future__ import annotations

import asyncio

import pytest

from papayya import CreditExhausted, NonRetriable
from papayya.durable import run as run_module
from papayya.durable.run import PapayyaRun
from papayya.durable.types import DurableRunConfig


# The real ladder, captured at import before conftest's autouse fixture zeroes
# it for the rest of the suite. This file is the one that asserts it.
_REAL_BASE = run_module.STEP_RETRY_BASE_SECONDS
_REAL_MAX = run_module.STEP_RETRY_MAX_SECONDS


@pytest.fixture(autouse=True)
def _no_real_sleeping(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Restore the real ladder, then record it instead of serving it.

    Returned as a fixture value so a test can assert the LADDER, which is the
    part a customer feels — the retry existing is worth little if it gives up
    while the outage is still going, which is exactly what the first cut did.

    Patches `run._sleep` / `run._async_sleep`, the module-level seams, and NOT
    `time.sleep`: `run.py` does `import time as _time`, so patching through it
    reaches every other test in the process.
    """
    monkeypatch.setattr(run_module, "STEP_RETRY_BASE_SECONDS", _REAL_BASE)
    monkeypatch.setattr(run_module, "STEP_RETRY_MAX_SECONDS", _REAL_MAX)

    waits: list[float] = []
    monkeypatch.setattr(run_module, "_sleep", waits.append)

    async def _fake_async_sleep(d: float) -> None:
        waits.append(d)

    monkeypatch.setattr(run_module, "_async_sleep", _fake_async_sleep)
    return waits


def _run() -> PapayyaRun:
    return PapayyaRun(DurableRunConfig(agent="docproc"))


class OcrUnavailable(RuntimeError):
    """Walk 57's own exception, verbatim: a bare RuntimeError with 503 in the
    message and no status_code attribute."""


class HttpErr(Exception):
    def __init__(self, status_code: int) -> None:
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


def _flaky(fails: int, exc_factory=lambda: OcrUnavailable(
    "ocr sidecar returned 503 for page 17 of DOC-2001")):
    """A function that raises `fails` times and then succeeds."""
    state = {"calls": 0}

    def fn():
        state["calls"] += 1
        if state["calls"] <= fails:
            raise exc_factory()
        return {"page": 17, "text": "page 17 body text"}

    fn.state = state  # type: ignore[attr-defined]
    return fn


# ── The headline ──────────────────────────────────────────────────────────

def test_the_walk_57_failure_now_recovers_without_a_human() -> None:
    """THE case. The sidecar is down for one attempt and back for the next."""
    run = _run()
    fn = _flaky(1)

    result = run.step("read-photo", fn)()

    assert result["page"] == 17
    assert fn.state["calls"] == 2, "the step must have run again"
    assert run._cache["read-photo"].result == result


def test_a_step_that_never_recovers_still_fails_and_stops(
    _no_real_sleeping: list[float],
) -> None:
    run = _run()
    fn = _flaky(99)

    with pytest.raises(OcrUnavailable):
        run.step("read-photo", fn)()

    # 1 first attempt + DEFAULT_STEP_RETRIES extras, and not one more.
    assert fn.state["calls"] == 1 + run_module.DEFAULT_STEP_RETRIES
    assert _no_real_sleeping == [1.0, 2.0, 4.0, 8.0], "exponential from a 1s base, capped at 8s"


def test_a_clean_step_is_called_exactly_once() -> None:
    run = _run()
    calls = []
    run.step("classify", lambda: calls.append(1) or "photo_report")()
    assert len(calls) == 1


# ── What stops a retry, and what must not ─────────────────────────────────

def test_retries_zero_opts_out(_no_real_sleeping: list[float]) -> None:
    """For a step whose side effect must happen at most once."""
    run = _run()
    fn = _flaky(99)

    with pytest.raises(OcrUnavailable):
        run.step("charge-card", fn, retries=0)()

    assert fn.state["calls"] == 1
    assert _no_real_sleeping == []


def test_non_retriable_stops_immediately(_no_real_sleeping: list[float]) -> None:
    run = _run()
    fn = _flaky(99, exc_factory=lambda: NonRetriable("unknown document kind"))

    with pytest.raises(NonRetriable):
        run.step("classify", fn)()

    assert fn.state["calls"] == 1
    assert _no_real_sleeping == []


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
def test_an_identified_permanent_status_stops(status: int) -> None:
    run = _run()
    fn = _flaky(99, exc_factory=lambda: HttpErr(status))

    with pytest.raises(HttpErr):
        run.step("call", fn)()

    assert fn.state["calls"] == 1, f"{status} is still {status} in four seconds"


def test_a_rate_limit_is_retried() -> None:
    """429 is a client-error status and the single most retryable thing a
    provider produces. It must not be swept up with 4xx."""
    run = _run()
    fn = _flaky(1, exc_factory=lambda: HttpErr(429))

    run.step("call", fn)()
    assert fn.state["calls"] == 2


def test_an_unrecognised_exception_is_retried() -> None:
    """UNKNOWN IS NOT EVIDENCE, and this is the test that would have caught the
    design plan 58 shipped first.

    `classify_provider_error` returns 'permanent' as its FALLTHROUGH, so it
    cannot distinguish "we saw a 401" from "we have never seen this exception".
    Gating retries on that verdict means retrying only exceptions raised by an
    LLM SDK we already recognise — which excludes every exception a customer
    writes, including walk 57's.
    """
    from papayya.classify import classify_provider_error

    boom = OcrUnavailable("ocr sidecar returned 503 for page 17 of DOC-2001")
    assert classify_provider_error(boom) == "permanent", (
        "premise: the classifier calls the walk's own exception permanent"
    )

    run = _run()
    fn = _flaky(1)
    run.step("read-photo", fn)()
    assert fn.state["calls"] == 2, "and it is retried anyway"


def test_credit_exhaustion_pauses_instead_of_retrying(
    _no_real_sleeping: list[float],
) -> None:
    """Retrying cannot fund an account. The promotion to CreditExhausted runs
    BEFORE the retry decision so the pause covenant wins."""
    run = _run()
    fn = _flaky(99, exc_factory=lambda: CreditExhausted("out of credits"))

    with pytest.raises(CreditExhausted):
        run.step("draft", fn)()

    assert fn.state["calls"] == 1
    assert _no_real_sleeping == []


# ── The two contracts the loop must not disturb ───────────────────────────

def test_a_retry_does_not_advance_the_attempt_counter() -> None:
    """`attempt` counts re-executions ACROSS PASSES — a resume, a replay, a
    re-lease — and is seeded from STORED rows (plan 41 R3). A failed in-process
    attempt stores nothing, so it must not advance it, or the number an
    operator reads stops meaning what it was built to mean.

    Plan 58's first draft said "N executions, N checkpoint rows, attempt 1..N".
    That is wrong: `_next_execution` is called from `_post_call_success`, so a
    failed execution has never written a row.
    """
    run = _run()
    run.step("read-photo", _flaky(2))()

    entry = run._cache["read-photo"]
    assert entry.attempt == 1, "three executions, one recorded, attempt 1"
    assert run._attempts["read-photo"] == 1


def test_a_retry_reuses_the_provider_idempotency_key() -> None:
    """`run.idempotency_key`'s own docstring settles this: "a crash before the
    checkpoint persisted leaves nothing recorded, so the resumed execution
    sends the SAME key and the provider replays its stored response. That is
    the point of the key — it is what stops the retry billing your provider
    account twice."

    A retry after a raise is exactly that case: if the provider had already
    done the work when the connection dropped, the same key gets the stored
    answer back instead of paying for it again. Plan 58's first draft said the
    key must advance per attempt; it must not.
    """
    run = _run()
    keys: list[str] = []

    def fn():
        # Called from INSIDE the step, so the label component is already
        # `draft#2` — `_pre_call` consumed the occurrence before fn ran, which
        # is the caveat idempotency_key's own docstring gives ("call it
        # immediately before the step it protects"). The component under test
        # is the trailing ATTEMPT number, and it is the one that must not move.
        keys.append(run.idempotency_key("draft"))
        if len(keys) < 3:
            raise OcrUnavailable("503")
        return "ok"

    run.step("draft", fn)()

    assert len(keys) == 3
    assert {k.rsplit(":", 1)[1] for k in keys} == {"1"}, (
        f"the attempt must not advance across in-process retries, got {keys}"
    )
    assert len(set(keys)) == 1, f"one key across all attempts, got {set(keys)}"


def test_a_retry_consumes_one_occurrence_of_a_looped_label() -> None:
    """The `label#N` occurrence counter is POSITIONAL and is consumed in
    `_pre_call`, which runs once per step call. A retry must not consume a
    second slot, or every later iteration's key shifts and a resumed run
    hands iteration 4 the result of iteration 3.
    """
    run = _run()
    for i in range(3):
        run.step("read-photo", _flaky(1 if i == 1 else 0))()

    assert sorted(run._cache) == ["read-photo", "read-photo#2", "read-photo#3"]


# ── Async parity ──────────────────────────────────────────────────────────

def test_async_steps_retry_too() -> None:
    run = _run()
    state = {"calls": 0}

    async def fn():
        state["calls"] += 1
        if state["calls"] == 1:
            raise OcrUnavailable("503")
        return "ok"

    assert asyncio.run(run.step("read-photo", fn)()) == "ok"
    assert state["calls"] == 2


def test_async_backoff_does_not_block_the_event_loop(
    _no_real_sleeping: list[float], monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`_time.sleep` in a coroutine stalls every other coroutine in the same
    @agent body, including the ones that are working."""
    run = _run()
    slept_sync: list[float] = []
    monkeypatch.setattr(run_module, "_sleep", slept_sync.append)

    async def fn():
        raise OcrUnavailable("503")

    with pytest.raises(OcrUnavailable):
        asyncio.run(run.step("read-photo", fn, retries=1)())

    assert _no_real_sleeping == [1.0]
    assert slept_sync == [], "the async path must not call the blocking sleep"


# ── R4: the retry reaches the RECORD, not just the log ────────────────────

def test_a_retried_step_records_what_it_cost() -> None:
    """A step that failed four times and succeeded on the fifth used to write a
    checkpoint byte-identical to one that worked immediately — same
    outcome_status, same attempt, same everything, because a failed in-process
    attempt writes no row at all. The platform spent five executions of the
    customer's function and every surface reported one.
    """
    run = _run()
    run.step("read-photo", _flaky(3), item_id="DOC-2001#photo-17")()

    entry = run._cache["read-photo"]
    assert entry.retry_count == 3
    assert entry.retry_reason is not None
    assert "OcrUnavailable" in entry.retry_reason
    assert "page 17" in entry.retry_reason, "the diagnostic, not just the count"
    # Still ONE row, still attempt 1 — R4 records the cost, it does not change
    # what `attempt` means.
    assert entry.attempt == 1


def test_a_clean_step_records_zero_not_null() -> None:
    """0 is a fact — "ran once, first time" — and a consumer that has to tell
    an absent field from a zero is a consumer that will get it wrong."""
    run = _run()
    run.step("classify", lambda: "photo_report")()

    entry = run._cache["classify"]
    assert entry.retry_count == 0
    assert entry.retry_reason is None


def test_the_last_failure_is_the_one_recorded() -> None:
    """A step that hit a 429 and then a 503 is better described by the failure
    that was still true when it finally worked."""
    run = _run()
    seq = [HttpErr(429), OcrUnavailable("ocr sidecar returned 503 for page 17")]
    state = {"n": 0}

    def fn():
        if state["n"] < len(seq):
            exc = seq[state["n"]]
            state["n"] += 1
            raise exc
        return "ok"

    run.step("read-photo", fn)()

    entry = run._cache["read-photo"]
    assert entry.retry_count == 2
    assert "503" in (entry.retry_reason or ""), entry.retry_reason


def test_a_second_loop_iteration_does_not_inherit_the_first_ones_tally() -> None:
    """The tally lives on the per-call ctx, not on a closure the wrapper reuses
    across invocations. `run.step("read-photo", ...)` called forty times is
    forty independent tallies."""
    run = _run()
    run.step("read-photo", _flaky(2))()   # retried twice
    run.step("read-photo", _flaky(0))()   # clean

    assert run._cache["read-photo"].retry_count == 2
    assert run._cache["read-photo#2"].retry_count == 0


def test_the_retry_reason_is_bounded() -> None:
    """A provider that returns a stack trace in its message must not push a row
    past the column bound."""
    run = _run()
    huge = "x" * 50_000
    run.step("call", _flaky(1, exc_factory=lambda: OcrUnavailable(huge)))()

    reason = run._cache["call"].retry_reason or ""
    assert len(reason) <= run_module._RETRY_REASON_MAX

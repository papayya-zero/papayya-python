"""Tests for ``run.step(kind="llm")`` behavior.

Covers the three new responsibilities the wrapper picks up when the
``kind`` hint is set:

1. Shape-based extraction of usage metadata from the returned response.
2. Classification of raised provider exceptions — credit-shaped ones are
   promoted to ``CreditExhausted`` so the runtime pauses.
3. Unknown / unrecognized shapes still complete and still record a step
   entry (with ``provider_shape="unknown"``).

No LLM SDK imports; exceptions use a duck-typed ``FakeErr``.
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


# ---------------------------------------------------------------------------
# Success path: usage extraction
# ---------------------------------------------------------------------------

def test_llm_kind_extracts_openai_usage():
    run = _make_run()

    def call():
        return SimpleNamespace(
            model="gpt-4o-mini",
            usage=SimpleNamespace(prompt_tokens=100, completion_tokens=30, total_tokens=130),
            choices=[SimpleNamespace(finish_reason="stop")],
        )

    wrapped = run.step("call-openai", call, kind="llm")
    wrapped()

    entry = run._cache["call-openai"]
    assert entry.kind == "llm"
    assert entry.llm_provider_shape == "openai"
    assert entry.llm_prompt_tokens == 100
    assert entry.llm_completion_tokens == 30
    assert entry.llm_total_tokens == 130
    assert entry.llm_model == "gpt-4o-mini"
    assert entry.llm_stop_reason == "stop"


def test_llm_kind_extracts_anthropic_usage():
    run = _make_run()

    def call():
        return SimpleNamespace(
            model="claude-sonnet-4-6",
            usage=SimpleNamespace(input_tokens=200, output_tokens=50),
            stop_reason="end_turn",
        )

    run.step("call-anthropic", call, kind="llm")()
    entry = run._cache["call-anthropic"]
    assert entry.llm_provider_shape == "anthropic"
    assert entry.llm_prompt_tokens == 200
    assert entry.llm_completion_tokens == 50
    assert entry.llm_total_tokens == 250
    assert entry.llm_stop_reason == "end_turn"


def test_llm_kind_extracts_gemini_usage():
    run = _make_run()

    def call():
        return SimpleNamespace(
            model_version="gemini-2.0-flash",
            usage_metadata=SimpleNamespace(
                prompt_token_count=40, candidates_token_count=10, total_token_count=50
            ),
            candidates=[SimpleNamespace(finish_reason="STOP")],
        )

    run.step("call-gemini", call, kind="llm")()
    entry = run._cache["call-gemini"]
    assert entry.llm_provider_shape == "gemini"
    assert entry.llm_prompt_tokens == 40
    assert entry.llm_completion_tokens == 10
    assert entry.llm_model == "gemini-2.0-flash"


def test_llm_kind_unknown_shape_still_completes():
    run = _make_run()

    def call():
        return "just a string, no usage fields"

    result = run.step("call-unknown", call, kind="llm")()
    entry = run._cache["call-unknown"]
    assert result == "just a string, no usage fields"
    assert entry.kind == "llm"
    assert entry.llm_provider_shape == "unknown"
    assert entry.llm_prompt_tokens is None


# ---------------------------------------------------------------------------
# Error path: classification + CreditExhausted promotion
# ---------------------------------------------------------------------------

def test_llm_kind_credit_shaped_429_promotes_to_credit_exhausted():
    run = _make_run()

    def call():
        raise FakeErr(
            status_code=429,
            body={"error": {"type": "insufficient_quota"}},
            message="You exceeded your current quota",
        )

    wrapped = run.step("call-oom", call, kind="llm")
    with pytest.raises(CreditExhausted):
        wrapped()


def test_llm_kind_402_promotes_to_credit_exhausted():
    run = _make_run()

    def call():
        raise FakeErr(status_code=402, message="Payment required")

    with pytest.raises(CreditExhausted):
        run.step("call-402", call, kind="llm")()


def test_llm_kind_plain_429_does_not_promote():
    run = _make_run()

    def call():
        raise FakeErr(
            status_code=429,
            body={"error": {"type": "rate_limit_exceeded"}},
            message="Rate limit exceeded",
        )

    wrapped = run.step("call-rate", call, kind="llm")
    # Transient error — caller sees the original exception, not CreditExhausted.
    with pytest.raises(FakeErr):
        wrapped()


def test_llm_kind_permanent_error_propagates_unchanged():
    run = _make_run()

    def call():
        raise FakeErr(status_code=401, message="Invalid API key")

    with pytest.raises(FakeErr):
        run.step("call-auth", call, kind="llm")()


def test_existing_credit_exhausted_not_rewrapped():
    run = _make_run()

    def call():
        raise CreditExhausted("already-classified")

    with pytest.raises(CreditExhausted) as excinfo:
        run.step("call-already-credit", call, kind="llm")()
    assert "already-classified" in str(excinfo.value)
    # Not double-wrapped.
    assert not isinstance(excinfo.value.__cause__, CreditExhausted)


# ---------------------------------------------------------------------------
# No-kind behavior: untouched by extractor / classifier
# ---------------------------------------------------------------------------

def test_no_kind_skips_extraction():
    run = _make_run()

    def call():
        return SimpleNamespace(usage=SimpleNamespace(prompt_tokens=100, completion_tokens=30))

    run.step("plain-step", call)()
    entry = run._cache["plain-step"]
    assert entry.kind is None
    assert entry.llm_prompt_tokens is None
    assert entry.llm_provider_shape is None


def test_no_kind_does_not_reclassify_exceptions():
    run = _make_run()

    def call():
        raise FakeErr(status_code=402, message="Payment required")

    # Without kind="llm", the 402 is NOT promoted to CreditExhausted.
    wrapped = run.step("plain-fail", call)
    with pytest.raises(FakeErr):
        wrapped()


# ---------------------------------------------------------------------------
# Replay: cached result returned, extractor/classifier not re-run
# ---------------------------------------------------------------------------

def test_llm_kind_replay_returns_cached_result():
    run = _make_run()
    call_count = {"n": 0}

    def call():
        call_count["n"] += 1
        return SimpleNamespace(
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            model="m",
        )

    wrapped = run.step("cached", call, kind="llm")
    first = wrapped()
    second = wrapped()
    assert call_count["n"] == 1
    assert first is second




# ---------------------------------------------------------------------------
# SQLite round-trip — regression for the serialization boundary bug where
# raw json.dumps(entry.result) crashed on any non-JSON-native provider
# response (SimpleNamespace, Pydantic, dataclass, etc.). Exercises the
# complete save → load cycle to ensure the full BYOF kind="llm" surface
# is durable against realistic LLM SDK return types.
# ---------------------------------------------------------------------------

def test_llm_kind_persists_to_sqlite_and_survives_reload(tmp_path):
    from papayya.durable.sqlite_store import SQLiteStore

    db_path = tmp_path / "local.db"
    store = SQLiteStore(str(db_path))
    run = PapayyaRun(DurableRunConfig(agent="gemini-persist-test", store=store))

    def call():
        return SimpleNamespace(
            model_version="gemini-2.0-flash",
            usage_metadata=SimpleNamespace(
                prompt_token_count=40,
                candidates_token_count=10,
                total_token_count=50,
            ),
            candidates=[SimpleNamespace(finish_reason="STOP")],
        )

    run.step("gemini-call", call, kind="llm")()
    run.complete("ok")

    # Reload from a fresh store pointing at the same file — proves the
    # write landed and the stored JSON survives json.loads on replay.
    reloaded = SQLiteStore(str(db_path))
    checkpoint = reloaded.load(run.run_id)
    assert checkpoint is not None
    assert len(checkpoint.tasks) == 1
    task = checkpoint.tasks[0]
    assert task.label == "gemini-call"
    assert task.kind == "llm"
    assert task.llm_provider_shape == "gemini"
    assert task.llm_total_tokens == 50
    assert task.llm_model == "gemini-2.0-flash"
    # The raw SimpleNamespace response degraded through the ladder to a
    # __dict__ representation; the extracted llm_* columns preserve the
    # signal we actually care about.
    assert isinstance(task.result, dict)
    assert task.result["model_version"] == "gemini-2.0-flash"

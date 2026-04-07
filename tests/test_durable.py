"""Tests for the durable execution wrapper."""

import tempfile

import pytest

from papayya.durable import (
    BudgetExceededError,
    DurableRunConfig,
    FileStore,
    MemoryStore,
    PapayyaRun,
    papayya,
)


# --------------------------------------------------------------------------- #
#  Helpers                                                                     #
# --------------------------------------------------------------------------- #


def search_web(query: str) -> list[str]:
    return [f"result for: {query}"]


def summarize(items: list[str]) -> str:
    return f"summary of {len(items)} items"


# --------------------------------------------------------------------------- #
#  PapayyaRun — basic execution                                                  #
# --------------------------------------------------------------------------- #


class TestPapayyaRun:
    def test_executes_tasks_and_returns_results(self) -> None:
        run = PapayyaRun(DurableRunConfig(agent="test-agent", store=MemoryStore()))

        search = run.task("search", search_web)
        sum_ = run.task("summarize", summarize)

        results = search("quantum computing")
        assert results == ["result for: quantum computing"]

        summary = sum_(results)
        assert summary == "summary of 1 items"

        result = run.complete(summary)
        assert result.status == "completed"
        assert len(result.tasks) == 2
        assert result.tasks[0].label == "search"
        assert result.tasks[1].label == "summarize"

    def test_generates_run_id_if_not_provided(self) -> None:
        run = PapayyaRun(DurableRunConfig(agent="test", store=MemoryStore()))
        assert run.run_id
        assert isinstance(run.run_id, str)

    def test_uses_provided_run_id(self) -> None:
        run = PapayyaRun(DurableRunConfig(agent="test", run_id="my-run-123", store=MemoryStore()))
        assert run.run_id == "my-run-123"


# --------------------------------------------------------------------------- #
#  Resume / replay                                                             #
# --------------------------------------------------------------------------- #


class TestResume:
    def test_replays_cached_tasks_on_resume(self) -> None:
        store = MemoryStore()
        run_id = "resume-test"

        # First execution
        run1 = PapayyaRun(DurableRunConfig(agent="test", run_id=run_id, store=store))
        call_count = 0

        def tracked_search(q: str) -> list[str]:
            nonlocal call_count
            call_count += 1
            return search_web(q)

        search1 = run1.task("search", tracked_search)
        search1("quantum computing")
        assert call_count == 1

        # Second execution — same runId
        run2 = PapayyaRun(DurableRunConfig(agent="test", run_id=run_id, store=store))
        call_count2 = 0

        def tracked_search2(q: str) -> list[str]:
            nonlocal call_count2
            call_count2 += 1
            return search_web(q)

        search2 = run2.task("search", tracked_search2)
        results = search2("quantum computing")

        # Should NOT have been called — result from cache
        assert call_count2 == 0
        assert results == ["result for: quantum computing"]

    def test_executes_new_tasks_after_replayed(self) -> None:
        store = MemoryStore()
        run_id = "partial-resume"

        # First run — only search
        run1 = PapayyaRun(DurableRunConfig(agent="test", run_id=run_id, store=store))
        search1 = run1.task("search", search_web)
        search1("test query")

        # Second run — search replayed, summarize fresh
        run2 = PapayyaRun(DurableRunConfig(agent="test", run_id=run_id, store=store))
        summarize_count = 0

        def tracked_summarize(items: list[str]) -> str:
            nonlocal summarize_count
            summarize_count += 1
            return summarize(items)

        search2 = run2.task("search", search_web)
        sum2 = run2.task("summarize", tracked_summarize)

        results = search2("test query")  # replayed
        summary = sum2(results)  # fresh

        assert summarize_count == 1
        assert summary == "summary of 1 items"


# --------------------------------------------------------------------------- #
#  Budget enforcement                                                          #
# --------------------------------------------------------------------------- #


class TestBudget:
    def test_throws_budget_exceeded_error(self) -> None:
        run = PapayyaRun(DurableRunConfig(agent="test", budget_usd=0.01, store=MemoryStore()))
        run.record_cost(0.02)

        search = run.task("search", search_web)
        with pytest.raises(BudgetExceededError):
            search("test")

    def test_tracks_budget_state(self) -> None:
        run = PapayyaRun(DurableRunConfig(agent="test", budget_usd=1.0, store=MemoryStore()))

        assert run.budget["consumed_usd"] == 0
        assert run.budget["limit_usd"] == 1.0
        assert run.budget["remaining"] == 1.0
        assert run.budget["exceeded"] is False

        run.record_cost(0.75)
        assert run.budget["consumed_usd"] == 0.75
        assert run.budget["remaining"] == 0.25
        assert run.budget["exceeded"] is False

        run.record_cost(0.30)
        assert run.budget["exceeded"] is True
        assert run.budget["remaining"] == 0

    def test_unlimited_budget(self) -> None:
        run = PapayyaRun(DurableRunConfig(agent="test", store=MemoryStore()))

        assert run.budget["limit_usd"] is None
        assert run.budget["remaining"] is None
        assert run.budget["exceeded"] is False

        run.record_cost(1000)
        assert run.budget["exceeded"] is False


# --------------------------------------------------------------------------- #
#  Task label handling                                                         #
# --------------------------------------------------------------------------- #


class TestLabels:
    def test_defaults_label_to_function_name(self) -> None:
        run = PapayyaRun(DurableRunConfig(agent="test", store=MemoryStore()))
        search = run.task(search_web)
        search("test")
        assert run.completed_tasks == ["search_web"]

    def test_throws_on_lambda_without_label(self) -> None:
        run = PapayyaRun(DurableRunConfig(agent="test", store=MemoryStore()))
        with pytest.raises(ValueError, match="Anonymous"):
            run.task(lambda: "hello")

    def test_accepts_explicit_label_with_lambda(self) -> None:
        run = PapayyaRun(DurableRunConfig(agent="test", store=MemoryStore()))
        greet = run.task("greet", lambda: "hello")
        result = greet()
        assert result == "hello"
        assert run.completed_tasks == ["greet"]

    def test_decorator_syntax(self) -> None:
        run = PapayyaRun(DurableRunConfig(agent="test", store=MemoryStore()))

        @run.task("greet")
        def greet(name: str) -> str:
            return f"hello {name}"

        result = greet("world")
        assert result == "hello world"
        assert run.completed_tasks == ["greet"]


# --------------------------------------------------------------------------- #
#  Lifecycle                                                                   #
# --------------------------------------------------------------------------- #


class TestLifecycle:
    def test_throws_after_complete(self) -> None:
        run = PapayyaRun(DurableRunConfig(agent="test", store=MemoryStore()))
        run.complete("done")

        search = run.task("search", search_web)
        with pytest.raises(RuntimeError, match="already finished"):
            search("test")

    def test_complete_returns_result(self) -> None:
        run = PapayyaRun(DurableRunConfig(agent="test", store=MemoryStore()))
        search = run.task("search", search_web)
        search("test")

        result = run.complete("done")
        assert result.run_id == run.run_id
        assert result.agent == "test"
        assert result.status == "completed"
        assert len(result.tasks) == 1

    def test_fail_returns_result(self) -> None:
        run = PapayyaRun(DurableRunConfig(agent="test", store=MemoryStore()))
        search = run.task("search", search_web)
        search("test")

        result = run.fail(RuntimeError("broke"))
        assert result.status == "failed"


# --------------------------------------------------------------------------- #
#  papayya() factory                                                             #
# --------------------------------------------------------------------------- #


class TestFactory:
    def test_creates_runs(self) -> None:
        t = papayya(store=MemoryStore())
        run = t.run(agent="my-agent")

        assert isinstance(run, PapayyaRun)
        assert run.agent == "my-agent"

    def test_passes_default_store(self) -> None:
        store = MemoryStore()
        t = papayya(store=store)
        run = t.run(agent="test", run_id="shared-store-test")

        task = run.task("hello", lambda: "world")
        task()
        run.complete()

        checkpoint = store.load("shared-store-test")
        assert checkpoint is not None
        assert len(checkpoint.tasks) == 1


# --------------------------------------------------------------------------- #
#  Stores                                                                      #
# --------------------------------------------------------------------------- #


class TestMemoryStore:
    def test_returns_none_for_unknown(self) -> None:
        store = MemoryStore()
        assert store.load("nonexistent") is None


class TestFileStore:
    def test_returns_none_for_unknown(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            store = FileStore(d)
            assert store.load("nonexistent") is None

    def test_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            store = FileStore(d)

            from papayya.durable.types import RunCheckpoint, TaskEntry

            store.create(
                RunCheckpoint(
                    run_id="r1",
                    agent="test",
                    tasks=[],
                    status="running",
                    budget_consumed_usd=0,
                    budget_limit_usd=None,
                    created_at="2024-01-01T00:00:00Z",
                    updated_at="2024-01-01T00:00:00Z",
                )
            )

            store.save_task(
                "r1",
                TaskEntry(
                    label="search",
                    result=["hello"],
                    cost_usd=0.01,
                    duration_ms=100,
                    completed_at="2024-01-01T00:00:01Z",
                ),
            )

            loaded = store.load("r1")
            assert loaded is not None
            assert len(loaded.tasks) == 1
            assert loaded.tasks[0].label == "search"
            assert loaded.budget_consumed_usd == 0.01

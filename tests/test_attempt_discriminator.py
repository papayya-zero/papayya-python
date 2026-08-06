"""Plan 41 R3 — the attempt discriminator.

``attempt`` separates a *retry of an execution that was never recorded* from
*a new execution of a step that was*. The first must reuse the provider
idempotency key — that is the entire point of the key, and reusing it is what
stops a crash-resume from double-billing the customer's LLM account. The
second must mint a new one.

    attempt = N means: at the moment this execution started, the highest
    attempt value the SDK could see recorded for this (run_id,
    effective_label) was N-1. Computed as max(visible attempt) + 1; 1 when
    nothing is visible.

Read plans/41-R3-attempt-discriminator.md §T1, §T4 and §T5 before changing
any of these. §T5 in particular: the seeding rule here is R3's contract with
R7, and getting it wrong re-creates the bug this unit exists to fix.
"""

from __future__ import annotations

from datetime import datetime, timezone

from papayya.durable.run import PapayyaRun, DurableRunConfig, MemoryStore
from papayya.durable.sqlite_store import SQLiteStore
from papayya.durable.types import RunCheckpoint, TaskEntry


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _entry(label: str, attempt: int = 1, result: object = "v") -> TaskEntry:
    return TaskEntry(
        label=label, result=result, duration_ms=1, completed_at=_now(),
        attempt=attempt,
    )


def _re_execute(store: MemoryStore, label: str) -> tuple[str, PapayyaRun]:
    """Drive one deliberate re-execution of ``label`` and return its key.

    Stands in for R7's cache invalidation, which is the only thing that
    creates a deliberate re-execution of a *recorded* step. Without it a
    recorded label is a cache hit and never re-executes — which is why R3's
    justification is "R7's prerequisite", not "the bugs are live today".
    """
    run = PapayyaRun(DurableRunConfig(agent="a", run_id="run", store=store))
    run.init()
    run._cache.pop(label, None)
    key = run.idempotency_key(label)
    run.step(label, lambda: "v")()
    return key, run


class TestIdempotencyKey:
    def test_key_carries_the_attempt_across_three_executions(self) -> None:
        """Three executions of one step yield three distinct provider keys.

        Catches the off-by-one that a single-execution test would pass.
        """
        store = MemoryStore()
        run = PapayyaRun(DurableRunConfig(agent="a", run_id="run", store=store))
        run.init()
        assert run.idempotency_key("draft") == "run:draft:1"
        run.step("draft", lambda: "v")()

        assert _re_execute(store, "draft")[0] == "run:draft:2"
        assert _re_execute(store, "draft")[0] == "run:draft:3"

    def test_key_is_stable_after_an_unrecorded_re_execution(self) -> None:
        """T1 row 1 — the case a naive fix breaks.

        A crash between the side effect and the checkpoint leaves nothing
        recorded. On resume the SDK sees no prior attempt, so it re-sends the
        *same* key and the provider replays its response instead of billing
        twice. A run-level episode counter would fail exactly here.
        """
        store = MemoryStore()
        run1 = PapayyaRun(DurableRunConfig(agent="a", run_id="run", store=store))
        key1 = run1.idempotency_key("draft")

        # Nothing was persisted — the crash happened before save_task.
        run2 = PapayyaRun(DurableRunConfig(agent="a", run_id="run", store=store))
        run2.init()
        assert run2.idempotency_key("draft") == key1

    def test_key_changes_after_a_recorded_execution(self) -> None:
        """What R3 is for: a deliberate re-execution of a *recorded* step."""
        store = MemoryStore()
        store.create(RunCheckpoint(
            run_id="run", agent="a", tasks=[_entry("draft", attempt=1)],
            status="running", created_at=_now(), updated_at=_now(),
        ))
        run = PapayyaRun(DurableRunConfig(agent="a", run_id="run", store=store))
        run.init()
        assert run.idempotency_key("draft") == "run:draft:2"

    def test_attempt_is_max_plus_one_not_count_plus_one(self) -> None:
        """T2's journal collision leaves two rows both at attempt 1.

        The key's only requirement is that it differ from every key already
        used, and both rows already used ``:1``. So the third execution is 2,
        not 3.
        """
        store = MemoryStore()
        store.create(RunCheckpoint(
            run_id="run", agent="a",
            tasks=[_entry("draft", attempt=1), _entry("draft", attempt=1)],
            status="running", created_at=_now(), updated_at=_now(),
        ))
        run = PapayyaRun(DurableRunConfig(agent="a", run_id="run", store=store))
        run.init()
        assert run.idempotency_key("draft") == "run:draft:2"

    def test_attempt_is_orthogonal_to_the_occurrence_suffix(self) -> None:
        """T4 — ``label#N`` is a different logical step in the same pass;
        ``attempt`` is a re-execution of the same logical step across passes.
        They compose, which only holds if _attempts is keyed on the
        *effective* label.
        """
        store = MemoryStore()
        store.create(RunCheckpoint(
            run_id="run", agent="a",
            tasks=[_entry("draft", attempt=1), _entry("draft#2", attempt=1)],
            status="running", created_at=_now(), updated_at=_now(),
        ))
        run = PapayyaRun(DurableRunConfig(agent="a", run_id="run", store=store))
        run.init()

        # Occurrence 1 of "draft" has one recorded attempt -> 2.
        assert run.idempotency_key("draft") == "run:draft:2"
        run.step("draft", lambda: "v1")()
        # Occurrence 2 keys "draft#2", which separately has one attempt -> 2.
        assert run.idempotency_key("draft") == "run:draft#2:2"


class TestAttemptSeeding:
    def test_attempts_survive_a_suppressed_cache_entry(self) -> None:
        """T5 — **R3's contract with R7.**

        R7's entire mechanism is removing invalidated entries from ``_cache``
        at hydration so the step re-executes. If ``_attempts`` were seeded
        from ``_cache``, the invalidated label would have no entry, attempt
        would compute as 1, and the re-execution would reuse the recorded
        attempt's provider key — the exact bug R3 exists to fix, back on the
        exact path R3 was built for.

        Seed from the raw loaded task list, before and independently of any
        cache filter.
        """
        store = MemoryStore()
        store.create(RunCheckpoint(
            run_id="run", agent="a", tasks=[_entry("draft", attempt=1)],
            status="running", created_at=_now(), updated_at=_now(),
        ))
        run = PapayyaRun(DurableRunConfig(agent="a", run_id="run", store=store))
        run.init()

        # Simulate R7: the entry is invalidated out of the cache so the step
        # re-executes. The attempt counter must not follow it.
        run._cache.pop("draft", None)

        assert run.idempotency_key("draft") == "run:draft:2", (
            "attempt must be seeded from the loaded task list, not from _cache"
        )

    def test_persisted_attempts_are_one_two_three(self) -> None:
        """Three genuine executions persist attempts [1, 2, 3]."""
        store = MemoryStore()
        run = PapayyaRun(DurableRunConfig(agent="a", run_id="run", store=store))
        run.init()
        run.step("draft", lambda: "v")()
        _re_execute(store, "draft")
        _re_execute(store, "draft")

        attempts = [t.attempt for t in store.load("run").tasks if t.label == "draft"]
        assert sorted(attempts) == [1, 2, 3]

    def test_each_execution_gets_its_own_token(self) -> None:
        """The row id is derived from the token, so two executions that
        shared one would collide and the second would trample the first.
        """
        store = MemoryStore()
        run = PapayyaRun(DurableRunConfig(agent="a", run_id="run", store=store))
        run.init()
        run.step("draft", lambda: "v")()
        _re_execute(store, "draft")

        tokens = [t.execution_token for t in store.load("run").tasks if t.label == "draft"]
        assert len(set(tokens)) == len(tokens) == 2

    def test_attempt_one_token_is_deterministic(self) -> None:
        """T2's cross-generation rule for ADR-0002 #8.

        With UNIQUE (run_id, label) gone, a legacy client (no token, server
        derives "{label}|1") and a new client (random token) would occupy
        disjoint id spaces, so #8 would hold inside each SDK generation and
        never across it — and the worker service rolls at 50% overlap, so
        mixed-SDK workers exist by design. Attempt 1 therefore uses the
        literal "{label}|1"; only attempt >= 2 goes random.
        """
        store = MemoryStore()
        run = PapayyaRun(DurableRunConfig(agent="a", run_id="run", store=store))
        run.init()
        run.step("draft", lambda: "v")()

        [first] = [t for t in store.load("run").tasks if t.label == "draft"]
        assert first.execution_token == "draft|1"

        _re_execute(store, "draft")
        second = [t for t in store.load("run").tasks if t.label == "draft"][1]
        assert second.execution_token != "draft|2", "attempt >= 2 must be random"


class TestStoreRoundTrip:
    def test_sqlite_second_execution_writes_a_second_row(self, tmp_path) -> None:
        """T11 — the local guard is a first-writer-wins early return keyed on
        (item_id, label), so a re-executed step's new result is never recorded
        locally at all. Keyed on the execution token instead, a genuine
        re-execution lands as its own row.
        """
        store = SQLiteStore(str(tmp_path / "attempt.db"))
        try:
            store.create(RunCheckpoint(
                run_id="r1", agent="a", tasks=[], status="running",
                created_at=_now(), updated_at=_now(),
            ))
            e1 = _entry("draft", attempt=1, result="first")
            e2 = _entry("draft", attempt=2, result="second")
            store.save_task("r1", e1)
            store.save_task("r1", e2)
            # A re-delivery of e1 must still be idempotent.
            store.save_task("r1", e1)

            loaded = store.load("r1")
            drafts = [t for t in loaded.tasks if t.label == "draft"]
            assert len(drafts) == 2, "two executions, two rows; re-delivery adds none"
            assert sorted(t.attempt for t in drafts) == [1, 2]
        finally:
            store.close()

    def test_file_store_round_trips_attempt_and_token(self, tmp_path) -> None:
        """T7 — FileStore._write serializes only four TaskEntry fields, so
        without this the attempt round-trips to nothing and every execution
        computes attempt 1 forever.
        """
        from papayya.durable.store import FileStore

        store = FileStore(str(tmp_path / "runs"))
        entry = _entry("draft", attempt=3)
        store.create(RunCheckpoint(
            run_id="r1", agent="a", tasks=[], status="running",
            created_at=_now(), updated_at=_now(),
        ))
        store.save_task("r1", entry)

        loaded = store.load("r1")
        assert loaded is not None
        [got] = [t for t in loaded.tasks if t.label == "draft"]
        assert got.attempt == 3
        assert got.execution_token == entry.execution_token

    def test_memory_store_round_trips_attempt(self) -> None:
        store = MemoryStore()
        store.create(RunCheckpoint(
            run_id="r1", agent="a", tasks=[], status="running",
            created_at=_now(), updated_at=_now(),
        ))
        store.save_task("r1", _entry("draft", attempt=2))
        [got] = [t for t in store.load("r1").tasks if t.label == "draft"]
        assert got.attempt == 2


class TestHydration:
    def test_hydration_takes_the_latest_execution_not_the_latest_inserted(self) -> None:
        """T5's ordering trap, seeded as the collision case.

        E1 executes, journals, and drains *after* E2 has already landed. A
        seq-ordered (INSERT-time) implementation has two rows and still hands
        every reader E1. Order by the client-stamped completed_at.
        """
        store = MemoryStore()
        early = "2026-01-01T00:00:00+00:00"
        late = "2026-01-01T00:05:00+00:00"
        e1 = TaskEntry(label="enrich", result="E1", duration_ms=1,
                       completed_at=early, attempt=1)
        e2 = TaskEntry(label="enrich", result="E2", duration_ms=1,
                       completed_at=late, attempt=1)
        # Insert order is E2 then E1 — the drain arrives last.
        store.create(RunCheckpoint(
            run_id="run", agent="a", tasks=[e2, e1], status="running",
            created_at=_now(), updated_at=_now(),
        ))
        run = PapayyaRun(DurableRunConfig(agent="a", run_id="run", store=store))
        run.init()

        assert run._cache["enrich"].result == "E2"
        assert run._task_call_order.count("enrich") == 1, (
            "a label hydrated from two executions must appear once in the order"
        )

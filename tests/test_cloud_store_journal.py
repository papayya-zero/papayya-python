"""CloudStore retry + journal sidecar integration tests (ADR-0002 #8).

Exercises the four behaviors that turn a transient hosted-DB outage
into a recoverable lineage delivery rather than a silent loss:

1. Bounded exponential retry on transient failures.
2. Append-only journal sidecar when the retry budget exhausts.
3. FIFO drain on the next successful POST.
4. Late-delivery audit fields injected into ``save_task`` replays.

Uses ``httpx.MockTransport`` so no network is involved; the journal
file lives in ``tmp_path`` via the ``PAPAYYA_LINEAGE_JOURNAL_PATH``
env var. Sleeps inside the retry loop are monkeypatched out so test
runs stay fast.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import httpx
import pytest

from papayya.durable.cloud_store import CloudStore, CloudStoreConfig
from papayya.durable.lineage_journal import LineageJournal
from papayya.durable.types import RunCheckpoint, TaskEntry


def _make_store(
    handler: Callable[[httpx.Request], httpx.Response],
    journal_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> CloudStore:
    monkeypatch.setenv("PAPAYYA_LINEAGE_JOURNAL_PATH", str(journal_path))
    # No real waiting between retries — tests assert behavior, not timing.
    monkeypatch.setattr("papayya.durable.cloud_store.time.sleep", lambda _s: None)
    config = CloudStoreConfig(api_key="cpk_test", base_url="http://mock")
    store = CloudStore(config)
    store._client = httpx.Client(
        base_url="http://mock",
        headers={"X-Api-Key": "cpk_test"},
        transport=httpx.MockTransport(handler),
    )
    return store


def _entry() -> TaskEntry:
    return TaskEntry(
        label="enrich",
        result={"out": 1},
        duration_ms=50,
        completed_at="2026-04-29T00:00:00+00:00",
        item_id="co_42",
        input_snapshot={"in": 0},
        output_snapshot={"out": 1},
    )


# --- 1. Transient retry succeeds within budget --------------------------- #


class TestTransientRetrySucceeds:
    def test_5xx_then_200_no_journal(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        attempts: list[int] = []

        def handler(request: httpx.Request) -> httpx.Response:
            attempts.append(1)
            if len(attempts) < 3:
                return httpx.Response(503, json={"error": "db restart"})
            return httpx.Response(201, json={})

        journal_path = tmp_path / "journal.jsonl"
        store = _make_store(handler, journal_path, monkeypatch)
        store.save_task("r1", _entry())

        assert len(attempts) == 3, "expected 2 retries before success"
        assert not journal_path.exists(), "no entry should be journaled"


# --- 2. Sustained 5xx exhausts retries → journal entry ------------------- #


class TestSustained5xxJournals:
    def test_five_5xx_writes_journal_entry(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        attempts: list[int] = []

        def handler(request: httpx.Request) -> httpx.Response:
            attempts.append(1)
            return httpx.Response(503, json={"error": "down"})

        journal_path = tmp_path / "journal.jsonl"
        store = _make_store(handler, journal_path, monkeypatch)
        # Must NOT raise — that's the whole contract.
        store.save_task("r1", _entry())

        assert len(attempts) == 5
        assert journal_path.exists()
        lines = journal_path.read_text().strip().splitlines()
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["kind"] == "save_task"
        assert record["idempotency_key"] == "r1:enrich"
        assert record["attempts"] == 5
        assert record["payload"]["label"] == "enrich"
        assert record["last_error"].startswith("HTTPStatusError")


# --- 3. Network down → journal entry ------------------------------------- #


class TestNetworkDownJournals:
    def test_connect_error_writes_journal_entry(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        attempts: list[int] = []

        def handler(request: httpx.Request) -> httpx.Response:
            attempts.append(1)
            raise httpx.ConnectError("connection refused")

        journal_path = tmp_path / "journal.jsonl"
        store = _make_store(handler, journal_path, monkeypatch)
        store.save_task("r1", _entry())

        assert len(attempts) == 5
        assert journal_path.exists()
        record = json.loads(journal_path.read_text().strip())
        assert record["kind"] == "save_task"
        assert record["last_error"].startswith("ConnectError")


# --- 4. Outage → restore → drain on next POST ---------------------------- #


class TestDrainOnNextSuccessfulPost:
    def test_journal_drains_in_fifo_order_on_recovery(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Two writes during the outage, then a third write after recovery.
        # Expectation: when the third write fires, the drain replays
        # entries 1 and 2 in order, then the new write goes through.
        request_log: list[tuple[str, str, dict[str, Any]]] = []
        outage_active = {"on": True}

        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content) if request.content else {}
            request_log.append((request.method, request.url.path, body))
            if outage_active["on"]:
                return httpx.Response(503)
            return httpx.Response(201, json={})

        journal_path = tmp_path / "journal.jsonl"
        store = _make_store(handler, journal_path, monkeypatch)

        # Two journaled writes during outage.
        e1 = TaskEntry(label="step-a", result={"v": 1}, duration_ms=10,
                       completed_at="t1", item_id="x")
        e2 = TaskEntry(label="step-b", result={"v": 2}, duration_ms=10,
                       completed_at="t2", item_id="x")
        store.save_task("r1", e1)
        store.save_task("r1", e2)
        assert journal_path.exists()
        assert len(journal_path.read_text().strip().splitlines()) == 2

        # Outage ends.
        outage_active["on"] = False
        request_log.clear()

        # Third live write.
        e3 = TaskEntry(label="step-c", result={"v": 3}, duration_ms=10,
                       completed_at="t3", item_id="x")
        store.save_task("r1", e3)

        # The first three POSTs the server saw post-recovery should be
        # the two drained entries in FIFO order, then the new write.
        labels = [body.get("label") for _m, _p, body in request_log[:3]]
        assert labels == ["step-a", "step-b", "step-c"]

        # Journal is gone now.
        assert not journal_path.exists()


# --- 5. 4xx terminal → raise immediately, no journal --------------------- #


class TestTerminalErrorRaisesNoJournal:
    def test_400_raises_no_retry_no_journal(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        attempts: list[int] = []

        def handler(request: httpx.Request) -> httpx.Response:
            attempts.append(1)
            return httpx.Response(400, json={"error": "bad payload"})

        journal_path = tmp_path / "journal.jsonl"
        store = _make_store(handler, journal_path, monkeypatch)
        with pytest.raises(httpx.HTTPStatusError):
            store.save_task("r1", _entry())

        assert len(attempts) == 1, "4xx must not retry"
        assert not journal_path.exists()


# --- 6. Reconciler injects audit fields on save_task replay -------------- #


class TestDrainInjectsAuditFields:
    def test_replay_carries_delivery_attempts_and_journaled_at(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        outage_active = {"on": True}
        bodies: list[dict[str, Any]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content) if request.content else {}
            bodies.append(body)
            if outage_active["on"]:
                return httpx.Response(503)
            return httpx.Response(201, json={})

        journal_path = tmp_path / "journal.jsonl"
        store = _make_store(handler, journal_path, monkeypatch)
        store.save_task("r1", _entry())  # journaled

        outage_active["on"] = False
        bodies.clear()

        # Force a drain via a second call. The first POST in `bodies`
        # is the replay of the journaled entry; the second is the new
        # save_task we just issued. The replay must carry the audit
        # fields.
        store.save_task("r1", TaskEntry(
            label="next", result={"v": 2}, duration_ms=10,
            completed_at="t2", item_id="x",
        ))

        assert len(bodies) >= 2
        replay_body = bodies[0]
        assert replay_body["label"] == "enrich"
        assert replay_body["delivery_attempts"] == 6  # 5 failed + 1 successful drain
        assert "journaled_at" in replay_body and replay_body["journaled_at"]

        new_body = bodies[1]
        assert new_body["label"] == "next"
        # Live writes carry no audit fields — they delivered first try.
        assert "delivery_attempts" not in new_body
        assert "journaled_at" not in new_body


# --- 7. Mid-drain transient halts and preserves remainder ---------------- #


class TestMidDrainTransientPreservesRemainder:
    def test_drain_stops_on_5xx_and_keeps_remaining_entries(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Pre-seed a journal with three entries, then run a drain where
        # entry 1 succeeds, entry 2 hits a 5xx, and entry 3 should not
        # be attempted.
        journal_path = tmp_path / "journal.jsonl"
        journal_path.parent.mkdir(parents=True, exist_ok=True)

        seed_journal = LineageJournal(journal_path)
        from papayya.durable.lineage_journal import JournalEntry
        for label in ("a", "b", "c"):
            seed_journal.append(JournalEntry(
                kind="save_task",
                method="POST",
                url="/v1/durable/runs/r1/checkpoints",
                payload={"label": label, "result": {}, "duration_ms": 0,
                         "item_id": None, "input_snapshot": None,
                         "output_snapshot": None, "agent_version": None},
                idempotency_key=f"r1:{label}",
                first_attempt_at="t0",
                attempts=5,
                journaled_at="t1",
                last_error="HTTPStatusError: 503",
            ))

        seen_labels: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content) if request.content else {}
            seen_labels.append(body.get("label"))
            if body.get("label") == "b":
                return httpx.Response(503)
            return httpx.Response(201, json={})

        store = _make_store(handler, journal_path, monkeypatch)
        # The new live write triggers the drain. The drain halts at "b"
        # (transient 503), so c is not attempted. The new "d" write
        # itself goes through — the handler only 503s on "b" — so the
        # journal at the end contains [b, c] in original order.
        store.save_task("r1", TaskEntry(
            label="d", result={}, duration_ms=0, completed_at="t",
        ))

        # Drain attempted a, then b (which 503'd, halting). c was not tried.
        # Then the new "d" write went through live.
        assert seen_labels == ["a", "b", "d"]
        remaining_lines = journal_path.read_text().strip().splitlines()
        labels_in_journal = [json.loads(l)["payload"]["label"] for l in remaining_lines]
        assert labels_in_journal == ["b", "c"]


# --- 8. Mid-drain terminal drops the bad entry but continues ------------- #


class TestMidDrainTerminalDropsAndContinues:
    def test_drain_drops_4xx_entry_and_continues(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        journal_path = tmp_path / "journal.jsonl"
        journal_path.parent.mkdir(parents=True, exist_ok=True)

        seed_journal = LineageJournal(journal_path)
        from papayya.durable.lineage_journal import JournalEntry
        for label in ("a", "b", "c"):
            seed_journal.append(JournalEntry(
                kind="save_task",
                method="POST",
                url="/v1/durable/runs/r1/checkpoints",
                payload={"label": label, "result": {}, "duration_ms": 0,
                         "item_id": None, "input_snapshot": None,
                         "output_snapshot": None, "agent_version": None},
                idempotency_key=f"r1:{label}",
                first_attempt_at="t0",
                attempts=5,
                journaled_at="t1",
                last_error="HTTPStatusError: 503",
            ))

        seen_labels: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content) if request.content else {}
            seen_labels.append(body.get("label"))
            if body.get("label") == "b":
                return httpx.Response(404, json={"error": "run gone"})
            return httpx.Response(201, json={})

        store = _make_store(handler, journal_path, monkeypatch)
        # Issue a fresh save_task that lands fine, just to drive the drain.
        store.save_task("r1", TaskEntry(
            label="d", result={}, duration_ms=0, completed_at="t",
        ))

        # All three journaled entries were attempted (b dropped, c continued),
        # then the new "d" write went through.
        assert seen_labels == ["a", "b", "c", "d"]
        # Journal cleared — every entry either delivered or was dropped
        # as terminal.
        assert not journal_path.exists()

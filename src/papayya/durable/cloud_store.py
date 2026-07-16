"""Checkpoint store backed by the Papayya control plane API.

POSTs are wrapped in a bounded retry with a local-journal fallback
(ADR-0002 #8). Transient failures (5xx, 429, network errors) retry
with exponential backoff; on exhaustion the request is appended to a
``LineageJournal`` sidecar and the call returns successfully — the
customer's agent function does not see the outage. The next successful
POST drains the journal in FIFO order before issuing the new request,
so eventually every step row lands server-side.

The retry rhythm matches ``runtime/worker.py::_report_complete`` (Phase
2 #4): 5 attempts, 0.1s → 2.0s exponential, ~3.1s wall ceiling. Same
mental model and same ceiling, picked to stay well under any per-item
soft timeout (#2).

Terminal failures (4xx with body, decode errors) raise immediately and
are *not* journaled — they almost always indicate an SDK-side bug, and
journaling a bug-rich payload would just keep failing on every drain.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable

import httpx

from papayya._defaults import DEFAULT_BASE_URL
from papayya._serialize import encode_user_value

from .lineage_journal import JournalEntry, LineageJournal, resolve_journal_path
from .types import CheckpointStore, RunCheckpoint, TaskEntry


log = logging.getLogger("papayya.durable.cloud_store")


# Retry budget — same shape as runtime/worker.py::_report_complete so
# operators only have to learn one rhythm. Worst-case wait is roughly
# 0.1 + 0.2 + 0.4 + 0.8 + (capped) 2.0 = 3.5s before journaling.
_MAX_ATTEMPTS = 5
_INITIAL_BACKOFF = 0.1
_MAX_BACKOFF = 2.0


# Cap on how many journaled entries a single piggyback drain attempts.
# Keeps the per-POST overhead bounded if a long outage left thousands of
# entries; the next POST after this one will keep draining where this
# stopped.
_DRAIN_BATCH = 100


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_transient(exc: BaseException) -> bool:
    """Should this exception be retried?

    Yes for connection-level errors and server-side hiccups (5xx, 429).
    No for 4xx-with-body (SDK bug — payload is wrong, retrying won't
    help) and for any non-network exception (e.g. JSON decode errors).
    """
    if isinstance(exc, (
        httpx.ConnectError,
        httpx.ConnectTimeout,
        httpx.ReadTimeout,
        httpx.WriteTimeout,
        httpx.PoolTimeout,
        httpx.NetworkError,
        httpx.RemoteProtocolError,
    )):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        code = exc.response.status_code
        return code >= 500 or code == 429
    return False


@dataclass
class CloudStoreConfig:
    """Configuration for the cloud checkpoint store."""

    api_key: str
    base_url: str = DEFAULT_BASE_URL
    timeout: float = 15.0
    # Route prefix for run + checkpoint writes. Defaults to the tenant-scoped
    # durable API (cpk_ project keys). The hosted worker pool overrides it to
    # "/v1/runtime/runs" — the platform-authed lane (Plan 37 Unit 1) that
    # resolves the tenant off the pre-created run row instead of the key.
    runs_base: str = "/v1/durable/runs"
    # Send the key as X-Api-Key regardless of prefix. The platform worker key
    # isn't a cpk_ key but auth.go matches it in the X-Api-Key header.
    platform_auth: bool = False


class CloudStore:
    """Checkpoint store that persists to the Papayya control plane via HTTP.

    Wraps every write in retry + journal-on-exhaust (ADR-0002 #8). The
    public ``CheckpointStore`` surface is unchanged — callers see no
    difference except that transient outages no longer raise.
    """

    def __init__(self, config: CloudStoreConfig) -> None:
        headers: dict[str, str] = {"Content-Type": "application/json", "Accept": "application/json"}
        if config.platform_auth or config.api_key.startswith("cpk_"):
            headers["X-Api-Key"] = config.api_key
        else:
            headers["Authorization"] = f"Bearer {config.api_key}"

        self._runs_base = config.runs_base.rstrip("/")
        self._client = httpx.Client(
            base_url=config.base_url,
            headers=headers,
            timeout=config.timeout,
        )
        self._journal = LineageJournal(resolve_journal_path())
        # Plan 33: pause reason per run_id, set from a SaveCheckpoint response
        # whose run_status came back 'paused' (a server fence tripped). Read by
        # PapayyaRun._pre_call at the next step boundary. Keyed by run_id so a
        # process-shared store never leaks one run's pause onto another's steps.
        # Journaled/offline saves never set it — degraded-mode behavior is
        # "keep working", correct for a reliability product.
        self._pending_pause: dict[str, str] = {}

    # --- public store surface ----------------------------------------- #

    def load(self, run_id: str) -> RunCheckpoint | None:
        # Reads have no journal path: a load that can't reach the server
        # is a fundamentally different failure mode (the *replay-from*
        # data isn't there yet). Surface the error to the caller.
        resp = self._client.get(f"{self._runs_base}/{run_id}")
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        body = resp.json()
        return RunCheckpoint(
            run_id=body["run_id"],
            agent=body["agent"],
            status=body["status"],
            tasks=[
                TaskEntry(
                    label=cp["label"],
                    result=cp["result"],
                    duration_ms=cp["duration_ms"],
                    completed_at=cp["completed_at"],
                    item_id=cp.get("item_id"),
                    input_snapshot=cp.get("input_snapshot"),
                    output_snapshot=cp.get("output_snapshot"),
                    agent_version=cp.get("agent_version"),
                    metadata=cp.get("metadata"),
                    partition_key=cp.get("partition_key"),
                    llm_prompt_tokens=cp.get("llm_prompt_tokens"),
                    llm_completion_tokens=cp.get("llm_completion_tokens"),
                    llm_total_tokens=cp.get("llm_total_tokens"),
                    llm_model=cp.get("llm_model"),
                    llm_stop_reason=cp.get("llm_stop_reason"),
                    llm_provider_shape=cp.get("llm_provider_shape"),
                    # Defaults match the SDK-side defaults so older control-pane
                    # versions (pre-Plan-03) round-trip cleanly.
                    outcome_status=cp.get("outcome_status", "ok"),
                    outcome_reason=cp.get("outcome_reason"),
                )
                for cp in body.get("checkpoints") or []
            ],
            item_id=body.get("item_id"),
            created_at=body["created_at"],
            updated_at=body["updated_at"],
            agent_version=body.get("agent_version"),
            metadata=body.get("metadata"),
            partition_key=body.get("partition_key"),
            parent_run_id=body.get("parent_run_id"),
            worst_outcome_status=body.get("worst_outcome_status", "ok"),
            degraded_count=body.get("degraded_count", 0),
        )

    def create(self, checkpoint: RunCheckpoint) -> None:
        payload = {
            "run_id": checkpoint.run_id,
            "agent": checkpoint.agent,
            "item_id": checkpoint.item_id,
            "agent_version": checkpoint.agent_version,
            "metadata": checkpoint.metadata,
            "partition_key": checkpoint.partition_key,
            "parent_run_id": checkpoint.parent_run_id,
        }
        self._execute(
            kind="create",
            method="POST",
            url=self._runs_base,
            payload=payload,
            idempotency_key=checkpoint.run_id,
        )

    def save_task(self, run_id: str, entry: TaskEntry) -> None:
        payload = {
            "label": entry.label,
            "result": json.loads(encode_user_value(entry.result)),
            "duration_ms": entry.duration_ms,
            "item_id": entry.item_id,
            "input_snapshot": json.loads(encode_user_value(entry.input_snapshot, strict=True)),
            "output_snapshot": json.loads(encode_user_value(entry.output_snapshot, strict=True)),
            "agent_version": entry.agent_version,
            "metadata": entry.metadata,
            "partition_key": entry.partition_key,
            "llm_prompt_tokens": entry.llm_prompt_tokens,
            "llm_completion_tokens": entry.llm_completion_tokens,
            "llm_total_tokens": entry.llm_total_tokens,
            "llm_model": entry.llm_model,
            "llm_stop_reason": entry.llm_stop_reason,
            "llm_provider_shape": entry.llm_provider_shape,
            "outcome_status": entry.outcome_status,
            "outcome_reason": entry.outcome_reason,
        }
        self._execute(
            kind="save_task",
            method="POST",
            url=f"{self._runs_base}/{run_id}/checkpoints",
            payload=payload,
            idempotency_key=f"{run_id}:{entry.label}",
        )

    def set_status(self, run_id: str, status: str, output: Any = None) -> None:
        payload = {"status": status, "output": json.loads(encode_user_value(output))}
        self._execute(
            kind="set_status",
            method="PATCH",
            url=f"{self._runs_base}/{run_id}",
            payload=payload,
            idempotency_key=run_id,
        )

    def close(self) -> None:
        self._client.close()

    # --- retry + journal core ----------------------------------------- #

    def _execute(
        self,
        *,
        kind: str,
        method: str,
        url: str,
        payload: dict[str, Any],
        idempotency_key: str,
    ) -> None:
        """Drain pending journal entries, then issue ``op`` with retry.

        On exhaustion the request is appended to the journal; the call
        does NOT raise. On a terminal (non-transient) failure the
        underlying exception bubbles up and nothing is journaled.
        """
        # Causal ordering — drain BEFORE this write so a queued create
        # for run R lands before any save_task(R, ...) we're about to
        # issue. Drain failures don't block the new write; if the server
        # is still sick the new write will journal too, in correct order.
        self._drain_journal()

        first_attempt_at = _utcnow_iso()
        backoff = _INITIAL_BACKOFF
        last_exc: BaseException | None = None
        for attempt in range(1, _MAX_ATTEMPTS + 1):
            try:
                self._dispatch(method, url, payload)
                return
            except BaseException as exc:  # noqa: BLE001
                if not _is_transient(exc):
                    raise
                last_exc = exc
                if attempt < _MAX_ATTEMPTS:
                    log.debug(
                        "lineage write attempt %d/%d failed: %s; retrying in %.2fs",
                        attempt, _MAX_ATTEMPTS, exc, backoff,
                    )
                    time.sleep(backoff)
                    backoff = min(backoff * 2.0, _MAX_BACKOFF)

        # Exhausted. Journal and return so the customer's agent
        # function does not see the transient outage.
        entry = JournalEntry(
            kind=kind,
            method=method,
            url=url,
            payload=payload,
            idempotency_key=idempotency_key,
            first_attempt_at=first_attempt_at,
            attempts=_MAX_ATTEMPTS,
            journaled_at=_utcnow_iso(),
            last_error=f"{type(last_exc).__name__}: {last_exc}",
        )
        self._journal.append(entry)
        log.warning(
            "lineage write %s journaled after %d failed attempts: %s",
            idempotency_key, _MAX_ATTEMPTS, last_exc,
        )

    def _drain_journal(self) -> None:
        """FIFO drain of journaled entries, bounded by ``_DRAIN_BATCH``.

        Stops at the first transient error (server still sick) so the
        new POST gets to journal in correct order. Drops entries that
        fail with a terminal error (rare — implies the journaled
        payload is now invalid, e.g. tenant deleted).
        """
        if self._journal.is_empty():
            return

        remaining: list[JournalEntry] = []
        drained = 0
        halt = False
        for entry in self._journal.iter_entries():
            if halt or drained >= _DRAIN_BATCH:
                remaining.append(entry)
                continue

            payload = self._payload_for_replay(entry)
            try:
                self._dispatch(entry.method, entry.url, payload)
                drained += 1
            except BaseException as exc:  # noqa: BLE001
                if _is_transient(exc):
                    log.debug(
                        "drain halted on %s after transient error: %s",
                        entry.idempotency_key, exc,
                    )
                    halt = True
                    remaining.append(entry)
                else:
                    log.warning(
                        "dropping journal entry %s after terminal error: %s",
                        entry.idempotency_key, exc,
                    )

        self._journal.rewrite(remaining)
        if drained > 0:
            log.info("drained %d journaled lineage write(s)", drained)

    def _payload_for_replay(self, entry: JournalEntry) -> dict[str, Any]:
        """Build the wire payload for a journaled entry being reissued.

        For ``save_task`` the persisted row carries the late-delivery
        audit columns, so inject ``delivery_attempts`` (total attempts
        including this drain attempt) and ``journaled_at``. For other
        kinds the audit lives only on per-step rows; replay payloads
        are unchanged.
        """
        if entry.kind != "save_task":
            return entry.payload
        payload = dict(entry.payload)
        payload["delivery_attempts"] = entry.attempts + 1
        payload["journaled_at"] = entry.journaled_at
        return payload

    def _dispatch(self, method: str, url: str, payload: dict[str, Any]) -> httpx.Response:
        """Issue the actual HTTP request and raise for non-2xx.

        Raising on 4xx/5xx is what makes ``_is_transient`` work for
        ``HTTPStatusError``. ``httpx.Client.request`` does not
        ``raise_for_status`` automatically, so we do.
        """
        resp = self._client.request(method, url, json=payload)
        resp.raise_for_status()
        self._note_pause_from_response(url, resp)
        return resp

    def _note_pause_from_response(self, url: str, resp: httpx.Response) -> None:
        """Record a server-signalled pause riding a SaveCheckpoint response
        (Plan 33 Decision 2). The pause is on the save *response*, not a
        rejection — the checkpoint was accepted. Best effort: a non-JSON or
        unexpected body just leaves no pending pause (keep working)."""
        if not url.endswith("/checkpoints"):
            return
        try:
            body = resp.json()
        except Exception:
            return
        if not isinstance(body, dict) or body.get("run_status") != "paused":
            return
        run_id = body.get("run_id")
        if run_id:
            self._pending_pause[run_id] = body.get("pause_reason") or "paused"

    def pending_pause(self, run_id: str) -> str | None:
        """The pause reason a fence set for this run, or None. Consulted by
        PapayyaRun._pre_call before resolving the next step."""
        return self._pending_pause.get(run_id)


def make_runtime_store(base_url: str, api_key: str, *, timeout: float = 15.0) -> CloudStore:
    """A CloudStore pointed at the platform-authed runtime lane (Plan 37
    Unit 1). Used by the hosted worker pool so customer @agent code running
    in-process writes its checkpoints + run status to
    ``/v1/runtime/runs/...`` with the shared platform worker key, instead of
    the tenant-scoped ``/v1/durable/...`` API (which the worker can't reach —
    it has no project key). The control-plane resolves the tenant off the
    pre-created run row. Same retry/journal/pending-pause machinery as the
    tenant CloudStore, only the route prefix + auth header differ.

    ``base_url`` is the control-plane root (e.g. ``http://control-pane-api:8090``);
    the ``runs_base`` supplies the ``/v1/runtime/runs`` path.
    """
    return CloudStore(
        CloudStoreConfig(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            runs_base="/v1/runtime/runs",
            platform_auth=True,
        )
    )


# Type-checkers want CheckpointStore conformance at the module surface,
# so a no-op assertion helps catch protocol drift at import time without
# costing anything at runtime.
_check: Callable[[CloudStore], CheckpointStore] = lambda store: store  # noqa: E731

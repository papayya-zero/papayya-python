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


def _is_schema_rejection(entry: "JournalEntry", exc: BaseException) -> bool:
    """Did the control pane reject a journaled write for its *shape*?

    Narrow on purpose: a 400 on a checkpoint write. That is the signature
    of an SDK sending a field the server does not know yet — the
    new-SDK-against-old-API direction of a rollout, or a rollback past the
    release that added the field.

    Such an entry is NOT invalid; the server is. Retaining it (rather than
    dropping it as an ordinary terminal error) is what keeps a rollback
    from silently destroying the lineage writes ADR-0002 #8 protects. A
    genuinely malformed payload also lands here, but it costs bounded disk
    and stays visible in the journal instead of vanishing into a log line.
    """
    if entry.kind != "save_task":
        return False
    return isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code == 400


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


def _raise_with_server_message(resp: "httpx.Response") -> None:
    """raise_for_status, but say what the server said (plan 56 F8 / plan 54 N2).

    httpx's own message is the request line and the status code. The
    control-plane always sends a body explaining itself — ``{"error":
    {"message": "invalid request body"}}`` — and raise_for_status throws it
    away, so a customer whose input was rejected got a traceback ending in

        httpx.HTTPStatusError: Client error '400 Bad Request' for url ...

    with the one sentence that names the problem sitting unread in the
    response. Plan 55 measured the cost: a dict-shaped input 400s (N1), and
    the error named neither the field nor the reason.

    Raises the SAME exception type with the same request/response attached, so
    ``_is_transient`` and every other caller inspecting ``.response`` keep
    working — this widens the message and changes nothing else.
    """
    if not resp.is_error:
        return
    detail = ""
    try:
        body = resp.json()
        if isinstance(body, dict):
            err = body.get("error")
            if isinstance(err, dict):
                detail = str(err.get("message") or "")
            elif isinstance(err, str):
                detail = err
            if not detail:
                detail = str(body.get("message") or "")
    except Exception:  # noqa: BLE001 — a non-JSON body is not worth failing on
        detail = ""
    if not detail:
        # Fall back to the raw body, trimmed. An HTML error page is still
        # more use than a bare status code.
        detail = (resp.text or "").strip()[:300]

    message = f"{resp.status_code} from {resp.request.method} {resp.request.url}"
    if detail:
        message += f": {detail}"
    # The server's own correlation id, echoed on every response since plan 56
    # F7. It is what turns "it 500'd" into a line an operator can actually
    # find — the API log was 100% idle-poll noise, so the deploy failure that
    # read as "logs nothing server-side" was in there and ungreppable.
    request_id = resp.headers.get("X-Request-Id")
    if request_id:
        message += f" (request_id={request_id})"
    raise httpx.HTTPStatusError(message, request=resp.request, response=resp)


def _attach_run_token(request: "httpx.Request") -> None:
    """Authenticate a hosted write with THIS run's token (plan 60 S1).

    The store is built once per worker process, before the customer's module is
    imported, and the credential changes per item — the same shape as the lease
    header below, and forced by the same fact. So the key is stamped per
    request from a contextvar rather than baked into the client's headers.

    ONLY ON THE PLATFORM LANE. The tenant-scoped CloudStore authenticates with
    the customer's own project key, which is theirs to hold and is set at
    construction; overwriting it here would break every non-worker client.

    WHEN NO TOKEN IS IN SCOPE THE REQUEST GOES OUT UNAUTHENTICATED AND THE
    SERVER REFUSES IT, deliberately. Before this the fallback was the platform
    worker key, which is precisely the credential this unit removes from
    customer reach — so "fall back to something that works" here means "fall
    back to the hole". The visible case is a customer thread: contextvars do
    not cross ``threading.Thread``, so a write from one 401s where it used to
    succeed with fleet-wide authority. That is a real behaviour change and it
    is the intended one; the same thread already lost its lease header, so its
    writes were already unfenced.
    """
    from papayya.agent import current_run_token

    token = current_run_token()
    if token:
        request.headers["X-Api-Key"] = token


def _attach_lease_header(request: "httpx.Request") -> None:
    """Stamp X-Papayya-Lease on outbound writes when one is in scope.

    Absent outside a hosted invocation — the local path and the tenant-scoped
    API have no lease, and the server only fences the /v1/runtime/* routes.

    Read from a contextvar rather than passed as an argument because the
    customer's function sits between the worker and this client; threading a
    lease id through it would put a platform concern into a user-facing
    signature, and any customer who spawned a thread would drop it.
    """
    # Local import: papayya.agent imports the durable package, so a
    # module-level import here is a cycle.
    from papayya.agent import current_lease_id

    lease_id = current_lease_id()
    if lease_id:
        request.headers["X-Papayya-Lease"] = lease_id


def _redact_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Strip injected secret values out of one outbound checkpoint body.

    Plan 67 S1. Serialize once, filter the text, and parse back only when
    something actually changed — a step whose snapshot happens to contain a key
    is rare, and the common path must not pay for a round trip through JSON.

    Encoding failures are swallowed on purpose. This is a redaction pass over a
    body the caller already built and is entitled to send; a payload we cannot
    re-encode is one we pass through unfiltered rather than one we drop.
    """
    from papayya.runtime.redact import redact

    try:
        raw = json.dumps(payload)
    except (TypeError, ValueError):
        return payload
    cleaned = redact(raw)
    if cleaned == raw:
        return payload
    try:
        return json.loads(cleaned)
    except ValueError:
        return payload


class CloudStore:
    """Checkpoint store that persists to the Papayya control plane via HTTP.

    Wraps every write in retry + journal-on-exhaust (ADR-0002 #8). The
    public ``CheckpointStore`` surface is unchanged — callers see no
    difference except that transient outages no longer raise.
    """

    def __init__(self, config: CloudStoreConfig) -> None:
        headers: dict[str, str] = {"Content-Type": "application/json", "Accept": "application/json"}
        if config.platform_auth:
            # No key at construction on the platform lane (plan 60 S1): the
            # per-request hook supplies this run's token. An empty header is
            # not set at all, so a request with no token in scope arrives
            # unauthenticated and is refused, rather than arriving with a
            # process-wide credential.
            if config.api_key:
                headers["X-Api-Key"] = config.api_key
        elif config.api_key.startswith("cpk_"):
            headers["X-Api-Key"] = config.api_key
        else:
            headers["Authorization"] = f"Bearer {config.api_key}"

        self._runs_base = config.runs_base.rstrip("/")
        self._client = httpx.Client(
            base_url=config.base_url,
            headers=headers,
            timeout=config.timeout,
            # Plan 56 F3. The lease this invocation runs under is stamped on
            # every write so the control-plane can reject one from a worker
            # whose lease was revoked — see _attach_lease_header.
            #
            # AN EVENT HOOK, NOT A CONSTRUCTOR HEADER, and that is forced: the
            # store is built ONCE per worker process, before the customer's
            # module is even imported, and the lease changes per item. A
            # header baked in here would pin every write for the process
            # lifetime to whichever lease happened to be first.
            event_hooks={"request": [_attach_run_token, _attach_lease_header]},
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
                    # Plan 44 / D6. A control pane predating 091 sends
                    # neither; None is "not tainted", which is the right
                    # reading of a row written before the concept existed.
                    tainted_by=cp.get("tainted_by"),
                    tainted_reason=cp.get("tainted_reason"),
                    # Plan 41 R3. A control pane predating 072 sends
                    # neither, and the defaults are exactly right there:
                    # one execution per label, attempt 1. The server does
                    # not echo the execution token (it is an input to the
                    # row-id derivation, not row state), so hydration
                    # leaves it empty — nothing downstream reads it off a
                    # loaded row, only off one being written.
                    attempt=cp.get("attempt", 1),
                    retry_count=cp.get("retry_count", 0),
                    retry_reason=cp.get("retry_reason"),
                    # Plan 58 U. A control pane predating 106 sends no such
                    # key, and None is exactly right there: nothing was
                    # reused, because nothing could be — that binary's
                    # replay recomputed everything by construction.
                    reused_from=cp.get("reused_from_run_id"),
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
            # Plan 41 R7. A control pane predating 077 sends no such key and
            # the empty default is exactly right there: nothing is
            # invalidated and the run resumes as it did before R7, which is
            # the behaviour that binary's own resume expects.
            pause_invalidated=[
                (e["label"], e["attempt"])
                for e in (body.get("pause_invalidated") or [])
                if isinstance(e, dict) and "label" in e and "attempt" in e
            ],
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
            "tainted_by": entry.tainted_by,
            "tainted_reason": entry.tainted_reason,
            # Plan 41 R3 — the execution identity. The server derives the
            # checkpoint row's primary key from (run_id, execution_token),
            # so this payload IS the idempotency of the write: a journal
            # drain reissues this exact dict and therefore lands on the
            # same row, which is what preserves ADR-0002 #8.
            #
            # completed_at is the client's own stamp, and sending it is
            # load-bearing rather than cosmetic — the server used to
            # substitute its own clock at request entry, which cannot
            # order a late-drained journal write against the execution
            # that replaced it.
            "execution_token": entry.execution_token,
            "attempt": entry.attempt,
            # Plan 58 R4 — what this execution cost. Omitted-safe on an older
            # control pane: it ignores unknown keys, and 0/None is the honest
            # reading of a row written before retries existed.
            "retry_count": entry.retry_count,
            "retry_reason": entry.retry_reason,
            "completed_at": entry.completed_at,
        }
        self._execute(
            kind="save_task",
            method="POST",
            url=f"{self._runs_base}/{run_id}/checkpoints",
            payload=payload,
            # Per EXECUTION, not per step: two executions of one label are
            # two distinct writes and must not dedupe into each other.
            idempotency_key=f"{run_id}:{entry.label}:{entry.attempt}",
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

        EXCEPT a 400 on a checkpoint write, which is RETAINED rather than
        dropped (Plan 41 R3 §T12). Dropping those makes rollback destroy
        lineage: if the control pane is rolled back past the release that
        accepts a field this SDK sends, every drained entry 400s,
        ``_is_transient`` says false, and we would permanently and
        silently discard exactly the writes ADR-0002 #8 exists to protect
        — at the worst possible moment, since the journal only has
        entries because the server was already failing. A retained entry
        costs a bounded amount of disk and drains itself once the schema
        catches up.
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
                elif _is_schema_rejection(entry, exc):
                    log.warning(
                        "retaining journal entry %s — the control pane rejected "
                        "its shape (%s). This is the rollback case: the write is "
                        "valid and will drain once the server accepts it again.",
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

        THE ONE PLACE EVERY CHECKPOINT BODY PASSES (plan 67 S1). Injected
        secret values are stripped here rather than at each of the six payload
        builders above, because a filter that has to be remembered at six call
        sites is a filter that will be missed at the seventh. Outside a hosted
        executor nothing is installed and :func:`_redact_payload` returns the
        same object.
        """
        resp = self._client.request(method, url, json=_redact_payload(payload))
        _raise_with_server_message(resp)
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


def make_runtime_store(base_url: str, api_key: str = "", *, timeout: float = 15.0) -> CloudStore:
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

    ``api_key`` DEFAULTS TO EMPTY AND SHOULD STAY THAT WAY (plan 60 S1). The
    credential now arrives per request as the run token the dispatcher minted
    with this item's lease — see ``_attach_run_token``. The parameter survives
    for the LocalDispatcher path and for tests that drive the lane directly;
    passing the platform worker key here re-opens exactly the hole S1 closes.
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

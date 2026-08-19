from __future__ import annotations

import json
import uuid as _uuid
from typing import Any, Iterator, TYPE_CHECKING

import httpx

from papayya.api import PapayyaAPIError

if TYPE_CHECKING:
    from papayya.api import APIClient


def _looks_like_uuid(value: str) -> bool:
    """Whether to spend a round trip treating `value` as OUR id first.

    Shape only. A customer's own key is allowed to be a uuid — order numbers
    from an upstream system often are — so this decides the ORDER of two
    lookups, never which one is authoritative.
    """
    try:
        _uuid.UUID(value)
    except (ValueError, AttributeError, TypeError):
        return False
    return True


def _ensure_run_id(resp: Any) -> Any:
    """Guarantee a stable ``run_id`` key on a submit response.

    The control-plane's durable-run record spells its identifier ``id``
    (the row PK); some routes carry ``durable_run_id`` instead. Rather than
    make callers guess which one a given deploy returns, copy whichever is
    present into ``run_id`` (without clobbering an existing one) so
    ``resp["run_id"]`` is ALWAYS valid and can be handed straight to
    ``.get()`` / ``.steps()`` / ``.stream()``. Purely additive — the raw
    server fields are left intact.
    """
    if isinstance(resp, dict) and "run_id" not in resp:
        for alias in ("id", "durable_run_id"):
            if alias in resp:
                return {**resp, "run_id": resp[alias]}
    return resp


class Items:
    """Per-item resource (Plan 34 rename of the old ``Runs`` resource).

    Every method here reads or mutates ONE ITEM — one record a run
    processed. The HTTP wire below is FROZEN at the pre-consolidation
    paths and field names (``/v1/durable/runs/{run_id}``, ``run_id`` /
    ``item_id`` / ``parent_run_id``) until Plan 34 Unit 5 renames the
    control-pane side; do not "fix" the paths to say items.

    Reachable as ``Papayya().items``. The old ``Papayya().runs`` name now
    addresses invocations (see ``resources/runs.py``) — that shift is a
    declared 0.3.0 breaking change, not an alias.
    """

    def __init__(self, api: APIClient) -> None:
        self._api = api

    def create(
        self,
        agent_id: str,
        input: Any,
        *,
        item_id: str | None = None,
        model: str | None = None,
        max_steps: int | None = None,
        budget_cents: int | None = None,
        callback_url: str | None = None,
        parent_run_id: str | None = None,
    ) -> dict[str, Any]:
        """Submit ONE run-of-one and return immediately, without blocking on
        execution — the agent runs off the request path on the worker pool.

        Returns the created durable-run record as a dict. The run identifier
        is guaranteed present under ``run_id`` (see :func:`_ensure_run_id`);
        pass it to :meth:`get`, :meth:`steps`, or :meth:`stream`, or supply a
        ``callback_url`` to have the terminal outcome POSTed to you instead of
        polling. For many items in one shot use ``Papayya().runs.create``.

        ``item_id`` is YOUR id for this record — an order id, a ticket id,
        whatever you key your world on. Optional, at most 256 bytes, and kept
        separate from ``input``: this route used to write the input onto that
        column, which is why a record could never be looked up by the id its
        owner actually knows it by.
        """
        body: dict[str, Any] = {"input": input}
        if item_id is not None:
            body["item_id"] = item_id
        if model:
            body["model"] = model
        if max_steps:
            body["max_steps"] = max_steps
        if budget_cents:
            body["budget_cents"] = budget_cents
        if callback_url:
            body["callback_url"] = callback_url
        # Sub-runs lineage (Layer 3 #7 Phase 2). Explicit kwarg wins;
        # else auto-pick the active @agent run's id when called from
        # inside an @agent body. Lazy import — keeps the resource module
        # importable without pulling in the agent contextvar machinery.
        resolved_parent = parent_run_id
        if resolved_parent is None:
            from papayya.agent import get_active_run_id
            resolved_parent = get_active_run_id()
        if resolved_parent:
            body["parent_run_id"] = resolved_parent
        return _ensure_run_id(
            self._api._request("POST", f"/v1/agents/{agent_id}/runs", json=body)
        )

    # v1→v2 cutover: a run is a durable_run. The read/poll surface below
    # targets /v1/durable/runs/*. The quarantine lifecycle (quarantine/
    # release/discard) targets the durable endpoints too. The v1 lifecycle
    # mutations (cancel / replay-from-step / dlq_*) retired with the v1 DROP
    # — their endpoints are gone and the durable triage-action lifecycle
    # (dlq_disposition) is a deferred follow-up.

    def get(self, item_id: str) -> dict[str, Any]:
        """Fetch ONE record, by our run id or by the customer's own key.

        The argument has been named ``item_id`` on the CLI since 0.3.0 and the
        help has said "fetch one hosted item by id" for as long — and it
        resolved ONLY a run uuid, so ``papayya items get DOC-2001`` answered
        "durable run not found" about a document the product had just named in
        the triage feed (plan 57 D5). Plan 43 B2a exists so a customer can
        address a record by their own key; this was the one command whose
        signature promised exactly that and the one that refused.

        Resolution order, and the ordering is a round-trip decision:

        * A uuid-shaped argument is tried as a run id first — that is what it
          almost always is, and the direct GET is one call.
        * Anything else goes straight to the key lookup; trying it as a run id
          would be a guaranteed 404.
        * A uuid-shaped argument that 404s falls through to the key lookup
          anyway, because a customer's own id is allowed to be a uuid.

        When a key names several records — which is the NORMAL state after a
        re-drive, since replay mints a new run id under the same key — the
        newest is returned. Use :meth:`resolve` to see all of them; the CLI
        does, and says so.
        """
        def by_run_id() -> dict[str, Any] | None:
            try:
                return self._api._request("GET", f"/v1/durable/runs/{item_id}")
            except PapayyaAPIError as e:
                # 404 only. A 403 means the caller has a real problem with THAT
                # run, and retrying it as a customer key would replace a clear
                # error with "not found".
                if e.status != 404:
                    raise
                return None

        def by_customer_key() -> dict[str, Any] | None:
            matches = self.resolve(item_id)
            return matches[0] if matches else None

        # BOTH are tried either way; the shape only decides which goes first.
        # run_id is a TEXT column, so "not uuid-shaped" is not proof it is not
        # a run id, and a customer's own key is allowed to be a uuid.
        order = (by_run_id, by_customer_key) if _looks_like_uuid(item_id) \
            else (by_customer_key, by_run_id)
        for lookup in order:
            found = lookup()
            if found is not None:
                return found

        raise PapayyaAPIError(
            404,
            f"no record found for {item_id!r} — it is not a run id, and no "
            f"item carries it as its item_id",
        )

    def resolve(self, item_id: str) -> list[dict[str, Any]]:
        """Every record carrying the customer's ``item_id``, newest first.

        More than one is the ordinary case for a document that was re-driven:
        the submission that failed and the replay that fixed it are two runs
        under one key, and until plan 57 D5 the only surface that joined them
        was the dashboard item page — which you can only reach if you already
        hold the uuid of the run that failed.
        """
        rows = self.list(item_id=item_id)
        if not isinstance(rows, list):
            return []
        return sorted(
            (r for r in rows if isinstance(r, dict)),
            key=lambda r: r.get("created_at") or "",
            reverse=True,
        )

    def list(
        self,
        *,
        run_id: str | None = None,
        agent: str | None = None,
        partition_key: str | None = None,
        item_id: str | None = None,
        item_id_prefix: str | None = None,
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        """List items, optionally scoped.

        THE SERVER HAS ALWAYS SUPPORTED THESE (plan 56 F6). ListRuns reads
        ``parent_run_id``, ``agent`` and ``partition_key`` off the query
        string; the filters grew for the dashboard, which is their only
        consumer, and this client was never brought forward. So
        ``papayya status`` printed "Items in this run: papayya items list",
        and ``items list`` took no options — the only route from a submitted
        run id to one wedged item was dumping every item in the project as
        NDJSON and reading for "running" (plan 54 N5, plan 55 D6.3).

        ``run_id`` maps to ``parent_run_id``: a submission's group id is the
        parent of the items it fanned out into (plan 42 — the group id IS the
        invocation id), so "the items in this run" is that filter.

        ``item_id`` / ``item_id_prefix`` / ``status`` ARE NEW ON BOTH SIDES
        (plan 57 D4). The server accepted those names and ignored them, so
        ``?item_id=DOC-2001`` returned every run in the project and a client
        that trusted it rendered one document's history under another's
        heading. They are honored now, and an unknown param is a 400 — so this
        method sending a name the server does not implement fails loudly
        instead of over-returning.

        ``item_id_prefix`` is the unit BELOW the document: ``DOC-2001#``
        selects every page-grained record of one photo report.
        """
        params: dict[str, str] = {}
        if run_id:
            params["parent_run_id"] = run_id
        if agent:
            params["agent"] = agent
        if partition_key:
            params["partition_key"] = partition_key
        if item_id:
            params["item_id"] = item_id
        if item_id_prefix:
            params["item_id_prefix"] = item_id_prefix
        if status:
            params["status"] = status
        path = "/v1/durable/runs"
        if params:
            from urllib.parse import urlencode

            path = f"{path}?{urlencode(params)}"
        return self._api._request("GET", path)

    def steps(self, run_id: str) -> list[dict[str, Any]]:
        # Durable checkpoints — {label, result, cost_usd, ...}, not the v1
        # {step_number, step_type, output} shape.
        return self._api._request("GET", f"/v1/durable/runs/{run_id}/checkpoints")

    def stream(
        self,
        run_id: str,
        *,
        from_step: int | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Stream steps for a run via Server-Sent Events.

        Yields one dict per event with keys ``event`` (``"step"``,
        ``"terminal"``, or ``"error"``) and ``data`` (decoded JSON payload).
        Step events also carry ``id`` — the step_number — usable as
        ``from_step`` to resume after a disconnect.

        The iterator exits when the run reaches a terminal status; a final
        ``terminal`` event is yielded first with ``data={"status": "..."}``.
        Backfill of existing steps happens before live tailing, so callers
        always see a complete history regardless of when they connect.

        Usage::

            for event in client.runs.stream(run_id):
                if event["event"] == "step":
                    print(f"step {event['id']}: {event['data']['step_type']}")
                elif event["event"] == "terminal":
                    print(f"run ended: {event['data']['status']}")

        Pass ``from_step`` with the highest step_number already observed to
        resume after a transient disconnect; the server skips backfill of
        those rows.
        """
        headers: dict[str, str] = {"Accept": "text/event-stream"}
        if from_step is not None:
            headers["Last-Event-ID"] = str(from_step)

        # Disable the read timeout for the stream body — SSE connections
        # can idle between steps for far longer than the default 30s. The
        # connect timeout stays in place so a dead server still fails fast.
        stream_timeout = httpx.Timeout(
            connect=self._api._config.timeout,
            read=None,
            write=self._api._config.timeout,
            pool=self._api._config.timeout,
        )
        with self._api._http.stream(
            "GET",
            f"/v1/durable/runs/{run_id}/events",
            headers=headers,
            timeout=stream_timeout,
        ) as response:
            if response.status_code != 200:
                body = response.read().decode("utf-8", errors="replace")
                from papayya.api import PapayyaAPIError

                raise PapayyaAPIError(response.status_code, body)
            yield from _parse_sse(response.iter_lines())

    # ── Quarantine ─────────────────────────────────────────────────────────
    # Quarantine is the non-terminal soft-pause lane (Plan 08/09): a run
    # paused mid-stream, in-flight state preserved, awaiting an operator
    # decision. Transitions: running ↔ quarantine, quarantine → cancelled.
    # Durable surface (/v1/durable/runs/*); Triage.list() surfaces the lane.

    def quarantine(self, run_id: str, reason: str) -> dict[str, Any]:
        """Move a durable run into the non-terminal quarantine lane.

        Reason is required; the server rejects an empty string. The run
        keeps its in-flight state — call ``release(run_id)`` to resume or
        ``discard(run_id)`` to abandon. 409 if the run isn't running.
        """
        return self._api._request(
            "POST",
            f"/v1/durable/runs/{run_id}/quarantine",
            json={"reason": reason},
        )

    def release(self, run_id: str) -> dict[str, Any]:
        """Exit quarantine by resuming the run in-place. Returns the
        updated run with ``quarantine_disposition='released'``. 409 if
        the run is not currently in quarantine."""
        return self._api._request("POST", f"/v1/durable/runs/{run_id}/release")

    def discard(self, run_id: str) -> dict[str, Any]:
        """Exit quarantine by abandoning the run. Returns the updated
        run with ``quarantine_disposition='discarded'`` and status
        ``cancelled``. 409 if the run is not currently in quarantine."""
        return self._api._request("POST", f"/v1/durable/runs/{run_id}/discard")

    # DLQ lane (Plan 41 R1) — the triage feed's other half: degraded/failed
    # runs that were never quarantined. These drain a row without changing
    # its status, because a terminal run's outcome is a historical fact and
    # the disposition is operator state. The third action the feed
    # advertises, "retry", is ``replay()`` — it writes
    # ``dlq_disposition='replayed'`` on the source run for you.

    def replay(
        self,
        run_id: str,
        *,
        tenant: str | None = None,
        latest: bool = False,
        force: bool = False,
    ) -> dict[str, Any]:
        """Re-drive a run's captured item as a NEW run, linked back via
        ``replayed_from``, and mark the source ``dlq_disposition='replayed'``
        so it drains from the triage feed.

        The lane-aware counterpart to ``release`` — this is what "retry"
        means for a terminal degraded/failed row. Delegates to the same
        endpoint as the top-level ``papayya replay`` command.
        """
        return self._api.replay_run(
            run_id, tenant=tenant, latest=latest, force=force
        )

    def dismiss(self, run_id: str) -> dict[str, Any]:
        """Drain a degraded/failed run out of the triage feed without
        re-driving it. Returns the updated run with
        ``dlq_disposition='skipped'``. 409 if the run isn't an un-triaged
        degraded/failed run, 404 if it doesn't exist."""
        return self._api._request("POST", f"/v1/durable/runs/{run_id}/dismiss")

    def acknowledge(self, run_id: str) -> dict[str, Any]:
        """Drain a degraded/failed run out of the triage feed, recording
        that it was seen rather than declined. Returns the updated run with
        ``dlq_disposition='acknowledged'``. Same preconditions as
        ``dismiss``; the difference is operator intent, preserved so a later
        audit can tell "I looked at this" from "I chose not to fix it"."""
        return self._api._request("POST", f"/v1/durable/runs/{run_id}/acknowledge")

    # Auto-pause (Plan 33) — the system-initiated counterpart to the
    # operator-initiated quarantine lane above. A run a degradation/budget
    # fence paused is resumed here; replay skips already-saved steps.

    def resume(self, run_id: str) -> dict[str, Any]:
        """Clear the pause on a run auto-paused by a fence (Plan 33).

        Transitions paused→running. 409 if the run is not paused.

        **This does not re-drive the run.** A paused run completes its
        lease with ``error_category="paused"``, so its item is no longer
        queued; clearing the status does not put it back. The run then
        reads as ``running`` with nothing working on it, which is worse
        than it reads. Use :meth:`replay` to actually re-drive the
        captured item. Plan 41 R6 makes this verb re-enqueue."""
        return self._api._request("POST", f"/v1/durable/runs/{run_id}/resume")

    def clusters(self, partition_key: str | None = None) -> dict[str, Any]:
        """Degraded/failed runs grouped by (reason, prompt-prefix) — the
        silent-failure clustering the dashboard's failure view renders.
        Optional ``partition_key`` narrows to one tenant."""
        params = {"partition_key": partition_key} if partition_key else {}
        return self._api._request("GET", "/v1/durable/runs/clusters", params=params)

    # Cohort lane (Plan 41 R4, ADR 0009 D7) — "every record where predicate P
    # held over window W". Distinct from every method above, which takes ONE
    # record's id. This is the input type of the recovery verbs.

    def cohort(
        self,
        *,
        agent: str | None = None,
        tenant: str | None = None,
        run_id: str | None = None,
        outcome: str | None = None,
        since: str | None = None,
        until: str | None = None,
        include_triaged: bool = False,
        limit: int | None = None,
    ) -> dict[str, Any]:
        """Select the records matching a predicate over a window.

        Returns ``{members, total, truncated}``. ``total`` is the FULL
        cohort size ignoring ``limit``, and ``truncated`` says plainly
        whether ``members`` is the whole of it — read both before acting on
        a cohort, or you will act on a page and believe it was the set.

        ``run_id`` is one optional term rather than the addressing scheme:
        nothing in the product mints a multi-item run, so a run-keyed
        selection returns that single record. ``tenant`` is the customer's
        own partition key. ``since`` / ``until`` are RFC3339. Records an
        operator already dispositioned are excluded unless
        ``include_triaged=True`` — re-driving those would refill the triage
        feed.
        """
        params: dict[str, Any] = {}
        for key, val in (
            ("agent", agent),
            ("tenant", tenant),
            ("run_id", run_id),
            ("outcome", outcome),
            ("from", since),
            ("to", until),
        ):
            if val is not None:
                params[key] = val
        if include_triaged:
            params["include_triaged"] = "true"
        if limit is not None:
            params["limit"] = str(limit)
        return self._api._request("GET", "/v1/durable/cohorts", params=params)


def _parse_sse(lines: Iterator[str]) -> Iterator[dict[str, Any]]:
    """Parse the SSE wire format into ``{event, data, id?}`` dicts.

    Minimal but correct: ignores comment frames (``:`` prefix), joins
    multi-line ``data:`` payloads with a literal newline, dispatches on
    blank line. Matches the subset of the SSE spec the control plane
    emits — no ``retry:`` handling, the caller is responsible for
    reconnect logic.
    """
    event_type = "message"
    data_lines: list[str] = []
    event_id: str | None = None
    for raw in lines:
        line = raw.rstrip("\r")
        if line == "":
            if data_lines:
                data_str = "\n".join(data_lines)
                try:
                    parsed: Any = json.loads(data_str)
                except json.JSONDecodeError:
                    parsed = data_str
                out: dict[str, Any] = {"event": event_type, "data": parsed}
                if event_id is not None:
                    out["id"] = event_id
                yield out
            event_type = "message"
            data_lines = []
            event_id = None
            continue
        if line.startswith(":"):
            continue
        if line.startswith("event:"):
            event_type = line[len("event:"):].strip()
        elif line.startswith("data:"):
            data_lines.append(line[len("data:"):].lstrip(" "))
        elif line.startswith("id:"):
            event_id = line[len("id:"):].strip()

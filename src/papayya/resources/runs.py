from __future__ import annotations

import json
from typing import Any, Iterator, TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    from papayya.api import APIClient


class Runs:
    def __init__(self, api: APIClient) -> None:
        self._api = api

    def create(
        self,
        agent_id: str,
        input: Any,
        *,
        model: str | None = None,
        max_steps: int | None = None,
        budget_cents: int | None = None,
        callback_url: str | None = None,
        parent_run_id: str | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {"input": input}
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
        return self._api._request("POST", f"/v1/agents/{agent_id}/runs", json=body)

    def get(self, run_id: str) -> dict[str, Any]:
        return self._api._request("GET", f"/v1/runs/{run_id}")

    def list(self) -> list[dict[str, Any]]:
        return self._api._request("GET", "/v1/runs")

    def cancel(self, run_id: str) -> dict[str, Any]:
        return self._api._request("POST", f"/v1/runs/{run_id}/cancel")

    def replay(
        self,
        run_id: str,
        *,
        from_step: int,
        latest: bool = False,
    ) -> dict[str, Any]:
        """Replay a durable run from a step.

        ``latest=False`` (the default) is the safe option: when the hosted
        version-mismatch gate (ADR-0002 #7) lands, the server will reject a
        replay whose registration's ``agent_version`` differs from the
        captured one. Pass ``latest=True`` to opt into running the current
        code anyway. Until the hosted gate is live the server ignores the
        flag — we ship the wire contract now so the SDK is forward-compatible.
        """
        return self._api._request(
            "POST",
            f"/v1/runs/{run_id}/replay",
            json={"from_step": from_step, "allow_version_mismatch": latest},
        )

    def steps(self, run_id: str) -> list[dict[str, Any]]:
        return self._api._request("GET", f"/v1/runs/{run_id}/steps")

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
            f"/v1/runs/{run_id}/events",
            headers=headers,
            timeout=stream_timeout,
        ) as response:
            if response.status_code != 200:
                body = response.read().decode("utf-8", errors="replace")
                from papayya.api import PapayyaAPIError

                raise PapayyaAPIError(response.status_code, body)
            yield from _parse_sse(response.iter_lines())

    # ── Dead Letter Queue ──────────────────────────────────────────────────
    # A failed/budget_exceeded run that belongs to a batch lands in the DLQ
    # until the operator triages it. Use one of:
    #   - dlq_skip       — accept the failure, don't replay
    #   - dlq_acknowledge — record review, don't replay
    #   - dlq_replay     — re-issue the run from input_snapshot
    # Once every failure in a batch has a disposition, the batch promotes
    # from 'partial' to 'completed'. See Batches.dlq() for the list endpoint.

    def dlq_skip(self, run_id: str) -> dict[str, Any]:
        """Mark a failed run as 'skipped' — accept the failure as terminal.
        Returns the updated run with dlq_disposition set."""
        return self._api._request("POST", f"/v1/runs/{run_id}/dlq/skip")

    def dlq_acknowledge(self, run_id: str) -> dict[str, Any]:
        """Mark a failed run as 'acknowledged' — record that the operator
        has reviewed the failure but is choosing to leave it. Functionally
        equivalent to skip; semantically distinct (skip ≈ "not worth
        looking at"; acknowledge ≈ "I've looked at this")."""
        return self._api._request("POST", f"/v1/runs/{run_id}/dlq/acknowledge")

    def dlq_replay(self, run_id: str) -> dict[str, Any]:
        """Re-issue the failed run from its input_snapshot as a new queued
        run. Marks the source as 'replayed' and links the new run via
        replayed_from. Returns the new run (HTTP 202)."""
        return self._api._request("POST", f"/v1/runs/{run_id}/dlq/replay")


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

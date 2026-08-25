"""HTTP client for the Papayya control plane API."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any

import httpx

from papayya._defaults import DEFAULT_BASE_URL


class PapayyaAPIError(Exception):
    def __init__(self, status: int, message: str):
        self.status = status
        super().__init__(f"HTTP {status}: {message}")


# Retry budget for transient control-plane failures. Same rhythm as
# durable/cloud_store.py so operators only learn one cadence: worst-case
# wait is ~0.1 + 0.2 + 0.4 + 0.8 + (capped) 2.0 = 3.5s before raising.
_MAX_ATTEMPTS = 5
_INITIAL_BACKOFF = 0.1
_MAX_BACKOFF = 2.0

# Connect-level exceptions where the request provably never reached the
# application, so a retry cannot double-apply a non-idempotent submit.
# Deliberately NARROWER than cloud_store._is_transient: read/write timeouts
# and RemoteProtocolError can fire *after* the server accepted the request,
# and run submission (unlike an idempotent checkpoint write) is not safe to
# replay in that window — a double-submit would re-pay every LLM call.
_TRANSIENT_CONNECT_EXC = (
    httpx.ConnectError,
    httpx.ConnectTimeout,
    httpx.PoolTimeout,
)

# Server-side statuses that mean the request was rejected/never processed by
# the app (rate-limited or gateway-level), so retrying is safe. 500 is
# excluded on purpose: it can be raised after partial processing.
_TRANSIENT_STATUS = frozenset({429, 502, 503, 504})


@dataclass
class APIConfig:
    api_key: str
    base_url: str = DEFAULT_BASE_URL
    timeout: float = 30.0


def resolve_config(
    api_key: str | None = None,
    base_url: str | None = None,
) -> APIConfig:
    """Resolve credentials for the hosted resource namespaces (.items/.runs).

    Resolution order matches the durable path
    (``papayya._resolve_durable_credentials``): explicit arg → ``PAPAYYA_API_KEY``
    → the CLI config file that ``papayya login`` writes
    (``~/.papayya/config.json``). Without the last step a developer who
    authed via the CLI would get a 401 the moment they touched ``.items``
    while the durable path silently succeeded — the asymmetry this closes.
    """
    # Lazy import: keep api.py free of package-level import ordering effects
    # (subprocess/WAL tests are sensitive to it) and avoid any cycle.
    from papayya._config import env_config, load_cli_config

    key = api_key or os.environ.get("PAPAYYA_API_KEY")
    url = base_url or os.environ.get("PAPAYYA_BASE_URL")
    if not key or not url:
        env = env_config(load_cli_config())
        # Pair key + base_url from the SAME env block so a CLI login to a
        # non-prod env doesn't hand back that env's key against prod's URL.
        if not key:
            key = env.get("api_key")
            url = url or env.get("base_url")
        if not url:
            url = env.get("base_url")

    if not key:
        raise PapayyaAPIError(
            401,
            "No API key. Pass api_key=..., set PAPAYYA_API_KEY, or run `papayya login`.",
        )
    return APIConfig(api_key=key, base_url=url or DEFAULT_BASE_URL)


class APIClient:
    """Thin wrapper around the control plane REST API."""

    def __init__(self, config: APIConfig) -> None:
        self._config = config
        headers: dict[str, str] = {
            "Accept": "application/json",
        }
        # API keys (cpk_...) use X-Api-Key header; JWTs use Authorization: Bearer
        if config.api_key.startswith("cpk_"):
            headers["X-Api-Key"] = config.api_key
        else:
            headers["Authorization"] = f"Bearer {config.api_key}"

        self._http = httpx.Client(
            base_url=config.base_url,
            timeout=config.timeout,
            headers=headers,
        )

    def close(self) -> None:
        self._http.close()

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        """Issue one control-plane request, retrying only provably-safe
        transient failures (connect errors + 429/502/503/504) with bounded
        exponential backoff. 4xx and 500 raise immediately — the former is a
        client bug, the latter may have partially applied and must not be
        blindly replayed. Non-transient failures raise on the first attempt.
        """
        backoff = _INITIAL_BACKOFF
        last_exc: PapayyaAPIError | None = None
        for attempt in range(1, _MAX_ATTEMPTS + 1):
            try:
                resp = self._http.request(method, path, **kwargs)
            except _TRANSIENT_CONNECT_EXC as exc:
                last_exc = PapayyaAPIError(0, f"connection error: {exc}")
            else:
                if resp.is_success:
                    return resp.json()
                if resp.status_code not in _TRANSIENT_STATUS:
                    raise PapayyaAPIError(resp.status_code, resp.text)
                last_exc = PapayyaAPIError(resp.status_code, resp.text)

            if attempt < _MAX_ATTEMPTS:
                time.sleep(backoff)
                backoff = min(backoff * 2, _MAX_BACKOFF)

        assert last_exc is not None  # loop runs ≥1 time and only breaks with a set exc
        raise last_exc

    # -- Auth ----------------------------------------------------------------

    def login(self, email: str, password: str) -> dict[str, Any]:
        return self._request("POST", "/v1/auth/login", json={"email": email, "password": password})

    def register(self, email: str, password: str, name: str) -> dict[str, Any]:
        return self._request("POST", "/v1/auth/register", json={"email": email, "password": password, "name": name})

    # -- Projects ------------------------------------------------------------

    def create_project(self, name: str, slug: str) -> dict[str, Any]:
        return self._request("POST", "/v1/projects", json={"name": name, "slug": slug})

    def list_projects(self) -> list[dict[str, Any]]:
        return self._request("GET", "/v1/projects")

    # -- API Keys ------------------------------------------------------------

    def create_api_key(self, project_id: str, name: str) -> dict[str, Any]:
        return self._request("POST", f"/v1/projects/{project_id}/api-keys", json={"name": name})

    # -- Agents --------------------------------------------------------------

    def deploy_agent(self, agent_def: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", "/v1/agents", json=agent_def)

    def create_agent(self, project_id: str, name: str, slug: str, config: dict[str, Any] | None = None) -> dict[str, Any]:
        body: dict[str, Any] = {"project_id": project_id, "name": name, "slug": slug}
        if config:
            body["config"] = config
        return self._request("POST", "/v1/agents", json=body)

    def list_agents(self, project_id: str | None = None) -> list[dict[str, Any]]:
        params = {}
        if project_id:
            params["project_id"] = project_id
        return self._request("GET", "/v1/agents", params=params)

    def get_agent(self, agent_id: str) -> dict[str, Any]:
        return self._request("GET", f"/v1/agents/{agent_id}")

    # -- Runs ----------------------------------------------------------------

    def trigger_run(
        self,
        agent_id: str,
        *,
        model: str,
        system_prompt: str,
        input_data: Any,
        max_steps: int = 50,
        budget_cents: int = 500,
        item_id: str | None = None,
        partition_key: str | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "agent_id": agent_id,
            "model": model,
            "system_prompt": system_prompt,
            "input": input_data,
            "max_steps": max_steps,
            "budget_cents": budget_cents,
        }
        # Omitted rather than sent as null (plan 67 S2): the route treats an
        # absent item_id as "the caller declared none" and mints a surrogate,
        # which is a different thing from declaring the empty string.
        if item_id:
            body["item_id"] = item_id
        # The tenant this work belongs to. Onboarding calls it the concept that
        # makes the product worth having, and neither cloud path could set it
        # until plan 67 S10.
        if partition_key:
            body["partition_key"] = partition_key
        return self._request("POST", "/v1/runs", json=body)

    def resume_run(self, run_id: str) -> dict[str, Any]:
        """Resume a paused run — clear the fence and re-drive what it stopped.

        The verb for a run something DECIDED to stop, as distinct from
        ``replay_run``, which is the verb for a run that ENDED badly. The
        server refuses each on the other's states (409), and until plan 53
        the CLI exposed only replay — so a fenced run had no way out of the
        terminal at all.

        The response carries two numbers worth reading together:
        ``redriven`` (was a parked item actually re-queued) and
        ``reexecuting`` (how many steps the fence objected to will run
        again). ``redriven=false`` means the pause was cleared but the lease
        had already been completed, so there is nothing to re-queue — replay
        it instead.
        """
        return self._request("POST", f"/v1/durable/runs/{run_id}/resume")

    def replay_run(
        self,
        run_id: str,
        *,
        tenant: str | None = None,
        latest: bool = False,
        force: bool = False,
        fresh: bool = False,
    ) -> dict[str, Any]:
        """Replay a run: mint a new run that re-drives its captured item.

        Hosted replay (Plan 37 Unit R) — the server-side rebuild of the
        wedge's "replay what didn't work". ``tenant`` narrows to one
        partition slice; ``latest`` re-runs on the agent's current version
        (default: the original run's version); ``force`` re-drives a clean
        run (default: only a run that didn't work is replayable).

        ``fresh`` re-executes every step instead of reusing the source run's
        completed ones (plan 58 U). Reuse is the default because the old
        behaviour was not the conservative one — it wrote 39 of 40 pages
        downstream a second time to fix the 40th.
        """
        params: dict[str, str] = {}
        if tenant is not None:
            params["tenant"] = tenant
        if latest:
            params["latest"] = "true"
        if fresh:
            params["fresh"] = "true"
        if force:
            params["force"] = "true"
        return self._request(
            "POST", f"/v1/durable/runs/{run_id}/replay", params=params
        )

    # -- Cohorts (Plan 41 R4 + Plan 39 S3; ADR 0009 D7 and §4 step 3) --------
    #
    # THERE IS EXACTLY ONE PARAMS BUILDER BELOW, AND THAT IS THE POINT. The
    # three cohort verbs take the same selection, and the server parses it with
    # one shared function so an operator cannot approve one cohort and re-drive
    # a different one. A client that re-encoded the rules per method would give
    # that guarantee back at the edge — and this file used to have two copies of
    # the loop already. Plan 39 S3 added two more selector doors, which would
    # have made six. One builder; every verb calls it.
    #
    # The dashboard's `predicateQuery` (dashboard/src/lib/api/cohorts.ts) is the
    # same function in TypeScript, with the same early-return shape. If you
    # change the rules, change all three.

    _COHORT_TERMS = ("agent", "tenant", "run_id", "outcome", "from", "to")

    @staticmethod
    def _cohort_params(
        *,
        probe: str | None = None,
        drift_episode: str | None = None,
        agent: str | None = None,
        tenant: str | None = None,
        run_id: str | None = None,
        outcome: str | None = None,
        since: str | None = None,
        until: str | None = None,
        flagged: bool = False,
        include_triaged: bool = False,
        limit: int | None = None,
    ) -> dict[str, str]:
        """Build the query params for any cohort verb.

        THREE DOORS, MUTUALLY EXCLUSIVE. A selection comes from exactly one
        of:

        * ``probe`` — a frozen by-example predicate (Plan 39 S3). Every term
          it stands for (bare label, window, condition token, borrowed floor)
          lives server-side on the ``cohort_probes`` row.
        * ``drift_episode`` — the change-detection handoff (Plan 40 U3 C6),
          resolved server-side off the episode row.
        * the descriptive terms — agent/tenant/run/outcome/window.

        The first two are opaque on purpose: emitting their terms for a
        client to pass back would make the preview and the re-drive two
        client-built selections that can disagree, and would put a
        server-learned floor on the wire for a verb that re-executes work.

        Passing a selector *and* a descriptive term raises here rather than
        travelling: the operator has two different selections in mind, and
        quietly letting one win is how someone approves one cohort and
        re-drives another. The server refuses it too — this is the earlier,
        better-worded copy of the same refusal, not the authority.

        ``since``/``until`` are the exception on the probe path, where they
        NARROW the frozen window rather than describing a selection. A
        by-example probe is unbounded by construction, and the release cap
        (1000) is reachable, so without them the server's own "narrow the
        window and release in passes" advice would name a flag that did not
        exist.

        ``flagged`` is legal alongside a selector on every door — it selects
        items a human said were wrong (plan 43 B1, ADR 0009 D3), and
        "everything like this record that someone also complained about" is the
        query backflow and search-by-example make together. On a selector door
        it can only ever remove items, because the resolved outcome is
        already ``any``.
        """
        if probe and drift_episode:
            raise ValueError(
                "pass either probe or drift_episode, not both — each selects "
                "its own items"
            )

        selector = "probe" if probe else ("drift_episode" if drift_episode else None)
        if selector is not None:
            described = {
                "agent": agent, "tenant": tenant, "run_id": run_id,
                "outcome": outcome,
            }
            # since/until narrow a probe's frozen window; they still describe a
            # selection on the drift path, where the episode owns the window.
            if selector == "drift_episode":
                described["since"] = since
                described["until"] = until
            offending = sorted(k for k, v in described.items() if v is not None)
            if offending:
                raise ValueError(
                    f"{selector} selects its own items; remove "
                    + ", ".join(offending)
                )

        params: dict[str, str] = {}
        if probe:
            params["probe"] = probe
            for key, val in (("from", since), ("to", until)):
                if val is not None:
                    params[key] = val
        elif drift_episode:
            params["drift_episode"] = drift_episode
        else:
            for key, val in (
                ("agent", agent), ("tenant", tenant), ("run_id", run_id),
                ("outcome", outcome), ("from", since), ("to", until),
            ):
                if val is not None:
                    params[key] = val

        # Legal alongside a selector for the reason in the docstring: on a door
        # that resolves its own predicate, `flagged` intersects an already-`any`
        # selection and can only remove records from it.
        if flagged:
            params["flagged"] = "true"

        # Neither of these describes WHICH records the predicate picks out, so
        # both are legal alongside a selector: include_triaged re-admits work
        # someone already dispositioned, and limit pages the response.
        if include_triaged:
            params["include_triaged"] = "true"
        if limit is not None:
            params["limit"] = str(limit)
        return params

    def get_cohort(
        self,
        *,
        probe: str | None = None,
        drift_episode: str | None = None,
        agent: str | None = None,
        tenant: str | None = None,
        run_id: str | None = None,
        outcome: str | None = None,
        since: str | None = None,
        until: str | None = None,
        flagged: bool = False,
        include_triaged: bool = False,
        limit: int | None = None,
    ) -> dict[str, Any]:
        """Select items by predicate over a window (Plan 41 R4, ADR 0009 D7).

        Returns ``{members, total, truncated}``. ``total`` is the FULL
        cohort size ignoring ``limit``; ``truncated`` says whether
        ``members`` is the whole of it. Read both — acting on a page while
        believing it is the set is the mistake this response shape exists
        to prevent.

        ``run_id`` is one optional predicate term, not the addressing
        scheme: nothing in the product mints a multi-item run, so a
        run-keyed selection returns that single item.

        ``flagged=True`` narrows to items a human said were wrong (plan 43
        B1). It IMPLIES ``outcome="any"`` unless you pass one: the items
        people flag are usually the ones that completed ok, so the default
        would answer zero and read as good news.

        See :meth:`_cohort_params` for the three mutually-exclusive doors.
        """
        return self._request("GET", "/v1/durable/cohorts", params=self._cohort_params(
            probe=probe, drift_episode=drift_episode, agent=agent, tenant=tenant,
            run_id=run_id, outcome=outcome, since=since, until=until,
            flagged=flagged, include_triaged=include_triaged, limit=limit,
        ))

    def record_verification(self, receipt: dict[str, Any]) -> dict[str, Any]:
        """Leave a receipt saying a cohort was verified (plan 64 D2).

        The one control-plane call ``papayya verify`` makes, and it happens
        AFTER the verdict is computed: verify runs the customer's new code
        locally and its answer is valid whether or not this succeeds. The
        caller is expected to swallow failures for exactly that reason — see
        the CLI's ``verify`` command.

        ``receipt`` carries counts and the cohort predicate. It carries no
        input, no output and no step trace; those are the customer's content
        and the browser does not need them to render a sentence.
        """
        return self._request("POST", "/v1/durable/verifications", json=receipt)

    def latest_verification(self, agent: str) -> dict[str, Any] | None:
        """The newest receipt for one agent, or ``None`` if nobody has verified.

        ``None`` rather than an exception: "not verified" is the commonest true
        answer, and it is the answer the Release button most needs to render.
        """
        payload = self._request(
            "GET", "/v1/durable/verifications/latest", params={"agent": agent}
        )
        return (payload or {}).get("receipt")

    def derive_probes(
        self,
        record: str,
        *,
        since: str | None = None,
        until: str | None = None,
    ) -> dict[str, Any]:
        """Search by example: what selects everything like this item?

        ADR 0009 §4 step 3 — *"paste one known-bad record, get everything
        like it across the last 60 days. Writing a predicate against history
        is the power-user path; pointing at a bad one is what actually
        happens at 2am."*

        Returns ``{item, agent, from, to, proposals, refusals,
        window_note}``. Each proposal carries a ``probe`` id to hand to
        :meth:`get_cohort` or :meth:`release_cohort`, and a ``count`` — the
        blast radius, which is what makes it something to decide about.

        ``refusals`` are conditions that could have applied and could not be
        derived, each with its reason (most often: the population's baseline
        is still forming, so there is no trustworthy floor to compare
        against). They are returned rather than dropped because a silently
        missing proposal is indistinguishable from a condition the item
        did not satisfy.

        Writes nothing but the saved predicates; re-executes nothing.
        """
        params = {"record": record}
        for key, val in (("from", since), ("to", until)):
            if val is not None:
                params[key] = val
        return self._request("POST", "/v1/durable/cohorts/like", params=params)

    def release_cohort(
        self,
        *,
        probe: str | None = None,
        drift_episode: str | None = None,
        agent: str | None = None,
        tenant: str | None = None,
        run_id: str | None = None,
        outcome: str | None = None,
        since: str | None = None,
        until: str | None = None,
        flagged: bool = False,
        include_triaged: bool = False,
        latest: bool = False,
    ) -> dict[str, Any]:
        """Re-drive every item the predicate selects (Plan 41 R4 C5, C6).

        Same predicate parser as :meth:`get_cohort` on the server, so what
        an operator previews is what they re-drive. Deliberately takes no
        ``limit``: a release acts on the WHOLE cohort or fails, and the
        server refuses one larger than it will re-drive in a single call
        rather than silently acting on a page.

        Returns ``{released, cohort_total, members, skipped_not_terminal,
        skipped_agent_missing, cost_note}``. Raises with a count when the
        cohort exceeds the remaining plan quota — nothing is released in
        that case, because a half-released cohort mid-incident is the worst
        available outcome.

        ON THE PROBE PATH, THIS SELECTS SLIGHTLY FEWER RECORDS THAN THE
        PREVIEW, deliberately. A by-example preview admits items that are
        themselves re-drives — an operator asking "what else looks like
        this" wants the truth about their whole history. Release excludes
        them, so nobody is offered a re-drive of a re-drive. It can only
        ever remove from what was previewed, never add, so nothing unseen
        is released.

        ``flagged`` DOES NOT IMPLY AN OUTCOME HERE, unlike
        :meth:`get_cohort`. On a hand-written predicate the flag is
        outcome-widening — the items people flag are usually the ones that
        completed ok — so a release that inherited the preview's implied
        ``any`` would re-drive items the operator's own predicate excludes.
        Say which you mean. The server refuses the same request; this is the
        earlier, better-worded copy.
        """
        if flagged and outcome is None and not (probe or drift_episode):
            raise ValueError(
                "flagged selects items that completed ok, so a re-drive "
                "would act on items the default predicate excludes. Pass "
                "outcome='any' to re-drive them, or outcome='not_ok' for the "
                "flagged items that also failed."
            )
        return self._request("POST", "/v1/durable/cohorts/replay", params=self._cohort_params(
            probe=probe, drift_episode=drift_episode, agent=agent, tenant=tenant,
            run_id=run_id, outcome=outcome, since=since, until=until,
            flagged=flagged, include_triaged=include_triaged,
        ) | ({"latest": "true"} if latest else {}))

    def get_run(self, run_id: str) -> dict[str, Any]:
        # v1→v2 cutover: a triggered run is now a durable_run. The v1
        # /v1/runs/{id} surface reads the (now-unfed) runs table, so poll
        # the durable run instead. Response carries status + checkpoints.
        return self._request("GET", f"/v1/durable/runs/{run_id}")

    def get_run_v2(self, run_id: str) -> dict[str, Any] | None:
        """The SUBMISSION rollup for a group id, or None when there is no such
        run. Distinct from get_run, which reads the per-ITEM surface — the two
        nouns live at different paths and a customer holds one id."""
        try:
            return self._request("GET", f"/v2/runs/{run_id}")
        except PapayyaAPIError as e:
            if getattr(e, "status", None) == 404:
                return None
            raise

    def get_steps(self, run_id: str) -> list[dict[str, Any]]:
        # Steps are durable checkpoints after the cutover. Each item is
        # {label, result, cost_usd, duration_ms, ...} — not the v1
        # {step_number, step_type, output} shape.
        return self._request("GET", f"/v1/durable/runs/{run_id}/checkpoints")

    # v1→v2 cutover: cancel_run AND the tool-call worker bridge
    # (poll_tool_calls/resolve_tool_call → /v1/tool-calls/*) retired with the
    # v1 DROP. Those routes and the tool_calls table are gone server-side
    # (control-pane slice 1 + migration 063); durable runs have no cancel verb
    # (use the quarantine→discard lifecycle instead).

    # -- Deployments ---------------------------------------------------------

    def upload_deployment(
        self,
        agent_id: str,
        tarball: bytes,
        runtime: str = "python",
        entrypoint: str = "agent.py",
    ) -> dict[str, Any]:
        """Upload a deployment artifact (multipart)."""
        import io
        resp = self._http.post(
            f"/v1/agents/{agent_id}/deploy",
            files={"file": ("artifact.tar.gz", io.BytesIO(tarball), "application/gzip")},
            data={"runtime": runtime, "entrypoint": entrypoint},
        )
        if not resp.is_success:
            raise PapayyaAPIError(resp.status_code, resp.text)
        return resp.json()

    def get_deployment(self, deployment_id: str) -> dict[str, Any]:
        return self._request("GET", f"/v1/deployments/{deployment_id}")

    def list_deployments(self, agent_id: str) -> list[dict[str, Any]]:
        return self._request("GET", f"/v1/agents/{agent_id}/deployments")

    # -- Secrets -------------------------------------------------------------

    def set_secret(self, project_id: str, name: str, value: str) -> dict[str, Any]:
        return self._request("POST", f"/v1/projects/{project_id}/secrets", json={"name": name, "value": value})

    def list_secrets(self, project_id: str) -> list[dict[str, Any]]:
        return self._request("GET", f"/v1/projects/{project_id}/secrets")

    def delete_secret(self, project_id: str, name: str) -> None:
        resp = self._http.request("DELETE", f"/v1/projects/{project_id}/secrets/{name}")
        if not resp.is_success:
            raise PapayyaAPIError(resp.status_code, resp.text)

    # -- Schedules -----------------------------------------------------------

    def list_schedules(self, agent_id: str) -> list[dict[str, Any]]:
        return self._request("GET", f"/v1/agents/{agent_id}/schedules")

    def create_schedule(
        self,
        agent_id: str,
        cron_expression: str,
        timezone: str = "UTC",
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            f"/v1/agents/{agent_id}/schedules",
            json={"cron_expression": cron_expression, "timezone": timezone},
        )

    def delete_schedule(self, schedule_id: str) -> None:
        resp = self._http.request("DELETE", f"/v1/schedules/{schedule_id}")
        if not resp.is_success:
            raise PapayyaAPIError(resp.status_code, resp.text)

    def put_schedules(
        self,
        agent_id: str,
        schedules: list[dict[str, Any]],
        *,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Replace all code-managed schedules for an agent in one call.

        ``schedules`` is a list of dicts each shaped like the POST body
        (``cron_expression`` + optional ``timezone`` / ``input`` /
        ``budget_cents``), sent verbatim. The server scopes its full-replace
        to ``managed_by='code'`` rows on its own — ``managed_by='api'`` rows
        (dashboard / direct-POST) are invisible to this call — and reports
        which scope it applied in the response.

        ``dry_run=True`` flips the server into preview mode: the same
        diff is computed against current ``managed_by='code'`` rows but
        no rows are mutated. The response shape changes to the diff
        envelope (``managed_by``, ``create``, ``update``, ``delete``,
        ``unmanaged_skipped``) — see Plan 13 for the consumer.

        Returns the server's apply-mode ``{items, summary}`` envelope by
        default, or the dry-run diff envelope when ``dry_run=True``.
        """
        # NOT `{**item, "managed_by": "code"}` (plan 53). The server decides
        # the scope and SAYS SO in its response envelope's `managed_by` field;
        # sending it was redundant, and its request struct has no such field,
        # so DisallowUnknownFields rejected the whole body:
        #
        #     PUT /v1/agents/{id}/schedules?dry_run=true
        #     400 {"error":{"message":"invalid request body"}}
        #
        # which is every `papayya deploy` of an agent carrying a @schedule.
        body = {"items": list(schedules)}
        path = f"/v1/agents/{agent_id}/schedules"
        if dry_run:
            path = f"{path}?dry_run=true"
        return self._request("PUT", path, json=body)

    # -- Webhooks ------------------------------------------------------------

    def list_webhooks(self, agent_id: str) -> list[dict[str, Any]]:
        return self._request("GET", f"/v1/agents/{agent_id}/webhooks")

    def create_webhook(self, agent_id: str, name: str) -> dict[str, Any]:
        """Create a webhook. Response `secret` + `trigger_url` are only visible here."""
        return self._request(
            "POST",
            f"/v1/agents/{agent_id}/webhooks",
            json={"name": name},
        )

    def delete_webhook(self, webhook_id: str) -> None:
        resp = self._http.request("DELETE", f"/v1/webhooks/{webhook_id}")
        if not resp.is_success:
            raise PapayyaAPIError(resp.status_code, resp.text)

    def put_webhooks(
        self,
        agent_id: str,
        webhooks: list[dict[str, Any]],
        *,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Replace all code-managed webhooks for an agent in one call.

        ``webhooks`` is a list of dicts each shaped like the POST body
        (``name`` + optional ``description``). Each item carries
        ``managed_by='code'`` on the wire — ``managed_by='api'`` rows
        are not touched.

        ``dry_run=True`` flips the server into preview mode: the proposed
        diff is computed and returned without generating any new webhook
        secrets and without mutating any row. Use Plan 13's CLI renderer
        to surface the diff to the operator.

        Returns the server's apply-mode ``{items, summary}`` envelope by
        default (newly-created rows carry ``secret`` + ``trigger_url``
        exactly once), or the dry-run diff envelope when
        ``dry_run=True`` (no secret in the response).
        """
        # Same as put_schedules: the server owns the scope and rejects an
        # unknown `managed_by` (plan 53). Every `papayya deploy` of an agent
        # carrying a @trigger 400'd on it.
        body = {"items": list(webhooks)}
        path = f"/v1/agents/{agent_id}/webhooks"
        if dry_run:
            path = f"{path}?dry_run=true"
        return self._request("PUT", path, json=body)

    # -- Rate card -----------------------------------------------------------

    def get_rate_card(self, project_id: str) -> dict[str, Any]:
        """Return the project's per-model rate card. Empty dict if unset."""
        return self._request("GET", f"/v1/projects/{project_id}/rate-card")

    def set_rate_card(self, project_id: str, rate_card: dict[str, Any]) -> dict[str, Any]:
        """Replace the project's rate card wholesale."""
        return self._request("PUT", f"/v1/projects/{project_id}/rate-card", json=rate_card)

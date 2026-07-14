"""Agent definition and @agent decorator for cloud deployment.

Papayya does NOT ship LLM provider adapters — you call your LLM SDK
(anthropic, openai, bedrock, ...) directly inside your agent function,
and decorate it with ``@agent`` so the platform knows how to deploy and
meter it.

Usage::

    from papayya import agent

    @agent(name="ops-assistant", model="gpt-4o-mini", budget_usd=1.0)
    def ops_assistant(input_data):
        from openai import OpenAI
        client = OpenAI()
        # ... your agent loop ...
        return result

The decorated function remains callable as a normal function. On deploy,
``papayya deploy`` discovers all ``@agent``-decorated functions in the
file and deploys each one.
"""

from __future__ import annotations

import functools
import inspect
import logging
import os
import shutil
import subprocess
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Callable

from papayya import _serialize
from papayya._config import ScheduleSpec, WebhookSpec
from papayya._otel_baggage import (
    annotate_current_span,
    clear_papayya_baggage,
    set_papayya_baggage,
)
from papayya.tools import ToolDefinition

log = logging.getLogger("papayya.agent")


# ---------------------------------------------------------------------------
# Per-call agent input snapshot.
#
# The @agent wrapper captures the function's call args here so that
# DurableRun.init() can populate runs.input_snapshot — the column
# `runs.replay()` / dlq replay / `papayya replay` all read.
#
# Without this bridge, every run is created with input_snapshot=NULL and
# replay surfaces error out with "no input_snapshot — cannot replay."
# ---------------------------------------------------------------------------

_AGENT_INPUT: ContextVar[Any] = ContextVar("papayya_agent_input", default=None)


def consume_agent_input_snapshot() -> Any:
    """Return the current agent's captured input args, or None.

    Called by DurableRun.init() when seeding a fresh RunCheckpoint. The
    contextvar stays set across the fn body so multiple runs created
    inside one @agent call all inherit the same input — that matches
    intent: the snapshot describes the *agent invocation*, not a run.
    """
    return _AGENT_INPUT.get()


# True when control is inside an @agent wrapper that took the legacy
# (non-injected) path — i.e. the fn does not declare `run` as its first
# positional parameter. Read by Papayya.run() so it can fire the Layer 3
# #9 deprecation warning at the exact call site the customer needs to
# delete.
_LEGACY_AGENT_PATH_ACTIVE: ContextVar[bool] = ContextVar(
    "papayya_legacy_agent_active", default=False
)


def legacy_agent_path_active() -> bool:
    """True when an @agent wrapper above this frame took the legacy
    (no `run` parameter) path. Used by Papayya.run() to gate the
    deprecation warning."""
    return _LEGACY_AGENT_PATH_ACTIVE.get()


# Sub-runs lineage (Layer 3 #7 Phase 2). When an @agent body is
# executing, this holds the active outer run's id; any Papayya.run()
# call made from inside that body picks it up as parent_run_id (unless
# the caller passes an explicit parent_run_id= to opt out / override).
# The wrapper sets it AFTER creating the outer run object — so the outer
# run itself stays parented to whatever was set when it was created
# (None at the top level, or the grandparent if @agent was somehow
# nested).
_ACTIVE_RUN_ID: ContextVar[str | None] = ContextVar(
    "papayya_active_run_id", default=None
)


def get_active_run_id() -> str | None:
    """Return the run_id of the @agent body currently on this stack,
    or None when called outside an @agent body. Used by Papayya.run()
    to auto-set parent_run_id on child runs."""
    return _ACTIVE_RUN_ID.get()


# v1→v2 execution cutover (Workstream C). The hosted worker pre-creates
# the durable run on submission (durable_run(queued)) and carries its
# run_id on the lease. Before invoking the @agent fn, the worker sets
# this contextvar to that run_id so the FIRST Papayya.run() inside the
# fn body adopts it — tying the SDK's checkpoints to the exact run the
# submission created (rather than minting a fresh, orphaned run_id).
#
# One-shot, exactly like _REPLAY_HYDRATION: consume_bootstrap_run_id()
# clears it on read so only the outer @agent run links to the lease;
# any sub-runs the body spawns get fresh ids. Unset (the default) on the
# local-dev path (LocalDispatcher leases carry no run_id), so local runs
# mint their own id as before — the guardrail stays green.
_BOOTSTRAP_RUN_ID: ContextVar[str | None] = ContextVar(
    "papayya_bootstrap_run_id", default=None
)


def set_bootstrap_run_id(run_id: str | None):
    """Set the lease's run_id for the next Papayya.run() to adopt.
    Returns the contextvar token; the worker resets it in a finally."""
    return _BOOTSTRAP_RUN_ID.set(run_id)


def reset_bootstrap_run_id(token) -> None:
    """Restore the bootstrap-run-id contextvar to its prior value."""
    _BOOTSTRAP_RUN_ID.reset(token)


def consume_bootstrap_run_id() -> str | None:
    """One-shot read of the worker-injected run_id. Returns the lease's
    run_id when a worker set it for this invocation, None otherwise.
    Clears on read so only the first Papayya.run() in the @agent body
    adopts it (sub-runs mint fresh ids)."""
    value = _BOOTSTRAP_RUN_ID.get()
    if value is not None:
        _BOOTSTRAP_RUN_ID.set(None)
    return value


# ---------------------------------------------------------------------------
# Module-level registry — maps function name → AgentRegistration
# ---------------------------------------------------------------------------

@dataclass
class AgentRegistration:
    """An @agent-decorated function and its metadata."""
    name: str
    model: str
    instructions: str
    fn: Callable
    tools: list[ToolDefinition]
    max_steps: int
    budget_usd: float | None
    durable: bool = False
    # Per-agent default for the worker's wall-clock watchdog. None
    # disables the watchdog entirely. Per-call overrides via the
    # dispatcher payload take priority. ADR-0002 #2.
    max_duration_seconds: float | None = None
    # Version tag that gets stamped on every run + step the worker
    # produces under this registration. Resolved at decoration time via
    # ``_resolve_agent_version``: explicit decorator arg → env var →
    # git short SHA → "unknown". Replay refuses to use a registration
    # whose version doesn't match the original run unless --latest.
    # ADR-0002 #7.
    agent_version: str = "unknown"
    # Cap on concurrent in-flight items per partition_key value (where
    # partition_key is the metadata field declared in papayya.yaml). When
    # the cap is hit, additional items stay in runtime_pending until an
    # in-flight one finishes. None disables the cap. Layer 3 #1.
    concurrency_per_key: int | None = None
    # Cap on lease throughput per partition_key value, in requests/min.
    # Sliding window. None disables the cap. Layer 3 #2. Source format
    # at the decorator is "N/min" or "N/sec" — parsed once and stored
    # here as int RPM.
    rate_limit_per_min: int | None = None
    # Decorator-attached schedule + webhook metadata (Plan 11). Populated
    # by papayya.decorators.schedule / .trigger when those decorators
    # wrap an already-@agent-wrapped function. Empty by default — agents
    # without @schedule / @trigger continue to declare triggers in
    # papayya.yaml. Plan 12's bundler-harvest path reads these lists to
    # synthesise the EnvSpec the reconciler consumes.
    schedules: list[ScheduleSpec] = field(default_factory=list)
    webhooks: list[WebhookSpec] = field(default_factory=list)
    # Plan 35 — customer outcome checks (deterministic callables and/or
    # llm_judge scaffolds). They run in the same _post_call_success pipeline
    # as the built-in inspectors on every step this agent's runs execute; the
    # worst verdict wins the run's worst_outcome_status. Empty by default.
    checks: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Rate-limit string parser (Layer 3 #2)
#
# Accepts ``"N/min"`` or ``"N/sec"``; normalises to integer requests per
# minute. Fails fast at decoration time so a typo doesn't ship silently
# disabled. ``"100/sec"`` → 6000; ``"100/min"`` → 100.
# ---------------------------------------------------------------------------

def _parse_rate_limit(value: str) -> int:
    raw = value.strip()
    if "/" not in raw:
        raise ValueError(
            f"rate_limit must be 'N/min' or 'N/sec', got {value!r}"
        )
    n_str, unit = (part.strip() for part in raw.split("/", 1))
    try:
        n = int(n_str)
    except ValueError as exc:
        raise ValueError(
            f"rate_limit numerator must be an integer, got {n_str!r}"
        ) from exc
    if n <= 0:
        raise ValueError(f"rate_limit must be > 0, got {n}")
    unit = unit.lower()
    if unit in ("min", "minute", "m"):
        return n
    if unit in ("sec", "second", "s"):
        return n * 60
    raise ValueError(
        f"rate_limit unit must be 'min' or 'sec', got {unit!r}"
    )


# ---------------------------------------------------------------------------
# OTel baggage extractors (Plan 07).
#
# The @agent wrapper has the workload name in scope (the decorator's
# ``name`` arg) but item_id and partition_key live in the call args /
# kwargs. Both helpers return ``None`` on absence — the baggage helper
# treats ``None`` as "skip this key" so the matching column on
# usage_events stays NULL instead of being stamped with a stringified
# placeholder.
# ---------------------------------------------------------------------------

def _extract_item_id(args: tuple[Any, ...]) -> Any:
    """Return the first positional arg, matching the ``inject_run``
    path's existing ``item_id = args[0] if args else None`` convention.
    Both the inject_run and legacy paths use the same extraction here so
    baggage stays consistent across the two call shapes.
    """
    return args[0] if args else None


def _extract_partition_key(kwargs: dict[str, Any]) -> Any:
    """Pull ``partition_key`` out of kwargs if the caller passed it.

    The dispatcher resolves partition_key from papayya.yaml's
    ``partition_key:`` field and threads it into the worker invocation;
    when uncertain we return ``None`` and the mapper writes NULL — the
    UsageEvent still records provider/model/tokens, only workload
    attribution is missing.
    """
    return kwargs.get("partition_key")


# ---------------------------------------------------------------------------
# Agent version resolution (ADR-0002 #7)
#
# Resolved once per process at decoration time. Order:
#   1. Explicit decorator arg (`@agent(..., agent_version="2.3.1")`)
#   2. Env var PAPAYYA_AGENT_VERSION (CI/CD injects the build tag)
#   3. `git rev-parse --short HEAD` from cwd
#   4. "unknown" sentinel
# Layers 2 + 3 memoize at module level so a project with N agents only
# does one git subprocess at boot, not N. Memoization is a process-level
# cache: tests can clear it with ``_clear_agent_version_cache()``.
# ---------------------------------------------------------------------------

_AGENT_VERSION_FALLBACK = "unknown"
_VERSION_RESOLVE_CACHE: dict[str, str | None] = {}


def _clear_agent_version_cache() -> None:
    """Reset the env+git memoization. Test-only helper."""
    _VERSION_RESOLVE_CACHE.clear()


def _resolve_env_version() -> str | None:
    if "env" in _VERSION_RESOLVE_CACHE:
        return _VERSION_RESOLVE_CACHE["env"]
    raw = os.environ.get("PAPAYYA_AGENT_VERSION", "")
    value = raw.strip() or None
    _VERSION_RESOLVE_CACHE["env"] = value
    return value


def _resolve_git_version() -> str | None:
    """Run `git rev-parse --short HEAD` once; memoize the answer.

    Silent on every failure mode (no git binary, not a repo, subprocess
    timeout, decoding error). The fallback chain handles the None case.
    """
    if "git" in _VERSION_RESOLVE_CACHE:
        return _VERSION_RESOLVE_CACHE["git"]
    git = shutil.which("git")
    if git is None:
        _VERSION_RESOLVE_CACHE["git"] = None
        return None
    try:
        out = subprocess.run(
            [git, "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2.0,
            check=False,
        )
    except (subprocess.SubprocessError, OSError):
        _VERSION_RESOLVE_CACHE["git"] = None
        return None
    if out.returncode != 0:
        _VERSION_RESOLVE_CACHE["git"] = None
        return None
    sha = out.stdout.strip()
    value = sha or None
    _VERSION_RESOLVE_CACHE["git"] = value
    return value


def _resolve_agent_version(explicit: str | None) -> tuple[str, str]:
    """Pick the agent version from the four-layer chain.

    Returns ``(version, source)`` where ``source`` is one of
    ``"decorator" | "env" | "git" | "unknown"``. The ``"unknown"``
    sentinel is returned when none of the layers resolve. Replay treats
    "unknown" strictly — if either side of the comparison is "unknown",
    the gate fires unless --latest is passed. That's the point of the
    sentinel: an un-tagged process should be visible, not silently
    equal to other un-tagged processes.
    """
    if explicit is not None:
        cleaned = explicit.strip()
        if cleaned:
            return cleaned, "decorator"
    env = _resolve_env_version()
    if env is not None:
        return env, "env"
    git = _resolve_git_version()
    if git is not None:
        return git, "git"
    return _AGENT_VERSION_FALLBACK, "unknown"


# Global registry, keyed by ``(agent_name, agent_version)`` so a hosted
# worker can hold v1 and v2 simultaneously and route each lease to the
# matching registration (ADR-0003 § Worker #4). Local-dev parity is
# preserved because ``get_agent(name)`` (no version) returns the
# most-recently-registered entry — the same overwrite-by-import semantics
# the slug-keyed dict gave pre-slice-3.
#
# Contract: ``@agent`` registration MUST happen at import time. The
# multi-version routing relies on the registration being keyed by
# ``(name, agent_version)`` *before* the worker resolves the lease;
# deferred or runtime registration would break dispatch silently.
_registry: dict[tuple[str, str], AgentRegistration] = {}


def get_registry() -> dict[tuple[str, str], AgentRegistration]:
    """Return the current module-level agent registry, keyed by
    ``(name, agent_version)``.

    Callers that just want one registration per name should use
    :func:`get_agent` instead — the tuple-keyed view is internal.
    """
    return _registry


def get_agent(name: str, version: str | None = None) -> AgentRegistration | None:
    """Look up a registered agent by name and (optional) version.

    ``version is None`` preserves the local-dev / single-resident
    semantics: returns the most-recently-registered entry for ``name``
    (insertion order in the dict, latest wins — same as the pre-slice-3
    slug-keyed overwrite behaviour). ``papayya dev`` and tests that
    register one agent per slug rely on this branch.

    ``version is not None`` does the multi-version lookup. Used by
    ``Worker._handle_lease`` so the dispatched fn matches the lease's
    ``agent_version``.
    """
    if version is not None:
        return _registry.get((name, version))
    candidates = [reg for (n, _v), reg in _registry.items() if n == name]
    if not candidates:
        return None
    # Latest insertion wins. Python dicts preserve insertion order
    # (3.7+), so this matches the legacy "last @agent on the same name
    # overwrites the previous" behaviour.
    return candidates[-1]


# ---------------------------------------------------------------------------
# @agent decorator
# ---------------------------------------------------------------------------

def agent(
    name: str,
    model: str = "",
    instructions: str = "",
    tools: list[ToolDefinition] | None = None,
    max_steps: int = 50,
    budget_usd: float | None = None,
    durable: bool = False,
    max_duration_seconds: float | None = None,
    agent_version: str | None = None,
    concurrency_per_key: int | None = None,
    rate_limit: str | None = None,
    checks: list | None = None,
) -> Callable:
    """Decorator that registers a function as a deployable agent.

    The decorated function keeps its original behavior — you can call it
    directly in local code. The metadata is stored in a registry that the
    CLI (``papayya deploy``) and the runtime shim use to discover and
    invoke agents.

    Args:
        name: Agent name (used as the slug for deploy lookup).
        model: Display label for the dashboard (not used for routing).
        instructions: System prompt / instructions (display only).
        tools: Optional list of ToolDefinition objects.
        max_steps: Max LLM calls per run (enforced by the runtime shim).
        budget_usd: Per-run budget cap.
        max_duration_seconds: Wall-clock soft timeout for one invocation
            of the agent fn, enforced by the runtime worker (ADR-0002 #2).
            ``None`` (the default) disables enforcement — existing agents
            keep their pre-timeout behavior. The dispatcher payload's
            ``max_duration_seconds`` field overrides this on a per-call
            basis.

            Caveats: signal-based watchdog (Unix only). Cannot interrupt
            blocking C calls (SSL handshakes, default ``requests.get``);
            pair this with explicit socket timeouts in your HTTP client
            for full coverage. Customer code that installs its own
            ``SIGALRM`` handler conflicts.
        agent_version: Opaque string stamped on every run + step this
            agent produces, used as the replay-mismatch gate (ADR-0002
            #7). Resolution chain when omitted: env
            ``PAPAYYA_AGENT_VERSION`` → ``git rev-parse --short HEAD``
            → ``"unknown"``. CI/CD injecting the env var is the
            recommended path.
        concurrency_per_key: Cap on concurrent in-flight items per
            partition_key value (Layer 3 #1). When the cap is hit, the
            hosted dispatcher keeps additional items in runtime_pending
            until one in-flight finishes. The bucket key is the
            partition_key value declared in papayya.yaml's
            ``partition_key:`` field; falls back to the calling
            account_id when no partition_key is set. None disables the
            cap.
        rate_limit: Cap on lease throughput per partition_key value
            (Layer 3 #2). Format: ``"N/min"`` or ``"N/sec"``. Sliding
            window enforced server-side; uses the same bucket key as
            ``concurrency_per_key``. None disables the cap.
        checks: Customer outcome checks (Plan 35) — deterministic callables
            ``Callable[[result], papayya.CheckVerdict | None]`` and/or
            :func:`papayya.llm_judge` scaffolds. They run in the same outcome
            pipeline as the built-in inspectors on every step; the worst
            verdict wins the run's outcome. A broken/slow check is a contained
            pass (an observer never fails the run).
    """
    if max_duration_seconds is not None and max_duration_seconds <= 0:
        raise ValueError(
            f"max_duration_seconds must be > 0 or None, got {max_duration_seconds!r}"
        )

    if concurrency_per_key is not None and concurrency_per_key <= 0:
        raise ValueError(
            f"concurrency_per_key must be > 0 or None, got {concurrency_per_key!r}"
        )

    rate_limit_per_min: int | None = None
    if rate_limit is not None:
        rate_limit_per_min = _parse_rate_limit(rate_limit)

    resolved_version, version_source = _resolve_agent_version(agent_version)
    if version_source == "unknown":
        log.warning(
            "registered '%s' v=unknown (no agent_version arg, no PAPAYYA_AGENT_VERSION, no git SHA available)",
            name,
        )
    else:
        log.info(
            "registered '%s' v=%s (source=%s)",
            name,
            resolved_version,
            version_source,
        )

    def decorator(fn: Callable) -> Callable:
        try:
            sig: inspect.Signature | None = inspect.signature(fn)
        except (TypeError, ValueError):
            # Builtins / C-level callables — no introspectable signature.
            # Snapshot capture skipped for these; runs still execute.
            sig = None

        # Layer 3 #9: inject the run as fn's first positional argument when
        # the customer declares `def process_note(run, ...)`. Detection is
        # by literal parameter name — keeps the new shape obvious from the
        # signature and avoids requiring type annotations.
        inject_run = False
        if sig is not None:
            positional = [
                p for p in sig.parameters.values()
                if p.name not in ("self", "cls")
                and p.kind in (
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                )
            ]
            inject_run = bool(positional) and positional[0].name == "run"

        # When injecting, build_input_snapshot must bind worker-provided
        # args to the user's *non-run* parameters, otherwise it tries to
        # bind args[0] to `run` and the snapshot ends up shaped wrong.
        sig_for_snapshot: inspect.Signature | None = sig
        if inject_run and sig is not None:
            sig_for_snapshot = sig.replace(parameters=[
                p for p in sig.parameters.values() if p.name != "run"
            ])

        if inspect.iscoroutinefunction(fn):
            @functools.wraps(fn)
            async def wrapper(*args, **kwargs):
                snapshot = _serialize.build_input_snapshot(
                    sig_for_snapshot, args, kwargs,
                )
                input_token = _AGENT_INPUT.set(snapshot)
                item_id_value = _extract_item_id(args)
                partition_key_value = _extract_partition_key(kwargs)
                baggage_token = set_papayya_baggage(
                    workload=name,
                    item_id=item_id_value,
                    partition_key=partition_key_value,
                )
                annotate_current_span(
                    workload=name,
                    item_id=item_id_value,
                    partition_key=partition_key_value,
                )
                # Lazy imports: papayya.iterators and papayya.durable both
                # import back into papayya.agent, so resolve them at call
                # time (both modules are fully loaded by the time an agent
                # body runs).
                from papayya import iterators as _iterators
                try:
                    # Case C detection peeks the enclosing isolate's minted
                    # run too — the mint no longer publishes on _ACTIVE_RUN
                    # (a set() from inside an asyncio.Task could never be
                    # reset from the wrapper's parent context).
                    existing = _iterators._peek_run()
                    if existing is not None:
                        # Case C: a run is already active (papayya.map opened
                        # it, or we're a nested agent). Reuse it — ambient
                        # verbs resolve against it — instead of opening a
                        # second run. The owning caller closes it.
                        if inject_run:
                            return await fn(existing, *args, **kwargs)
                        return await fn(*args, **kwargs)

                    if inject_run:
                        # `def f(run, …)`: create the run and inject it (as
                        # before); additionally publish it as the ambient run
                        # so in-body verbs resolve too. Customer owns completion.
                        # partition_key passes explicitly (possibly None): the
                        # customer doesn't control this run() call, so the
                        # strict-metadata contract can't bind them here.
                        from papayya.durable import papayya as _papayya_factory
                        run_obj = _papayya_factory().run(
                            agent=name,
                            item_id=item_id_value,
                            partition_key=_iterators._coerce_partition_key(
                                partition_key_value
                            ),
                        )
                        active_id_token = _ACTIVE_RUN_ID.set(run_obj.run_id)
                        active_run_token = _iterators._ACTIVE_RUN.set(run_obj)
                        try:
                            return await fn(run_obj, *args, **kwargs)
                        finally:
                            _iterators._ACTIVE_RUN.reset(active_run_token)
                            _ACTIVE_RUN_ID.reset(active_id_token)

                    # Clean `def f(item)` path. Publish a lazy isolate so an
                    # ambient @papayya.llm / mark_degraded mints + resolves a
                    # run (the front door the wedge previously missed) without
                    # eagerly opening one — legacy bodies that call
                    # papayya().run() themselves keep their bootstrap/replay
                    # adoption. On the hosted worker the worker/control-plane
                    # own terminal status, so don't self-complete there.
                    hosted = _BOOTSTRAP_RUN_ID.get() is not None
                    legacy_token = _LEGACY_AGENT_PATH_ACTIVE.set(True)
                    try:
                        return await _iterators.drive_ambient_async(
                            name, item_id_value, partition_key_value,
                            lambda: fn(*args, **kwargs),
                            own_completion=not hosted,
                        )
                    finally:
                        _LEGACY_AGENT_PATH_ACTIVE.reset(legacy_token)
                finally:
                    clear_papayya_baggage(baggage_token)
                    _AGENT_INPUT.reset(input_token)
        else:
            @functools.wraps(fn)
            def wrapper(*args, **kwargs):
                snapshot = _serialize.build_input_snapshot(
                    sig_for_snapshot, args, kwargs,
                )
                input_token = _AGENT_INPUT.set(snapshot)
                item_id_value = _extract_item_id(args)
                partition_key_value = _extract_partition_key(kwargs)
                baggage_token = set_papayya_baggage(
                    workload=name,
                    item_id=item_id_value,
                    partition_key=partition_key_value,
                )
                annotate_current_span(
                    workload=name,
                    item_id=item_id_value,
                    partition_key=partition_key_value,
                )
                from papayya import iterators as _iterators
                try:
                    existing = _iterators._peek_run()
                    if existing is not None:
                        # Case C — see the async wrapper above.
                        if inject_run:
                            return fn(existing, *args, **kwargs)
                        return fn(*args, **kwargs)

                    if inject_run:
                        # See the async wrapper above for the partition_key
                        # rationale.
                        from papayya.durable import papayya as _papayya_factory
                        run_obj = _papayya_factory().run(
                            agent=name,
                            item_id=item_id_value,
                            partition_key=_iterators._coerce_partition_key(
                                partition_key_value
                            ),
                        )
                        active_id_token = _ACTIVE_RUN_ID.set(run_obj.run_id)
                        active_run_token = _iterators._ACTIVE_RUN.set(run_obj)
                        try:
                            return fn(run_obj, *args, **kwargs)
                        finally:
                            _iterators._ACTIVE_RUN.reset(active_run_token)
                            _ACTIVE_RUN_ID.reset(active_id_token)

                    # Clean `def f(item)` path — see the async wrapper above.
                    hosted = _BOOTSTRAP_RUN_ID.get() is not None
                    legacy_token = _LEGACY_AGENT_PATH_ACTIVE.set(True)
                    try:
                        return _iterators.drive_ambient_sync(
                            name, item_id_value, partition_key_value,
                            lambda: fn(*args, **kwargs),
                            own_completion=not hosted,
                        )
                    finally:
                        _LEGACY_AGENT_PATH_ACTIVE.reset(legacy_token)
                finally:
                    clear_papayya_baggage(baggage_token)
                    _AGENT_INPUT.reset(input_token)

        # Register the *wrapper*, not the raw fn — the runtime worker
        # calls registration.fn(item_id) directly, and the wrapper is
        # what sets the input-snapshot contextvar that DurableRun.init()
        # reads when seeding runs.input_snapshot. Storing the raw fn
        # would silently bypass that bridge for every worker-driven run.
        registration = AgentRegistration(
            name=name,
            model=model,
            instructions=instructions,
            fn=wrapper,
            tools=tools or [],
            max_steps=max_steps,
            budget_usd=budget_usd,
            durable=durable,
            max_duration_seconds=max_duration_seconds,
            agent_version=resolved_version,
            concurrency_per_key=concurrency_per_key,
            rate_limit_per_min=rate_limit_per_min,
            checks=checks or [],
        )
        # Attach metadata so callers can inspect without the registry
        wrapper._papayya_agent = registration
        # ADR-0003 § Worker #4 — keyed by (name, version) so a hosted
        # worker can hold multiple versions of the same agent slug
        # resident at once. Local-dev parity is preserved by
        # ``get_agent(name, version=None)`` falling back to the
        # latest-registered entry for ``name``.
        _registry[(name, resolved_version)] = registration
        return wrapper

    return decorator


def durable(fn: Callable | None = None, *, name: str | None = None, **kwargs: Any) -> Callable:
    """The isolate boundary — register a function as a deployable durable workload.

    This is :func:`agent` with the signature freed and the name optional.
    Durability is *ambient* inside the body: an ``@papayya.llm`` /
    ``mark_degraded`` / structural outcome inspection resolves against the run
    the decorator opens, with no ``run`` parameter threaded through your
    signature and no per-call bookkeeping.

    Usage::

        @papayya.durable
        def enrich(company): ...

        @papayya.durable(name="enrich", budget_usd=1.0)
        def enrich(company): ...

    ``name`` defaults to the function's ``__name__``. All other keyword
    arguments pass straight through to :func:`agent`. ``@agent`` remains as a
    lower-level alias (it requires an explicit ``name``); the public lead
    decorator is ``@papayya.durable``.
    """
    # Support @papayya.durable("slug") as a positional-name form.
    if isinstance(fn, str):
        name = name or fn
        fn = None

    def decorate(f: Callable) -> Callable:
        resolved_name = name or getattr(f, "__name__", None) or "workload"
        return agent(name=resolved_name, **kwargs)(f)

    if fn is not None:
        return decorate(fn)
    return decorate


# ---------------------------------------------------------------------------
# Agent dataclass (internal representation, used by shim + API)
# ---------------------------------------------------------------------------

@dataclass
class Agent:
    """Declarative description of an agent.

    Primarily used internally by the shim and API layer. Most users should
    use the ``@agent`` decorator instead.
    """

    name: str
    model: str
    instructions: str = ""
    tools: list[ToolDefinition] = field(default_factory=list)
    max_steps: int = 50
    budget_usd: float | None = None
    project_id: str | None = None

    def run(self, input_data: str | Any) -> str:
        raise NotImplementedError(
            "Agent.run() has been removed. Use the @agent decorator instead:\n\n"
            "    @agent(name='my-agent', model='gpt-4o-mini')\n"
            "    def my_agent(input_data):\n"
            "        # call your LLM SDK directly\n"
            "        ...\n"
        )

    def to_definition(self) -> dict[str, Any]:
        """Serialize to the API format for deployment."""
        defn: dict[str, Any] = {
            "name": self.name,
            "slug": self.name.lower().replace(" ", "-"),
            "description": "",
            "config": {
                "model": self.model,
                "max_steps": self.max_steps,
                "tools": [t.to_schema() for t in self.tools],
            },
        }
        if self.budget_usd is not None:
            defn["config"]["budget_usd"] = self.budget_usd
        if self.project_id:
            defn["project_id"] = self.project_id
        return defn

    def get_tool(self, name: str) -> ToolDefinition | None:
        for t in self.tools:
            if t.name == name:
                return t
        return None

    @property
    def tool_map(self) -> dict[str, ToolDefinition]:
        return {t.name: t for t in self.tools}

    @property
    def budget_cents(self) -> int:
        if self.budget_usd is None:
            return 500  # default $5
        return int(self.budget_usd * 100)

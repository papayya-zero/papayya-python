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


# Global registry, keyed by agent name (slug)
_registry: dict[str, AgentRegistration] = {}


def get_registry() -> dict[str, AgentRegistration]:
    """Return the current module-level agent registry."""
    return _registry


def get_agent(name: str) -> AgentRegistration | None:
    """Look up a registered agent by name."""
    return _registry.get(name)


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
    """
    if max_duration_seconds is not None and max_duration_seconds <= 0:
        raise ValueError(
            f"max_duration_seconds must be > 0 or None, got {max_duration_seconds!r}"
        )

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

        if inspect.iscoroutinefunction(fn):
            @functools.wraps(fn)
            async def wrapper(*args, **kwargs):
                snapshot = _serialize.build_input_snapshot(sig, args, kwargs)
                token = _AGENT_INPUT.set(snapshot)
                try:
                    return await fn(*args, **kwargs)
                finally:
                    _AGENT_INPUT.reset(token)
        else:
            @functools.wraps(fn)
            def wrapper(*args, **kwargs):
                snapshot = _serialize.build_input_snapshot(sig, args, kwargs)
                token = _AGENT_INPUT.set(snapshot)
                try:
                    return fn(*args, **kwargs)
                finally:
                    _AGENT_INPUT.reset(token)

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
        )
        _registry[name] = registration

        # Attach metadata so callers can inspect without the registry
        wrapper._papayya_agent = registration
        return wrapper

    return decorator


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

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
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Callable

from papayya import _serialize
from papayya.tools import ToolDefinition


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


# Global registry, keyed by agent name (slug)
_registry: dict[str, AgentRegistration] = {}


def get_registry() -> dict[str, AgentRegistration]:
    """Return the current module-level agent registry."""
    return _registry


def get_agent(name: str) -> AgentRegistration | None:
    """Look up a registered agent by name."""
    return _registry.get(name)


# ---------------------------------------------------------------------------
# Input snapshot capture
# ---------------------------------------------------------------------------

def _build_input_snapshot(
    sig: inspect.Signature | None,
    args: tuple,
    kwargs: dict,
) -> Any:
    """Build the snapshot dict for an @agent function call, or None.

    Returns None — never raises — if any of the following fail:
      - The signature couldn't be introspected at decoration time.
      - bind() rejects the call (TypeError will surface from fn() anyway).
      - The bound args aren't JSON-encodable (e.g. a custom class instance
        with no __dict__). The run still executes; replay just isn't
        available for that invocation.
    """
    if sig is None:
        return None
    try:
        bound = sig.bind(*args, **kwargs)
    except TypeError:
        return None
    # Pull defaults onto the snapshot so replay stays deterministic if
    # the source code's default values change after a run was captured.
    bound.apply_defaults()
    snap = dict(bound.arguments)
    try:
        _serialize.encode_user_value(snap, strict=True)
    except (TypeError, ValueError):
        return None
    return snap


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
    """
    def decorator(fn: Callable) -> Callable:
        try:
            sig: inspect.Signature | None = inspect.signature(fn)
        except (TypeError, ValueError):
            # Builtins / C-level callables — no introspectable signature.
            # Snapshot capture skipped for these; runs still execute.
            sig = None

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            snapshot = _build_input_snapshot(sig, args, kwargs)
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

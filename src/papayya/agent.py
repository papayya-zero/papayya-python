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
from dataclasses import dataclass, field
from typing import Any, Callable

from papayya.tools import ToolDefinition


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
        registration = AgentRegistration(
            name=name,
            model=model,
            instructions=instructions,
            fn=fn,
            tools=tools or [],
            max_steps=max_steps,
            budget_usd=budget_usd,
            durable=durable,
        )
        _registry[name] = registration

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            return fn(*args, **kwargs)

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

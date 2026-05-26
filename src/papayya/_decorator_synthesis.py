"""Synthesise an EnvSpec from yaml-sourced + decorator-attached metadata.

Plan 11 attaches :class:`ScheduleSpec` / :class:`WebhookSpec` lists to
:class:`AgentRegistration` via ``@schedule`` and ``@trigger``. Plan 12's
reconciler reads from an :class:`EnvSpec` sourced from ``papayya.yaml``
today. This helper bridges the two: it harvests the decorator metadata
via Plan 11's :func:`harvest_decorator_specs` and merges the results
into the yaml-sourced :class:`EnvSpec` for the reconciler to consume
identically.

Semantics:

- yaml-sourced schedules/webhooks pass through unchanged.
- Decorator-sourced schedules/webhooks are appended to the matching
  agent's spec (or land in a fresh :class:`AgentSpec` when no yaml block
  exists for that slug).
- The ``managed_by='code'`` marker the server uses to scope full-replace
  is attached at the API-call site in
  :func:`papayya._reconcile.apply_plan` (specifically in
  :meth:`papayya.api.APIClient.put_schedules` /
  :meth:`papayya.api.APIClient.put_webhooks`), not here.

Import note: this module transitively imports ``papayya.decorators``,
which pulls ``croniter`` + ``zoneinfo``. ``papayya/__init__.py`` defers
its public re-export of ``schedule`` / ``trigger`` via ``__getattr__``
for the same reason — eager package-init imports of the decorator chain
change module-init ordering enough to mask cross-process SQLite WAL
writes inside the worker subprocess test. Callers should import this
module lazily inside the function that uses it (see ``cli.py``'s deploy
flow for the canonical splice).
"""

from __future__ import annotations

from papayya._config import AgentSpec, EnvSpec
from papayya.agent import AgentRegistration
from papayya.decorators import harvest_decorator_specs


def env_spec_from_registry_and_yaml(
    yaml_env: EnvSpec | None,
    registry: dict[tuple[str, str], AgentRegistration],
) -> EnvSpec:
    """Return an EnvSpec that fuses yaml-sourced + decorator-sourced specs.

    Args:
        yaml_env: The yaml-sourced :class:`EnvSpec`, or ``None`` when
            the project has no ``papayya.yaml`` (decorator-only deploy).
        registry: ``{(name, version): AgentRegistration}`` — the module-
            level dict :func:`papayya.agent.get_registry` returns. The
            registry MUST already be populated by the agent-discovery
            import (the deploy flow calls ``_discover_agents`` first).

    Returns:
        An :class:`EnvSpec` whose ``agents[slug]`` is the union of:

        - The yaml-sourced :class:`AgentSpec` for ``slug`` (if any).
        - The decorator-attached schedules and webhooks for ``slug``
          (if any), appended to the yaml lists.

        Agents present in yaml only — with no decorator metadata — pass
        through unchanged. Agents present in the registry only — with no
        yaml block — land as a fresh :class:`AgentSpec` carrying just
        the decorator-attached schedules and webhooks.

    The output shape is identical to what the yaml-only loader produced
    pre-Plan 12, so :func:`papayya._reconcile.diff_env` consumes it
    without changes.
    """
    decorator_specs = harvest_decorator_specs(registry)

    # Start from yaml (when present) so the order of explicit yaml entries
    # is preserved in dict insertion order for downstream deterministic
    # iteration (CLI output, tests).
    agents: dict[str, AgentSpec] = {}
    if yaml_env is not None:
        for slug, agent_spec in yaml_env.agents.items():
            agents[slug] = agent_spec

    for slug, (decorator_schedules, decorator_webhooks) in decorator_specs.items():
        if slug in agents:
            existing = agents[slug]
            # pydantic frozen-model extension uses model_copy(update=...).
            # Preserves every other field on AgentSpec (none today, but
            # future-additive).
            agents[slug] = existing.model_copy(update={
                "schedules": list(existing.schedules) + list(decorator_schedules),
                "webhooks": list(existing.webhooks) + list(decorator_webhooks),
            })
        else:
            agents[slug] = AgentSpec(
                schedules=list(decorator_schedules),
                webhooks=list(decorator_webhooks),
            )

    return EnvSpec(agents=agents)

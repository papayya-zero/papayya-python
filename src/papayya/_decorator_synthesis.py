"""Synthesise an EnvSpec from decorator-attached trigger metadata.

Plan 11 attaches :class:`ScheduleSpec` / :class:`WebhookSpec` lists to
:class:`AgentRegistration` via ``@schedule`` and ``@trigger``. Plan 12's
reconciler consumes an :class:`EnvSpec`. This helper bridges the two: it
harvests the decorator metadata via Plan 11's
:func:`harvest_decorator_specs` and shapes it into the :class:`EnvSpec`
the reconciler reads.

Decorators are the sole source of truth — ``papayya.yaml`` is no longer
read by the CLI (the code-first direction removed the yaml surface).

Semantics:

- Every agent carrying decorator-sourced schedules/webhooks lands as an
  :class:`AgentSpec` keyed by its slug.
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


def env_spec_from_registry(
    registry: dict[tuple[str, str], AgentRegistration],
) -> EnvSpec:
    """Return an EnvSpec built from decorator-attached trigger metadata.

    Args:
        registry: ``{(name, version): AgentRegistration}`` — the module-
            level dict :func:`papayya.agent.get_registry` returns. The
            registry MUST already be populated by the agent-discovery
            import (the deploy flow calls ``_discover_agents`` first).

    Returns:
        An :class:`EnvSpec` whose ``agents[slug]`` carries the
        decorator-attached schedules and webhooks for ``slug``. Agents
        with no decorator metadata do not appear.

    The output shape is what :func:`papayya._reconcile.diff_env` consumes.
    """
    decorator_specs = harvest_decorator_specs(registry)

    agents: dict[str, AgentSpec] = {}
    for slug, (decorator_schedules, decorator_webhooks) in decorator_specs.items():
        agents[slug] = AgentSpec(
            schedules=list(decorator_schedules),
            webhooks=list(decorator_webhooks),
        )

    return EnvSpec(agents=agents)

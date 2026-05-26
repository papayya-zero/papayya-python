"""``@schedule`` and ``@trigger`` decorators (Plan 11).

Decorators that attach :class:`ScheduleSpec` / :class:`WebhookSpec`
metadata to an ``@agent``-decorated function's
:class:`AgentRegistration`. The metadata is inert at decoration time —
Plan 12's modified reconciler harvests it at deploy time via
:func:`harvest_decorator_specs` and reconciles against the control
plane.

Naming asymmetry (decision D3 in plans/11-schedule-and-trigger-decorators.md):
    * User-facing decorator is ``@trigger`` (verb chosen in the 08-18
      breakout for symmetry with ``@schedule``).
    * Persisted metadata class stays :class:`WebhookSpec` (existing name
      in ``_config.py``, wired into ``_reconcile.py`` + ``api.create_webhook``
      today; Plan 11 anti-scope forbids the rename).

Decoration order: ``@schedule`` / ``@trigger`` MUST be applied above
``@agent`` (i.e. outermost) — the inner ``@agent`` produces the wrapper
that carries the ``_papayya_agent`` back-reference these decorators read.
Wrong order is a clear :class:`DecoratorTargetError`.
"""

from __future__ import annotations

import re
from typing import Callable
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from croniter import croniter

from papayya._config import ScheduleSpec, WebhookSpec
from papayya.agent import AgentRegistration


class DecoratorValidationError(ValueError):
    """Raised at decoration time when a ``@schedule`` or ``@trigger``
    argument is syntactically invalid (bad cron, bad timezone, bad
    webhook name, bad env var name).
    """


class DecoratorTargetError(TypeError):
    """Raised when ``@schedule`` or ``@trigger`` is applied to a
    function that wasn't first wrapped by ``@agent`` — i.e. no
    ``_papayya_agent`` attribute. Tells the customer to swap order.
    """


class DecoratorConflictError(ValueError):
    """Raised at harvest time when an agent has duplicate cron values
    or duplicate webhook names across its stacked decorators.

    Harvest-time rather than decoration-time because decoration order
    across multiple files is non-deterministic; we want the failure to
    happen at deploy with a useful summary, not at first import.
    """


_WEBHOOK_NAME_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
_ENV_VAR_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")


def _validate_cron(cron: str) -> None:
    if not isinstance(cron, str) or not cron.strip():
        raise DecoratorValidationError(
            f"@schedule cron must be a non-empty string, got {cron!r}"
        )
    if not croniter.is_valid(cron):
        raise DecoratorValidationError(
            f"@schedule cron expression is not valid: {cron!r}"
        )


def _validate_timezone(tz: str) -> None:
    if not isinstance(tz, str) or not tz:
        raise DecoratorValidationError(
            f"@schedule timezone must be a non-empty string, got {tz!r}"
        )
    try:
        ZoneInfo(tz)
    except ZoneInfoNotFoundError as exc:
        raise DecoratorValidationError(
            f"@schedule timezone is not a known IANA zone: {tz!r}"
        ) from exc


def _validate_webhook_name(name: str) -> None:
    if not isinstance(name, str) or not _WEBHOOK_NAME_RE.match(name):
        raise DecoratorValidationError(
            f"@trigger name must match {_WEBHOOK_NAME_RE.pattern}, got {name!r}"
        )


def _validate_secret_env(env: str) -> None:
    if not isinstance(env, str) or not _ENV_VAR_RE.match(env):
        raise DecoratorValidationError(
            f"@trigger secret_env must be a valid env var name "
            f"matching {_ENV_VAR_RE.pattern}, got {env!r}"
        )


def _registration_for(fn: Callable) -> AgentRegistration:
    reg = getattr(fn, "_papayya_agent", None)
    if reg is None:
        raise DecoratorTargetError(
            "@schedule / @trigger must be applied ABOVE @agent — e.g. "
            "`@schedule(...)`\n`@agent(name=...)`. The decorated function "
            "had no _papayya_agent attribute."
        )
    return reg


def schedule(cron: str, *, timezone: str = "UTC") -> Callable:
    """Attach a :class:`ScheduleSpec` to the underlying agent's
    registration.

    Stacks additively: multiple ``@schedule`` decorators on one function
    all fire. Duplicate cron values are caught at harvest time, not
    here.

    Args:
        cron: Cron expression. Validated syntactically via ``croniter``
            at decoration time — invalid expressions raise
            :class:`DecoratorValidationError`.
        timezone: IANA timezone name. Defaults to ``"UTC"``. Validated
            via :class:`zoneinfo.ZoneInfo`.
    """
    _validate_cron(cron)
    _validate_timezone(timezone)

    def apply(fn: Callable) -> Callable:
        reg = _registration_for(fn)
        reg.schedules.append(ScheduleSpec(cron=cron, timezone=timezone))
        return fn

    return apply


def trigger(*, name: str, secret_env: str) -> Callable:
    """Attach a :class:`WebhookSpec` to the underlying agent's
    registration.

    Args:
        name: Webhook name. Must match ``^[a-zA-Z0-9_-]{1,64}$`` — the
            same constraint the server enforces. URL is server-assigned
            at create time (``api.create_webhook`` returns
            ``trigger_url``); the decorator does NOT accept a ``url``
            kwarg today (decision D3 in the plan file).
        secret_env: Process env var holding the HMAC shared secret.
            Must look like a real env var name (``^[A-Z][A-Z0-9_]*$``)
            — protects against ``secret_env="my-secret"`` literals.
    """
    _validate_webhook_name(name)
    _validate_secret_env(secret_env)

    def apply(fn: Callable) -> Callable:
        reg = _registration_for(fn)
        reg.webhooks.append(WebhookSpec(name=name, secret_env=secret_env))
        return fn

    return apply


def harvest_decorator_specs(
    registry: dict[tuple[str, str], AgentRegistration],
) -> dict[str, tuple[list[ScheduleSpec], list[WebhookSpec]]]:
    """Collapse the ``(name, version)``-keyed registry to ``slug → (schedules, webhooks)``.

    Plan 12 consumes this to synthesise the :class:`EnvSpec` the
    reconciler reads. The slug is the agent name lowercased with spaces
    converted to dashes — mirroring ``cli.py``'s slug derivation.

    Versioning collapse: when multiple registrations share a name
    (multi-version routing, see ``agent.py:308-310``), the
    latest-inserted entry's specs win. This matches
    :func:`get_agent`'s "latest wins" semantic for the no-version
    lookup; deploy is always against the latest registration the
    bundler captured.

    Raises:
        DecoratorConflictError: when one agent has duplicate cron
            values across stacked ``@schedule`` decorators, or
            duplicate webhook names across stacked ``@trigger``
            decorators.
    """
    by_slug: dict[str, tuple[list[ScheduleSpec], list[WebhookSpec]]] = {}
    # Iterate in insertion order so "latest wins" is deterministic.
    seen_latest: dict[str, AgentRegistration] = {}
    for (name, _version), reg in registry.items():
        seen_latest[name] = reg

    for name, reg in seen_latest.items():
        slug = name.lower().replace(" ", "-")
        cron_seen: set[str] = set()
        for s in reg.schedules:
            if s.cron in cron_seen:
                raise DecoratorConflictError(
                    f"agent {slug!r} has duplicate @schedule cron {s.cron!r}"
                )
            cron_seen.add(s.cron)
        webhook_seen: set[str] = set()
        for w in reg.webhooks:
            if w.name in webhook_seen:
                raise DecoratorConflictError(
                    f"agent {slug!r} has duplicate @trigger name {w.name!r}"
                )
            webhook_seen.add(w.name)
        by_slug[slug] = (list(reg.schedules), list(reg.webhooks))
    return by_slug


__all__ = [
    "schedule",
    "trigger",
    "harvest_decorator_specs",
    "DecoratorValidationError",
    "DecoratorTargetError",
    "DecoratorConflictError",
]

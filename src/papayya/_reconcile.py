"""Reconcile papayya.yaml triggers against control plane state.

Pure diff/apply for the declarative-config reconciler wired into
`papayya deploy`. No I/O beyond the APIClient passed in; the CLI
is responsible for printing.

Keying rules (from launch_yaml_envs_v1.md):
- schedules: (agent_slug, normalized_cron) — rename = delete + create.
- webhooks:  (agent_slug, name)           — rename = delete + create (URL rotates).

Apply order per agent is deletes-before-creates so renamed keys don't
collide on server-side unique constraints during the transition.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Protocol, Union

from papayya._config import AgentSpec, EnvSpec
from papayya.api import PapayyaAPIError


class _APILike(Protocol):
    def list_schedules(self, agent_id: str) -> list[dict[str, Any]]: ...
    def create_schedule(self, agent_id: str, cron_expression: str, timezone: str = "UTC") -> dict[str, Any]: ...
    def delete_schedule(self, schedule_id: str) -> None: ...
    def list_webhooks(self, agent_id: str) -> list[dict[str, Any]]: ...
    def create_webhook(self, agent_id: str, name: str) -> dict[str, Any]: ...
    def delete_webhook(self, webhook_id: str) -> None: ...


class ReconcileError(Exception):
    """Yaml-vs-bundle mismatches raised before any server mutation."""


@dataclass(frozen=True)
class ScheduleOp:
    kind: Literal["create", "delete"]
    agent_slug: str
    agent_id: str
    cron: str
    remote_id: str | None = None


@dataclass(frozen=True)
class WebhookOp:
    kind: Literal["create", "delete"]
    agent_slug: str
    agent_id: str
    name: str
    secret_env: str | None = None
    remote_id: str | None = None
    reason: Literal["missing", "rename", "removed"] | None = None


Op = Union[ScheduleOp, WebhookOp]


@dataclass
class AgentPlan:
    slug: str
    agent_id: str
    schedule_ops: list[ScheduleOp] = field(default_factory=list)
    webhook_ops: list[WebhookOp] = field(default_factory=list)

    @property
    def is_noop(self) -> bool:
        return not self.schedule_ops and not self.webhook_ops


@dataclass
class ReconcilePlan:
    agents: list[AgentPlan] = field(default_factory=list)

    @property
    def is_noop(self) -> bool:
        return all(a.is_noop for a in self.agents)

    @property
    def total_ops(self) -> int:
        return sum(len(a.schedule_ops) + len(a.webhook_ops) for a in self.agents)


@dataclass
class ApplyResult:
    applied: int
    total: int
    failed_op: Op | None = None
    error: PapayyaAPIError | None = None
    created_webhooks: list[dict[str, Any]] = field(default_factory=list)


def _normalize_cron(cron: str) -> str:
    return " ".join(cron.split())


def diff_env(
    env_spec: EnvSpec,
    deployed: dict[str, str],
    api: _APILike,
) -> ReconcilePlan:
    """Compute the reconcile plan for one env against current server state.

    `deployed` maps agent_slug -> agent_id for agents the deploy step just
    resolved or created. Any yaml slug not in `deployed` raises ReconcileError
    before a single server call.
    """
    missing = [slug for slug in env_spec.agents if slug not in deployed]
    if missing:
        raise ReconcileError(
            f"papayya.yaml references agent(s) {sorted(missing)!r} "
            "which are not in the bundle. Add an @agent for each, or remove the block."
        )

    plan = ReconcilePlan()
    for slug, agent_spec in env_spec.agents.items():
        agent_id = deployed[slug]
        agent_plan = AgentPlan(slug=slug, agent_id=agent_id)
        _diff_schedules(agent_plan, agent_spec, agent_id, api)
        _diff_webhooks(agent_plan, agent_spec, agent_id, api)
        plan.agents.append(agent_plan)
    return plan


def _diff_schedules(
    agent_plan: AgentPlan,
    agent_spec: AgentSpec,
    agent_id: str,
    api: _APILike,
) -> None:
    remote = api.list_schedules(agent_id) or []
    remote_by_cron: dict[str, dict[str, Any]] = {}
    for row in remote:
        cron = row.get("cron_expression") or ""
        remote_by_cron[_normalize_cron(cron)] = row

    wanted = {_normalize_cron(s.cron) for s in agent_spec.schedules}

    # Deletes first.
    for key, row in remote_by_cron.items():
        if key not in wanted:
            agent_plan.schedule_ops.append(
                ScheduleOp(
                    kind="delete",
                    agent_slug=agent_plan.slug,
                    agent_id=agent_id,
                    cron=row.get("cron_expression", key),
                    remote_id=row.get("id"),
                )
            )

    # Then creates.
    for s in agent_spec.schedules:
        if _normalize_cron(s.cron) not in remote_by_cron:
            agent_plan.schedule_ops.append(
                ScheduleOp(
                    kind="create",
                    agent_slug=agent_plan.slug,
                    agent_id=agent_id,
                    cron=s.cron,
                )
            )


def _diff_webhooks(
    agent_plan: AgentPlan,
    agent_spec: AgentSpec,
    agent_id: str,
    api: _APILike,
) -> None:
    remote = api.list_webhooks(agent_id) or []
    remote_by_name: dict[str, dict[str, Any]] = {}
    for row in remote:
        name = row.get("name") or ""
        remote_by_name[name] = row

    wanted_names = {w.name for w in agent_spec.webhooks}

    # Deletes first. A delete whose name is absent from yaml is either a rename
    # (paired create below) or a removal. We can't tell them apart in isolation,
    # so the reason is set on the create side and left as "removed" here; the
    # CLI reads the companion create's `reason` to decide rotation copy.
    for name, row in remote_by_name.items():
        if name not in wanted_names:
            agent_plan.webhook_ops.append(
                WebhookOp(
                    kind="delete",
                    agent_slug=agent_plan.slug,
                    agent_id=agent_id,
                    name=name,
                    remote_id=row.get("id"),
                    reason="removed",
                )
            )

    # Then creates. "rename" detection: if there's any delete in this agent and
    # this create is net-new, mark rotation so the CLI prints the warning.
    has_pending_delete = any(op.kind == "delete" for op in agent_plan.webhook_ops)
    for w in agent_spec.webhooks:
        if w.name not in remote_by_name:
            reason: Literal["missing", "rename"] = "rename" if has_pending_delete else "missing"
            agent_plan.webhook_ops.append(
                WebhookOp(
                    kind="create",
                    agent_slug=agent_plan.slug,
                    agent_id=agent_id,
                    name=w.name,
                    secret_env=w.secret_env,
                    reason=reason,
                )
            )


def apply_plan(
    plan: ReconcilePlan,
    api: _APILike,
    *,
    printer: Callable[[str], None] | None = None,
) -> ApplyResult:
    """Execute ops fail-fast; return ApplyResult for the CLI to format."""
    out = printer or (lambda _m: None)
    total = plan.total_ops
    applied = 0
    created_webhooks: list[dict[str, Any]] = []

    for agent_plan in plan.agents:
        # Deletes first, creates second — within each resource, preserve the
        # diff's own ordering so deterministic tests stay deterministic.
        schedule_deletes = [o for o in agent_plan.schedule_ops if o.kind == "delete"]
        schedule_creates = [o for o in agent_plan.schedule_ops if o.kind == "create"]
        webhook_deletes = [o for o in agent_plan.webhook_ops if o.kind == "delete"]
        webhook_creates = [o for o in agent_plan.webhook_ops if o.kind == "create"]

        for op in schedule_deletes + schedule_creates + webhook_deletes + webhook_creates:
            try:
                if isinstance(op, ScheduleOp):
                    if op.kind == "delete":
                        assert op.remote_id is not None
                        api.delete_schedule(op.remote_id)
                    else:
                        api.create_schedule(op.agent_id, op.cron)
                else:
                    if op.kind == "delete":
                        assert op.remote_id is not None
                        api.delete_webhook(op.remote_id)
                    else:
                        created = api.create_webhook(op.agent_id, op.name)
                        created_webhooks.append({
                            "agent_slug": op.agent_slug,
                            "name": op.name,
                            "secret_env": op.secret_env,
                            "reason": op.reason,
                            **created,
                        })
            except PapayyaAPIError as e:
                return ApplyResult(
                    applied=applied,
                    total=total,
                    failed_op=op,
                    error=e,
                    created_webhooks=created_webhooks,
                )
            applied += 1
            _log_op(out, op)

    return ApplyResult(applied=applied, total=total, created_webhooks=created_webhooks)


def _log_op(printer: Callable[[str], None], op: Op) -> None:
    if isinstance(op, ScheduleOp):
        verb = "created" if op.kind == "create" else "deleted"
        printer(f"    schedule {op.cron:<22} {verb}")
    else:
        verb = "created" if op.kind == "create" else "deleted"
        printer(f"    webhook  {op.name:<22} {verb}")

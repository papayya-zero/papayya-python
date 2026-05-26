"""Reconcile papayya.yaml triggers against control plane state.

Pure diff/apply for the declarative-config reconciler wired into
`papayya deploy`. No I/O beyond the APIClient passed in; the CLI
is responsible for printing.

Keying rules (from launch_yaml_envs_v1.md):
- schedules: (agent_slug, normalized_cron) — rename = delete + create.
- webhooks:  (agent_slug, name)           — rename = delete + create (URL rotates).

The reconciler operates exclusively on ``managed_by='code'`` rows on
the server side. Rows created via the dashboard / direct POST land as
``managed_by='api'`` and are filtered out of the diff client-side:
they are not deleted, not updated, not even visible to the create/
unchanged/delete classifier. This keeps yaml-driven and dashboard-driven
operators coexisting on the same agent — the wedge-blocker the original
N-call reconciler had (silently nuking dashboard rows that weren't in
yaml) is gone by construction.

Apply path: a single ``put_schedules`` / ``put_webhooks`` call per
agent per resource type replaces the previous N-call create/delete
loop. The server applies the diff atomically inside one transaction;
the previous half-converged-state failure mode (mid-loop network blip
leaves the operator to fix by re-running) cannot happen.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Protocol, Union

from papayya._config import AgentSpec, EnvSpec
from papayya.api import PapayyaAPIError


class _APILike(Protocol):
    def list_schedules(self, agent_id: str) -> list[dict[str, Any]]: ...
    def list_webhooks(self, agent_id: str) -> list[dict[str, Any]]: ...
    def put_schedules(
        self, agent_id: str, schedules: list[dict[str, Any]],
    ) -> dict[str, Any]: ...
    def put_webhooks(
        self, agent_id: str, webhooks: list[dict[str, Any]],
    ) -> dict[str, Any]: ...


class ReconcileError(Exception):
    """Yaml-vs-bundle mismatches raised before any server mutation."""


@dataclass(frozen=True)
class ScheduleOp:
    kind: Literal["create", "update", "delete", "unchanged"]
    agent_slug: str
    agent_id: str
    cron: str
    remote_id: str | None = None


@dataclass(frozen=True)
class WebhookOp:
    kind: Literal["create", "update", "delete", "unchanged"]
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
        # An agent plan is a no-op when every op is `unchanged` (or there are
        # no ops at all). Unchanged ops carry signal for Plan 13's dry-run
        # but never produce a mutation, so they don't count as "work".
        if not self.schedule_ops and not self.webhook_ops:
            return True
        return all(
            op.kind == "unchanged"
            for op in (*self.schedule_ops, *self.webhook_ops)
        )


@dataclass
class ReconcilePlan:
    agents: list[AgentPlan] = field(default_factory=list)

    @property
    def is_noop(self) -> bool:
        return all(a.is_noop for a in self.agents)

    @property
    def total_ops(self) -> int:
        # Count only non-`unchanged` ops — these are what gets applied.
        # `unchanged` ops are diagnostic-only for Plan 13's dry-run.
        return sum(
            sum(1 for op in a.schedule_ops if op.kind != "unchanged")
            + sum(1 for op in a.webhook_ops if op.kind != "unchanged")
            for a in self.agents
        )


@dataclass
class ApplyResult:
    applied: int
    total: int
    failed_op: Op | None = None
    error: PapayyaAPIError | None = None
    created_webhooks: list[dict[str, Any]] = field(default_factory=list)


def _normalize_cron(cron: str) -> str:
    return " ".join(cron.split())


def _is_code_managed(row: dict[str, Any]) -> bool:
    """True iff the server row is owned by code-driven reconciliation.

    Missing ``managed_by`` key is treated as ``'api'`` (defensive — old
    server, or a pre-migration row). The reconciler never touches api-
    managed rows: not in the diff, not in the apply.
    """
    return row.get("managed_by") == "code"


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
    # Server-side filter is server-applied; we still filter client-side so
    # an old server (no managed_by column) doesn't silently delete rows
    # the reconciler shouldn't own.
    remote_by_cron: dict[str, dict[str, Any]] = {}
    for row in remote:
        if not _is_code_managed(row):
            continue
        cron = row.get("cron_expression") or ""
        remote_by_cron[_normalize_cron(cron)] = row

    wanted = {_normalize_cron(s.cron) for s in agent_spec.schedules}

    # Deletes first — code-managed remote rows whose cron isn't in yaml.
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

    # Then creates + unchanged. Unchanged ops carry signal for Plan 13's
    # dry-run (no-op output rows like `= schedule 0 9 * * *`).
    for s in agent_spec.schedules:
        key = _normalize_cron(s.cron)
        if key not in remote_by_cron:
            agent_plan.schedule_ops.append(
                ScheduleOp(
                    kind="create",
                    agent_slug=agent_plan.slug,
                    agent_id=agent_id,
                    cron=s.cron,
                )
            )
        else:
            agent_plan.schedule_ops.append(
                ScheduleOp(
                    kind="unchanged",
                    agent_slug=agent_plan.slug,
                    agent_id=agent_id,
                    cron=s.cron,
                    remote_id=remote_by_cron[key].get("id"),
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
        if not _is_code_managed(row):
            continue
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

    # Then creates + unchanged. "rename" detection: if there's any delete in
    # this agent and this create is net-new, mark rotation so the CLI prints
    # the warning.
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
        else:
            agent_plan.webhook_ops.append(
                WebhookOp(
                    kind="unchanged",
                    agent_slug=agent_plan.slug,
                    agent_id=agent_id,
                    name=w.name,
                    secret_env=w.secret_env,
                    remote_id=remote_by_name[w.name].get("id"),
                )
            )


def apply_plan(
    plan: ReconcilePlan,
    api: _APILike,
    *,
    printer: Callable[[str], None] | None = None,
) -> ApplyResult:
    """Execute ops fail-fast; return ApplyResult for the CLI to format.

    Per-agent strategy is one PUT per resource type, scoped to
    ``managed_by='code'`` rows. The PUT body is built from the union of
    create + unchanged ops (everything yaml wants) — delete ops are
    implicit (absent crons / names get deleted server-side). The PUT
    is skipped entirely when only `unchanged` ops exist for that
    resource, to avoid a no-op write that would still bump
    ``updated_at`` on every row.
    """
    out = printer or (lambda _m: None)
    total = plan.total_ops
    applied = 0
    created_webhooks: list[dict[str, Any]] = []

    for agent_plan in plan.agents:
        sched_non_noop = [
            o for o in agent_plan.schedule_ops if o.kind != "unchanged"
        ]
        wh_non_noop = [
            o for o in agent_plan.webhook_ops if o.kind != "unchanged"
        ]

        if sched_non_noop:
            desired: list[dict[str, Any]] = [
                {"cron_expression": op.cron}
                for op in agent_plan.schedule_ops
                if op.kind in ("create", "update", "unchanged")
            ]
            try:
                api.put_schedules(agent_plan.agent_id, desired)
            except PapayyaAPIError as e:
                return ApplyResult(
                    applied=applied,
                    total=total,
                    failed_op=sched_non_noop[0],
                    error=e,
                    created_webhooks=created_webhooks,
                )
            for op in sched_non_noop:
                applied += 1
                _log_op(out, op)

        if wh_non_noop:
            desired_wh: list[dict[str, Any]] = [
                {"name": op.name}
                for op in agent_plan.webhook_ops
                if op.kind in ("create", "update", "unchanged")
            ]
            try:
                resp = api.put_webhooks(agent_plan.agent_id, desired_wh)
            except PapayyaAPIError as e:
                return ApplyResult(
                    applied=applied,
                    total=total,
                    failed_op=wh_non_noop[0],
                    error=e,
                    created_webhooks=created_webhooks,
                )
            # Surface secret-bearing rows so the CLI's "rotation note:
            # copy this URL once" path works identically to the old
            # per-create code path. Server only returns `secret` on
            # rows newly created by this PUT.
            for item in resp.get("items", []) or []:
                if not item.get("secret"):
                    continue
                matching_create = next(
                    (
                        op for op in agent_plan.webhook_ops
                        if op.kind == "create" and op.name == item.get("name")
                    ),
                    None,
                )
                created_webhooks.append({
                    "agent_slug": agent_plan.slug,
                    "name": item.get("name"),
                    "secret_env": (
                        matching_create.secret_env if matching_create else None
                    ),
                    "reason": (
                        matching_create.reason if matching_create else "missing"
                    ),
                    **item,
                })
            for op in wh_non_noop:
                applied += 1
                _log_op(out, op)

    return ApplyResult(
        applied=applied, total=total, created_webhooks=created_webhooks,
    )


def _log_op(printer: Callable[[str], None], op: Op) -> None:
    # `unchanged` is silent in the normal apply log — Plan 13's dry-run
    # path renders it through its own printer.
    if op.kind == "unchanged":
        return
    verb = {
        "create": "created",
        "update": "updated",
        "delete": "deleted",
    }.get(op.kind, op.kind)
    if isinstance(op, ScheduleOp):
        printer(f"    schedule {op.cron:<22} {verb}")
    else:
        printer(f"    webhook  {op.name:<22} {verb}")

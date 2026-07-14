"""Papayya — Durable background jobs for AI agents."""

from papayya.agent import Agent, agent, get_registry, get_agent
from papayya.papayya import Papayya
from papayya.client import Client, RunResult
from papayya.tools import tool
from papayya.durable import papayya, Item, PapayyaRun
from papayya.errors import CreditExhausted, WorkloadPaused
from papayya.classify import is_credit_exhaustion_error, classify_provider_error
from papayya.checks import CheckVerdict, llm_judge

# Plan 10: wrapper-shaped adoption surface. Import deferred until first
# attribute access via __getattr__ — eager import of iterators.py here
# would pull in papayya.outcomes at package-import time, which changes
# the order side-effecting modules (durable, runtime) initialize in for
# subprocess tests that re-import the whole package. ``iter`` shadows
# the builtin inside this module's namespace by design — the package-
# level name is ``papayya.iter``.

def __getattr__(name: str):
    if name in ("iter", "map", "mark_degraded", "mark_outcome", "llm", "step",
                "active_item", "active_run_id"):
        from papayya.iterators import (
            iter as _iter,
            map as _map,
            mark_degraded as _mark_degraded,
            mark_outcome as _mark_outcome,
            llm as _llm,
            step as _step,
            active_item as _active_item,
            active_run_id as _active_run_id,
        )
        return {
            "iter": _iter,
            "map": _map,
            "mark_degraded": _mark_degraded,
            "mark_outcome": _mark_outcome,
            "llm": _llm,
            "step": _step,
            "active_item": _active_item,
            "active_run_id": _active_run_id,
        }[name]
    if name in ("schedule", "trigger"):
        # Plan 11 decorators. Deferred for the same reason as Plan 10's
        # iter/mark_* — pulling papayya.decorators (which transitively
        # imports croniter + zoneinfo + papayya._config) at package
        # import time changes module-init ordering in the worker
        # subprocess enough to mask cross-process SQLite WAL writes.
        from papayya.decorators import schedule as _schedule, trigger as _trigger
        return _schedule if name == "schedule" else _trigger
    raise AttributeError(f"module 'papayya' has no attribute {name!r}")

__all__ = [
    "agent", "durable", "get_registry", "get_agent",
    "schedule", "trigger",
    "Papayya", "Client", "RunResult",
    "papayya", "Item", "PapayyaRun",
    "CreditExhausted", "WorkloadPaused",
    "is_credit_exhaustion_error", "classify_provider_error",
    "CheckVerdict", "llm_judge",
    "iter", "map", "mark_degraded", "mark_outcome",
    "llm", "step", "active_item", "active_run_id",
]

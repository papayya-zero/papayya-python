"""Papayya — Durable background jobs for AI agents."""

from papayya.agent import Agent, agent, get_registry, get_agent
from papayya.papayya import Papayya
from papayya.client import Client, RunResult
from papayya.tools import tool
from papayya.durable import papayya, PapayyaRun
from papayya.errors import CreditExhausted, BudgetExceeded
from papayya.classify import is_credit_exhaustion_error, classify_provider_error

# Plan 10: wrapper-shaped adoption surface. Import deferred until first
# attribute access via __getattr__ — eager import of iterators.py here
# would pull in papayya.outcomes at package-import time, which changes
# the order side-effecting modules (durable, runtime) initialize in for
# subprocess tests that re-import the whole package. ``iter`` shadows
# the builtin inside this module's namespace by design — the package-
# level name is ``papayya.iter``.

def __getattr__(name: str):
    if name in ("iter", "mark_degraded", "mark_outcome"):
        from papayya.iterators import iter as _iter, mark_degraded as _mark_degraded, mark_outcome as _mark_outcome
        if name == "iter":
            return _iter
        if name == "mark_degraded":
            return _mark_degraded
        if name == "mark_outcome":
            return _mark_outcome
    raise AttributeError(f"module 'papayya' has no attribute {name!r}")

__all__ = [
    "agent", "get_registry", "get_agent",
    "Papayya", "Client", "RunResult",
    "papayya", "PapayyaRun",
    "CreditExhausted", "BudgetExceeded",
    "is_credit_exhaustion_error", "classify_provider_error",
    "iter", "mark_degraded", "mark_outcome",
]

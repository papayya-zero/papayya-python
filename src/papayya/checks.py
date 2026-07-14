"""Customer-defined outcome checks (Plan 35 Unit 3).

A check is a ``Callable[[result], CheckVerdict | None]`` that runs inside the
SAME outcome pipeline as the built-in structural inspectors (``outcomes.py``):
``None`` means pass, a verdict flips the step's ``outcome_status``. The worst
verdict across built-in + custom checks wins the run's
``worst_outcome_status`` rollup — one pipeline, no second data path, no second
UI. The dashboard tells a custom verdict from a built-in one only by the
``user:`` reason prefix (auto-applied here) so the reason histogram can group
them.

Two kinds ship in v1, both returning the same verdict shape:

1. **Deterministic** — a plain callable the customer writes (length, schema,
   regex, business rule).
2. **LLM-judge** — a scaffold (:func:`llm_judge`) the customer parameterizes
   with a rubric + their own model-invoking callable (BYO key). Papayya owns
   the scaffold (format rubric+result → invoke your callable → parse a
   pass/fail → verdict), never a model. It runs INLINE on a sampled fraction
   of runs with a hard timeout — a slow/broken judge is a contained pass.

Invariant: a check is an OBSERVER. A check that raises, times out, or can't be
parsed is logged and treated as a pass — it must NEVER fail the run.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import logging
from typing import Any, Callable

from papayya.outcomes import OutcomeVerdict

log = logging.getLogger("papayya.checks")

# A check verdict is the same shape the built-in inspectors produce, so a
# custom verdict and a structural one flow through one pipeline (Decision 3).
CheckVerdict = OutcomeVerdict
Check = Callable[[Any], "OutcomeVerdict | None"]

_SEVERITY = {"ok": 0, "degraded": 1, "failed": 2}


def _severity(status: str) -> int:
    return _SEVERITY.get(status, 0)


def _namespace(reason: str | None) -> str | None:
    """Prefix a customer reason with ``user:`` so the dashboard histogram can
    group custom-check reasons (idempotent if already prefixed)."""
    if reason is None:
        return None
    return reason if reason.startswith("user:") else f"user:{reason}"


def degraded(reason: str) -> OutcomeVerdict:
    """A degraded verdict — the step ran but the output isn't good enough.
    The reason is auto-namespaced under ``user:``."""
    return OutcomeVerdict("degraded", _namespace(reason))


def failed(reason: str) -> OutcomeVerdict:
    """A failed verdict — the output is unacceptable. The reason is
    auto-namespaced under ``user:``."""
    return OutcomeVerdict("failed", _namespace(reason))


def _run_sampled(run_id: str, sample_rate: float) -> bool:
    """Deterministic per-RUN sampling gate (amendment #6). Checks fire
    per-step, so 'a fraction of runs' must key on run-level state — a hash of
    run_id — not a per-step coin flip that would silently sample per step."""
    if sample_rate >= 1.0:
        return True
    if sample_rate <= 0.0:
        return False
    h = int(hashlib.sha1(run_id.encode()).hexdigest()[:8], 16)
    return (h % 1000) < sample_rate * 1000


def run_checks(
    checks: list[Check], result: Any, base: OutcomeVerdict, run_id: str
) -> OutcomeVerdict:
    """Fold custom checks into the built-in verdict; worst severity wins.

    ``base`` is the built-in verdict (``ok`` when structural detection is off).
    Each check is contained: an exception is logged and treated as a pass.
    Checks carrying a ``_papayya_sample_rate`` (the judge) are gated per run.
    """
    worst = base
    for check in checks:
        sample_rate = getattr(check, "_papayya_sample_rate", None)
        if sample_rate is not None and not _run_sampled(run_id, sample_rate):
            continue
        try:
            verdict = check(result)
        except Exception:
            log.exception("papayya: outcome check raised; treating as pass (observer)")
            continue
        if verdict is None or verdict.status == "ok":
            continue
        verdict = OutcomeVerdict(verdict.status, _namespace(verdict.reason))
        if _severity(verdict.status) > _severity(worst.status):
            worst = verdict
    return worst


def _call_with_timeout(fn: Callable[[str], Any], arg: str, timeout: float) -> Any:
    """Invoke ``fn(arg)`` with a hard timeout. On timeout the worker thread is
    abandoned (shutdown wait=False) rather than joined, so a hung model call
    can't stall the run past ``timeout`` — that's the point of sampling."""
    ex = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    fut = ex.submit(fn, arg)
    try:
        return fut.result(timeout=timeout)
    finally:
        ex.shutdown(wait=False)


def _format_judge_prompt(rubric: str, result: Any) -> str:
    return (
        f"{rubric}\n\n"
        f"Output to evaluate:\n{result!r}\n\n"
        "Respond with PASS if the output meets the rubric, or FAIL if it does not."
    )


def _parse_judge(raw: Any, fail_keywords: tuple[str, ...], pass_keywords: tuple[str, ...]) -> str:
    """Parse a pass/fail from the model response. FAIL wins ties; an
    unparseable response is 'unknown' → a contained pass upstream."""
    text = str(raw).strip().lower()
    if any(kw in text for kw in fail_keywords):
        return "fail"
    if any(kw in text for kw in pass_keywords):
        return "pass"
    return "unknown"


def llm_judge(
    *,
    name: str,
    model: Callable[[str], Any],
    rubric: str,
    sample_rate: float = 0.2,
    timeout: float = 10.0,
    fail_keywords: tuple[str, ...] = ("fail",),
    pass_keywords: tuple[str, ...] = ("pass",),
) -> Check:
    """Build an LLM-judge check (Decision 4, amendments #4/#5).

    You supply ``model`` — a callable that invokes your model on YOUR key and
    returns the raw response — and a ``rubric``. Papayya supplies the scaffold:
    format the rubric + step result into a judge prompt, invoke your callable,
    parse a PASS/FAIL, map FAIL to a ``user:judge:<name>`` degraded verdict.

    Honesty constraints baked in: **BYO key only** (the callable is yours — we
    never spend our tokens judging your traffic) and **sampling** (``sample_rate``
    defaults < 1.0 to bound cost; sampled per run). The judge runs **inline**
    with a **hard timeout**; a timeout, error, or unparseable response is a
    **contained pass**, never a run failure.
    """
    verdict_reason = f"judge:{name}"

    def _check(result: Any) -> OutcomeVerdict | None:
        prompt = _format_judge_prompt(rubric, result)
        try:
            raw = _call_with_timeout(model, prompt, timeout)
        except Exception:
            log.warning("papayya: judge %r timed out or raised — contained pass", name)
            return None
        if _parse_judge(raw, fail_keywords, pass_keywords) == "fail":
            return OutcomeVerdict("degraded", f"user:{verdict_reason}")
        return None

    _check._papayya_sample_rate = sample_rate  # type: ignore[attr-defined]
    _check._papayya_check_name = name  # type: ignore[attr-defined]
    return _check

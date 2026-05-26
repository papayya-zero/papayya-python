"""Structural outcome inspectors.

Pure functions that look at a task's return value (and optionally the LLM
usage extracted from it) and decide whether the outcome is 'ok' or
'degraded'. Used by durable/run.py to populate TaskEntry.outcome_status
without any customer code change.

Each inspector is independent. The orchestrator (inspect_result) runs them
in order and returns the first 'degraded' verdict; if all return 'ok',
the overall outcome is 'ok'.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


# Module-level switch. Reading the value at integration time in
# _post_call_success lets tests monkey-patch this to disable structural
# detection without needing a YAML/config schema. A future plan will
# replace this with proper configuration.
ENABLE_STRUCTURAL_DETECTION: bool = True


@dataclass(frozen=True)
class OutcomeVerdict:
    status: str           # 'ok' | 'degraded'
    reason: str | None    # short token; None when status == 'ok'


OK = OutcomeVerdict("ok", None)


def inspect_empty(result: Any) -> OutcomeVerdict:
    """Flag absent/empty results as degraded.

    Numeric ``0`` / ``0.0`` are legitimate outputs, not degraded.
    """
    if result is None:
        return OutcomeVerdict("degraded", "empty_none")
    if result is False:
        return OutcomeVerdict("degraded", "empty_false")
    if isinstance(result, (str, bytes)) and len(result) == 0:
        return OutcomeVerdict("degraded", "empty_string")
    if isinstance(result, (list, tuple)) and len(result) == 0:
        return OutcomeVerdict("degraded", "empty_sequence")
    if isinstance(result, dict) and len(result) == 0:
        return OutcomeVerdict("degraded", "empty_dict")
    return OK


def _all_zero_numeric_sequence(seq: Any) -> bool:
    """True iff ``seq`` is a non-empty sequence of numbers all equal to 0.

    Conservative: returns False for empty sequences (empty is flagged by
    inspect_empty, not here) and for any element that isn't a plain
    ``int`` / ``float``.
    """
    if not isinstance(seq, (list, tuple)):
        return False
    if len(seq) == 0:
        return False
    for x in seq:
        if isinstance(x, bool):
            return False
        if not isinstance(x, (int, float)):
            return False
        if x != 0:
            return False
    return True


def inspect_degenerate_embedding(result: Any) -> OutcomeVerdict:
    """Flag zero embeddings (all-zero vectors) as degraded.

    Looks at three shapes:
      - A bare list/tuple of numbers.
      - A dict with an ``"embedding"`` or ``"embeddings"`` key holding the
        same shape.
      - Anything else (including numpy arrays we can't cheaply introspect)
        falls through to OK.
    """
    if _all_zero_numeric_sequence(result):
        return OutcomeVerdict("degraded", "degenerate_embedding")
    if isinstance(result, dict):
        for key in ("embedding", "embeddings"):
            if key in result and _all_zero_numeric_sequence(result[key]):
                return OutcomeVerdict("degraded", "degenerate_embedding")
    return OK


_DEGENERATE_STOP_REASONS = frozenset({"length", "content_filter", "refusal", "error"})


def inspect_llm_stop_reason(usage: Any) -> OutcomeVerdict:
    """Flag degenerate LLM stop reasons as degraded.

    ``usage`` is expected to be an :class:`~papayya.llm_extract.LlmUsage`
    or ``None`` when the step wasn't LLM-shaped. The function duck-types
    on ``.stop_reason`` so non-LLM callers can pass ``None`` safely.
    """
    if usage is None:
        return OK
    stop_reason = getattr(usage, "stop_reason", None)
    if stop_reason in _DEGENERATE_STOP_REASONS:
        return OutcomeVerdict("degraded", f"llm_stop_reason:{stop_reason}")
    return OK


def inspect_result(result: Any, *, usage: Any = None) -> OutcomeVerdict:
    """Run all inspectors and return the first degraded verdict.

    Order: stop-reason first (so LLM-shape signal wins over empty/zero
    checks on the response object), then empty, then degenerate
    embedding. Returns :data:`OK` when all inspectors pass.
    """
    verdict = inspect_llm_stop_reason(usage)
    if verdict.status != "ok":
        return verdict
    verdict = inspect_empty(result)
    if verdict.status != "ok":
        return verdict
    verdict = inspect_degenerate_embedding(result)
    if verdict.status != "ok":
        return verdict
    return OK

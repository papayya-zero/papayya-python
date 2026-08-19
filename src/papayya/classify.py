"""Shape-based classification for LLM provider exceptions.

These classifiers are provider-agnostic: they inspect exception shape
(``status_code``, ``body.error.type``, message substrings) rather than
isinstance-checking provider-specific SDK types. This is what lets the
same rules cover OpenAI, Anthropic, DeepSeek, Cohere, Fireworks, and
(via keyword heuristics) unknown providers without enumerating SDKs.

Tier structure:
  Tier 1 — HTTP 402 always means billing.
  Tier 2 — Specific error codes inside 4xx bodies (``insufficient_quota``,
           ``billing_hard_limit_reached``, ``credit_balance_too_low``, etc.).
  Tier 3 — Free-text keyword heuristics for unknown providers.
"""

from __future__ import annotations

from typing import Any


_CREDIT_KEYWORDS = (
    "payment method",
    "payment required",
    "billing limit",
    "billing capacity",
    "out of credits",
    "out of balance",
    "add credits",
    "insufficient credits",
    "insufficient funds",
    "prepaid balance",
    "exceeded your current quota, please check your plan and billing",
)

_TRANSIENT_SIGNALS = (
    "timeout",
    "timed out",
    "connection",
    "reset by peer",
    "broken pipe",
    "eof",
    "temporarily unavailable",
    "service unavailable",
)


def is_credit_exhaustion_error(exc: BaseException) -> bool:
    """Detect whether a provider exception indicates credit/quota exhaustion.

    Returns ``True`` for errors that mean "top up your account", not
    transient rate limits or server errors. A plain 429 with
    ``rate_limit_exceeded`` returns ``False``; a 429 with
    ``insufficient_quota`` returns ``True``.
    """
    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    message = str(exc).lower()

    # Structured error.type / error.code inside the body (OpenAI, Anthropic, DeepSeek)
    error_type = ""
    error_code = ""
    body: Any = getattr(exc, "body", None) or {}
    if isinstance(body, dict):
        error_obj = body.get("error", {})
        if isinstance(error_obj, dict):
            error_type = (error_obj.get("type", "") or "").lower()
            error_code = (error_obj.get("code", "") or "").lower()

    top_code = getattr(exc, "code", "") or ""
    if isinstance(top_code, str):
        top_code = top_code.lower()
    else:
        top_code = ""

    all_codes = f"{error_type} {error_code} {top_code}"

    # Tier 1 — HTTP 402 Payment Required is unambiguous billing.
    if status == 402:
        return True

    # Tier 2 — Specific error codes / types.
    _CODE_SIGNALS = (
        "insufficient_quota",           # OpenAI / Azure OpenAI
        "billing_hard_limit_reached",   # OpenAI
        "credit_balance_too_low",       # Anthropic
        "billing_error",                # Anthropic (also caught by 402)
        "insufficient_balance",         # DeepSeek (also caught by 402)
    )
    for signal in _CODE_SIGNALS:
        if signal in all_codes or signal in message:
            return True

    # Tier 3 — keyword heuristic for unknown providers.
    for keyword in _CREDIT_KEYWORDS:
        if keyword in message:
            return True

    return False


def classify_provider_error(exc: BaseException) -> str:
    """Classify a provider exception into one of three action categories.

    * ``"credit"``    — account billing / quota exhaustion → pause run.
    * ``"transient"`` — rate limit, server error, timeout → retry with backoff.
    * ``"permanent"`` — bad request, auth, not found → fail immediately.

    Credit detection wins over transient — a 429 with ``insufficient_quota``
    is credit, not a rate limit.
    """
    if is_credit_exhaustion_error(exc):
        return "credit"

    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)

    # Rate limits (429), server errors (5xx), overloaded (529).
    if status in (429, 500, 502, 503, 529):
        return "transient"

    # Connection / timeout errors (no HTTP status) are transient.
    if status is None:
        exc_type = type(exc).__name__.lower()
        exc_msg = str(exc).lower()
        for signal in _TRANSIENT_SIGNALS:
            if signal in exc_type or signal in exc_msg:
                return "transient"

    # Everything else (400, 401, 403, 404, ...) is permanent.
    return "permanent"


# Statuses that are POSITIVE evidence a second attempt is pointless. A bad
# request is still bad in four seconds; a 401 is still a 401.
#
# 429 IS DELIBERATELY ABSENT even though it is a client-error status — a rate
# limit is the single most retryable thing a provider produces.
_PERMANENT_STATUSES = frozenset({400, 401, 403, 404, 405, 409, 410, 422})


def is_retryable(exc: BaseException) -> bool:
    """Whether ``run.step`` should run this step again (plan 58 R1).

    **This deliberately does NOT delegate to :func:`classify_provider_error`,
    and the reason is the measurement that shaped the whole unit.** That
    function's final line is ``return "permanent"`` — so ``permanent`` means
    *either* "we saw a 401" *or* "we have never seen this exception before",
    and it cannot tell you which. Executed against the exception this feature
    exists to fix::

        >>> classify_provider_error(OcrUnavailable(
        ...     "ocr sidecar returned 503 for page 17 of DOC-2001"))
        'permanent'

    A customer-raised exception carries no ``status_code`` attribute and does
    not match the keyword list, so it falls through to the default. Gating
    retries on that verdict means **retrying only exceptions raised by an LLM
    SDK we already recognise** — which is every case except the ones customers
    actually write.

    So the rule inverts: **retry unless there is positive evidence it is
    pointless.** Unknown is not evidence.

    Three things stop a retry, each for its own reason:

    * :class:`NonRetriable` — the customer said so, and they know their domain.
    * Credit exhaustion — retrying cannot fund an account. The existing
      ``CreditExhausted`` path pauses the run instead, which is a better
      outcome than either failing or spinning.
    * An identified permanent HTTP status — the only case where the platform
      itself has grounds.

    ``classify_provider_error`` keeps its job of labelling
    ``error_category`` on the record; it is just no longer the gate.
    """
    from papayya.errors import CreditExhausted, NonRetriable, WorkloadPaused

    # WorkloadPaused is control flow, not a failure: a fence stopped the run
    # between steps. Retrying it would fight the fence.
    if isinstance(exc, (NonRetriable, WorkloadPaused, CreditExhausted)):
        return False
    if is_credit_exhaustion_error(exc):
        return False
    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    if status in _PERMANENT_STATUSES:
        return False
    return True

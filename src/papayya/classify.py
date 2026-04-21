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

"""Error classification for local step capture.

Classifies an error message into one of four categories so the local
dashboard can colour badges and cluster failures meaningfully:

- ``provider`` — the model provider (rate limit, overloaded, auth, credit)
- ``tool``     — a user-defined tool raised during a tool call
- ``timeout``  — execution ran past a configured budget or the wall-clock
- ``logic``    — everything else (ValueError, KeyError, assertion, etc.)

The classifier is intentionally **string-pattern based** rather than
exception-type based. Errors arrive here as already-stringified messages
from provider SDKs whose exception hierarchies we do not control.

Column names and category values are contracts with the hosted product
(see ``memory/credit_exhaustion_detection.md``). Do not rename without
coordinating the hosted side.
"""

from __future__ import annotations

import re
from typing import Final

CATEGORY_PROVIDER: Final = "provider"
CATEGORY_TOOL: Final = "tool"
CATEGORY_TIMEOUT: Final = "timeout"
CATEGORY_LOGIC: Final = "logic"


# Each tuple: (error_code, pattern) where pattern is applied case-insensitively.
# Order matters — first match wins. Keep patterns tight; we'd rather mis-bucket
# a rare error to "logic" than false-positive a common error into "provider".
_PROVIDER_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("provider_rate_limit", re.compile(r"\b(rate.?limit|429|too many requests)\b", re.I)),
    ("provider_overloaded", re.compile(r"\b(overloaded|529|service unavailable|503)\b", re.I)),
    ("provider_credit",     re.compile(r"(insufficient.{0,3}(credits?|balance|quota)|payment required|\b402\b|billing)", re.I)),
    ("provider_auth",       re.compile(r"\b(invalid.?api.?key|unauthorized|401|authentication)\b", re.I)),
    ("provider_context",    re.compile(r"\b(context.?length|maximum context|token.?limit)\b", re.I)),
    ("provider_bad_request",re.compile(r"\b(bad request|400|invalid request)\b", re.I)),
    ("provider_server",     re.compile(r"\b(internal server error|500|502)\b", re.I)),
]

_TIMEOUT_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("budget_exceeded", re.compile(r"\bbudget exceeded\b", re.I)),
    ("timeout",         re.compile(r"\b(timed out|timeout|deadline exceeded)\b", re.I)),
]

_TOOL_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("tool_execution", re.compile(r"\btool (call|execution|function) (failed|error)\b", re.I)),
    ("tool_json",      re.compile(r"\b(malformed|invalid).?(json|tool.?arg)", re.I)),
]


def classify_error(message: str | None) -> tuple[str | None, str | None]:
    """Return ``(error_code, error_category)`` for a stringified error.

    Returns ``(None, None)`` for empty / whitespace-only input so callers
    can safely pass through non-error paths.
    """
    if not message or not message.strip():
        return (None, None)

    for code, pattern in _PROVIDER_PATTERNS:
        if pattern.search(message):
            return (code, CATEGORY_PROVIDER)

    for code, pattern in _TIMEOUT_PATTERNS:
        if pattern.search(message):
            return (code, CATEGORY_TIMEOUT)

    for code, pattern in _TOOL_PATTERNS:
        if pattern.search(message):
            return (code, CATEGORY_TOOL)

    return ("logic_error", CATEGORY_LOGIC)

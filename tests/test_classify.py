"""Shape-based classifier tests — provider-agnostic.

These tests assemble duck-typed exception objects and feed them through
``is_credit_exhaustion_error`` and ``classify_provider_error``. They do
not import any LLM SDK; the whole point of the design is that the rules
work on shape, so the tests should work on shape too.

Covers Tier 1 (HTTP 402), Tier 2 (specific error codes), Tier 3 (keyword
heuristics), plus negatives (rate-limit, auth, bad-request) to guard
against false-positive pauses.
"""

from __future__ import annotations

import pytest

from papayya.classify import classify_provider_error, is_credit_exhaustion_error


class FakeErr(Exception):
    """Duck-typed stand-in for provider SDK exceptions.

    Providers expose ``status_code`` (OpenAI, Anthropic) or ``status``
    (some older SDKs); ``body`` as a dict with nested ``error.type`` /
    ``error.code`` (OpenAI, Anthropic, DeepSeek); and ``code`` at the
    top level (some flattened SDKs). The classifier duck-types all of
    them via ``getattr``.
    """

    def __init__(
        self,
        *,
        message: str = "",
        status_code: int | None = None,
        body: dict | None = None,
        code: str = "",
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.body = body
        self.code = code


# (case_id, exception, expect_credit, expect_category)
_CASES = [
    # ── Tier 1: HTTP 402 Payment Required ──
    (
        "tier1-402-anthropic-billing",
        FakeErr(status_code=402, body={"error": {"type": "billing_error"}}),
        True,
        "credit",
    ),
    (
        "tier1-402-deepseek-balance",
        FakeErr(status_code=402, body={"error": {"type": "insufficient_balance"}}),
        True,
        "credit",
    ),
    (
        "tier1-402-no-body",
        FakeErr(status_code=402, message="Payment required"),
        True,
        "credit",
    ),

    # ── Tier 2: Specific error codes ──
    (
        "tier2-openai-insufficient-quota",
        FakeErr(status_code=429, body={"error": {"type": "insufficient_quota"}}),
        True,
        "credit",
    ),
    (
        "tier2-openai-billing-hard-limit",
        FakeErr(status_code=429, body={"error": {"type": "billing_hard_limit_reached"}}),
        True,
        "credit",
    ),
    (
        "tier2-anthropic-balance-too-low",
        FakeErr(status_code=400, body={"error": {"type": "credit_balance_too_low"}}),
        True,
        "credit",
    ),
    (
        "tier2-top-level-code-field",
        FakeErr(status_code=400, code="insufficient_quota"),
        True,
        "credit",
    ),

    # ── Tier 3: keyword heuristics for unknown providers ──
    (
        "tier3-keyword-out-of-credits",
        FakeErr(message="You are out of credits. Please add a payment method."),
        True,
        "credit",
    ),
    (
        "tier3-keyword-payment-required",
        FakeErr(message="Payment required to continue."),
        True,
        "credit",
    ),

    # ── Negatives: rate limit is NOT credit ──
    (
        "neg-openai-rate-limit-exceeded",
        FakeErr(status_code=429, body={"error": {"type": "rate_limit_exceeded"}}),
        False,
        "transient",
    ),
    (
        "neg-bare-429",
        FakeErr(status_code=429, message="Rate limit exceeded"),
        False,
        "transient",
    ),
    (
        "neg-quota-exceeded-alone-is-ambiguous",
        # Deliberately rejected — "quota exceeded" alone is often a
        # transient rate-limit message, not a billing signal.
        FakeErr(status_code=429, message="Quota exceeded, try again later"),
        False,
        "transient",
    ),

    # ── Negatives: permanent errors ──
    (
        "perm-401-auth",
        FakeErr(status_code=401, message="Invalid API key"),
        False,
        "permanent",
    ),
    (
        "perm-400-bad-request",
        FakeErr(status_code=400, body={"error": {"type": "invalid_request_error"}}),
        False,
        "permanent",
    ),
    (
        "perm-404-not-found",
        FakeErr(status_code=404, message="Model not found"),
        False,
        "permanent",
    ),

    # ── Negatives: transient server/network ──
    (
        "trans-500-server",
        FakeErr(status_code=500, message="Internal server error"),
        False,
        "transient",
    ),
    (
        "trans-503-unavailable",
        FakeErr(status_code=503, message="Service unavailable"),
        False,
        "transient",
    ),
    (
        "trans-connection-no-status",
        FakeErr(message="Connection reset by peer"),
        False,
        "transient",
    ),
]


@pytest.mark.parametrize(
    "exc, expect_credit, expect_category",
    [(exc, credit, cat) for _, exc, credit, cat in _CASES],
    ids=[case_id for case_id, *_ in _CASES],
)
def test_shape_based_classification(exc, expect_credit, expect_category):
    assert is_credit_exhaustion_error(exc) is expect_credit
    assert classify_provider_error(exc) == expect_category


def test_non_dict_body_does_not_crash():
    # Some SDKs return bytes or strings in body; classifier must not blow up.
    exc = FakeErr(status_code=429, message="limit")
    exc.body = b"opaque bytes"  # type: ignore[assignment]
    assert is_credit_exhaustion_error(exc) is False


def test_non_string_top_level_code_does_not_crash():
    # ``code`` is sometimes an int on weird SDKs; classifier must coerce.
    exc = FakeErr(status_code=429, message="limit")
    exc.code = 42  # type: ignore[assignment]
    assert is_credit_exhaustion_error(exc) is False
    assert classify_provider_error(exc) == "transient"

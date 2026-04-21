"""Papayya exception types.

These are the user-facing exception classes for BYOF flows. The runtime
shim's interceptor raises :class:`CreditExhausted` automatically for
providers whose exception shape we can classify (OpenAI, Anthropic).
Users with other providers can raise :class:`CreditExhausted` themselves
to trigger the same pause-and-resume behavior::

    from papayya import CreditExhausted

    try:
        response = my_custom_llm.chat(messages)
    except MyProviderOutOfCredits as e:
        raise CreditExhausted(f"custom provider out of credits: {e}") from e
"""

from __future__ import annotations


class CreditExhausted(Exception):
    """Raised when the LLM provider reports credit/quota exhaustion.

    Pauses the run rather than failing it. The user tops up their provider
    account and resumes; all durable checkpoints are preserved.
    """
    pass


class BudgetExceeded(Exception):
    """Raised when a run exceeds its per-run budget cap.

    Enforced at the control-plane side (InsertStep) and, when enabled,
    proactively by the shim interceptor via pre-call reservations.
    """
    pass

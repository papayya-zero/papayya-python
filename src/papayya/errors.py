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


class WorkloadPaused(Exception):
    """Raised at a step boundary when a fence has paused the run (Plan 33).

    The system stopped spending on your behalf — a degraded-output streak, a
    budget breach, or a workload-level degraded-rate threshold — after the
    just-completed step was safely checkpointed. This is not a failure: the
    run's server-side (or local) status is ``paused``, in-flight work is
    preserved, and an operator resume + replay picks up exactly where the
    pause landed. Named and catchable so a customer body can special-case it
    (e.g. log and exit cleanly) instead of treating it as a crash.

    ``reason`` carries the trigger detail ("3 consecutive degraded steps:
    llm_empty_content", "budget", "11 of last 20 runs degraded"); ``run_id``
    is the paused run.
    """

    def __init__(self, reason: str, run_id: str | None = None) -> None:
        super().__init__(reason)
        self.reason = reason
        self.run_id = run_id

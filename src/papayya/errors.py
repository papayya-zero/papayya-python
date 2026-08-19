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

    Pauses the run rather than failing it, and all durable checkpoints are
    preserved — so no completed work is lost and nothing is re-paid for.

    The worker releases its lease back to the queue (parked) and marks the
    run ``paused`` with ``pause_reason="credit"``. Top up the provider
    account and resume the run: it re-drives from where it stopped, skipping
    every checkpointed step.

    Papayya cannot know when your provider account is funded again, so it
    does not retry on a timer. The resume is the signal.
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


class NonRetriable(Exception):
    """Raise this to tell the runtime NOT to retry the step (plan 58 R1).

    ``run.step`` retries a raising step by default, because a transient
    provider outage should not become an operator's incident. That default is
    wrong for work you know cannot succeed on a second attempt — a malformed
    document, a validation failure, an unroutable record — and re-running it
    costs the customer time and money for a certainty::

        from papayya import NonRetriable

        def classify(doc):
            if doc["kind"] not in KNOWN:
                raise NonRetriable(f"unknown document kind: {doc['kind']}")

    Wrap the real cause rather than replacing it::

        except MyValidationError as e:
            raise NonRetriable(str(e)) from e

    The exception surfaces to the run exactly as any other failure — the run
    fails, the record enters triage, `error` carries this message. The only
    thing this class changes is how many times the step ran first.

    For provider credit exhaustion use :class:`CreditExhausted` instead: that
    one pauses the run rather than failing it, which is a different and better
    outcome for a condition the operator can actually fix.
    """
    pass

"""Runtime-side hooks for SDK code running inside the papayya runtime.

When user code runs inside the papayya runtime container, the shim
installs a :class:`LlmCallReporter` into a context variable so the SDK
can emit telemetry through the same channel the interceptor uses. When
user code runs outside the runtime (local ``python agent.py`` / pytest /
anywhere else), the context variable stays unset and SDK hooks are
no-ops — nothing to report to.

This module is intentionally small and dependency-free; the shim
provides its own :class:`LlmCallReporter` implementation that wires
into its reporter + usage tracker.
"""

from __future__ import annotations

from contextvars import ContextVar
from typing import Protocol, runtime_checkable

from papayya.llm_extract import LlmUsage


@runtime_checkable
class LlmCallReporter(Protocol):
    """Runtime hook for emitting LLM-call telemetry from SDK-wrapped calls.

    The shim implements this interface and installs an instance via
    :func:`set_current_reporter` before running the user's agent. SDK
    code (currently ``run.step(kind="llm")``) consults the contextvar
    to decide whether to emit.

    Dedupe is per-call: the SDK opens a scope via :meth:`begin_call`
    before invoking the wrapped fn and asks :meth:`was_emitted_for`
    after. The implementation tracks, for each token, whether its
    interceptor recorded a step while that token was active — typically
    by stashing the active token in a ``ContextVar`` that the
    interceptor reads on every emission. Per-call scoping is correct
    under ``asyncio.gather`` (each task gets its own contextvar copy);
    the legacy global counter (:meth:`intercepted_call_count`) was not.
    """

    def begin_call(self, label: str) -> object:
        """Open a per-call dedupe scope.

        Returns an opaque token the caller passes back to
        :meth:`was_emitted_for`. Implementations MAY tag concurrent
        emissions from their interceptor against the active token so
        the SDK can ask, post-call, whether the interceptor handled
        this particular call.

        Tokens are opaque to callers — they MUST NOT inspect them.
        """
        ...

    def was_emitted_for(self, token: object) -> bool:
        """Return True iff the interceptor already emitted a step for
        the call associated with ``token``. The SDK uses this to skip
        a second emission for the same underlying call.

        Callers MUST invoke this exactly once per :meth:`begin_call`
        (cleanup is idempotent on the implementation side, but the
        single-call contract makes leak audits easy).
        """
        ...

    def intercepted_call_count(self) -> int:
        """DEPRECATED — use :meth:`begin_call` + :meth:`was_emitted_for`.

        Returns the monotonic count of LLM calls recorded by the
        interceptor so far. Implementations SHOULD keep this method
        for one release so older SDK builds (which snapshot pre/post
        counts) keep working against newer shims.
        """
        ...

    def report_llm_call(
        self,
        *,
        label: str,
        usage: LlmUsage,
        duration_ms: int,
        error_category: str | None = None,
    ) -> None:
        """Emit one LLM-call step. Must be idempotent and safe to no-op
        when the interceptor already handled the underlying call.
        """
        ...


_current_reporter: ContextVar[LlmCallReporter | None] = ContextVar(
    "papayya_current_llm_reporter",
    default=None,
)


def set_current_reporter(reporter: LlmCallReporter | None) -> object:
    """Install the runtime reporter into the current context.

    Returns the :class:`contextvars.Token` from ``ContextVar.set`` so the
    caller can ``reset`` it later (or discard if the process exits).
    """
    return _current_reporter.set(reporter)


def get_current_reporter() -> LlmCallReporter | None:
    """Return the runtime reporter for the current context, if any.

    Returns ``None`` in local/non-runtime contexts — SDK callers should
    treat ``None`` as "no telemetry channel; proceed silently".
    """
    return _current_reporter.get()


def reset_current_reporter(token: object) -> None:
    """Restore the prior contextvar state (inverse of
    :func:`set_current_reporter`). Safe to call in a ``finally`` block.
    """
    _current_reporter.reset(token)  # type: ignore[arg-type]

"""OpenTelemetry baggage helpers.

OTel stays an *optional* SDK dependency. Customers without ``papayya[otel]``
installed see every helper here turn into a no-op. Customers who install
the extra get workload/item/tenant context propagated as OTel baggage and
annotated on the active recording span, so that the control-pane mapper
can route per-call cost back to the right workload row.

These helpers exist to bridge two halves of the system that meet at the
span attribute layer:

- The control-pane mapper reads ``papayya.workload`` / ``papayya.item_id`` /
  ``papayya.partition_key`` off received spans (see ``control-pane/internal/otel/mapper.go``).
- The SDK ``@agent`` wrapper has the values at call time but no way to
  hand them to whichever LLM SDK the customer uses unless we plant them
  in the OTel context.

We do both: set baggage (so downstream instrumented calls inherit it) and
annotate the current span (so the very next span exported also carries
the attrs — not every instrumentation library copies baggage onto exported
span attributes automatically).
"""

from __future__ import annotations

from typing import Any

try:
    from opentelemetry import baggage, context, trace as _trace

    _OTEL_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised by the not-installed test branch
    _OTEL_AVAILABLE = False


BAGGAGE_WORKLOAD = "papayya.workload"
BAGGAGE_ITEM_ID = "papayya.item_id"
BAGGAGE_PARTITION_KEY = "papayya.partition_key"


def set_papayya_baggage(
    *,
    workload: Any = None,
    item_id: Any = None,
    partition_key: Any = None,
) -> Any:
    """Set Papayya baggage on the current OTel context.

    Returns an opaque token to pass to :func:`clear_papayya_baggage` in
    the matching ``finally`` block. Returns ``None`` when OTel is not
    installed — callers must hand the same ``None`` back to ``clear``
    (which no-ops) so the wrapper code is identical on both paths.
    """
    if not _OTEL_AVAILABLE:
        return None
    ctx = context.get_current()
    if workload:
        ctx = baggage.set_baggage(BAGGAGE_WORKLOAD, str(workload), context=ctx)
    if item_id is not None:
        ctx = baggage.set_baggage(BAGGAGE_ITEM_ID, str(item_id), context=ctx)
    if partition_key is not None:
        ctx = baggage.set_baggage(BAGGAGE_PARTITION_KEY, str(partition_key), context=ctx)
    return context.attach(ctx)


def clear_papayya_baggage(token: Any) -> None:
    """Detach the baggage context returned by :func:`set_papayya_baggage`.

    No-op when ``token`` is ``None`` (either OTel is not installed, or
    the caller never attached anything).
    """
    if token is None or not _OTEL_AVAILABLE:
        return
    context.detach(token)


def annotate_current_span(
    *,
    workload: Any = None,
    item_id: Any = None,
    partition_key: Any = None,
) -> None:
    """Set the three Papayya attributes on the current active recording span.

    Baggage flows through context but not every instrumentation library
    copies baggage onto exported span attributes. Annotating the current
    span directly ensures the control-pane mapper finds these values on
    the spans it processes — without this, baggage round-trips only when
    a downstream library happens to bridge baggage → attributes.
    """
    if not _OTEL_AVAILABLE:
        return
    span = _trace.get_current_span()
    if span is None or not span.is_recording():
        return
    if workload:
        span.set_attribute(BAGGAGE_WORKLOAD, str(workload))
    if item_id is not None:
        span.set_attribute(BAGGAGE_ITEM_ID, str(item_id))
    if partition_key is not None:
        span.set_attribute(BAGGAGE_PARTITION_KEY, str(partition_key))

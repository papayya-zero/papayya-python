"""OTel baggage helper tests (Plan 07).

Two halves:

1. ``OTel-not-installed`` branch — exercised by monkey-patching
   ``_OTEL_AVAILABLE`` to ``False``. Confirms the helpers degrade to
   silent no-ops so customers without ``papayya[otel]`` installed
   import cleanly and the ``@agent`` wrapper keeps working unchanged.

2. ``OTel-installed`` branch — exercised when ``opentelemetry-api``
   is importable. Confirms baggage actually lands in the OTel context
   and that the current recording span gets the three Papayya
   attributes annotated on it.

The installed-branch tests use ``pytest.importorskip`` so the suite
passes on environments without the extra installed.
"""

from __future__ import annotations

import pytest

from papayya import _otel_baggage


# ---------------------------------------------------------------------------
# OTel not installed — degrades to no-ops
# ---------------------------------------------------------------------------


def test_set_papayya_baggage_returns_none_when_otel_missing(monkeypatch):
    monkeypatch.setattr(_otel_baggage, "_OTEL_AVAILABLE", False)
    token = _otel_baggage.set_papayya_baggage(workload="x", item_id="y", partition_key="z")
    assert token is None


def test_clear_papayya_baggage_is_noop_on_none(monkeypatch):
    monkeypatch.setattr(_otel_baggage, "_OTEL_AVAILABLE", False)
    # No exception, no return value — just a quiet no-op so callers can
    # always finally-clear without branching on OTel availability.
    _otel_baggage.clear_papayya_baggage(None)


def test_annotate_current_span_is_noop_when_otel_missing(monkeypatch):
    monkeypatch.setattr(_otel_baggage, "_OTEL_AVAILABLE", False)
    # Must not raise even though no span exists; the wrapper must be
    # safe to call unconditionally on the not-installed branch.
    _otel_baggage.annotate_current_span(workload="x", item_id="y", partition_key="z")


# ---------------------------------------------------------------------------
# OTel installed — baggage and span annotation actually happen
# ---------------------------------------------------------------------------
#
# Each test guards itself with importorskip so the not-installed tests
# above always run, while these are silently skipped when the optional
# extra is absent.


def _skip_unless_otel():
    pytest.importorskip(
        "opentelemetry.baggage",
        reason="papayya[otel] not installed — skipping OTel-installed branch",
    )


def test_set_papayya_baggage_populates_keys():
    _skip_unless_otel()
    from opentelemetry import baggage as ot_baggage

    token = _otel_baggage.set_papayya_baggage(
        workload="extract",
        item_id="msg_123",
        partition_key="org_47",
    )
    try:
        assert ot_baggage.get_baggage(_otel_baggage.BAGGAGE_WORKLOAD) == "extract"
        assert ot_baggage.get_baggage(_otel_baggage.BAGGAGE_ITEM_ID) == "msg_123"
        assert ot_baggage.get_baggage(_otel_baggage.BAGGAGE_PARTITION_KEY) == "org_47"
    finally:
        _otel_baggage.clear_papayya_baggage(token)
    # After detach, the keys must be gone — the baggage scope ends with
    # the wrapper's finally block in agent.py.
    assert ot_baggage.get_baggage(_otel_baggage.BAGGAGE_WORKLOAD) is None


def test_set_papayya_baggage_skips_none_values():
    _skip_unless_otel()
    from opentelemetry import baggage as ot_baggage

    token = _otel_baggage.set_papayya_baggage(workload="only_wl")
    try:
        assert ot_baggage.get_baggage(_otel_baggage.BAGGAGE_WORKLOAD) == "only_wl"
        # item_id / partition_key were not provided — must stay NULL on
        # the eventual usage_event row, which means absent from baggage.
        assert ot_baggage.get_baggage(_otel_baggage.BAGGAGE_ITEM_ID) is None
        assert ot_baggage.get_baggage(_otel_baggage.BAGGAGE_PARTITION_KEY) is None
    finally:
        _otel_baggage.clear_papayya_baggage(token)


def test_annotate_current_span_sets_three_attributes():
    _skip_unless_otel()
    pytest.importorskip("opentelemetry.sdk.trace", reason="opentelemetry-sdk required")
    from opentelemetry import trace as ot_trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("papayya-test")

    with tracer.start_as_current_span("test-span") as span:
        _otel_baggage.annotate_current_span(
            workload="extract",
            item_id="msg_123",
            partition_key="org_47",
        )
        # End-of-block exports the span.

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    attrs = dict(spans[0].attributes or {})
    assert attrs.get(_otel_baggage.BAGGAGE_WORKLOAD) == "extract"
    assert attrs.get(_otel_baggage.BAGGAGE_ITEM_ID) == "msg_123"
    assert attrs.get(_otel_baggage.BAGGAGE_PARTITION_KEY) == "org_47"
    # Suppress unused-import warnings: keep refs alive for clarity.
    _ = (ot_trace,)


def test_annotate_current_span_is_safe_with_no_active_span():
    _skip_unless_otel()
    # No active span on the stack — get_current_span returns the
    # INVALID_SPAN sentinel whose is_recording() is False. Must not
    # raise, must not annotate anything.
    _otel_baggage.annotate_current_span(workload="x")


# ---------------------------------------------------------------------------
# Integration: @agent body annotates the active span
# ---------------------------------------------------------------------------


def test_agent_decorator_annotates_workload_on_active_span():
    _skip_unless_otel()
    pytest.importorskip("opentelemetry.sdk.trace", reason="opentelemetry-sdk required")
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    from papayya.agent import agent

    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("papayya-integration-test")

    @agent(name="extract_invoices")
    def extract(item_id):  # noqa: ARG001 - test fixture
        return "ok"

    with tracer.start_as_current_span("outer-span"):
        assert extract("invoice-42") == "ok"

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    attrs = dict(spans[0].attributes or {})
    assert attrs.get(_otel_baggage.BAGGAGE_WORKLOAD) == "extract_invoices"
    assert attrs.get(_otel_baggage.BAGGAGE_ITEM_ID) == "invoice-42"

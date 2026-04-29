"""Dispatcher event log shape — Phase 2 ADR-0002 #16.

The Phase 1.5 mechanical feel-test (2026-04-28) found the dispatcher's
``completed`` event lacked per-item duration, forcing operators to
subtract leased→completed timestamps by hand. This test pins the new
field on both the event payload and the formatted log line.

Computing duration dispatcher-side (from `_LeasedRecord.leased_at`)
keeps the measurement single-clock — worker skew can't mislead reads.
"""

from __future__ import annotations

import time

from papayya.runtime.dispatcher import LocalDispatcher, _format_event


def test_completed_event_includes_duration_ms():
    events: list[tuple[str, dict]] = []
    dispatcher = LocalDispatcher(
        port=0,
        on_event=lambda kind, data: events.append((kind, data)),
        # Long TTL so the reaper is irrelevant to this test.
        lease_ttl_seconds=60.0,
    )
    try:
        lease_id = dispatcher.enqueue(agent="t", item_id="i1")
        item = dispatcher._take_lease(worker_id="w1")
        assert item is not None
        # Brief gap so the duration is measurable, not zero.
        time.sleep(0.05)
        dispatcher._mark_complete(
            lease_id=lease_id,
            status="completed",
            error=None,
            worker_id="w1",
        )
    finally:
        dispatcher.shutdown()

    completed = [d for kind, d in events if kind == "completed"]
    assert len(completed) == 1
    duration = completed[0].get("duration_ms")
    assert duration is not None, "duration_ms missing from completed event"
    # Sanity bounds: positive, but well under the test's wallclock budget.
    assert 0 < duration < 5000, f"duration_ms={duration}ms out of plausible bounds"


def test_completed_log_line_renders_duration_field():
    line = _format_event("completed", {
        "lease_id": "abcdef0123456789",
        "status": "completed",
        "worker_id": "w1",
        "duration_ms": 47,
        "error": None,
    })
    assert "duration=47ms" in line, line


def test_completed_log_line_omits_duration_when_absent():
    """Back-compat: callers that don't pass duration_ms (e.g. older
    test harnesses) shouldn't get a stray ``duration=None`` field."""
    line = _format_event("completed", {
        "lease_id": "abcdef0123456789",
        "status": "completed",
        "worker_id": "w1",
        "error": None,
    })
    assert "duration" not in line, line

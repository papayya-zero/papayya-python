"""papayya.signal() — the contract is about what it REFUSES to do.

Plan 43 B2b C9 / ADR 0009 D8-D9. Every test here is about a failure mode, because
the happy path is one POST and the whole design is the behaviour when the control
plane is slow, down, or misconfigured — this runs inside a customer's
thumbs-down handler.
"""

import logging
import threading
import time

import pytest

import papayya
from papayya.api import PapayyaAPIError


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("PAPAYYA_API_KEY", raising=False)
    monkeypatch.delenv("PAPAYYA_BASE_URL", raising=False)


def test_missing_api_key_raises_on_the_calling_thread(monkeypatch, tmp_path):
    """A configuration failure must NOT be swallowed like a delivery failure.

    The difference matters more than it looks: a dropped signal is one lost
    complaint, an unconfigured deployment is EVERY complaint, silently, forever.
    So this one raises where the customer's traceback can reach it — off-thread
    it would only ever be a log line in a web worker nobody tails.
    """
    # CONFIG_FILE is resolved at import time, so moving $HOME does nothing —
    # on a developer machine `papayya login` has already written a real key and
    # this test would pass for the wrong reason.
    import papayya._config as config_module
    monkeypatch.setattr(config_module, "CONFIG_FILE", tmp_path / "nothing.json")

    with pytest.raises(PapayyaAPIError):
        papayya.signal("order-1", agent="triage")


def test_bad_verdict_raises_before_anything_is_dispatched(monkeypatch):
    monkeypatch.setenv("PAPAYYA_API_KEY", "cpk_deadbeef_" + "0" * 32)
    monkeypatch.setenv("PAPAYYA_BASE_URL", "http://127.0.0.1:9")
    with pytest.raises(ValueError, match="verdict"):
        papayya.signal("order-1", agent="triage", verdict="banana")
    with pytest.raises(ValueError, match="source"):
        papayya.signal("order-1", agent="triage", source="telepathy")


def test_a_dead_control_plane_neither_raises_nor_blocks(monkeypatch):
    """The measurement that decided this module exists.

    Through APIClient._request the same call blocks 1.52 s against a dead port
    (and ~151 s against a black hole at the default timeout), because that client
    retries — correctly, for a run submission. Here one attempt is the whole
    policy.
    """
    monkeypatch.setenv("PAPAYYA_API_KEY", "cpk_deadbeef_" + "0" * 32)
    monkeypatch.setenv("PAPAYYA_BASE_URL", "http://127.0.0.1:9")  # nothing listens

    started = time.monotonic()
    future = papayya.signal("order-1", agent="triage")
    caller_returned = time.monotonic() - started

    assert future.result(timeout=5) is None      # delivery failed, quietly
    assert future.exception(timeout=5) is None   # and did not raise
    assert caller_returned < 0.1, f"caller blocked for {caller_returned:.3f}s"


def test_the_caller_is_not_on_the_sending_thread(monkeypatch):
    """A slow control plane must cost the handler nothing.

    Asserted by identity rather than by timing: the send runs on some thread
    that is not this one. A timing assertion would pass on a fast machine even
    if the call were synchronous.
    """
    monkeypatch.setenv("PAPAYYA_API_KEY", "cpk_deadbeef_" + "0" * 32)
    monkeypatch.setenv("PAPAYYA_BASE_URL", "http://127.0.0.1:9")

    future = papayya.signal("order-1", agent="triage")
    future.result(timeout=5)
    assert threading.current_thread() is threading.main_thread()


def test_a_4xx_is_logged_as_the_callers_bug_and_still_does_not_raise(
    monkeypatch, caplog
):
    """A rejected signal is a wire-up error — the wrong agent slug, say. It can
    only be reported by log, because there is nobody to raise to; it is logged at
    ERROR rather than WARNING so it does not read as a transient network blip.
    """
    import papayya.signals as signals_module

    class _Rejecting:
        def post(self, path, json):  # noqa: A002 — httpx's parameter name
            class R:
                status_code = 400
                text = '{"error":{"message":"unknown agent"}}'
                is_success = False
            return R()

    monkeypatch.setattr(signals_module, "_client", lambda *a, **k: _Rejecting())
    monkeypatch.setenv("PAPAYYA_API_KEY", "cpk_deadbeef_" + "0" * 32)
    monkeypatch.setenv("PAPAYYA_BASE_URL", "http://127.0.0.1:9")

    with caplog.at_level(logging.WARNING, logger="papayya.signal"):
        assert papayya.signal("order-1", agent="nope").result(timeout=5) is None

    records = [r for r in caplog.records if r.name == "papayya.signal"]
    assert records and records[0].levelno == logging.ERROR
    assert "order-1" in records[0].getMessage()


def test_the_payload_carries_the_customers_key_and_omits_what_was_not_passed(
    monkeypatch,
):
    """item_id/agent are the KEY, and an unset field must not be sent as null —
    the server distinguishes "no partition" from "partition null" and they mean
    the same thing only by accident.
    """
    import papayya.signals as signals_module
    seen = {}

    class _Capturing:
        def post(self, path, json):  # noqa: A002
            seen["path"] = path
            seen["payload"] = json
            class R:
                status_code = 201
                is_success = True
                @staticmethod
                def json():
                    return {"id": "s-1", "resolved_run_id": None}
            return R()

    monkeypatch.setattr(signals_module, "_client", lambda *a, **k: _Capturing())
    monkeypatch.setenv("PAPAYYA_API_KEY", "cpk_deadbeef_" + "0" * 32)
    monkeypatch.setenv("PAPAYYA_BASE_URL", "http://127.0.0.1:9")

    papayya.signal("order-9", agent="triage").result(timeout=5)

    assert seen["path"] == "/v1/durable/signals"
    assert seen["payload"] == {
        "agent": "triage",
        "item_id": "order-9",
        "verdict": "bad",      # the default IS the common case: a thumbs-down
        "source": "thumbs",
    }

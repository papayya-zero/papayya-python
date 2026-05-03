"""Test wrapper around papayya.runtime.dispatcher.LocalDispatcher.

The dispatcher used in tests is the same one shipped to users (one
source of truth, one wire format). This module exists so the test
suite has a fixture-shaped name (``FakeDispatcher``) and an optional
seam for adding test-only helpers (chaos, lease-release, etc.) without
polluting the public class.
"""

from __future__ import annotations

from papayya.runtime.dispatcher import LocalDispatcher


class FakeDispatcher(LocalDispatcher):
    """Test-named alias. Identical behavior to LocalDispatcher.

    Future test-only helpers (force-release leases, simulate worker
    death, chaos hooks) will land here. Phase 1 acceptance test needs
    nothing beyond what LocalDispatcher already exposes.
    """

    def __init__(self, *, expected_api_key: str | None = None) -> None:
        super().__init__(host="127.0.0.1", port=0, expected_api_key=expected_api_key)

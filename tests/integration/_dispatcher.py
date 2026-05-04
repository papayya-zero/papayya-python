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

    def failed_completions(self) -> list[dict]:
        """Return the full ``_completed`` records for terminal failures.

        ``LocalDispatcher.failed()`` returns ``(lease_id, error_msg)``
        pairs only; tests need ``error_category`` and friends to assert
        on categorised paths (slice 2 introduces
        ``error_category="version_not_found"``). Snapshot under the lock
        so the caller doesn't see torn rows.
        """
        with self._lock:  # type: ignore[attr-defined]
            return [
                dict(record, lease_id=lease_id)
                for lease_id, record in self._completed.items()  # type: ignore[attr-defined]
                if record.get("status") != "completed"
            ]

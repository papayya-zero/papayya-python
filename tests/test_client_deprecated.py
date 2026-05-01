"""Unit 3: Client deprecation.

papayya.Client is the legacy HTTP run-trigger client. Papayya consolidates
trigger + monitor onto Papayya.runs (resource namespace), so Client emits
a DeprecationWarning on construction. Behavior is unchanged for one
release — removal is scheduled for the next minor.
"""

from __future__ import annotations

import warnings

import pytest


def test_client_emits_deprecation_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PAPAYYA_API_KEY", "cpk_test")
    monkeypatch.setenv("PAPAYYA_BASE_URL", "http://mock")

    from papayya.client import Client

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        Client()

    deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert deprecations, "expected a DeprecationWarning on Client(...)"
    msg = str(deprecations[0].message)
    assert "Client" in msg and "Papayya" in msg


def test_client_message_points_at_papayya_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The deprecation message names the replacement so users have a
    one-line migration recipe."""
    monkeypatch.setenv("PAPAYYA_API_KEY", "cpk_test")
    monkeypatch.setenv("PAPAYYA_BASE_URL", "http://mock")

    from papayya.client import Client

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        Client()

    deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    msg = str(deprecations[0].message)
    assert "runs.create" in msg

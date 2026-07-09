"""Plan 34: papayya.Client is folded into Papayya.

The legacy HTTP run-trigger Client (deprecated with a removal notice in
the previous release) is gone in 0.3.0; ``papayya.Client`` now resolves
to the ``Papayya`` class itself, so old imports keep constructing a
working client while the v1-trigger method surface is honestly removed.
"""

from __future__ import annotations

import pytest


def test_client_is_papayya(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PAPAYYA_API_KEY", "cpk_test")
    monkeypatch.setenv("PAPAYYA_BASE_URL", "http://mock")

    from papayya import Papayya
    from papayya.client import Client

    assert Client is Papayya
    client = Client()
    assert isinstance(client, Papayya)


def test_client_importable_from_package_root() -> None:
    import papayya
    from papayya import Papayya

    assert papayya.Client is Papayya


def test_run_result_still_importable() -> None:
    """RunResult stays importable for old type annotations."""
    from papayya.client import RunResult

    r = RunResult("out", run_id="r1", status="completed")
    assert r == "out"
    assert r.run_id == "r1"
    assert r.status == "completed"

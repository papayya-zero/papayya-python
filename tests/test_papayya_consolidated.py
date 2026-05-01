"""Unit 3: consolidated Papayya class.

Verifies the single ``Papayya`` class exposes both surfaces:
* Durable runtime via ``.run(agent=...)`` returning a ``PapayyaRun``.
* Platform resource namespaces (``.runs``, ``.batches``, etc.) lazy-resolving
  the API client only when accessed.

Plus the back-compat shims: ``papayya()`` factory and ``PapayyaClient``
alias both produce a ``Papayya`` instance.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from papayya import Papayya, papayya
from papayya.durable.client import PapayyaClient, PapayyaClientConfig
from papayya.durable.run import PapayyaRun
from papayya.durable.sqlite_store import SQLiteStore


@pytest.fixture
def isolated_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Run inside a clean cwd so no stray papayya.yaml interferes."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _local_store(tmp_path: Path) -> SQLiteStore:
    return SQLiteStore(str(tmp_path / "local.db"))


class TestUnifiedSurface:
    def test_run_returns_papayya_run(
        self, tmp_path: Path, isolated_cwd: Path
    ) -> None:
        client = Papayya(store=_local_store(tmp_path))
        run = client.run(agent="enrich")
        assert isinstance(run, PapayyaRun)
        assert run.agent == "enrich"

    def test_local_only_construction_without_api_key(
        self, tmp_path: Path, isolated_cwd: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A Papayya instance with no api_key still works for durable
        execution against a local store. Resource namespaces would fail
        on first access, but constructing the client doesn't require
        credentials."""
        monkeypatch.delenv("PAPAYYA_API_KEY", raising=False)
        client = Papayya(store=_local_store(tmp_path))
        # No raise here — the durable surface is fully usable.
        run = client.run(agent="enrich", run_id="r1")
        run.step("enrich", lambda: {"out": 1})()

    def test_resource_namespaces_lazy(
        self, tmp_path: Path, isolated_cwd: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Resource namespaces only construct the API client on first
        access. A run() call that never touches them must not require
        an api_key."""
        monkeypatch.delenv("PAPAYYA_API_KEY", raising=False)
        # Point ~ at a fresh dir so any saved CLI config is invisible.
        monkeypatch.setenv("HOME", str(tmp_path))
        client = Papayya(store=_local_store(tmp_path))
        # Construction succeeded with no credentials. Only when we access
        # a resource namespace does the lookup fail.
        from papayya.api import PapayyaAPIError

        with pytest.raises(PapayyaAPIError, match="No API key"):
            _ = client.runs

    def test_resource_namespace_cached_on_repeat_access(
        self, isolated_cwd: Path
    ) -> None:
        client = Papayya(api_key="cpk_test", base_url="http://mock")
        first = client.runs
        second = client.runs
        assert first is second


class TestFactoryAlias:
    def test_papayya_factory_returns_papayya_instance(
        self,
        tmp_path: Path,
        isolated_cwd: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("PAPAYYA_API_KEY", raising=False)
        monkeypatch.setenv("HOME", str(tmp_path))
        client = papayya()
        assert isinstance(client, Papayya)


class TestPapayyaClientShim:
    def test_papayya_client_subclass_of_papayya(
        self, isolated_cwd: Path
    ) -> None:
        # Tests that previously imported PapayyaClient + Config keep working.
        assert issubclass(PapayyaClient, Papayya)

    def test_papayya_client_with_config(
        self, tmp_path: Path, isolated_cwd: Path
    ) -> None:
        store = _local_store(tmp_path)
        client = PapayyaClient(PapayyaClientConfig(store=store))
        run = client.run(agent="enrich")
        assert isinstance(run, PapayyaRun)

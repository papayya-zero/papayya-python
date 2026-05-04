"""Tarball-hash recycle trigger — ADR-0003 § Worker #6, extends ADR-0002 #6.

When a deploy ships a new ``requirements.txt`` (or ``pyproject.toml``)
hash that differs from the resident version's, the worker can't safely
load the new bundle in-process — ``importlib.reload()`` is unreliable
for transitively-imported modules + new C extensions. The right answer
is a clean process recycle: fail the triggering lease with
``error_category="recycle_pending"``, set ``_running=False`` so the
main loop exits, and let the orchestrator bring up a fresh process.

These tests exercise:
  • ``_bundle_cache._compute_dep_hash`` returns the SHA256 of
    requirements.txt (preferred) or pyproject.toml (fallback), or
    None when neither is present.
  • ``_bundle_cache.ensure_bundle`` writes ``.papayya_dep_hash`` next
    to ``.papayya_entrypoint`` on extract.
  • A second cache miss whose dep hash differs from the resident's
    triggers ``_RecyclePending`` and flips ``_recycle_pending`` /
    ``_running``.
  • Same-hash second miss does NOT trigger recycle (deps unchanged).
  • Bundle without a manifest (no requirements.txt / pyproject.toml)
    skips the recycle check entirely.
"""

from __future__ import annotations

import hashlib
import io
import tarfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from papayya.runtime import _bundle_cache
from papayya.runtime._bundle_cache import (
    BundleEntry,
    FetchedBundle,
    _compute_dep_hash,
)
from papayya.runtime.worker import (
    Lease,
    Worker,
    _LoadedBundle,
    _RecyclePending,
)


# ── helpers ────────────────────────────────────────────────────────────


def _make_tarball(files: dict[str, bytes]) -> bytes:
    """Build a gzipped tarball whose top level contains ``files``."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, body in files.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(body)
            tar.addfile(info, io.BytesIO(body))
    return buf.getvalue()


def _fetched(tarball: bytes, *, entrypoint: str = "agent.py") -> FetchedBundle:
    return FetchedBundle(
        tarball_bytes=tarball,
        entrypoint=entrypoint,
        artifact_hash=hashlib.sha256(tarball).hexdigest(),
        account_id="acc",
        agent_id="agent",
        deployment_id="dep",
    )


# ── _compute_dep_hash unit ─────────────────────────────────────────────


def test_compute_dep_hash_prefers_requirements_txt(tmp_path: Path) -> None:
    """When both files exist, requirements.txt wins (the dominant
    Python manifest)."""
    (tmp_path / "requirements.txt").write_bytes(b"openai==1.0\n")
    (tmp_path / "pyproject.toml").write_bytes(b"[project]\nname = 'x'\n")

    h = _compute_dep_hash(tmp_path)
    assert h == hashlib.sha256(b"openai==1.0\n").hexdigest()


def test_compute_dep_hash_falls_back_to_pyproject(tmp_path: Path) -> None:
    """No requirements.txt → pyproject.toml is the source of truth."""
    (tmp_path / "pyproject.toml").write_bytes(b"[project]\nname = 'x'\n")

    h = _compute_dep_hash(tmp_path)
    assert h == hashlib.sha256(b"[project]\nname = 'x'\n").hexdigest()


def test_compute_dep_hash_returns_none_when_neither_present(tmp_path: Path) -> None:
    """No manifest → ``None`` → recycle is skipped on subsequent
    loads. Single-file bundles (just ``agent.py``) fall here."""
    assert _compute_dep_hash(tmp_path) is None


# ── ensure_bundle persists the sidecar ────────────────────────────────


def test_ensure_bundle_writes_dep_hash_sidecar(tmp_path: Path) -> None:
    """After extraction, the ``.papayya_dep_hash`` file lands in the
    final cache directory and ``BundleEntry.dep_hash`` is populated."""
    tarball = _make_tarball({
        "agent.py": b"# agent\n",
        "requirements.txt": b"openai==1.0\n",
    })

    entry = _bundle_cache.ensure_bundle(
        agent_slug="enrich",
        version=1,
        fetch=lambda: _fetched(tarball),
        root=tmp_path,
    )

    assert entry.dep_hash == hashlib.sha256(b"openai==1.0\n").hexdigest()
    sidecar = tmp_path / "enrich" / "v1" / ".papayya_dep_hash"
    assert sidecar.exists()
    assert sidecar.read_text(encoding="utf-8") == entry.dep_hash


def test_ensure_bundle_no_manifest_no_sidecar(tmp_path: Path) -> None:
    """Bundles without ``requirements.txt`` *and* without
    ``pyproject.toml`` skip the sidecar; ``dep_hash`` is None."""
    tarball = _make_tarball({"agent.py": b"# agent\n"})

    entry = _bundle_cache.ensure_bundle(
        agent_slug="enrich",
        version=1,
        fetch=lambda: _fetched(tarball),
        root=tmp_path,
    )

    assert entry.dep_hash is None
    assert not (tmp_path / "enrich" / "v1" / ".papayya_dep_hash").exists()


def test_entry_from_disk_reads_dep_hash_sidecar(tmp_path: Path) -> None:
    """Hot-path hits don't go through the fetch closure; the dep hash
    must be reconstructed from the on-disk sidecar."""
    tarball = _make_tarball({
        "agent.py": b"# agent\n",
        "requirements.txt": b"openai==1.0\n",
    })
    expected = hashlib.sha256(b"openai==1.0\n").hexdigest()

    # First call extracts + writes the sidecar.
    _bundle_cache.ensure_bundle(
        agent_slug="enrich",
        version=1,
        fetch=lambda: _fetched(tarball),
        root=tmp_path,
    )

    # Second call hits the hot-path — fetch closure must not be
    # invoked, and dep_hash must be sourced from the sidecar.
    def _no_fetch() -> FetchedBundle:
        raise AssertionError("fetch should not run on cache hit")

    entry = _bundle_cache.ensure_bundle(
        agent_slug="enrich",
        version=1,
        fetch=_no_fetch,
        root=tmp_path,
    )
    assert entry.dep_hash == expected


# ── Worker._ensure_loaded recycle decision ────────────────────────────


def _make_worker_for_recycle_test(monkeypatch: pytest.MonkeyPatch) -> Worker:
    """Construct a Worker without booting heartbeat / agent module.

    The recycle decision lives entirely inside ``_ensure_loaded`` and
    operates on ``self._loaded_versions`` + the cache's
    ``BundleEntry``. We can sidestep the full ``__init__`` by
    instantiating with ``__new__`` and setting just the fields the
    method reads.
    """
    w = Worker.__new__(Worker)
    w._loaded_versions = {}
    w._recycle_pending = False
    w._running = True
    w._bundle_url_base = "http://unused"
    w.dispatcher_url = "http://unused"
    w._api_key = None
    return w


def test_recycle_triggers_on_dep_hash_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """v1 resident with hash A; v2 fetch with hash B → ``_RecyclePending``
    raised, ``_recycle_pending`` set, ``_running`` cleared."""
    worker = _make_worker_for_recycle_test(monkeypatch)
    worker._loaded_versions[("enrich", "1")] = _LoadedBundle(
        agent_name="enrich",
        agent_version="1",
        bundle_path=str(tmp_path / "v1"),
        module_name="_papayya_user_agent__v1",
        dep_hash="aaaa" * 16,
    )

    # Stub ensure_bundle to return a v2 entry with a different dep_hash.
    def _stub_ensure_bundle(*, agent_slug: str, version: int, fetch):
        return BundleEntry(
            path=tmp_path / "v2",
            agent_slug=agent_slug,
            version=version,
            entrypoint="agent.py",
            dep_hash="bbbb" * 16,
        )

    monkeypatch.setattr(_bundle_cache, "ensure_bundle", _stub_ensure_bundle)

    lease = Lease(
        lease_id="L",
        agent="enrich",
        item_id="co_42",
        agent_version="2",
    )

    with pytest.raises(_RecyclePending) as excinfo:
        worker._ensure_loaded(lease)

    assert "recycling worker" in str(excinfo.value)
    assert worker._recycle_pending is True
    assert worker._running is False
    # v2 must NOT have been added to the registry — load was aborted.
    assert ("enrich", "2") not in worker._loaded_versions


def test_recycle_skipped_when_dep_hash_matches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """v1 resident with hash X; v2 fetch with same hash X → no
    recycle. The two versions co-resident, both in the registry."""
    worker = _make_worker_for_recycle_test(monkeypatch)
    same_hash = "cccc" * 16
    worker._loaded_versions[("enrich", "1")] = _LoadedBundle(
        agent_name="enrich",
        agent_version="1",
        bundle_path=str(tmp_path / "v1"),
        module_name="_papayya_user_agent__v1",
        dep_hash=same_hash,
    )

    def _stub_ensure_bundle(*, agent_slug: str, version: int, fetch):
        return BundleEntry(
            path=tmp_path / "v2",
            agent_slug=agent_slug,
            version=version,
            entrypoint="agent.py",
            dep_hash=same_hash,
        )

    # Stub the import path too; we don't want it touching the real
    # filesystem during this test.
    monkeypatch.setattr(_bundle_cache, "ensure_bundle", _stub_ensure_bundle)
    monkeypatch.setattr(
        Worker,
        "_import_bundle_module",
        lambda self, **_kw: "_papayya_user_agent__v2",
    )

    lease = Lease(
        lease_id="L",
        agent="enrich",
        item_id="co_42",
        agent_version="2",
    )

    worker._ensure_loaded(lease)

    assert worker._recycle_pending is False
    assert worker._running is True
    # v1 still resident, v2 added.
    assert ("enrich", "1") in worker._loaded_versions
    assert ("enrich", "2") in worker._loaded_versions


def test_recycle_skipped_when_no_manifest_in_either_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No manifest = unknown deps; conservative behaviour is to NOT
    recycle. Single-file bundles (just ``agent.py``) hit this branch."""
    worker = _make_worker_for_recycle_test(monkeypatch)
    worker._loaded_versions[("enrich", "1")] = _LoadedBundle(
        agent_name="enrich",
        agent_version="1",
        bundle_path=str(tmp_path / "v1"),
        module_name="_papayya_user_agent__v1",
        dep_hash=None,
    )

    def _stub_ensure_bundle(*, agent_slug: str, version: int, fetch):
        return BundleEntry(
            path=tmp_path / "v2",
            agent_slug=agent_slug,
            version=version,
            entrypoint="agent.py",
            dep_hash=None,
        )

    monkeypatch.setattr(_bundle_cache, "ensure_bundle", _stub_ensure_bundle)
    monkeypatch.setattr(
        Worker,
        "_import_bundle_module",
        lambda self, **_kw: "_papayya_user_agent__v2",
    )

    lease = Lease(
        lease_id="L",
        agent="enrich",
        item_id="co_42",
        agent_version="2",
    )
    worker._ensure_loaded(lease)

    assert worker._recycle_pending is False
    assert worker._running is True

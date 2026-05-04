"""On-disk cache for deployment bundles fetched by hosted workers.

ADR-0003 § 2 + § Worker #2 — when a lease arrives carrying an
``agent_version`` the worker hasn't seen, it fetches the tarball from
the control-pane bundle download endpoint, extracts it under
``<root>/<account_id>/<agent_id>/v<N>/``, and re-uses that extracted
directory on every subsequent lease for the same tuple. Slice 2 ships
with no eviction; the operator manages disk for now (ADR-0003 Q5).

Concurrency contract:
    Two worker processes on the same host *can* race to populate the
    same cache key (autoscale brings up replicas in parallel; both pick
    up the first lease of a new agent_version simultaneously). The
    extraction is therefore staged through a per-tuple ``.partial-<uuid>``
    directory and atomically renamed into place. A flock-style file
    lock serializes the rename so a half-extracted tree is never the
    one returned to ``_ensure_loaded``.

Cache layout:
    <root>/
      <agent_slug>/
        v<N>/                             ← stable, returned to caller
        v<N>.partial-<uuid>/              ← in-flight extraction
        v<N>.lock                         ← flock target

ADR-0003 § 2 specifies ``<root>/<account_id>/<agent_id>/v<N>/`` —
slice 2 deviates and uses the agent slug as the middle key because
the worker is project-scoped (one API key, one project) and only
learns ``account_id`` / ``agent_id`` after fetching from the bundle
endpoint. Re-keying the cache after the network round-trip would
mean every cache hit pays an HTTP HEAD to discover the path. Slice
D (multi-tenant ECS workers) will revisit when shared workers land.

The default ``root`` is ``~/.papayya/bundles/``. ECS sets
``PAPAYYA_BUNDLE_CACHE_ROOT`` to ``/var/cache/papayya/bundles/`` (Slice
D); slice 2 doesn't ship that wiring but reads the env var so the
deploy plumbing can land separately.
"""

from __future__ import annotations

import fcntl
import hashlib
import logging
import os
import shutil
import tarfile
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

log = logging.getLogger("papayya.runtime.bundle_cache")


# Module-level so tests can monkeypatch a tmp path without touching the
# filesystem of the developer running the suite.
_DEFAULT_ROOT = Path.home() / ".papayya" / "bundles"


def _resolve_root() -> Path:
    """Return the cache root, honouring ``PAPAYYA_BUNDLE_CACHE_ROOT``."""
    raw = os.environ.get("PAPAYYA_BUNDLE_CACHE_ROOT")
    if raw:
        return Path(raw).expanduser()
    return _DEFAULT_ROOT


@dataclass
class BundleEntry:
    """Path of an extracted bundle plus the metadata the worker needs.

    ``ensure_bundle`` returns this so callers can build a ``ModuleSpec``
    without re-querying the bundle endpoint or re-parsing the tarball.

    ``account_id`` and ``agent_id`` are populated from the response
    headers of the original fetch when available; on a hot-path hit
    (no fetch), they are read from the entrypoint sidecar and may be
    None until slice 3 starts persisting them.

    ``dep_hash`` is the SHA256 of the bundle's ``requirements.txt``
    (or ``pyproject.toml`` if the former is absent) computed at
    extract time and persisted in ``.papayya_dep_hash``. The worker
    uses it to decide whether a new version's pip dependency graph
    differs from the resident version's — if so, it self-recycles
    rather than trying to ``importlib.reload`` (ADR-0003 § Worker #6,
    extends ADR-0002 #6). ``None`` when neither dep file is present
    in the bundle, in which case recycle is skipped.
    """
    path: Path
    agent_slug: str
    version: int
    entrypoint: str
    account_id: str | None = None
    agent_id: str | None = None
    deployment_id: str | None = None
    artifact_hash: str | None = None
    dep_hash: str | None = None


class BundleVerificationError(RuntimeError):
    """Raised when a fetched tarball's SHA256 disagrees with its ETag.

    Worker maps this to a failed lease — in-flight tampering or
    truncation must not silently land in the cache and run customer
    code. The cache directory is left in its ``.partial-<uuid>`` state
    and gc'd on next process boot (slice 3 will cover the gc; slice 2
    just leaves the partial in place — the next attempt creates a fresh
    UUID-suffixed staging dir).
    """


# ── Fetch result type ─────────────────────────────────────────────────


@dataclass
class FetchedBundle:
    """Raw result from the bundle-download endpoint.

    The fetcher (worker side) constructs this from the HTTP response;
    ``_bundle_cache`` then writes it to disk and returns a
    ``BundleEntry``. Keeping the fetch IO in the caller and the
    filesystem IO here makes both halves trivially testable in isolation.
    """
    tarball_bytes: bytes
    entrypoint: str
    artifact_hash: str | None  # ETag value (un-quoted) if the server set one
    account_id: str | None = None
    agent_id: str | None = None
    deployment_id: str | None = None


# ── Public API ────────────────────────────────────────────────────────


def ensure_bundle(
    *,
    agent_slug: str,
    version: int,
    fetch: Callable[[], FetchedBundle],
    root: Path | None = None,
) -> BundleEntry:
    """Return an extracted bundle directory, fetching + extracting on miss.

    On hit: short-circuit. On miss: call ``fetch()`` to obtain bytes,
    verify SHA256 against the server-supplied ETag (if any), extract
    under ``<root>/<agent_slug>/v<N>.partial-<uuid>/``, then atomically
    rename to ``v<N>/``.

    Concurrency: a per-tuple file lock (``v<N>.lock``) serializes the
    extract+rename so two workers on the same host don't both produce
    half-trees. The lock is dropped immediately after the rename
    succeeds; reads of an existing ``v<N>/`` are lock-free (the rename
    is atomic on POSIX, so seeing the directory is sufficient).

    Args:
        agent_slug: Agent slug from the lease (``Lease.agent``). One
            slug → one cache subtree per worker; the worker is
            project-scoped so slug uniqueness holds.
        version: Integer deployment version.
        fetch: Zero-arg callable returning the freshly-fetched
            ``FetchedBundle``. Only invoked on cache miss; receiver
            doesn't pay the network cost on hot path.
        root: Override cache root. Defaults to ``_resolve_root()``.

    Returns:
        Resolved ``BundleEntry`` pointing at the on-disk extraction.

    Raises:
        BundleVerificationError: SHA256 mismatch.
        OSError: filesystem permission / disk-full failures bubble up.
        Anything ``fetch()`` raises bubbles unchanged.
    """
    if version < 1:
        raise ValueError(f"version must be a positive integer, got {version}")

    base = (root or _resolve_root()) / agent_slug
    final_path = base / f"v{version}"
    lock_path = base / f"v{version}.lock"

    # Hot path: cache hit. The extracted directory is the canonical
    # signal — its existence (post-atomic-rename) means *some* prior
    # writer finished. We do not re-verify the hash on every hit; the
    # rename is the commit point.
    if final_path.is_dir():
        return _entry_from_disk(final_path, agent_slug, version)

    base.mkdir(parents=True, exist_ok=True)

    # Take the per-tuple lock before re-checking + extracting. Two
    # workers racing past the hot-path check both arrive here; the
    # second one will see the directory after the first releases the
    # lock and bail to the inner hot-path check.
    with _file_lock(lock_path):
        if final_path.is_dir():
            return _entry_from_disk(final_path, agent_slug, version)

        fetched = fetch()
        if fetched.artifact_hash:
            actual = hashlib.sha256(fetched.tarball_bytes).hexdigest()
            if actual != fetched.artifact_hash:
                raise BundleVerificationError(
                    f"sha256 mismatch for {agent_slug}/v{version}: "
                    f"expected {fetched.artifact_hash}, got {actual}"
                )

        partial = base / f"v{version}.partial-{uuid.uuid4().hex}"
        partial.mkdir(parents=True, exist_ok=False)
        try:
            _extract_tarball(fetched.tarball_bytes, partial)
            # Persist the entrypoint into the staging dir *before* the
            # rename so it rides into the final tree atomically. Without
            # this a hot-path hit (no fetch + no response headers) would
            # have no way to recover the entrypoint name.
            if fetched.entrypoint:
                write_entrypoint_sidecar(partial, fetched.entrypoint)
            # ADR-0003 § Worker #6 — compute the bundle's dep-list hash
            # and persist a sidecar so the worker can detect dep-graph
            # changes across deploys without re-extracting. ``None``
            # (neither requirements.txt nor pyproject.toml present)
            # means recycle is skipped on subsequent loads.
            dep_hash = _compute_dep_hash(partial)
            if dep_hash is not None:
                _write_dep_hash_sidecar(partial, dep_hash)
            # Atomic rename. If a concurrent writer somehow won this
            # race despite the lock (e.g., NFS-style lock failure), the
            # rename will fail with EEXIST or ENOTEMPTY; drop our partial
            # and fall through to the cache-hit path.
            try:
                os.rename(partial, final_path)
            except OSError:
                if final_path.is_dir():
                    shutil.rmtree(partial, ignore_errors=True)
                else:
                    raise
        except BaseException:
            shutil.rmtree(partial, ignore_errors=True)
            raise

    return _entry_from_disk(
        final_path,
        agent_slug,
        version,
        entrypoint_override=fetched.entrypoint,
        account_id=fetched.account_id,
        agent_id=fetched.agent_id,
        deployment_id=fetched.deployment_id,
        artifact_hash=fetched.artifact_hash,
        dep_hash_override=dep_hash,
    )


# ── Internals ─────────────────────────────────────────────────────────


def _entry_from_disk(
    final_path: Path,
    agent_slug: str,
    version: int,
    *,
    entrypoint_override: str | None = None,
    account_id: str | None = None,
    agent_id: str | None = None,
    deployment_id: str | None = None,
    artifact_hash: str | None = None,
    dep_hash_override: str | None = None,
) -> BundleEntry:
    """Build a BundleEntry pointing at an already-extracted bundle.

    Slice 2 records entrypoint as a sidecar file
    (``<v<N>>/.papayya_entrypoint``) on extract, so a hot-path hit can
    reconstruct the entry without the response-header context that
    triggered the original fetch. Falls back to ``entrypoint_override``
    when called from the freshly-fetched path.

    Slice 3 adds the same pattern for the dep-hash sidecar
    (``.papayya_dep_hash``), so a hot-path hit can compare the
    resident bundle's dep graph against a newly-fetched one without
    re-hashing.
    """
    entrypoint = entrypoint_override
    if entrypoint is None:
        sidecar = final_path / ".papayya_entrypoint"
        if sidecar.exists():
            entrypoint = sidecar.read_text(encoding="utf-8").strip()
    dep_hash = dep_hash_override
    if dep_hash is None:
        dep_sidecar = final_path / ".papayya_dep_hash"
        if dep_sidecar.exists():
            dep_hash = dep_sidecar.read_text(encoding="utf-8").strip() or None
    return BundleEntry(
        path=final_path,
        agent_slug=agent_slug,
        version=version,
        entrypoint=entrypoint or "",
        account_id=account_id,
        agent_id=agent_id,
        deployment_id=deployment_id,
        artifact_hash=artifact_hash,
        dep_hash=dep_hash,
    )


def _extract_tarball(data: bytes, dest: Path) -> None:
    """Eager-extract a gzipped tarball into ``dest`` (must exist).

    ``dest`` is the ``.partial-<uuid>`` staging dir; the caller renames
    it into place after this returns. We extract via a tempfile rather
    than ``BytesIO`` so the tarfile module can do its random-access
    reads cheaply on large bundles without holding the entire file in
    memory twice.

    Path-traversal defence: ``tarfile``'s default ``data_filter``
    (Python 3.12+) rejects entries that escape the destination via
    ``..`` or absolute paths. Older versions get an explicit guard.
    """
    with tempfile.NamedTemporaryFile(suffix=".tar.gz") as tmp:
        tmp.write(data)
        tmp.flush()
        tmp.seek(0)
        with tarfile.open(tmp.name, mode="r:gz") as tar:
            # Python 3.12+: ``filter="data"`` enforces extraction-safety
            # rules (no absolute paths, no ``..``, no symlinks pointing
            # outside the destination).
            try:
                tar.extractall(path=dest, filter="data")  # type: ignore[arg-type]
            except TypeError:
                _safe_extractall(tar, dest)


def _safe_extractall(tar: tarfile.TarFile, dest: Path) -> None:
    """Path-traversal-safe extractall for Python < 3.12.

    Rejects absolute paths and ``..`` components before extraction. The
    branch is rarely taken in production (we ship 3.12+) but keeps the
    fallback honest for local dev on older interpreters.
    """
    dest = dest.resolve()
    for member in tar.getmembers():
        member_path = (dest / member.name).resolve()
        if not str(member_path).startswith(str(dest)):
            raise BundleVerificationError(
                f"refusing to extract path-escaping tar entry: {member.name!r}"
            )
    tar.extractall(path=dest)


class _file_lock:
    """Per-tuple flock context manager.

    The lock file is a sentinel (zero-byte). We hold an exclusive
    ``fcntl.flock`` on it for the duration of the extract+rename, so
    concurrent ``ensure_bundle`` calls for the same tuple serialize.
    Used as a context manager so the FD always closes — leaking it
    would silently keep the lock past the rename on long-lived
    processes.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._fd: int | None = None

    def __enter__(self) -> "_file_lock":
        # Create the lock file with O_RDWR|O_CREAT — exclusive flock
        # below provides serialization, not the open() mode.
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._fd = os.open(self._path, os.O_RDWR | os.O_CREAT, 0o644)
        fcntl.flock(self._fd, fcntl.LOCK_EX)
        return self

    def __exit__(self, *_: object) -> None:
        if self._fd is not None:
            try:
                fcntl.flock(self._fd, fcntl.LOCK_UN)
            finally:
                os.close(self._fd)
                self._fd = None


def _compute_dep_hash(bundle_path: Path) -> str | None:
    """SHA256 of the bundle's pip-dep manifest, or None if absent.

    Prefers ``requirements.txt`` (still the dominant manifest in the
    Python world); falls back to ``pyproject.toml`` so projects that
    only declare deps via PEP 621 / Poetry still get recycle behaviour.
    Both files are tiny so a single read+hash is fine.

    The hash is over the *raw bytes* of the file. Re-ordering lines or
    whitespace edits to ``requirements.txt`` will trigger a recycle
    even if the resolved package set is unchanged — accepted false
    positive. The alternative (parsing + canonicalising) bakes in
    pip-resolver semantics we don't want to track.
    """
    for candidate in ("requirements.txt", "pyproject.toml"):
        path = bundle_path / candidate
        if path.is_file():
            return hashlib.sha256(path.read_bytes()).hexdigest()
    return None


def _write_dep_hash_sidecar(bundle_path: Path, dep_hash: str) -> None:
    """Persist the dep-list hash beside the entrypoint sidecar.

    Called from ``ensure_bundle`` right before the atomic rename so
    the file rides into the final tree atomically with the rest of
    the bundle. Worker hot-path hits read this back via
    ``_entry_from_disk`` to populate ``BundleEntry.dep_hash``.
    """
    (bundle_path / ".papayya_dep_hash").write_text(dep_hash, encoding="utf-8")


def write_entrypoint_sidecar(bundle_path: Path, entrypoint: str) -> None:
    """Persist the entrypoint into the extracted bundle dir.

    Called from ``ensure_bundle`` right before the atomic rename so the
    sidecar lands in the staging dir and rides into the final tree
    atomically. Worker hot-path hits read this back via
    ``_entry_from_disk`` — the response headers from the original
    fetch aren't available in steady-state.

    Public so the test suite can construct cache entries directly
    without going through ``ensure_bundle``.
    """
    (bundle_path / ".papayya_entrypoint").write_text(entrypoint, encoding="utf-8")

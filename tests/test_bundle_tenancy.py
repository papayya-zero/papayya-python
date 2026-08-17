"""Account isolation for resident bundles — plan 47 S1.

A shared platform worker leases across accounts. Before this, every
residency key was ``(agent_slug, version)``:

  - the on-disk cache was ``<root>/<slug>/v<N>/``
  - ``Worker._loaded_versions`` was keyed ``(slug, version)``
  - the ``sys.modules`` name was ``_papayya_user_<stem>__v<N>``
  - the bundle loader registered its root under the bare version

So two accounts deploying a slug both customers happened to choose —
``triage``, ``enrich`` — at the same version aliased onto one entry, and
whichever landed first served both. That was observed end to end: one
account's 200 items executed the other account's code.

The cache-hit path returns *before* ``fetch()``, so the collision also
bypassed the bundle endpoint's cross-account check, which lives inside
the fetch. These tests pin each layer separately, because any one of
them reverting re-opens the hole on its own.
"""

from __future__ import annotations

import io
import tarfile
from pathlib import Path

import pytest

from papayya.runtime import _bundle_cache
from papayya.runtime._bundle_cache import LOCAL_SCOPE, FetchedBundle
from papayya.runtime.worker import _residency_token

ACCT_A = "11111111-1111-4111-8111-111111111111"
ACCT_B = "22222222-2222-4222-8222-222222222222"

SHARED_SLUG = "triage"


def _make_tarball(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, data in files.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def _fetched(tarball: bytes, entrypoint: str = "agent.py") -> FetchedBundle:
    return FetchedBundle(
        tarball_bytes=tarball,
        entrypoint=entrypoint,
        artifact_hash=None,
    )


# ── on-disk cache ─────────────────────────────────────────────────────


def test_same_slug_and_version_across_accounts_do_not_share_a_cache_entry(
    tmp_path: Path,
) -> None:
    """The headline case. Two accounts, one slug, one version — each
    must extract and read back *its own* source."""
    tar_a = _make_tarball({"agent.py": b"MARKER = 'account-a'\n"})
    tar_b = _make_tarball({"agent.py": b"MARKER = 'account-b'\n"})

    entry_a = _bundle_cache.ensure_bundle(
        account_id=ACCT_A,
        agent_slug=SHARED_SLUG,
        version=1,
        fetch=lambda: _fetched(tar_a),
        root=tmp_path,
    )
    entry_b = _bundle_cache.ensure_bundle(
        account_id=ACCT_B,
        agent_slug=SHARED_SLUG,
        version=1,
        fetch=lambda: _fetched(tar_b),
        root=tmp_path,
    )

    assert entry_a.path != entry_b.path
    assert (entry_a.path / "agent.py").read_bytes() == b"MARKER = 'account-a'\n"
    assert (entry_b.path / "agent.py").read_bytes() == b"MARKER = 'account-b'\n"


def test_second_account_is_a_cache_miss_and_must_fetch(tmp_path: Path) -> None:
    """The fetch is where the control plane's cross-account check runs.
    A cache hit short-circuits it, so a second account arriving at a
    populated slug+version MUST still reach the network."""
    tar_a = _make_tarball({"agent.py": b"# a\n"})
    tar_b = _make_tarball({"agent.py": b"# b\n"})

    _bundle_cache.ensure_bundle(
        account_id=ACCT_A,
        agent_slug=SHARED_SLUG,
        version=1,
        fetch=lambda: _fetched(tar_a),
        root=tmp_path,
    )

    fetched_for_b = False

    def _fetch_b() -> FetchedBundle:
        nonlocal fetched_for_b
        fetched_for_b = True
        return _fetched(tar_b)

    _bundle_cache.ensure_bundle(
        account_id=ACCT_B,
        agent_slug=SHARED_SLUG,
        version=1,
        fetch=_fetch_b,
        root=tmp_path,
    )

    assert fetched_for_b, "account B served from account A's cache entry"


def test_same_account_still_hits_the_cache(tmp_path: Path) -> None:
    """Partitioning must not cost the hot path — a repeat lease for the
    same account still short-circuits without fetching."""
    tarball = _make_tarball({"agent.py": b"# a\n"})

    _bundle_cache.ensure_bundle(
        account_id=ACCT_A,
        agent_slug=SHARED_SLUG,
        version=1,
        fetch=lambda: _fetched(tarball),
        root=tmp_path,
    )

    def _no_fetch() -> FetchedBundle:
        raise AssertionError("cache hit must not re-fetch")

    entry = _bundle_cache.ensure_bundle(
        account_id=ACCT_A,
        agent_slug=SHARED_SLUG,
        version=1,
        fetch=_no_fetch,
        root=tmp_path,
    )
    assert entry.path.is_dir()


def test_local_scope_is_partitioned_from_real_accounts(tmp_path: Path) -> None:
    """Local dev has no account id. Its partition must not alias one."""
    tar_local = _make_tarball({"agent.py": b"# local\n"})
    tar_hosted = _make_tarball({"agent.py": b"# hosted\n"})

    local = _bundle_cache.ensure_bundle(
        account_id=LOCAL_SCOPE,
        agent_slug=SHARED_SLUG,
        version=1,
        fetch=lambda: _fetched(tar_local),
        root=tmp_path,
    )
    hosted = _bundle_cache.ensure_bundle(
        account_id=ACCT_A,
        agent_slug=SHARED_SLUG,
        version=1,
        fetch=lambda: _fetched(tar_hosted),
        root=tmp_path,
    )

    assert local.path != hosted.path
    assert (local.path / "agent.py").read_bytes() == b"# local\n"


# ── key-component hardening ───────────────────────────────────────────


@pytest.mark.parametrize(
    "account_id, agent_slug",
    [
        (ACCT_A, "../" + ACCT_B + "/triage"),
        ("..", "triage"),
        (ACCT_A, ".."),
        (ACCT_A, "nested/slug"),
        ("", "triage"),
    ],
)
def test_path_escaping_key_components_are_rejected(
    tmp_path: Path, account_id: str, agent_slug: str
) -> None:
    """A slug that walks out of its account defeats the partition
    entirely, so it is refused rather than normalised."""
    with pytest.raises(ValueError):
        _bundle_cache.ensure_bundle(
            account_id=account_id,
            agent_slug=agent_slug,
            version=1,
            fetch=lambda: _fetched(_make_tarball({"agent.py": b"# x\n"})),
            root=tmp_path,
        )


# ── in-process residency token ────────────────────────────────────────


def test_residency_token_differs_across_accounts_at_same_version() -> None:
    """The token names the ``sys.modules`` entry and the loader root.
    Equal tokens mean one account's module object serves the other."""
    assert _residency_token(ACCT_A, "1") != _residency_token(ACCT_B, "1")


def test_residency_token_differs_across_versions_within_an_account() -> None:
    """Multi-version residency (ADR-0003 § Worker #4) must survive the
    account partition."""
    assert _residency_token(ACCT_A, "1") != _residency_token(ACCT_A, "2")


def test_residency_token_is_identifier_safe() -> None:
    """It lands in a module name; UUID hyphens must not leak through."""
    token = _residency_token(ACCT_A, "1")
    assert "-" not in token
    assert token.replace("_", "").isalnum()

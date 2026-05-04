"""Worker fetches + extracts + executes a versioned bundle (ADR-0003 slice 2).

End-to-end coverage of the slice-2 cutover: a lease arrives carrying
``agent_version="1"``, the worker hits the bundle download endpoint
served by ``_bundle_server.FakeBundleServer``, extracts the tarball
into ``PAPAYYA_BUNDLE_CACHE_ROOT`` (pointed at ``tmp_path`` so the
test never touches ``~/.papayya/``), imports the entrypoint module
which calls ``@agent``, and dispatches the lease to the freshly-loaded
fn.

The stub ``--agent-module`` file registers nothing — it only exists
because the worker CLI requires the flag (Slice 2 punts on bootstrap
mode per ADR-0003 § Worker #5). The bundle's ``@agent`` is what
populates the registry under ``"enrich"``.

Two paths:
  • happy: bundle served → completion lands → lineage row written
  • 404: bundle endpoint returns 404 → completion is failed +
    ``error_category="version_not_found"``
"""

from __future__ import annotations

import time as _t

from papayya.bundler import bundle_project

from ._bundle_server import FakeBundleServer
from ._dispatcher import FakeDispatcher
from .conftest import write_test_agent


# Stub --agent-module: registers nothing. The bundle's @agent is what
# populates the registry under "enrich". A registered fn here under a
# *different* name is fine; the lease's "enrich" only resolves once the
# bundle is loaded.
_STUB_AGENT = '''\
"""Stub agent module — slice 2 worker boots with this; the bundle's
@agent registers "enrich" on first lease."""

# No imports of papayya — keeps this file deliberately empty so the
# only @agent registration in the worker process is the one from the
# fetched bundle.
'''


# Customer-shaped bundle source — the file the worker imports after
# fetching+extracting the tarball.
_BUNDLE_AGENT_SOURCE = '''\
"""Bundle agent.py: the real customer code."""

from papayya import agent
from papayya.durable import papayya


# Explicit ``agent_version="1"`` makes registration deterministic under
# slice 3's tuple-keyed registry. The worker also injects
# ``PAPAYYA_AGENT_VERSION`` from the lease, so the env-driven path would
# resolve to the same value — explicit decorator arg is just clearer
# for readers of the test.
@agent(name="enrich", agent_version="1")
def enrich(item_id: str) -> dict:
    run = papayya().run("enrich", item_id=item_id)
    step = run.step("extract", lambda: {"id": item_id, "loaded_from": "bundle_v1"})
    result = step()
    run.complete(result)
    return result
'''


def _build_bundle(tmp_path) -> tuple[bytes, str]:
    """Materialize a real bundle tarball (CLI-shaped) for the test.

    Uses ``papayya.bundler.bundle_project`` — same code path
    ``papayya deploy`` uses on real customer bundles — so the on-disk
    layout the worker extracts matches what the control-pane stores in
    S3. Returns ``(tarball_bytes, sha256_hex)``.
    """
    project_dir = tmp_path / "bundle_src"
    project_dir.mkdir()
    (project_dir / "agent.py").write_text(_BUNDLE_AGENT_SOURCE)
    return bundle_project(str(project_dir), entrypoint="agent.py")


def test_worker_fetches_bundle_and_completes(
    tmp_path,
    in_memory_store,
    worker_subprocess,
):
    """Full happy path: fetched bundle's @agent fn handles the lease."""
    tarball, sha = _build_bundle(tmp_path)

    bundle_server = FakeBundleServer()
    try:
        bundle_server.register(
            agent="enrich",
            version=1,
            tarball=tarball,
            entrypoint="agent.py",
            artifact_hash=sha,
        )

        dispatcher = FakeDispatcher()
        try:
            dispatcher.enqueue(agent="enrich", item_id="co_42", agent_version="1")

            stub_path = write_test_agent(tmp_path, _STUB_AGENT)
            cache_root = tmp_path / "bundle_cache"
            worker = worker_subprocess(
                agent_module=stub_path,
                dispatcher=dispatcher,
                store=in_memory_store,
                bundle_url_base=bundle_server.url,
                env_overrides={"PAPAYYA_BUNDLE_CACHE_ROOT": str(cache_root)},
            )

            dispatcher.wait_until_drained(timeout=15.0)

            deadline = _t.monotonic() + 2.0
            run = in_memory_store.run_for_item("co_42")
            while run is None and _t.monotonic() < deadline:
                _t.sleep(0.02)
                run = in_memory_store.run_for_item("co_42")
            assert run is not None, (
                "run never landed in the store; worker log:\n"
                + worker.stderr_tail(8192)
            )
            assert [t.label for t in run.tasks] == ["extract"]

            # Bundle was actually requested (not a stub satisfying it
            # from sys.path).
            assert ("enrich", 1) in bundle_server.calls

            # Cache layout: <root>/<agent_slug>/v<N>/agent.py present.
            assert (cache_root / "enrich" / "v1" / "agent.py").exists()
            assert (cache_root / "enrich" / "v1" / ".papayya_entrypoint").read_text() == "agent.py"

            worker.stop(timeout=5.0)
            assert worker.exit_code == 0
        finally:
            dispatcher.shutdown()
    finally:
        bundle_server.shutdown()


def test_worker_reports_version_not_found_on_404(
    tmp_path,
    in_memory_store,
    worker_subprocess,
):
    """Bundle endpoint 404 → failed completion + error_category set.

    ADR-0003 Q4: deployment row exists but the artifact upload hasn't
    landed (or the version legitimately doesn't exist). Worker must
    fail the lease with ``error_category="version_not_found"`` so
    operators can distinguish it from a customer-code crash.
    """
    bundle_server = FakeBundleServer()
    try:
        # Don't register any bundle — every fetch will 404.

        dispatcher = FakeDispatcher()
        try:
            dispatcher.enqueue(agent="enrich", item_id="co_404", agent_version="9")

            stub_path = write_test_agent(tmp_path, _STUB_AGENT)
            cache_root = tmp_path / "bundle_cache"
            worker = worker_subprocess(
                agent_module=stub_path,
                dispatcher=dispatcher,
                store=in_memory_store,
                bundle_url_base=bundle_server.url,
                env_overrides={"PAPAYYA_BUNDLE_CACHE_ROOT": str(cache_root)},
            )

            # Wait for the lease to be completed (failed). FakeDispatcher's
            # wait_until_drained considers failures terminal too.
            dispatcher.wait_until_drained(timeout=10.0)

            stats = dispatcher.stats()
            assert stats["pending"] == 0
            assert stats["leased"] == 0
            # The lease completed terminally; verify it was a failure
            # categorised as version_not_found.
            failures = dispatcher.failed_completions()
            assert len(failures) == 1, (
                f"expected 1 failed completion, got {failures}; worker log:\n"
                + worker.stderr_tail(8192)
            )
            assert failures[0]["error_category"] == "version_not_found", failures[0]

            # No customer code ran → no lineage row.
            assert in_memory_store.run_for_item("co_404") is None

            # Cache directory was never finalised — only the lock file
            # exists (or nothing at all). v9/ must not be present.
            v9 = cache_root / "enrich" / "v9"
            assert not v9.is_dir(), f"v9 should not have been finalised, but {v9} exists"

            worker.stop(timeout=5.0)
        finally:
            dispatcher.shutdown()
    finally:
        bundle_server.shutdown()

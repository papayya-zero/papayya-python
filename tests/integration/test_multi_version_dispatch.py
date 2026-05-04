"""Worker holds two versions of the same agent slug resident at once.

ADR-0003 § Worker #4 (slice 3): a single hosted worker fetches v1 and
v2 of the same agent slug, registers each under
``(name, agent_version)`` in the agent registry, and dispatches each
lease to its matching registration. This is the end-to-end signal —
exec'd inside a real worker subprocess against the FakeBundleServer
+ FakeDispatcher fixtures.

Sibling sanity check: bundles sharing a sibling file (``helpers.py``)
must resolve to *their own* version's copy at function-body import
time. The MetaPathFinder + ``activate(version)`` scope from
``_bundle_loader.py`` is what makes that work; this test exercises it
end-to-end.
"""

from __future__ import annotations

import time as _t

from papayya.bundler import bundle_project

from ._bundle_server import FakeBundleServer
from ._dispatcher import FakeDispatcher
from .conftest import write_test_agent


# Stub --agent-module: registers nothing. Each fetched bundle's
# ``@agent`` is what populates the registry under
# ``("enrich", "<version>")``.
_STUB_AGENT = '''\
"""Stub — slice 3 worker boots with this; the bundle's @agent
registers per-version on each fetch."""
'''


def _build_versioned_bundle(tmp_path, version: str, marker: str) -> tuple[bytes, str]:
    """Materialize a bundle whose ``agent.py`` imports a sibling
    ``helpers.py`` that carries the version's marker.

    The customer-shaped layout exercises ADR-0003 § Worker #4 + #B
    together: the @agent function does ``from helpers import VERSION``
    at module scope (during exec_module) AND uses it at call time
    (during _handle_lease) — both must resolve to *this* version's
    helpers.
    """
    project = tmp_path / f"bundle_v{version}"
    project.mkdir()
    (project / "helpers.py").write_text(f'VERSION = "{marker}"\n')
    (project / "agent.py").write_text(
        '"""Bundle agent.py: imports a sibling helper module."""\n'
        "\n"
        "from papayya import agent\n"
        "from papayya.durable import papayya\n"
        "\n"
        "from helpers import VERSION\n"
        "\n"
        "\n"
        f'@agent(name="enrich", agent_version="{version}")\n'
        "def enrich(item_id: str) -> dict:\n"
        "    run = papayya().run('enrich', item_id=item_id)\n"
        f"    step = run.step('extract_v{version}', lambda: {{'id': item_id, 'version_marker': VERSION}})\n"
        "    result = step()\n"
        "    run.complete(result)\n"
        "    return result\n"
    )
    return bundle_project(str(project), entrypoint="agent.py")


def test_worker_holds_two_versions_resident(
    tmp_path,
    in_memory_store,
    worker_subprocess,
):
    """v1 + v2 of the same slug both run on a single worker.

    The single-threaded poll loop processes one lease at a time, so
    the test enqueues both before booting the worker; the registry
    must hold v1 *and* v2 simultaneously after the second bundle
    loads (slice 2 would have overwritten v1).
    """
    v1_tarball, v1_sha = _build_versioned_bundle(tmp_path, "1", "marker_v1")
    v2_tarball, v2_sha = _build_versioned_bundle(tmp_path, "2", "marker_v2")

    bundle_server = FakeBundleServer()
    try:
        bundle_server.register(
            agent="enrich",
            version=1,
            tarball=v1_tarball,
            entrypoint="agent.py",
            artifact_hash=v1_sha,
        )
        bundle_server.register(
            agent="enrich",
            version=2,
            tarball=v2_tarball,
            entrypoint="agent.py",
            artifact_hash=v2_sha,
        )

        dispatcher = FakeDispatcher()
        try:
            dispatcher.enqueue(agent="enrich", item_id="co_v1", agent_version="1")
            dispatcher.enqueue(agent="enrich", item_id="co_v2", agent_version="2")

            stub_path = write_test_agent(tmp_path, _STUB_AGENT)
            cache_root = tmp_path / "bundle_cache"
            worker = worker_subprocess(
                agent_module=stub_path,
                dispatcher=dispatcher,
                store=in_memory_store,
                bundle_url_base=bundle_server.url,
                env_overrides={"PAPAYYA_BUNDLE_CACHE_ROOT": str(cache_root)},
            )

            dispatcher.wait_until_drained(timeout=20.0)

            # Stop the worker before querying so the worker's
            # SQLiteStore commits its final WAL frames and releases
            # the write lock. Without this, ``wal_checkpoint`` can't
            # merge the v2 row into the main DB file because the
            # worker's connection still holds it. macOS APFS
            # exacerbates the visibility lag — Linux ext4 generally
            # doesn't need the explicit stop, but the test must run
            # on both.
            worker.stop(timeout=5.0)
            assert worker.exit_code == 0

            deadline = _t.monotonic() + 3.0
            run_v1 = in_memory_store.run_for_item("co_v1")
            run_v2 = in_memory_store.run_for_item("co_v2")
            while (run_v1 is None or run_v2 is None) and _t.monotonic() < deadline:
                _t.sleep(0.05)
                run_v1 = in_memory_store.run_for_item("co_v1")
                run_v2 = in_memory_store.run_for_item("co_v2")

            assert run_v1 is not None, (
                "v1 run never landed; worker log:\n" + worker.stderr_tail(8192)
            )
            assert run_v2 is not None, (
                "v2 run never landed; worker log:\n" + worker.stderr_tail(8192)
            )

            # Step labels carry the version — confirms each lease was
            # dispatched to the matching registration.
            assert [t.label for t in run_v1.tasks] == ["extract_v1"]
            assert [t.label for t in run_v2.tasks] == ["extract_v2"]

            # Both bundles were actually fetched (not satisfied from a
            # shared sys.path or stale sys.modules entry).
            assert ("enrich", 1) in bundle_server.calls
            assert ("enrich", 2) in bundle_server.calls

            # Both extracted on disk under the slug-keyed cache layout.
            assert (cache_root / "enrich" / "v1" / "agent.py").exists()
            assert (cache_root / "enrich" / "v2" / "agent.py").exists()
            # And both expose the helpers sibling that the entrypoint
            # imported — proves the MetaPathFinder did not collapse them.
            assert (cache_root / "enrich" / "v1" / "helpers.py").exists()
            assert (cache_root / "enrich" / "v2" / "helpers.py").exists()
        finally:
            dispatcher.shutdown()
    finally:
        bundle_server.shutdown()

"""Bootstrap-mode worker boots without --agent-module (ADR-0003 § Worker #5).

Hosted ECS workers can't take an --agent-module flag because they don't
know which customer code to load until the first lease arrives. The
worker CLI gained a --bootstrap flag (and PAPAYYA_BOOTSTRAP=1 env var)
that's mutually exclusive with --agent-module: in bootstrap mode the
worker skips the eager module import at construction time and lets the
slice-2 _ensure_loaded path handle the first import on the first lease.

Two paths covered here:
  • happy: bootstrap worker drains a lease carrying agent_version="1"
    by fetching the bundle from FakeBundleServer; the bundle's @agent
    handles the lease. Companion to test_ensure_loaded_bundle.py's
    happy path, but with no stub --agent-module file in play.
  • misuse: bootstrap worker receives a lease with agent_version=None
    (LocalDispatcher pointed at a hosted worker by mistake). The
    worker has nothing to dispatch, so it must fail the lease with a
    distinct error_category="no_agent_module" instead of the generic
    "unknown agent" path — operators need to tell apart "I forgot to
    pass --agent-module" from "the registry doesn't have this slug".
"""

from __future__ import annotations

import time as _t

from papayya.bundler import bundle_project

from ._bundle_server import FakeBundleServer
from ._dispatcher import FakeDispatcher


_BUNDLE_AGENT_SOURCE = '''\
"""Bundle agent.py: the customer code a bootstrap worker fetches."""

from papayya import agent
from papayya.durable import papayya


@agent(name="enrich", agent_version="1")
def enrich(item_id: str) -> dict:
    run = papayya().run("enrich", item_id=item_id)
    step = run.step("extract", lambda: {"id": item_id, "loaded_from": "bootstrap"})
    result = step()
    run.complete(result)
    return result
'''


def _build_bundle(tmp_path) -> tuple[bytes, str]:
    project_dir = tmp_path / "bundle_src"
    project_dir.mkdir()
    (project_dir / "agent.py").write_text(_BUNDLE_AGENT_SOURCE)
    return bundle_project(str(project_dir), entrypoint="agent.py")


def test_bootstrap_worker_drains_versioned_lease(
    tmp_path,
    in_memory_store,
    worker_subprocess,
):
    """Worker boots with --bootstrap (no --agent-module) and handles a
    lease whose agent_version="1" triggers the bundle fetch path."""
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
            dispatcher.enqueue(agent="enrich", item_id="co_1", agent_version="1")

            cache_root = tmp_path / "bundle_cache"
            worker = worker_subprocess(
                agent_module=None,
                bootstrap=True,
                dispatcher=dispatcher,
                store=in_memory_store,
                bundle_url_base=bundle_server.url,
                env_overrides={"PAPAYYA_BUNDLE_CACHE_ROOT": str(cache_root)},
            )

            dispatcher.wait_until_drained(timeout=15.0)

            deadline = _t.monotonic() + 2.0
            run = in_memory_store.run_for_item("co_1")
            while run is None and _t.monotonic() < deadline:
                _t.sleep(0.02)
                run = in_memory_store.run_for_item("co_1")
            assert run is not None, (
                "run never landed in the store; worker log:\n"
                + worker.stderr_tail(8192)
            )
            assert [t.label for t in run.tasks] == ["extract"]

            # The bundle was fetched on demand — no stub agent module in play.
            assert ("enrich", 1) in bundle_server.calls

            worker.stop(timeout=5.0)
            assert worker.exit_code == 0
        finally:
            dispatcher.shutdown()
    finally:
        bundle_server.shutdown()


def test_bootstrap_worker_fails_versionless_lease_with_category(
    tmp_path,
    in_memory_store,
    worker_subprocess,
):
    """Misuse path: bootstrap worker + agent_version=None -> distinct
    error_category="no_agent_module" so operators can tell it apart
    from the generic unknown-agent failure."""
    bundle_server = FakeBundleServer()
    try:
        # No bundle registered — and we wouldn't reach the fetch path anyway,
        # since the lease has no agent_version. Bundle server only exists
        # because the worker requires --bundle-url-base to be addressable.

        dispatcher = FakeDispatcher()
        try:
            dispatcher.enqueue(agent="enrich", item_id="co_oops")

            cache_root = tmp_path / "bundle_cache"
            worker = worker_subprocess(
                agent_module=None,
                bootstrap=True,
                dispatcher=dispatcher,
                store=in_memory_store,
                bundle_url_base=bundle_server.url,
                env_overrides={"PAPAYYA_BUNDLE_CACHE_ROOT": str(cache_root)},
            )

            dispatcher.wait_until_drained(timeout=10.0)

            failures = dispatcher.failed_completions()
            assert len(failures) == 1, (
                f"expected 1 failed completion, got {failures}; worker log:\n"
                + worker.stderr_tail(8192)
            )
            assert failures[0]["error_category"] == "no_agent_module", failures[0]

            assert in_memory_store.run_for_item("co_oops") is None

            worker.stop(timeout=5.0)
        finally:
            dispatcher.shutdown()
    finally:
        bundle_server.shutdown()

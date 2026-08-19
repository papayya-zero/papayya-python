"""Plan 61 U1 — customer code runs in a child that holds no fleet credential.

These drive REAL child processes against REAL bundles on disk. The property
under test is about a process image and an address space, and neither can be
observed by a test that stubs the process away.

The shape of the adversarial case is plan 60's: a bundle whose ``@agent`` tries
to read the platform key and reports what it found.
"""

from __future__ import annotations

import json
import os
import tarfile
import textwrap
from pathlib import Path

import pytest

from papayya.runtime.worker import Lease, Worker

PLATFORM_KEY = "papayya_platform_TESTSECRET_do_not_leak"
ACCOUNT_A = "11111111-1111-1111-1111-111111111111"
ACCOUNT_B = "22222222-2222-2222-2222-222222222222"


def _write_bundle(root: Path, name: str, body: str) -> Path:
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "agent.py").write_text(textwrap.dedent(body))
    return d


class _FakeBundle:
    """What Worker._ensure_bundle returns, without the fetch."""

    def __init__(self, path: Path):
        self.path = path
        self.entrypoint = "agent.py"
        self.dep_hash = None


def _make_worker(tmp_path, **kw) -> Worker:
    w = Worker(
        dispatcher_url="http://127.0.0.1:1/v1/runtime",
        store_path="",
        agent_module_path=None,
        api_key=PLATFORM_KEY,
        heartbeat_interval_seconds=3600,
        **kw,
    )
    w.stop()
    w._hb_stop.set()
    w._stop_heartbeat()
    return w


def _lease(agent="probe", version="1", account=ACCOUNT_A, payload=None, lease_id="l-1"):
    return Lease(
        lease_id=lease_id,
        agent=agent,
        item_id="item-1",
        payload=payload if payload is not None else {"run_id": "run-1", "input": "hello"},
        agent_version=version,
        account_id=account,
        project_id="33333333-3333-3333-3333-333333333333",
        run_token="prt_fake_token_for_this_run",
    )


def _drive(w: Worker, lease: Lease, bundle_dir: Path, max_duration=None):
    """Send one lease through the executor and return the raw report."""
    return w._run_in_executor(lease, _FakeBundle(bundle_dir), max_duration)


# --- the security property -------------------------------------------------


def test_the_child_cannot_see_the_platform_key(tmp_path):
    """The whole reason the split exists.

    Plan 60 closed the env var, both argvs and /proc/*/environ, and an @agent
    still read `Worker._api_key` off the heap with a gc.get_objects() attribute
    walk. Here that walk happens in a process where the object does not exist.
    """
    bundle = _write_bundle(tmp_path, "probe", '''
        import gc, os
        from papayya import agent

        # Split so the full literal is never a module-level STRING — otherwise
        # the walk below finds the probe's own copy of what it is hunting for,
        # which is a true positive about the wrong process.
        _S = ("%s", "%s")
        # The PREFIX matches prose as well as secrets — papayya's own module
        # docstrings quote an example key — so a prefix hit in a __doc__ is
        # documentation, not a leak. The exact value is the real question.
        _P = ("papayya_", "platform_")

        @agent("probe")
        def probe(payload):
            secret = _S[0] + _S[1]
            prefix = _P[0] + _P[1]
            env = [k for k, v in os.environ.items() if prefix in str(v)]
            exact, prefixed = [], []
            for obj in gc.get_objects():
                # Guarded: some libraries (pydantic) raise from __getattr__,
                # and a probe that dies partway through would report a clean
                # heap it never finished searching.
                try:
                    d = object.__getattribute__(obj, "__dict__")
                except Exception:
                    continue
                if not isinstance(d, dict):
                    continue
                try:
                    items = list(d.items())
                except Exception:
                    continue
                for k, v in items:
                    if not isinstance(v, str):
                        continue
                    if secret in v:
                        exact.append(f"{type(obj).__name__}.{k}")
                    elif prefix in v and k != "__doc__":
                        prefixed.append(f"{type(obj).__name__}.{k}")
            return {
                "env": env, "exact": exact, "prefixed": prefixed, "scanned": True,
            }
    ''' % (PLATFORM_KEY[:12], PLATFORM_KEY[12:]))

    w = _make_worker(tmp_path)
    try:
        report = _drive(w, _lease(), bundle)
    finally:
        w._close_executor(grace=1)

    assert report["kind"] == "completed", report
    assert report["output"].get("scanned") is True, "the probe did not finish its walk"
    assert report["output"]["env"] == [], "the key is in the child's environment"
    assert report["output"]["exact"] == [], (
        f"the key is reachable on the child's heap: {report['output']['exact']}"
    )
    assert report["output"]["prefixed"] == [], (
        f"something key-shaped is on the child's heap: {report['output']['prefixed']}"
    )


def test_the_supervisor_still_holds_the_key(tmp_path):
    """The split moves the key away from customer code, not out of the fleet.

    A supervisor that lost its own credential could not lease, complete or
    heartbeat — so this asserts the thing that must NOT have changed.
    """
    w = _make_worker(tmp_path)
    try:
        assert w._api_key == PLATFORM_KEY
        assert w._auth_headers()["X-Api-Key"] == PLATFORM_KEY
    finally:
        w._close_executor(grace=1)


def test_two_accounts_do_not_share_an_interpreter(tmp_path):
    """Plan 60 S3: tenant A's module must not be resident for tenant B.

    Each bundle records its own identity into a module global. If they shared a
    process, the second import would find the first's marker still in
    sys.modules under a sibling name — and more to the point, would be able to
    reach it.
    """
    body = '''
        import sys
        from papayya import agent

        MARKER = "%s"

        @agent("probe")
        def probe(payload):
            others = [
                m for m in list(sys.modules)
                if m.startswith("_papayya_user_") and "%s" not in m
            ]
            return {"marker": MARKER, "other_tenant_modules": others}
    '''
    a = _write_bundle(tmp_path, "a", body % ("ACCOUNT-A", ACCOUNT_A.replace("-", "_")))
    b = _write_bundle(tmp_path, "b", body % ("ACCOUNT-B", ACCOUNT_B.replace("-", "_")))

    w = _make_worker(tmp_path)
    try:
        ra = _drive(w, _lease(account=ACCOUNT_A, lease_id="l-a"), a)
        rb = _drive(w, _lease(account=ACCOUNT_B, lease_id="l-b"), b)
    finally:
        w._close_executor(grace=1)

    assert ra["output"]["marker"] == "ACCOUNT-A"
    assert rb["output"]["marker"] == "ACCOUNT-B", (
        "account B ran account A's code — the residency collision plan 47 S1 found"
    )
    assert rb["output"]["other_tenant_modules"] == [], (
        f"account A's module was resident for account B: {rb['output']['other_tenant_modules']}"
    )


def test_the_executor_is_recycled_when_the_account_changes(tmp_path):
    """The key leads with the account, so a tenant change is a new process."""
    body = '''
        import os
        from papayya import agent

        @agent("probe")
        def probe(payload):
            return {"pid": os.getpid()}
    '''
    a = _write_bundle(tmp_path, "a", body)
    b = _write_bundle(tmp_path, "b", body)

    w = _make_worker(tmp_path)
    try:
        first = _drive(w, _lease(account=ACCOUNT_A, lease_id="l-1"), a)
        again = _drive(w, _lease(account=ACCOUNT_A, lease_id="l-2"), a)
        other = _drive(w, _lease(account=ACCOUNT_B, lease_id="l-3"), b)
    finally:
        w._close_executor(grace=1)

    assert first["output"]["pid"] == again["output"]["pid"], (
        "the same tenant's next item paid for a fresh interpreter"
    )
    assert other["output"]["pid"] != first["output"]["pid"], (
        "a different account was served by the same process"
    )


def test_reuse_item_gives_every_item_its_own_process(tmp_path):
    bundle = _write_bundle(tmp_path, "probe", '''
        import os
        from papayya import agent

        @agent("probe")
        def probe(payload):
            return {"pid": os.getpid()}
    ''')
    w = _make_worker(tmp_path, executor_reuse="item")
    try:
        one = _drive(w, _lease(lease_id="l-1"), bundle)
        two = _drive(w, _lease(lease_id="l-2"), bundle)
    finally:
        w._close_executor(grace=1)

    assert one["output"]["pid"] != two["output"]["pid"]


# --- the outcome taxonomy survives the pipe --------------------------------


def test_a_raising_agent_reports_failed_with_a_category(tmp_path):
    bundle = _write_bundle(tmp_path, "probe", '''
        from papayya import agent

        @agent("probe")
        def probe(payload):
            raise ValueError("the customer's own bug")
    ''')
    w = _make_worker(tmp_path)
    try:
        report = _drive(w, _lease(), bundle)
    finally:
        w._close_executor(grace=1)

    assert report["kind"] == "failed"
    assert "ValueError: the customer's own bug" in report["error"]
    assert report.get("error_category")


def test_a_bundle_that_will_not_import_fails_its_item_not_the_pool(tmp_path):
    """The property test_worker_stall_survival asserts for the in-process path.

    Same behaviour, now across the pipe: the import blows up in the child and
    the supervisor is still standing to report it.
    """
    bundle = _write_bundle(tmp_path, "probe", '''
        import pandas  # noqa: F401 — deliberately absent
        from papayya import agent

        @agent("probe")
        def probe(payload):
            return {}
    ''')
    w = _make_worker(tmp_path)
    try:
        report = _drive(w, _lease(), bundle)
        # And the pool is still usable afterwards.
        good = _write_bundle(tmp_path, "good", '''
            from papayya import agent

            @agent("probe")
            def probe(payload):
                return {"ok": True}
        ''')
        after = _drive(w, _lease(lease_id="l-2"), good)
    finally:
        w._close_executor(grace=1)

    assert report["kind"] == "failed"
    assert "pandas" in report["error"]
    assert report["error_category"] == "deps"
    assert after["kind"] == "completed", "one bad bundle took the pool with it"


def test_a_paused_agent_is_released_not_completed(tmp_path):
    """plan 41 R6: a completed lease is terminal, so a pause must not complete."""
    bundle = _write_bundle(tmp_path, "probe", '''
        from papayya import agent
        from papayya.errors import WorkloadPaused

        @agent("probe")
        def probe(payload):
            raise WorkloadPaused("budget fence")
    ''')
    w = _make_worker(tmp_path)
    try:
        report = _drive(w, _lease(), bundle)
    finally:
        w._close_executor(grace=1)

    assert report["kind"] == "released"
    assert report["reason"] == "paused"


def test_an_async_agent_works(tmp_path):
    bundle = _write_bundle(tmp_path, "probe", '''
        from papayya import agent

        @agent("probe")
        async def probe(payload):
            return {"async": True}
    ''')
    w = _make_worker(tmp_path)
    try:
        report = _drive(w, _lease(), bundle)
    finally:
        w._close_executor(grace=1)

    assert report["kind"] == "completed"
    assert report["output"] == {"async": True}


def test_the_agent_is_called_with_the_submitted_input(tmp_path):
    """plan 43 B2a — fn(item_id) instead of fn(input) 400'd every hosted step."""
    bundle = _write_bundle(tmp_path, "probe", '''
        from papayya import agent

        @agent("probe")
        def probe(payload):
            return {"got": payload}
    ''')
    w = _make_worker(tmp_path)
    try:
        report = _drive(
            w, _lease(payload={"run_id": "run-1", "input": {"doc": "D-1"}}), bundle
        )
    finally:
        w._close_executor(grace=1)

    assert report["output"]["got"] == {"doc": "D-1"}


def test_a_non_serialisable_return_still_completes(tmp_path):
    """An empty output is a better answer than a dead executor."""
    bundle = _write_bundle(tmp_path, "probe", '''
        from papayya import agent

        @agent("probe")
        def probe(payload):
            return object()
    ''')
    w = _make_worker(tmp_path)
    try:
        report = _drive(w, _lease(), bundle)
    finally:
        w._close_executor(grace=1)

    assert report["kind"] == "completed"
    assert report.get("output") is None


# --- the supervisor's timeout cannot be starved ---------------------------


@pytest.mark.slow
def test_a_gil_holding_agent_is_killed_by_the_supervisor(tmp_path):
    """Plan 55 D1's shape: compute that never returns to the interpreter.

    The child's own SIGALRM is the graceful path and cannot fire through a C
    loop. The supervisor is a different process and is not blocked, so its
    deadline is the one that holds.
    """
    bundle = _write_bundle(tmp_path, "probe", '''
        import re
        from papayya import agent

        @agent("probe")
        def probe(payload):
            re.match(r"(a+)+$", "a" * 29 + "b")
            return {"finished": True}
    ''')
    w = _make_worker(tmp_path)
    try:
        report = _drive(w, _lease(), bundle, max_duration=1)
    finally:
        w._close_executor(grace=1)

    assert report["kind"] == "failed"
    assert report["error_category"] == "timeout"


def test_a_customer_print_cannot_forge_a_result(tmp_path):
    """stdout is theirs; the protocol has its own descriptor.

    Without the separate fd this test's agent would complete the item as
    somebody else's success.
    """
    bundle = _write_bundle(tmp_path, "probe", '''
        import json, sys
        from papayya import agent

        @agent("probe")
        def probe(payload):
            print(json.dumps({"kind": "completed", "output": {"forged": True}}))
            sys.stdout.flush()
            raise ValueError("but it actually failed")
    ''')
    w = _make_worker(tmp_path)
    try:
        report = _drive(w, _lease(), bundle)
    finally:
        w._close_executor(grace=1)

    assert report["kind"] == "failed", "a print() on stdout was read as the result"
    assert "but it actually failed" in report["error"]

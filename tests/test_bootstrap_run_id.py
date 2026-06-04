"""v1→v2 cutover (Workstream C): the hosted worker injects the lease's
run_id via the one-shot _BOOTSTRAP_RUN_ID contextvar so the @agent's run
links its checkpoints to the durable_run the submission pre-created.

These pin the seam consumed inside Papayya.run() without standing up a
full worker/HTTP harness."""

from __future__ import annotations

from papayya import papayya
from papayya.agent import (
    consume_bootstrap_run_id,
    reset_bootstrap_run_id,
    set_bootstrap_run_id,
)
from papayya.durable.store import MemoryStore


def test_bootstrap_run_id_is_adopted_then_one_shot():
    """The first run() after the worker sets the contextvar adopts the
    lease's run_id; a second run() (a sub-run) mints a fresh id."""
    client = papayya()
    token = set_bootstrap_run_id("lease-run-123")
    try:
        run = client.run("enrich", store=MemoryStore())
        assert run.run_id == "lease-run-123"

        # One-shot: the contextvar was cleared on the first adoption, so a
        # sub-run spawned in the same body does not reuse the lease id.
        sub = client.run("enrich", store=MemoryStore())
        assert sub.run_id != "lease-run-123"
    finally:
        reset_bootstrap_run_id(token)


def test_explicit_run_id_wins_over_bootstrap():
    client = papayya()
    token = set_bootstrap_run_id("lease-run-123")
    try:
        run = client.run("enrich", run_id="explicit-id", store=MemoryStore())
        assert run.run_id == "explicit-id"
    finally:
        reset_bootstrap_run_id(token)


def test_no_bootstrap_mints_fresh_id():
    """Local-dev guardrail: with no worker-injected run_id, run() mints its
    own uuid exactly as before the cutover."""
    assert consume_bootstrap_run_id() is None  # unset by default
    run = papayya().run("enrich", store=MemoryStore())
    assert run.run_id and run.run_id != "lease-run-123"

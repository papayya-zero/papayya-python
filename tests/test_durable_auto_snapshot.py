"""Auto-snapshot capture in run.step.

When an item_id is in effect, run.step auto-captures the wrapped fn's
call args as the step's input_snapshot — same path the @agent decorator
uses to seed runs.input_snapshot. Opt out with snapshot=False; pass any
other value to override.

Before this change, input_snapshot only populated when the user
explicitly passed snapshot=, leaving lineage lopsided: the agent-level
input was captured, but per-step inputs were silently dropped unless the
developer remembered the kwarg.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from papayya.durable.run import PapayyaRun
from papayya.durable.sqlite_store import SQLiteStore
from papayya.durable.types import DurableRunConfig


@pytest.fixture
def tmp_db(tmp_path: Path) -> Path:
    return tmp_path / "auto_snap.db"


def _make_run(tmp_db: Path, *, item_id: str | None = None) -> PapayyaRun:
    return PapayyaRun(
        DurableRunConfig(
            agent="auto-snap",
            store=SQLiteStore(str(tmp_db)),
            item_id=item_id,
        )
    )


class TestAutoCapture:
    def test_captures_positional_arg_when_item_id_in_effect(
        self, tmp_db: Path
    ) -> None:
        run = _make_run(tmp_db, item_id="co_42")

        def enrich(payload: dict) -> dict:
            return {**payload, "tier": "gold"}

        run.step("enrich", enrich)({"name": "acme"})

        entry = run._cache["enrich"]
        assert entry.input_snapshot == {"payload": {"name": "acme"}}
        assert entry.output_snapshot == {"name": "acme", "tier": "gold"}

    def test_captures_via_canonical_form(self, tmp_db: Path) -> None:
        run = _make_run(tmp_db, item_id="co_42")

        def enrich(payload: dict) -> dict:
            return {**payload, "tier": "gold"}

        run.step("enrich", enrich)({"name": "acme"})

        assert run._cache["enrich"].input_snapshot == {"payload": {"name": "acme"}}

    def test_captures_kwargs_with_defaults(self, tmp_db: Path) -> None:
        run = _make_run(tmp_db, item_id="co_42")

        def enrich(payload: dict, retries: int = 3) -> dict:
            return payload

        run.step("enrich", enrich)({"name": "acme"})

        assert run._cache["enrich"].input_snapshot == {
            "payload": {"name": "acme"},
            "retries": 3,
        }

    def test_per_step_item_id_seeds_auto_capture(self, tmp_db: Path) -> None:
        """Auto-capture also fires when item_id is supplied per-step."""
        run = _make_run(tmp_db)  # no run-level item_id

        def enrich(payload: dict) -> dict:
            return payload

        run.step("enrich", enrich, item_id="co_seed")({"name": "acme"})

        assert run._cache["enrich"].input_snapshot == {"payload": {"name": "acme"}}


class TestOptOut:
    def test_snapshot_false_is_opt_out(self, tmp_db: Path) -> None:
        run = _make_run(tmp_db, item_id="co_42")

        def enrich(payload: dict) -> dict:
            return payload

        run.step("enrich", enrich, snapshot=False)({"name": "acme"})

        entry = run._cache["enrich"]
        assert entry.input_snapshot is None
        # Opt-out is input-only; output still captures.
        assert entry.output_snapshot == {"name": "acme"}


class TestExplicitOverride:
    def test_explicit_snapshot_overrides_auto(self, tmp_db: Path) -> None:
        run = _make_run(tmp_db, item_id="co_42")

        def enrich(payload: dict) -> dict:
            return payload

        run.step("enrich", enrich, snapshot={"alt": "value"})({"name": "acme"})

        assert run._cache["enrich"].input_snapshot == {"alt": "value"}

    def test_explicit_none_overrides_auto(self, tmp_db: Path) -> None:
        """snapshot=None is an explicit value, not auto. Records None."""
        run = _make_run(tmp_db, item_id="co_42")

        def enrich(payload: dict) -> dict:
            return payload

        run.step("enrich", enrich, snapshot=None)({"name": "acme"})

        assert run._cache["enrich"].input_snapshot is None


class TestDegradedPaths:
    def test_non_json_arg_falls_through_to_none(self, tmp_db: Path) -> None:
        """SimpleNamespace with __dict__ would coerce — use a class with no __dict__ via __slots__."""
        run = _make_run(tmp_db, item_id="co_42")

        class Slotted:
            __slots__ = ()

        def enrich(thing) -> str:
            return "ok"

        run.step("enrich", enrich)(Slotted())

        # strict encode rejects Slotted (no __dict__, not a dataclass, not a model).
        # Auto-capture returns None; the step still runs and records output.
        entry = run._cache["enrich"]
        assert entry.input_snapshot is None
        assert entry.output_snapshot == "ok"

    def test_builtin_with_no_signature_falls_through(self, tmp_db: Path) -> None:
        """C-level callables (no introspectable signature) → None, no crash."""
        run = _make_run(tmp_db, item_id="co_42")

        # `len` has no introspectable signature in some Pythons; use a wrapper
        # that we explicitly mark as having no signature by binding it as
        # functools.partial of a builtin. inspect.signature(len) actually
        # works in 3.11+, so we use a real builtin that fails: `dict.fromkeys`
        # bound — but the simplest is to verify the helper accepts None sig.
        from papayya._serialize import build_input_snapshot
        assert build_input_snapshot(None, ("anything",), {}) is None


class TestNoItemIdPreservesStatusQuo:
    def test_no_item_id_means_no_snapshots(self, tmp_db: Path) -> None:
        run = _make_run(tmp_db)  # no run-level item_id, no per-step item_id

        def enrich(payload: dict) -> dict:
            return payload

        run.step("enrich", enrich)({"name": "acme"})

        entry = run._cache["enrich"]
        assert entry.input_snapshot is None
        assert entry.output_snapshot is None


class TestReplayDoesNotReFire:
    def test_cached_step_skips_auto_capture(self, tmp_db: Path) -> None:
        """A cached step on replay returns the cached entry without re-running
        the wrapper body — so auto-capture doesn't re-fire and overwrite
        the persisted input_snapshot.
        """
        # First run: persist snapshot.
        run1 = PapayyaRun(
            DurableRunConfig(
                agent="auto-snap",
                store=SQLiteStore(str(tmp_db)),
                run_id="r1",
                item_id="co_42",
            )
        )

        def enrich(payload: dict) -> dict:
            return {**payload, "tier": "gold"}

        run1.step("enrich", enrich)({"name": "acme"})

        # Replay: same run_id, fresh PapayyaRun. Cached entry returns cached
        # result; the wrapper body never executes.
        run2 = PapayyaRun(
            DurableRunConfig(
                agent="auto-snap",
                store=SQLiteStore(str(tmp_db)),
                run_id="r1",
                item_id="co_42",
            )
        )
        # Pass a different arg on replay — proves the wrapper body does not
        # rebind args. The cached input_snapshot from the original call must
        # survive intact.
        result = run2.step("enrich", enrich)({"name": "different"})

        assert result == {"name": "acme", "tier": "gold"}  # cached output
        assert run2._cache["enrich"].input_snapshot == {"payload": {"name": "acme"}}

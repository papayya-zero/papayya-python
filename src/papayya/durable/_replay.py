"""Local-DB replay — Python entry point for the durable-run replay flow.

The CLI's ``papayya replay`` is a thin wrapper over this module.
SDK callers can re-drive a failed run from a notebook or REPL::

    from papayya.durable import client
    result = client.replay("run-id")

Setup/contract violations (missing run, NULL snapshot, version
mismatch, unknown agent) raise :class:`ReplayError`. The agent's own
exception during invoke is re-raised as-is so callers can distinguish
"I couldn't even start the replay" from "the replay ran and the
agent failed". The original run is marked ``disposition='replayed'``
on either outcome so a fresh failure shows up as its own dead letter.

ADR-0002 #7 version gate: a captured ``agent_version`` that differs
from the registration's current value blocks the replay; pass
``latest=True`` to opt out the same way the CLI's ``--latest`` flag
does. Pre-#7 runs (NULL ``agent_version``) replay without the gate.
"""

from __future__ import annotations

import importlib.util
import inspect
import json
import os
import sqlite3
import uuid
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Callable

from papayya.durable import _schema
from papayya.durable.sqlite_store import SQLiteStore
from papayya.durable.types import TaskEntry


class ReplayError(Exception):
    """Raised when replay setup fails — bad run id, NULL snapshot,
    version mismatch, missing registration. The agent's own exception
    during invoke is re-raised unwrapped, not as ReplayError."""


# Phase 3 step-rewind hydration transport. _replay.replay() sets this
# before invoking the customer's @agent function; Papayya.run() reads
# and clears it on the first construction inside that function so
# only the outermost run picks up the seeded cache. Subsequent intra-
# fn run() calls (rare but supported) construct normal fresh runs.
_REPLAY_HYDRATION: ContextVar[tuple[str, list[TaskEntry]] | None] = ContextVar(
    "papayya_replay_hydration", default=None
)


def consume_replay_hydration() -> tuple[str, list[TaskEntry]] | None:
    """One-shot read of the replay hydration tuple. Returns
    ``(new_run_id, [TaskEntry, ...])`` when ``_replay.replay()`` is
    driving the call, ``None`` otherwise. Resets the contextvar to
    None on read so only the first ``Papayya.run()`` call inside the
    replayed ``@agent`` body picks up the hydration."""
    value = _REPLAY_HYDRATION.get()
    if value is not None:
        _REPLAY_HYDRATION.set(None)
    return value


def _resolve_from_step(
    from_step: str | int,
    checkpoint_tasks: list[TaskEntry],
    run_id: str,
) -> int:
    """Normalise from_step (str|int) to a hydration prefix count
    (number of cached TaskEntry rows to seed before re-execution).

    Semantics:
    - **str** that matches a cached label → prefix = position of the
      first matching entry. Re-executing this label means "redo a
      previously-successful step." Hydrates everything strictly
      before it.
    - **str** that doesn't match any cached label → prefix = all
      cached entries. This is the natural failure-replay case: the
      step that died never made it into cache, so its label can't be
      validated against stored data; hydrate everything we have and
      let the agent fn pick up where it left off.
    - **int** (1-indexed step number) in ``[1, len(cached) + 1]`` →
      prefix = ``from_step - 1``. ``len + 1`` means "hydrate all
      cached and re-execute the first uncached step." Out of range
      raises ``ReplayError``.

    Raised before any side effect so the original run's
    ``dlq_disposition`` stays NULL on validation failure.
    """
    labels = [t.label for t in checkpoint_tasks]
    if isinstance(from_step, str):
        if from_step in labels:
            return labels.index(from_step)
        # Unmatched label is the natural failure-replay case (the step
        # that failed isn't cached). Hydrate everything we have. We
        # cannot detect typos here without knowing the agent's full
        # label set, which is only discoverable by executing it.
        return len(labels)
    if isinstance(from_step, int) and not isinstance(from_step, bool):
        if not (1 <= from_step <= len(labels) + 1):
            raise ReplayError(
                f"from_step={from_step} is out of range for run "
                f"{run_id!r} (run has {len(labels)} cached step(s); "
                f"valid range is 1..{len(labels) + 1})"
            )
        return from_step - 1
    raise ReplayError(
        f"from_step must be str (label) or int (1-indexed step number); "
        f"got {type(from_step).__name__}"
    )


def _resolve_db_path() -> Path:
    env_path = os.environ.get("PAPAYYA_LOCAL_DB_PATH")
    return Path(env_path) if env_path else Path(".papayya/local.db")


def _discover_agents(path: str | Path) -> list[Any]:
    from papayya.agent import AgentRegistration, _registry, get_registry

    _registry.clear()

    filepath = Path(path).resolve()
    if not filepath.exists():
        raise ReplayError(f"File not found: {filepath}")

    spec = importlib.util.spec_from_file_location("_agent_module", filepath)
    if spec is None or spec.loader is None:
        raise ReplayError(f"Cannot load module from: {filepath}")

    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    agents = list(get_registry().values())

    if not agents:
        agent_obj = getattr(mod, "agent", None)
        main_fn = getattr(mod, "main", None)
        if agent_obj is not None and main_fn is not None and callable(main_fn):
            agents.append(AgentRegistration(
                name=getattr(agent_obj, "name", "unknown"),
                model=getattr(agent_obj, "model", ""),
                instructions=getattr(agent_obj, "instructions", ""),
                fn=main_fn,
                tools=getattr(agent_obj, "tools", []),
                max_steps=getattr(agent_obj, "max_steps", 50),
                budget_usd=getattr(agent_obj, "budget_usd", None),
            ))

    return agents


def _replay_invoke(fn: Any, snapshot: Any) -> Any:
    if isinstance(snapshot, dict):
        try:
            inspect.signature(fn).bind(**snapshot)
        except (TypeError, ValueError):
            pass
        else:
            return fn(**snapshot)
    return fn(snapshot)


def _replay_with_handler(
    run_id: str,
    handler: Callable[[Any], Any],
    snapshot: Any,
    workload: str,
    db_path: Path,
) -> Any:
    """Re-drive a run's captured item through a caller-supplied callable.

    iter-runs carry no registration, so the customer supplies the same
    callable their loop drives. We feed the snapshot through
    ``papayya.iter`` once — reusing its per-item run lifecycle — so the
    replay is itself a fully-recorded run (new ``run_id``, leaf-decorator
    capture, outcome inspection, its own input_snapshot) exactly like a
    fresh iter pass. The original is then marked ``disposition='replayed'``
    so it leaves the dead-letter queue; a handler that raises records the
    replay run as failed (its own dead letter) and the exception
    propagates, mirroring registration-mode semantics.

    Unlike registration mode, the snapshot is always passed *positionally*
    (``handler(item)``) — never unpacked as kwargs. The iter contract is
    ``body(item)`` on the whole item; an item dict is the argument, not a
    bag of keyword arguments.
    """
    # Lazy import avoids a circular import (iterators imports papayya.durable).
    # SQLiteStore is already imported at module scope.
    from papayya import iterators

    load_store = SQLiteStore(str(db_path))
    try:
        checkpoint = load_store.load(run_id)
    finally:
        load_store.close()
    # The run row exists (read at the top of replay()), so load() should
    # not return None; guard anyway and fall back to empty attribution.
    item_id = "" if checkpoint is None or checkpoint.item_id is None else str(
        checkpoint.item_id
    )
    partition_key = (
        ""
        if checkpoint is None or checkpoint.partition_key is None
        else str(checkpoint.partition_key)
    )

    run_store = SQLiteStore(str(db_path))
    replay_error: BaseException | None = None
    result: Any = None
    try:
        for item in iterators.iter(
            [snapshot],
            agent=workload,
            item_id=lambda _it: item_id,
            partition_key=lambda _it: partition_key,
            store=run_store,
        ):
            result = handler(item)
    except Exception as exc:  # noqa: BLE001 — handler's exception class is unknown
        replay_error = exc
    finally:
        run_store.close()

    mark_store = SQLiteStore(str(db_path))
    try:
        mark_store.mark_dlq_disposition(run_id, _schema.DLQ_REPLAYED)
    finally:
        mark_store.close()

    if replay_error is not None:
        raise replay_error
    return result


def replay(
    run_id: str,
    *,
    handler: Callable[[Any], Any] | None = None,
    agent_module: str | Path | None = None,
    db_path: str | Path | None = None,
    latest: bool = False,
    from_step: str | int | None = None,
) -> Any:
    """Re-drive a failed durable run from its captured input snapshot.

    Two resolution modes for *what to re-run*:

    * **Registration mode (default).** Loads the run's input_snapshot,
      finds the matching ``@agent`` registration in the agent module, and
      re-invokes it. Dict snapshots whose keys bind to the agent's
      parameters are unpacked as kwargs; everything else is passed
      positionally. This is the path for ``@agent``-created runs.
    * **Handler mode (``handler=``).** ``papayya.iter`` runs have no
      ``@agent`` registration to discover — the loop body is a suspended
      frame, not a callable. Pass ``handler=`` (the same callable your
      loop drives) and the captured item is re-driven through
      ``papayya.iter`` once: the replay is itself a fully-recorded run
      (fresh ``run_id``, leaf-decorator capture, outcome inspection),
      exactly like the original, and the original is marked
      ``disposition='replayed'``. Item-granularity only — the whole item
      re-runs; there is no step cache at this tier (that's the
      ``@agent`` + ``run.step`` upgrade), so ``from_step=`` is rejected.
      ``handler=`` and ``agent_module=`` are mutually exclusive.

    Version gate (ADR-0002 #7) refuses to replay when the captured
    ``agent_version`` differs from the registration's current value.
    Pass ``latest=True`` to opt out. NULL captured value (legacy runs)
    replays without the gate.

    The original run is marked ``disposition='replayed'`` on either
    outcome. A failed replay re-raises after marking; a fresh dead
    letter created by the agent's own ``run.fail(...)`` path will show
    up alongside.

    Phase 3 step-level rewind: pass ``from_step`` (label string or
    1-indexed step number) to skip re-execution of cached predecessor
    steps. The replay constructs a *new* run with a freshly-generated
    ``run_id`` whose in-memory cache is pre-seeded with TaskEntry rows
    for every step strictly before ``from_step`` in the original run's
    task list. The wrapped agent's first ``step()`` calls then return
    the cached values directly without re-invoking the wrapped fns;
    ``from_step`` itself and every step after it re-execute fresh.
    Hydrated cache entries are NOT persisted to the new run's tasks
    table — only re-executed steps write rows. Local-SQLite-only;
    hosted callers use ``Papayya(...).runs.replay(run_id, from_step=N)``
    against the server-side endpoint.

    Common gotcha — bounded by captured input: the cached predecessor
    results are byte-for-byte identical to the originals. If the
    customer's ``from_step`` (or later) function body has been edited
    to read a field that the cached predecessor never produced, the
    function will raise ``KeyError`` / ``AttributeError`` against the
    cached payload. The exception propagates as-is — rewind further
    by choosing an earlier ``from_step`` so the missing-data step
    re-executes and produces the new shape.
    """
    db_path = Path(db_path) if db_path is not None else _resolve_db_path()
    if not db_path.exists():
        raise ReplayError(f"No local database at {db_path.resolve()}")

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            f"""SELECT {_schema.COL_ITEM_ID} AS run_id, agent, status,
                       {_schema.COL_ITEM_DLQ_DISPOSITION} AS disp,
                       {_schema.COL_ITEM_INPUT_SNAPSHOT} AS input_snapshot,
                       {_schema.COL_ITEM_AGENT_VERSION} AS agent_version
                FROM {_schema.TBL_ITEMS} WHERE {_schema.COL_ITEM_ID} = ?""",
            (run_id,),
        ).fetchone()
    finally:
        conn.close()

    if row is None:
        raise ReplayError(f"Run '{run_id}' not found in {db_path}")
    if row["status"] != "failed":
        raise ReplayError(
            f"Run '{run_id}' has status {row['status']!r}, not 'failed'. "
            "Only failed runs can be replayed."
        )
    if row["disp"] is not None:
        raise ReplayError(
            f"Run '{run_id}' is already resolved (disposition={row['disp']!r})."
        )

    raw = row["input_snapshot"]
    if raw is None:
        raise ReplayError(
            f"Run '{run_id}' has no input_snapshot — cannot replay. "
            "Input must be captured at run creation time; older runs "
            "predate this feature and are not replayable."
        )
    try:
        input_snapshot = json.loads(raw)
    except (TypeError, ValueError):
        input_snapshot = raw

    # Handler mode: re-drive the captured item through the caller's own
    # callable. Branches before @agent discovery / version gate / from_step
    # hydration — none of those apply when there's no registration. The
    # failed/undisposed/has-snapshot gates above still ran, so handler mode
    # inherits them.
    if handler is not None:
        if agent_module is not None:
            raise ReplayError(
                "Pass either handler= (re-drive the captured item through your "
                "own callable) or agent_module= (discover an @agent "
                "registration), not both."
            )
        if from_step is not None:
            raise ReplayError(
                "from_step= is not supported with handler=. iter-style replay is "
                "item-granularity: the whole item re-runs. Step-level rewind "
                "requires the @agent + run.step path."
            )
        return _replay_with_handler(
            run_id, handler, input_snapshot, row["agent"], db_path
        )

    agent_name = row["agent"]

    if agent_module is None:
        if Path("agent.py").exists():
            agent_module = "agent.py"
        else:
            raise ReplayError(
                "No agent.py in cwd. Pass agent_module= to point at "
                "the agent module."
            )

    registrations = _discover_agents(agent_module)
    matching = next((r for r in registrations if r.name == agent_name), None)
    if matching is None:
        names = ", ".join(r.name for r in registrations) or "(none)"
        raise ReplayError(
            f"No @agent with name {agent_name!r} found in {agent_module}. "
            f"Registered agents: {names}"
        )

    captured_version = row["agent_version"]
    current_version = matching.agent_version
    if (
        not latest
        and captured_version is not None
        and captured_version != current_version
    ):
        raise ReplayError(
            f"Run {run_id!r} captured agent_version="
            f"{captured_version!r}; current registration is at "
            f"{current_version!r}. Replay would run different code than "
            "the original. Pass latest=True (SDK) or --latest (CLI) to "
            "replay on the current version."
        )

    # Phase 3 from_step resolution + hydration setup. Done after every
    # other validation gate so an out-of-range from_step on a run that
    # would already have been rejected by the version gate fails on
    # the version gate (better signal than "step 5 doesn't exist" when
    # the real problem is "you're replaying the wrong code").
    hydration_token = None
    if from_step is not None:
        load_store = SQLiteStore(str(db_path))
        try:
            checkpoint = load_store.load(run_id)
        finally:
            load_store.close()
        # Defensive — the run row exists (we read it at the top of this
        # function) so load() should never return None here. Guard
        # anyway because store.load() could legally widen its contract
        # to filter on status in the future.
        if checkpoint is None:
            raise ReplayError(
                f"Run {run_id!r} could not be loaded for from_step rewind."
            )
        prefix_count = _resolve_from_step(from_step, checkpoint.tasks, run_id)
        prepopulated = checkpoint.tasks[:prefix_count]
        new_run_id = str(uuid.uuid4())
        hydration_token = _REPLAY_HYDRATION.set((new_run_id, prepopulated))

    replay_error: BaseException | None = None
    result: Any = None
    try:
        result = _replay_invoke(matching.fn, input_snapshot)
    except Exception as exc:  # noqa: BLE001 — agent's exception class is unknown
        replay_error = exc
    finally:
        if hydration_token is not None:
            _REPLAY_HYDRATION.reset(hydration_token)

    store = SQLiteStore(str(db_path))
    try:
        store.mark_dlq_disposition(run_id, _schema.DLQ_REPLAYED)
    finally:
        store.close()

    if replay_error is not None:
        raise replay_error
    return result


def replay_slice(
    run_id: str,
    *,
    tenant: str | None = None,
    handler: Callable[[Any], Any] | None = None,
    agent_module: str | Path | None = None,
    db_path: str | Path | None = None,
    latest: bool = False,
) -> dict[str, Any]:
    """Replay a RUN's not-ok slice: open a new run over the items of
    ``run_id`` whose ``worst_outcome_status != 'ok'`` (Plan 34 Unit 2b).

    This is the recovery verb of the acceptance sentence: *"you can replay
    the ones that didn't work."* Selection, minting and linkage:

    * selects the run's items where ``worst_outcome_status != 'ok'``;
      ``tenant=`` narrows the slice to one ``partition_key`` value
    * mints a NEW run row whose ``replayed_from`` is the source run's id
    * re-drives each selected item into the new run; every fresh item row
      carries ``replayed_from = <source item id>``
    * source items that were ``failed`` and untriaged are marked
      ``disposition='replayed'`` (degraded-but-completed sources have no
      DLQ state to update — the item-level ``replayed_from`` chain is the
      link)

    ``handler=`` / ``agent_module=`` resolve the callable exactly like
    :func:`replay` (handler for ``papayya.iter`` items, ``@agent``
    discovery otherwise, mutually exclusive). Unlike single-item replay,
    a raising item does NOT abort the slice — recovery over N items keeps
    going and the failures land as fresh dead letters in the new run.
    Items without a captured ``input_snapshot`` are skipped and counted.

    Returns a summary dict: ``{new_run_id, agent, selected, replayed_ok,
    replay_failed, skipped_no_snapshot}``.

    Single-item replay (:func:`replay`) is unchanged and remains available
    for one record at a time.
    """
    db_path = Path(db_path) if db_path is not None else _resolve_db_path()
    if not db_path.exists():
        raise ReplayError(f"No local database at {db_path.resolve()}")
    if handler is not None and agent_module is not None:
        raise ReplayError(
            "Pass either handler= (re-drive items through your own callable) "
            "or agent_module= (discover an @agent registration), not both."
        )

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        run_row = conn.execute(
            f"SELECT * FROM {_schema.TBL_RUNS} WHERE {_schema.COL_RUN_ID} = ?",
            (run_id,),
        ).fetchone()
        if run_row is None:
            raise ReplayError(f"Run '{run_id}' not found in {db_path}")

        query = f"""SELECT {_schema.COL_ITEM_ID} AS id, agent, status,
                           {_schema.COL_ITEM_ITEM_ID} AS item_id,
                           {_schema.COL_ITEM_PARTITION_KEY} AS partition_key,
                           {_schema.COL_ITEM_INPUT_SNAPSHOT} AS input_snapshot,
                           {_schema.COL_ITEM_AGENT_VERSION} AS agent_version,
                           {_schema.COL_ITEM_DLQ_DISPOSITION} AS disp
                    FROM {_schema.TBL_ITEMS}
                    WHERE {_schema.COL_ITEM_RUN_ID} = ?
                      AND {_schema.COL_ITEM_WORST_OUTCOME_STATUS} != 'ok'"""
        args: list[Any] = [run_id]
        if tenant is not None:
            query += f" AND {_schema.COL_ITEM_PARTITION_KEY} = ?"
            args.append(tenant)
        query += " ORDER BY created_at"
        item_rows = [dict(r) for r in conn.execute(query, args).fetchall()]
    finally:
        conn.close()

    if not item_rows:
        scope = f" for tenant {tenant!r}" if tenant is not None else ""
        raise ReplayError(
            f"Run '{run_id}' has no items with worst_outcome_status != 'ok'"
            f"{scope} — nothing to replay."
        )

    agent_name = run_row["agent"]

    # Resolve the callable once for the whole slice.
    if handler is not None:
        invoke: Callable[[Any], Any] = handler
    else:
        if agent_module is None:
            if Path("agent.py").exists():
                agent_module = "agent.py"
            else:
                raise ReplayError(
                    "No agent.py in cwd. Pass agent_module= to point at "
                    "the agent module, or handler= for papayya.iter items."
                )
        registrations = _discover_agents(agent_module)
        matching = next((r for r in registrations if r.name == agent_name), None)
        if matching is None:
            names = ", ".join(r.name for r in registrations) or "(none)"
            raise ReplayError(
                f"No @agent with name {agent_name!r} found in {agent_module}. "
                f"Registered agents: {names}"
            )
        if not latest:
            captured_versions = {
                r["agent_version"] for r in item_rows if r["agent_version"] is not None
            }
            mismatched = captured_versions - {matching.agent_version}
            if mismatched:
                raise ReplayError(
                    f"Run {run_id!r} has items captured at agent_version(s) "
                    f"{sorted(mismatched)!r}; current registration is at "
                    f"{matching.agent_version!r}. Replay would run different "
                    "code than the original. Pass latest=True (SDK) or "
                    "--latest (CLI) to replay on the current version."
                )
        invoke = lambda payload: _replay_invoke(matching.fn, payload)  # noqa: E731

    # Lazy imports — iterators imports papayya.durable, so a module-level
    # import here would recurse through the package init.
    from papayya import iterators
    from papayya.durable.run import Item
    from papayya.durable.types import DurableRunConfig
    from papayya.outcomes import OutcomeVerdict

    store = SQLiteStore(str(db_path))
    new_run_id = str(uuid.uuid4())
    replayed_ok = 0
    replay_failed = 0
    skipped_no_snapshot = 0
    try:
        store.create_run(
            new_run_id,
            agent_name,
            sum(1 for r in item_rows if r["input_snapshot"] is not None),
            replayed_from=run_id,
        )
        for source in item_rows:
            raw = source["input_snapshot"]
            if raw is None:
                skipped_no_snapshot += 1
                continue
            try:
                payload = json.loads(raw)
            except (TypeError, ValueError):
                payload = raw

            fresh = Item(
                DurableRunConfig(
                    agent=agent_name,
                    item_id=source["item_id"],
                    partition_key=source["partition_key"],
                    store=store,
                    invocation_id=new_run_id,
                    input_snapshot=payload,
                )
            )
            fresh.init()
            store.set_item_replayed_from(fresh.run_id, source["id"])
            token = iterators._ACTIVE_RUN.set(fresh)
            try:
                invoke(payload)
            except Exception as exc:  # noqa: BLE001 — customer code, class unknown
                replay_failed += 1
                iterators._write_synthetic_entry(
                    fresh, OutcomeVerdict("failed", "replay_body_exception")
                )
                try:
                    fresh.fail(error=str(exc))
                except Exception:
                    pass
            else:
                replayed_ok += 1
                try:
                    fresh.complete()
                except Exception:
                    pass
            finally:
                iterators._ACTIVE_RUN.reset(token)

            # Source item leaves the DLQ once it has been re-driven. Only
            # failed+untriaged items carry DLQ state; degraded-but-completed
            # sources are linked via replayed_from alone.
            if source["status"] == "failed" and source["disp"] is None:
                store.mark_dlq_disposition(source["id"], _schema.DLQ_REPLAYED)
    finally:
        store.close()

    return {
        "new_run_id": new_run_id,
        "agent": agent_name,
        "selected": len(item_rows),
        "replayed_ok": replayed_ok,
        "replay_failed": replay_failed,
        "skipped_no_snapshot": skipped_no_snapshot,
    }


__all__ = ["replay", "replay_slice", "ReplayError", "consume_replay_hydration"]

"""Local-DB replay — Python entry point for the durable-run replay flow.

The CLI's ``papayya dlq replay`` is a thin wrapper over this module.
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
from pathlib import Path
from typing import Any

from papayya.durable import _schema
from papayya.durable.sqlite_store import SQLiteStore


class ReplayError(Exception):
    """Raised when replay setup fails — bad run id, NULL snapshot,
    version mismatch, missing registration. The agent's own exception
    during invoke is re-raised unwrapped, not as ReplayError."""


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


def replay(
    run_id: str,
    *,
    agent_module: str | Path | None = None,
    db_path: str | Path | None = None,
    latest: bool = False,
    from_step: Any = None,
) -> Any:
    """Re-drive a failed durable run from its captured input snapshot.

    Loads the run's input_snapshot from the local SQLite DB, finds the
    matching ``@agent`` registration in the agent module, and re-invokes
    it. Dict snapshots whose keys bind to the agent's parameters are
    unpacked as kwargs; everything else is passed positionally.

    Version gate (ADR-0002 #7) refuses to replay when the captured
    ``agent_version`` differs from the registration's current value.
    Pass ``latest=True`` to opt out. NULL captured value (legacy runs)
    replays without the gate.

    The original run is marked ``disposition='replayed'`` on either
    outcome. A failed replay re-raises after marking; a fresh dead
    letter created by the agent's own ``run.fail(...)`` path will show
    up alongside.

    ``from_step`` is reserved for Phase 3 (step-level rewind) and must
    be ``None`` in Phase 1.
    """
    if from_step is not None:
        raise NotImplementedError(
            "from_step= is reserved for Phase 3 (step-level rewind); "
            "Phase 1 only supports full replay from the top."
        )

    db_path = Path(db_path) if db_path is not None else _resolve_db_path()
    if not db_path.exists():
        raise ReplayError(f"No local database at {db_path.resolve()}")

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            f"""SELECT run_id, agent, status,
                       {_schema.COL_RUN_DLQ_DISPOSITION} AS disp,
                       {_schema.COL_RUN_INPUT_SNAPSHOT} AS input_snapshot,
                       {_schema.COL_RUN_AGENT_VERSION} AS agent_version
                FROM {_schema.TBL_RUNS} WHERE run_id = ?""",
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
            "the original. Pass latest=True to replay on the current version."
        )

    replay_error: BaseException | None = None
    result: Any = None
    try:
        result = _replay_invoke(matching.fn, input_snapshot)
    except Exception as exc:  # noqa: BLE001 — agent's exception class is unknown
        replay_error = exc

    store = SQLiteStore(str(db_path))
    try:
        store.mark_dlq_disposition(run_id, _schema.DLQ_REPLAYED)
    finally:
        store.close()

    if replay_error is not None:
        raise replay_error
    return result


__all__ = ["replay", "ReplayError"]

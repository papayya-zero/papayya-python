"""``python -m papayya.runtime`` — boot a worker process.

Phase 1 prototype CLI. Argument surface is intentionally small; the
worker takes everything it needs at boot and never re-reads config. A
restart is the way to change behavior — that matches the recycle model
described in adr/0001-worker-pool-design-decisions.md.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

from .worker import (
    _DEFAULT_DRAIN_TIMEOUT_SECONDS,
    _DEFAULT_HEARTBEAT_INTERVAL,
    Worker,
)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m papayya.runtime",
        description="Long-running Papayya worker — pulls items, runs @agent functions.",
    )
    p.add_argument(
        "--agent-module",
        required=True,
        help="Absolute path to a .py file with @agent-decorated function(s).",
    )
    p.add_argument(
        "--dispatcher",
        required=True,
        help="Dispatcher base URL (e.g. http://127.0.0.1:8765).",
    )
    p.add_argument(
        "--store",
        required=True,
        help="Path to the SQLite file customer code should write through.",
    )
    p.add_argument(
        "--worker-id",
        default=None,
        help="Stable id for this worker (default: random short id).",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        help="Python logging level (default: INFO).",
    )
    p.add_argument(
        "--heartbeat-interval-seconds",
        type=float,
        default=_DEFAULT_HEARTBEAT_INTERVAL,
        help=(
            "Seconds between /heartbeat POSTs to the dispatcher while a "
            f"lease is in flight (default: {_DEFAULT_HEARTBEAT_INTERVAL})."
        ),
    )
    p.add_argument(
        "--drain-timeout-seconds",
        type=float,
        default=_DEFAULT_DRAIN_TIMEOUT_SECONDS,
        help=(
            "Seconds to let an in-flight item finish after SIGTERM "
            "before the worker force-exits (lease TTL recovers the "
            "orphaned item). 0 or negative disables the watchdog "
            f"(default: {_DEFAULT_DRAIN_TIMEOUT_SECONDS})."
        ),
    )
    p.add_argument(
        "--api-key",
        default=None,
        help=(
            "Project-scoped Papayya API key, sent as X-Api-Key on every "
            "dispatcher request. Falls back to the PAPAYYA_API_KEY env "
            "var when omitted. Required for the hosted dispatcher; "
            "optional for the local dispatcher."
        ),
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    # CLI flag wins; env var is the containerized-deploy fallback (ECS
    # task secret injection). Read here rather than via argparse default
    # so changes to the env between import and parse take effect.
    api_key = args.api_key or os.environ.get("PAPAYYA_API_KEY")
    worker = Worker(
        dispatcher_url=args.dispatcher,
        store_path=args.store,
        agent_module_path=args.agent_module,
        worker_id=args.worker_id,
        heartbeat_interval_seconds=args.heartbeat_interval_seconds,
        drain_timeout_seconds=args.drain_timeout_seconds,
        api_key=api_key,
    )
    worker.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())

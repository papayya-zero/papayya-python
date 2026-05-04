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
    _DEFAULT_MAX_ITEMS_BEFORE_RECYCLE,
    _DEFAULT_MAX_RSS_PERCENT_BEFORE_RECYCLE,
    Worker,
)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m papayya.runtime",
        description="Long-running Papayya worker — pulls items, runs @agent functions.",
    )
    p.add_argument(
        "--agent-module",
        required=False,
        default=None,
        help=(
            "Absolute path to a .py file with @agent-decorated function(s). "
            "Required for local dev (`papayya dev`); omit when --bootstrap "
            "or PAPAYYA_BOOTSTRAP=1 is set (hosted ECS workers load every "
            "bundle on demand via lease.agent_version)."
        ),
    )
    p.add_argument(
        "--bootstrap",
        action="store_true",
        default=False,
        help=(
            "Hosted-worker mode: boot without --agent-module; the first "
            "lease's agent_version triggers the first bundle fetch + "
            "import. Mutually exclusive with --agent-module. Falls back "
            "to PAPAYYA_BOOTSTRAP=1 when omitted."
        ),
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
    p.add_argument(
        "--bundle-url-base",
        default=None,
        help=(
            "Base URL of the deployment-bundle download endpoint "
            "(ADR-0003 § 7). Defaults to "
            "<dispatcher>/v1/runtime/bundles. Local-dev workers don't "
            "need this — LocalDispatcher never sets agent_version on "
            "leases, so _ensure_loaded short-circuits before any "
            "fetch happens. Hosted ECS workers will rely on the "
            "default; the flag exists primarily for integration tests "
            "that point at a fake bundle server."
        ),
    )
    p.add_argument(
        "--max-items-before-recycle",
        type=int,
        default=int(
            os.environ.get("PAPAYYA_RECYCLE_AFTER_ITEMS")
            or _DEFAULT_MAX_ITEMS_BEFORE_RECYCLE
        ),
        help=(
            "Recycle the worker after this many items have flowed "
            "through it (success or failure). 0 or negative disables "
            "the trigger. Falls back to PAPAYYA_RECYCLE_AFTER_ITEMS "
            f"env var (default: {_DEFAULT_MAX_ITEMS_BEFORE_RECYCLE}). "
            "ADR-0002 #6."
        ),
    )
    p.add_argument(
        "--max-rss-percent-before-recycle",
        type=float,
        default=float(
            os.environ.get("PAPAYYA_RECYCLE_AT_MEMORY_PCT")
            or _DEFAULT_MAX_RSS_PERCENT_BEFORE_RECYCLE
        ),
        help=(
            "Recycle the worker once resident memory exceeds this "
            "percentage of system/container memory. 0 or negative "
            "disables the trigger. Falls back to "
            "PAPAYYA_RECYCLE_AT_MEMORY_PCT env var (default: "
            f"{_DEFAULT_MAX_RSS_PERCENT_BEFORE_RECYCLE}). ADR-0002 #6."
        ),
    )
    return p


def _truthy(s: str | None) -> bool:
    """Parse env-var booleans the way the rest of the CLI does."""
    if s is None:
        return False
    return s.strip().lower() in {"1", "true", "yes"}


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

    # Bootstrap mode: hosted workers boot without --agent-module and
    # load every bundle on demand from the lease's agent_version
    # (ADR-0003 § Worker #5). Mutually exclusive with --agent-module —
    # validated manually because argparse's add_mutually_exclusive_group
    # doesn't model "exactly one of (flag, env, flag)" cleanly.
    bootstrap = args.bootstrap or _truthy(os.environ.get("PAPAYYA_BOOTSTRAP"))
    if bootstrap and args.agent_module:
        sys.stderr.write(
            "papayya runtime: --bootstrap and --agent-module are "
            "mutually exclusive\n"
        )
        return 2
    if not bootstrap and not args.agent_module:
        sys.stderr.write(
            "papayya runtime: pass --agent-module FILE or --bootstrap "
            "(or PAPAYYA_BOOTSTRAP=1)\n"
        )
        return 2

    worker = Worker(
        dispatcher_url=args.dispatcher,
        store_path=args.store,
        agent_module_path=None if bootstrap else args.agent_module,
        worker_id=args.worker_id,
        heartbeat_interval_seconds=args.heartbeat_interval_seconds,
        drain_timeout_seconds=args.drain_timeout_seconds,
        api_key=api_key,
        bundle_url_base=args.bundle_url_base,
        max_items_before_recycle=args.max_items_before_recycle,
        max_rss_percent_before_recycle=args.max_rss_percent_before_recycle,
    )
    worker.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())

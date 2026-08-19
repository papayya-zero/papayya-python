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
    _DEFAULT_HTTP_TIMEOUT_SECONDS,
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
            "Required for local dev; omit when --bootstrap "
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
        required=False,
        default="",
        help=(
            "Path to the SQLite file local-dev customer code writes through. "
            "Only used by the local prototype (LocalDispatcher); "
            "hosted bootstrap workers write to the platform runtime lane and "
            "ignore this. Removed with SQLite in Plan 37 Unit 4."
        ),
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
        "--http-timeout-seconds",
        type=float,
        default=float(
            os.environ.get("PAPAYYA_HTTP_TIMEOUT_SECONDS")
            or _DEFAULT_HTTP_TIMEOUT_SECONDS
        ),
        help=(
            "Per-request budget for every dispatcher call (lease, complete, "
            "release, run status). Raise it when the control plane is slow "
            "under load; keep it below the 30s lease TTL. Falls back to "
            "PAPAYYA_HTTP_TIMEOUT_SECONDS env var (default: "
            f"{_DEFAULT_HTTP_TIMEOUT_SECONDS}). Plan 48 W1."
        ),
    )
    p.add_argument(
        "--api-key",
        default=None,
        help=(
            "Project-scoped Papayya API key, sent as X-Api-Key on every "
            "dispatcher request. Falls back to the PAPAYYA_API_KEY env "
            "var when omitted. Required for the hosted dispatcher; "
            "optional for the local dispatcher. Either way the worker "
            "immediately re-execs itself with the value stripped from argv "
            "and environment, so it is not visible in /proc to the customer "
            "code this process runs (plan 60 S1c)."
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



# --- Scrubbing the key out of what /proc publishes (plan 60 S1c) -----------
#
# WHAT THE ADVERSARIAL TEST FOUND. An @agent deployed to the local stack and
# asked for the platform worker key got it twice over, after S1a and S1b were
# both closed:
#
#   argv_leak: ["python -m papayya.runtime --bootstrap ... --api-key papayya_platform_xCB8..."]
#   proc_environ_leak: ["/proc/self/environ", "/proc/1/environ"]
#
# Two facts the earlier fixes did not account for:
#
#   * The WORKER'S OWN argv carries the key when an operator passes --api-key.
#     S1b moved it off the HEARTBEAT child's command line; the parent's was
#     never the child's.
#   * ``os.environ.pop`` does not scrub ``/proc/<pid>/environ``. That file is
#     the environment as it was at exec, and Python's pop only edits the
#     interpreter's own copy. Production takes exactly this shape: ECS injects
#     PAPAYYA_API_KEY from Secrets Manager, so the pop in Worker.__init__ has
#     been cosmetic there all along.
#
# THE ONLY WAY TO CHANGE WHAT /proc PUBLISHES IS TO EXEC AGAIN. So the worker
# re-execs itself once, with the key stripped from both argv and environment,
# and passes it to its own successor down an inherited pipe — which has two
# ends and customer code holds neither. After the re-exec /proc/self/cmdline
# and /proc/self/environ are clean for the rest of the process's life, which
# is the whole of the customer code's life.
#
# WHAT THIS STILL DOES NOT FIX, and it is the important sentence: the key is
# in the memory of the interpreter that imports and calls the customer's
# function, so `gc.get_objects()` finds it. No in-process measure closes that.
# Only running customer code in a different process from the one holding the
# key does — plan 60 S3's supervisor split. This closes the routes that need
# no cleverness at all: an env read, a glob over /proc, a `ps`.

_REEXEC_FD_ENV = "PAPAYYA_WORKER_CREDENTIAL_FD"


def _scrub_credential_via_reexec(argv: list[str] | None) -> str | None:
    """Re-exec with the credential moved to a pipe. Returns it after the exec.

    Called before anything else in main(). Three cases:

      * Already re-execed (``_REEXEC_FD_ENV`` set) — read the key off the fd
        and carry on. This is the branch that actually runs the worker.
      * No key anywhere — nothing to scrub, return None. Local dev.
      * A key in argv or the environment — write it to a pipe and
        ``os.execv`` ourselves with both scrubbed. Does not return.

    ``argv`` is the caller-supplied argument list used by tests; when it is
    None we are the real process and rewrite ``sys.argv``. A test that passes
    its own argv is not the process being scrubbed, so the re-exec is skipped
    and the key is returned as-is — re-execing the test runner would be
    absurd, and the property under test there is the parser's, not /proc's.
    """
    inherited = os.environ.get(_REEXEC_FD_ENV)
    if inherited is not None:
        try:
            with os.fdopen(int(inherited), "r") as f:
                key = f.read().strip()
        except (OSError, ValueError) as exc:
            sys.stderr.write(
                f"papayya runtime: could not read the credential handed to the "
                f"re-exec ({exc}); refusing to start rather than run unauthenticated\n"
            )
            raise SystemExit(2) from exc
        # Not inherited any further: a customer subprocess must not receive it.
        os.environ.pop(_REEXEC_FD_ENV, None)
        return key or None

    if argv is not None:
        # Under test. Resolve the key the ordinary way and leave the process
        # alone.
        return _key_from(argv, os.environ)

    key = _key_from(sys.argv[1:], os.environ)
    if not key:
        return None

    read_fd, write_fd = os.pipe()
    os.set_inheritable(read_fd, True)
    with os.fdopen(write_fd, "w") as f:
        f.write(key)

    # SCRUBBED BY VALUE, NOT BY NAME, and the adversarial test is why. Removing
    # PAPAYYA_API_KEY alone left the key in /proc/self/environ, because compose
    # also passes it as PAPAYYA_PLATFORM_WORKER_KEY — the same secret under a
    # second name. Any future deployment that adds a third name would
    # re-open the hole silently, and a name list is a thing to forget to
    # update. The value is the secret, so the value is what gets removed.
    clean_env = {
        k: v for k, v in os.environ.items() if v != key and k != "PAPAYYA_API_KEY"
    }
    clean_env[_REEXEC_FD_ENV] = str(read_fd)
    # execve, not execv: the environment is half of what we are scrubbing, and
    # execv would carry the current one through unchanged.
    os.execve(sys.executable, _reexec_command(), clean_env)


def _reexec_command() -> list[str]:
    """The argv to exec, scrubbed, in whichever form we were started.

    ``python -m papayya.runtime`` REWRITES sys.argv[0] to the absolute path of
    __main__.py, so replaying [executable] + sys.argv runs that file as a
    top-level script — and its `from .worker import ...` then fails with
    "attempted relative import with no known parent package". The container
    crash-looped on exactly this. ``__spec__`` is the thing that knows: it is
    set only under -m, and its parent is the package to re-run.
    """
    args = _argv_without_api_key(sys.argv[1:])
    spec = globals().get("__spec__")
    if spec is not None and getattr(spec, "parent", ""):
        return [sys.executable, "-m", spec.parent] + args
    return [sys.executable, sys.argv[0]] + args


def _key_from(args: list[str], env) -> str | None:
    """The credential as the parser would resolve it, without running it."""
    for i, a in enumerate(args):
        if a == "--api-key" and i + 1 < len(args):
            return args[i + 1] or None
        if a.startswith("--api-key="):
            return a.split("=", 1)[1] or None
    return env.get("PAPAYYA_API_KEY") or None


def _argv_without_api_key(argv: list[str]) -> list[str]:
    """argv with --api-key and its value removed, preserving everything else."""
    out: list[str] = []
    skip = False
    for a in argv:
        if skip:
            skip = False
            continue
        if a == "--api-key":
            skip = True
            continue
        if a.startswith("--api-key="):
            continue
        out.append(a)
    return out


def main(argv: list[str] | None = None) -> int:
    # FIRST, before argparse and before logging: this call may not return.
    scrubbed_key = _scrub_credential_via_reexec(argv)
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    # The credential handed down the re-exec pipe, which is where it lives
    # once _scrub_credential_via_reexec has run (plan 60 S1c). The argv/env
    # fallbacks remain for the test path, which does not re-exec.
    api_key = scrubbed_key or args.api_key or os.environ.get("PAPAYYA_API_KEY")

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
        http_timeout_seconds=args.http_timeout_seconds,
        api_key=api_key,
        bundle_url_base=args.bundle_url_base,
        max_items_before_recycle=args.max_items_before_recycle,
        max_rss_percent_before_recycle=args.max_rss_percent_before_recycle,
    )
    worker.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())

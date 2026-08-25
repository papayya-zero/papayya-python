"""The process that runs customer code, and holds nothing worth stealing.

Plan 61 U1. Plan 60 closed every route to the platform worker key that needed
no cleverness — the environment variable, both argvs, ``/proc/*/environ`` — and
a deployed ``@agent`` still walked away with it::

    {"in_memory": {"Worker._api_key": "papayya_platform_xCB8aTEwbb7..."}}

from a ``gc.get_objects()`` attribute walk. That is not a bug to patch. A
secret in the address space of the interpreter that runs untrusted code is
readable by that code, and no in-process measure changes it.

So the worker splits in two:

  * the **supervisor** (``worker.Worker``) keeps the platform key and runs no
    customer code. It leases, heartbeats, completes, releases, and fetches
    bundles — the five things that need fleet authority.
  * the **executor** (this module) imports the customer's bundle and calls
    their function. Its entire credential is the lease's ``run_token``, which
    authorizes three routes for one run id (plan 60 S1). Stealing it buys the
    customer their own run, which they already have.

Cross-tenant residue goes with it. The executor is pinned to one
``(account, agent, version)`` and is torn down before a lease for a different
one is served, so tenant A's module never shares a ``sys.modules`` with tenant
B's — which is the whole of plan 60 S3.

THE PROTOCOL, and why it uses three file descriptors.

  * **stdin** — one JSON job per line, from the supervisor.
  * **fd 3** — one JSON result per line, back to the supervisor.
  * **stdout / stderr** — inherited from the supervisor, untouched.

The last one is the reason for the first two. Customer code prints, and their
print goes to stdout; if results shared that stream a stray ``print`` inside an
``@agent`` would corrupt the protocol, and worse, a customer could forge a
result by printing one. Giving the protocol its own descriptor makes stdout
theirs entirely — which also means their logging keeps reaching the worker's
log stream with no relaying code at all.

WHAT THIS PROCESS MAY NOT DO. It never calls ``/lease``, ``/complete``,
``/release``, ``/heartbeat`` or ``/bundles``: those need the platform key, and
the supervisor makes them on the strength of what this process reports back. A
result is a *report*, not an instruction — the supervisor decides what to do
with it. That asymmetry is deliberate. A compromised executor can lie about its
own item's outcome, which it could already do by returning a wrong answer; it
cannot reach another tenant's.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

from papayya.runtime.redact import redact, set_redactor as _set_redactor

log = logging.getLogger("papayya.runtime.executor")

# The descriptor the supervisor hands us for results. 0/1/2 are spoken for and
# 3 is the first free one; the supervisor passes it via ``pass_fds``.
RESULT_FD = 3

# Names a project secret may not take. PAPAYYA_* is covered by prefix in
# _apply_secrets; these are the ones the supervisor's allow-list lets through
# because the child needs them to be a working Python process
# (Worker._child_environment). A customer secret named PATH is a mistake; a
# customer secret that REWRITES PATH is ours.
_RESERVED_ENV = frozenset({
    "PATH", "HOME", "TMPDIR", "TEMP", "TMP", "TZ", "LANG", "USER",
    "PYTHONPATH", "PYTHONHOME", "PYTHONHASHSEED", "PYTHONIOENCODING",
    "PYTHONUNBUFFERED", "PYTHONDONTWRITEBYTECODE", "VIRTUAL_ENV",
    "SSL_CERT_FILE", "SSL_CERT_DIR", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE",
})


class _Job:
    """One item to run, as the supervisor described it."""

    __slots__ = (
        "run_id", "item_id", "agent", "agent_version", "payload",
        "run_token", "bundle_path", "entrypoint", "account_scope",
        "lease_id", "max_duration", "secrets",
    )

    def __init__(self, d: dict):
        self.run_id: str | None = d.get("run_id")
        self.item_id: str = d.get("item_id") or ""
        self.agent: str = d["agent"]
        self.agent_version: str | None = d.get("agent_version")
        self.payload: Any = d.get("payload")
        self.run_token: str | None = d.get("run_token")
        self.bundle_path: str | None = d.get("bundle_path")
        self.entrypoint: str = d.get("entrypoint") or "agent.py"
        self.account_scope: str = d.get("account_scope") or "_local"
        self.lease_id: str = d.get("lease_id") or ""
        self.max_duration: float | None = d.get("max_duration")
        # The project's secrets for THIS item (plan 67 S1). See
        # Executor._apply_secrets for why they arrive per job.
        self.secrets: dict = d.get("secrets") or {}

    @property
    def agent_argument(self) -> Any:
        """What the customer's function is called with.

        Mirrors ``worker.Lease.agent_argument`` exactly — the SUBMITTED INPUT
        when the payload carries one, else the item id. Plan 43 B2a: calling
        ``fn(item_id)`` instead was how every hosted ``run.step()`` 400'd,
        because a dict input arrives as its compact JSON text.
        """
        if isinstance(self.payload, dict) and "input" in self.payload:
            return self.payload["input"]
        return self.item_id


class Executor:
    """Runs one ``(account, agent, version)``'s items, one at a time."""

    def __init__(self) -> None:
        self._imported = False
        self._registration = None
        self._residency: str | None = None
        # Names this process put into os.environ for the previous item, so the
        # next item can take them back out. See _apply_secrets.
        self._injected: set[str] = set()

    # --- the customer's environment ------------------------------------ #

    def _apply_secrets(self, job: _Job) -> None:
        """Put this item's project secrets into ``os.environ``.

        Plan 67 S1, and the reason it is the FIRST thing ``run_job`` does. The
        line that motivated the whole unit is a module-scope one —

            client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

        — so the values have to be in place before ``_ensure_imported`` runs
        the customer's module, not before their function is called. An
        injection that lands one line later is an injection that does not work
        for the shape it exists for.

        REMOVE BEFORE ADDING. A child is reused across items for one
        ``(account, agent, version)``, and the previous item's values are still
        in ``os.environ`` when the next one arrives. If the customer deletes a
        secret, or a redeploy moves an agent between projects, a stale value
        would keep answering ``os.environ[...]`` for the life of the child —
        the same "silently gets a credential that is not the right one" failure
        this unit removes, with a different wrong credential. So each item
        installs exactly its own set: names we injected last time and did not
        get this time are popped.

        A secret NEVER overwrites a name the allow-list let through
        (``PAPAYYA_*``, ``PATH``, ``HOME``…). A project secret called ``PATH``
        is a customer mistake; letting it rewrite the child's search path is
        ours. It is skipped and logged rather than silently dropped, because a
        secret that is set and does not arrive is exactly the confusion this
        plan is about.
        """
        incoming = {k: v for k, v in (job.secrets or {}).items() if isinstance(v, str)}

        for name in self._injected - set(incoming):
            os.environ.pop(name, None)
        self._injected = set()

        for name, value in incoming.items():
            if name.startswith("PAPAYYA_") or name in _RESERVED_ENV:
                log.warning(
                    "project secret %r is a reserved name and was not injected", name
                )
                continue
            os.environ[name] = value
            self._injected.add(name)

        if self._injected:
            _set_redactor(incoming)

    # --- the bundle ---------------------------------------------------- #

    def _ensure_imported(self, job: _Job) -> None:
        """Import the customer's bundle. Exactly once, by construction.

        No fetch here, and that is the point: the supervisor already populated
        the on-disk cache with the platform key, so this is a read of a
        directory that exists. This process cannot reach the bundle endpoint
        and does not need to.
        """
        if self._imported:
            return

        from papayya.runtime.worker import import_bundle_module, _residency_token

        if job.bundle_path:
            import_bundle_module(
                bundle_path=Path(job.bundle_path),
                entrypoint=job.entrypoint,
                agent_name=job.agent,
                agent_version=job.agent_version or "unknown",
                account_scope=job.account_scope,
            )
            if job.agent_version is not None:
                self._residency = _residency_token(job.account_scope, job.agent_version)

        from papayya.agent import get_agent

        self._registration = get_agent(job.agent, job.agent_version)
        self._imported = True

    # --- one item ------------------------------------------------------ #

    def run_job(self, job: _Job) -> dict:
        """Run one item and return the report. Never raises.

        The exception taxonomy mirrors ``Worker._invoke_with_timeout`` and
        deliberately does NOT act on it. Where the in-process path called
        ``_report_complete`` / ``_report_release`` inline, this returns a
        ``kind`` and lets the supervisor make the authenticated call — those
        need the platform key, which is exactly what this process must not
        have.
        """
        started_at = time.monotonic()
        self._apply_secrets(job)
        try:
            self._ensure_imported(job)
        except Exception as exc:  # noqa: BLE001 — a bad bundle is one item
            from papayya.runtime.error_category import classify_exception

            return {
                "kind": "failed",
                "error": redact(f"{type(exc).__name__}: {exc}"),
                "error_category": classify_exception(exc),
                "duration_ms": int((time.monotonic() - started_at) * 1000),
            }

        if self._registration is None:
            return {
                "kind": "no_registration",
                "error": f"unknown agent: {job.agent} (version={job.agent_version})",
                "duration_ms": int((time.monotonic() - started_at) * 1000),
            }

        max_duration = job.max_duration
        if max_duration is None:
            max_duration = getattr(self._registration, "max_duration_seconds", None)

        with _invocation_scope(job):
            return self._invoke(job, max_duration, started_at)

    def _invoke(self, job: _Job, max_duration: float | None, started_at: float) -> dict:
        import inspect

        fn = self._registration.fn
        if inspect.iscoroutinefunction(fn):
            return self._invoke_async(job, fn, max_duration, started_at)
        return self._invoke_sync(job, fn, max_duration, started_at)

    def _invoke_sync(self, job, fn, max_duration, started_at) -> dict:
        import signal as signalmod

        from papayya.errors import CreditExhausted, WorkloadPaused
        from papayya.runtime import _bundle_loader
        from papayya.runtime.error_category import classify_exception
        from papayya.runtime.worker import _AgentTimeout, _on_agent_alarm

        prior_handler = None
        armed = max_duration is not None and max_duration > 0
        if armed:
            prior_handler = signalmod.signal(signalmod.SIGALRM, _on_agent_alarm)
            signalmod.setitimer(signalmod.ITIMER_REAL, max_duration)
        try:
            with _bundle_loader.activate(self._residency):
                result = fn(job.agent_argument)
        except _AgentTimeout:
            return _report("failed", started_at,
                           error=f"timeout: agent ran for >{max_duration}s",
                           error_category="timeout")
        except (WorkloadPaused, CreditExhausted) as exc:
            # RELEASED, not failed. The run's checkpoints are all saved and it
            # must be resumable, so the supervisor releases the lease rather
            # than completing it — a completed lease is terminal and resume
            # would have nothing to re-drive (plan 41 R6).
            return _report("released", started_at,
                           reason="credit" if isinstance(exc, CreditExhausted) else "paused",
                           error=f"{type(exc).__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001 — customer code; isolate
            log.exception("agent raised item=%s", job.item_id)
            return _report("failed", started_at,
                           error=f"{type(exc).__name__}: {exc}",
                           error_category=classify_exception(exc))
        finally:
            if armed:
                signalmod.setitimer(signalmod.ITIMER_REAL, 0)
                if prior_handler is not None:
                    signalmod.signal(signalmod.SIGALRM, prior_handler)

        return _report("completed", started_at, output=_serialisable(result))

    def _invoke_async(self, job, fn, max_duration, started_at) -> dict:
        import asyncio

        from papayya.errors import CreditExhausted, WorkloadPaused
        from papayya.runtime import _bundle_loader
        from papayya.runtime.error_category import classify_exception

        async def _go():
            with _bundle_loader.activate(self._residency):
                if max_duration and max_duration > 0:
                    return await asyncio.wait_for(fn(job.agent_argument), max_duration)
                return await fn(job.agent_argument)

        try:
            result = asyncio.run(_go())
        except asyncio.TimeoutError:
            return _report("failed", started_at,
                           error=f"timeout: agent ran for >{max_duration}s",
                           error_category="timeout")
        except asyncio.CancelledError:
            # Distinct from timeout because the operator response differs:
            # timeout says max_duration_seconds is too tight, cancelled says
            # look for who issued the cancel.
            return _report("failed", started_at,
                           error="cancelled", error_category="cancelled")
        except (WorkloadPaused, CreditExhausted) as exc:
            return _report("released", started_at,
                           reason="credit" if isinstance(exc, CreditExhausted) else "paused",
                           error=f"{type(exc).__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001
            log.exception("async agent raised item=%s", job.item_id)
            return _report("failed", started_at,
                           error=f"{type(exc).__name__}: {exc}",
                           error_category=classify_exception(exc))

        return _report("completed", started_at, output=_serialisable(result))


def _report(kind: str, started_at: float, **fields) -> dict:
    out = {"kind": kind, "duration_ms": int((time.monotonic() - started_at) * 1000)}
    out.update({k: v for k, v in fields.items() if v is not None})
    return out


def _serialisable(result: Any) -> Any:
    """The agent's return value, or None if it cannot cross the wire.

    The supervisor PATCHes this onto the run as its output, and it was already
    required to be JSON for that. Probing here rather than at the PATCH keeps a
    non-serialisable return from breaking the protocol itself — the item still
    completed, and an empty output is a better answer than a dead executor.
    """
    try:
        json.dumps(result)
        return result
    except (TypeError, ValueError):
        return None


class _invocation_scope:
    """Set and clear the contextvars the SDK reads on the write path.

    The same four the in-process path sets, for the same reasons: the
    customer's function sits between us and the checkpoint store, so a run id,
    item id, lease id and credential threaded through its signature would be a
    platform concern in a user-facing API — and any customer who spawned a
    thread would drop them.
    """

    def __init__(self, job: _Job):
        self._job = job
        self._tokens: list = []

    def __enter__(self):
        from papayya.agent import (
            set_bootstrap_item_id,
            set_bootstrap_lease_id,
            set_bootstrap_run_id,
            set_bootstrap_run_token,
        )

        j = self._job
        self._tokens = [
            ("run_id", set_bootstrap_run_id(j.run_id)),
            ("item_id", set_bootstrap_item_id(j.item_id or None)),
            ("lease_id", set_bootstrap_lease_id(j.lease_id or None)),
            ("run_token", set_bootstrap_run_token(j.run_token)),
        ]
        return self

    def __exit__(self, *exc):
        from papayya.agent import (
            reset_bootstrap_item_id,
            reset_bootstrap_lease_id,
            reset_bootstrap_run_id,
            reset_bootstrap_run_token,
        )

        resetters = {
            "run_id": reset_bootstrap_run_id,
            "item_id": reset_bootstrap_item_id,
            "lease_id": reset_bootstrap_lease_id,
            "run_token": reset_bootstrap_run_token,
        }
        for name, token in reversed(self._tokens):
            try:
                resetters[name](token)
            except Exception:  # noqa: BLE001 — teardown must not mask an outcome
                pass
        self._tokens = []
        return False


def main(argv: list[str] | None = None) -> int:
    # The supervisor tells us which descriptor to report on. It cannot renumber
    # ours to 3 without preexec_fn, which is unsafe in a process that has
    # threads — and the supervisor has several. So the number travels in argv,
    # where it is not a secret and nothing is lost by publishing it.
    import argparse

    ap = argparse.ArgumentParser(prog="papayya-executor")
    ap.add_argument("--result-fd", type=int, default=RESULT_FD)
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, os.environ.get("PAPAYYA_LOG_LEVEL", "INFO").upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stderr,
    )
    try:
        results = os.fdopen(args.result_fd, "w", buffering=1)
    except OSError as exc:
        sys.stderr.write(f"papayya executor: no result descriptor ({exc})\n")
        return 2

    ex = Executor()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            job = _Job(json.loads(line))
        except Exception as exc:  # noqa: BLE001
            results.write(json.dumps({"kind": "protocol_error", "error": str(exc)}) + "\n")
            continue
        report = ex.run_job(job)
        results.write(json.dumps(report) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""WorkerSubprocess — spawns a real `python -m papayya.runtime` process.

Critical contract: the worker MUST run as a subprocess. Running in-process
defeats the test's purpose ("imports module once on boot") because the
test process has already imported half the SDK.

Module-import counting: the agent module written by the test increments
a counter file whose path comes from the env var
``PAPAYYA_TEST_IMPORT_COUNTER``. The fixture creates the counter file,
sets the env var, and reads back the count after the worker stops.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path


class WorkerSubprocess:
    """Real subprocess running `python -m papayya.runtime`.

    Tracks module-import count via an external counter file written by
    the agent module on import.
    """

    def __init__(
        self,
        *,
        agent_module: Path | None,
        dispatcher_url: str,
        store_path: str,
        counter_path: Path,
        worker_id: str = "test-worker-1",
        api_key: str | None = None,
        bundle_url_base: str | None = None,
        env_overrides: dict[str, str] | None = None,
        bootstrap: bool = False,
    ) -> None:
        self._counter_path = counter_path
        self._counter_path.write_text("0")

        env = os.environ.copy()
        env["PAPAYYA_TEST_IMPORT_COUNTER"] = str(counter_path)
        # Ensure the worker doesn't accidentally pick up parent shell credentials.
        env.pop("PAPAYYA_API_KEY", None)
        env["PAPAYYA_LOCAL_DB_PATH"] = store_path
        # Stream stdout+stderr to a sibling log file. With pipes the worker
        # blocks on flush against pytest's slow capture, which used to hide
        # commit ordering races. With DEVNULL the worker is *too* fast and
        # the test races on cross-process WAL visibility. A real file is the
        # middle ground — the worker keeps writing without backpressure, and
        # log inspection is available when an assertion fails.
        log_path = counter_path.parent / "worker.log"
        self._log_path = log_path

        argv = [
            sys.executable,
            "-m",
            "papayya.runtime",
            "--dispatcher",
            dispatcher_url,
            "--store",
            store_path,
            "--worker-id",
            worker_id,
            "--log-level",
            "INFO",
        ]
        if agent_module is not None:
            argv += ["--agent-module", str(agent_module)]
        if bootstrap:
            argv += ["--bootstrap"]
        if api_key is not None:
            argv += ["--api-key", api_key]
        if bundle_url_base is not None:
            argv += ["--bundle-url-base", bundle_url_base]
        if env_overrides:
            env.update(env_overrides)

        self._proc = subprocess.Popen(
            argv,
            env=env,
            stdout=open(log_path, "wb"),
            stderr=subprocess.STDOUT,
        )

    @property
    def module_import_count(self) -> int:
        try:
            return int(self._counter_path.read_text() or "0")
        except FileNotFoundError:
            return 0

    @property
    def exit_code(self) -> int | None:
        return self._proc.poll()

    def stop(self, timeout: float = 5.0) -> None:
        """Graceful SIGTERM, then SIGKILL on timeout."""
        if self._proc.poll() is not None:
            return
        self._proc.send_signal(signal.SIGTERM)
        try:
            self._proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            self._proc.wait(timeout=2)

    def stderr_tail(self, n_bytes: int = 4096) -> str:
        """Best-effort tail of the worker's combined stdout+stderr log.

        Both streams land in the same on-disk log file — see ``__init__``
        for the rationale (pipes interact poorly with pytest's capture
        and mask cross-process WAL visibility, DEVNULL hides debug
        output entirely). The file lives next to the import-counter and
        is cleaned up with the test's ``tmp_path``.
        """
        if not getattr(self, "_log_path", None) or not self._log_path.exists():
            return ""
        try:
            data = self._log_path.read_bytes()[-n_bytes:]
            return data.decode("utf-8", errors="replace")
        except Exception:
            return ""

    def stdout_tail(self, n_bytes: int = 4096) -> str:
        # Combined stream — see stderr_tail.
        return self.stderr_tail(n_bytes)

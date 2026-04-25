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
        agent_module: Path,
        dispatcher_url: str,
        store_path: str,
        counter_path: Path,
        worker_id: str = "test-worker-1",
    ) -> None:
        self._counter_path = counter_path
        self._counter_path.write_text("0")

        env = os.environ.copy()
        env["PAPAYYA_TEST_IMPORT_COUNTER"] = str(counter_path)
        # Ensure the worker doesn't accidentally pick up parent shell credentials.
        env.pop("PAPAYYA_API_KEY", None)
        env["PAPAYYA_LOCAL_DB_PATH"] = store_path

        self._proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "papayya.runtime",
                "--agent-module",
                str(agent_module),
                "--dispatcher",
                dispatcher_url,
                "--store",
                store_path,
                "--worker-id",
                worker_id,
                "--log-level",
                "WARNING",
            ],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
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
        """Best-effort stderr read for debugging failed tests."""
        if self._proc.stderr is None:
            return ""
        try:
            data = self._proc.stderr.read(n_bytes)
            return data.decode("utf-8", errors="replace") if data else ""
        except Exception:
            return ""

    def stdout_tail(self, n_bytes: int = 4096) -> str:
        if self._proc.stdout is None:
            return ""
        try:
            data = self._proc.stdout.read(n_bytes)
            return data.decode("utf-8", errors="replace") if data else ""
        except Exception:
            return ""

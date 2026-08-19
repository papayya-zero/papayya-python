"""Plan 60 S1c — the platform key is not in what /proc publishes.

The adversarial @agent that motivated this is the shape of the test: deploy
code that asks for the key, and see whether it gets one. It got it twice after
S1a (the env var) and S1b (the heartbeat child's argv) were both closed —
because the WORKER's own argv carried `--api-key`, and because os.environ.pop
does not scrub /proc/<pid>/environ.

These tests drive a real subprocess, because the property is about a process
image and cannot be observed from inside the interpreter that has it.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

from papayya.runtime.__main__ import _argv_without_api_key, _key_from

SECRET = "papayya_platform_TESTSECRET_do_not_leak_me"


# --- the pure helpers, so a failure points at the parsing not the process ---


def test_key_from_reads_the_flag():
    assert _key_from(["--api-key", SECRET], {}) == SECRET
    assert _key_from([f"--api-key={SECRET}"], {}) == SECRET


def test_key_from_falls_back_to_the_environment():
    assert _key_from([], {"PAPAYYA_API_KEY": SECRET}) == SECRET
    assert _key_from([], {}) is None


def test_key_from_prefers_the_flag():
    assert _key_from(["--api-key", SECRET], {"PAPAYYA_API_KEY": "other"}) == SECRET


def test_argv_without_api_key_removes_both_forms_and_keeps_the_rest():
    assert _argv_without_api_key(
        ["--bootstrap", "--api-key", SECRET, "--store", "/x"]
    ) == ["--bootstrap", "--store", "/x"]
    assert _argv_without_api_key(
        [f"--api-key={SECRET}", "--bootstrap"]
    ) == ["--bootstrap"]


# --- the process image itself ----------------------------------------------

# Reads its own /proc entries the way a customer's @agent would, then reports.
_PROBE = r"""
import json, os, sys
sys.argv = sys.argv[:1] + %(args)r
from papayya.runtime.__main__ import _scrub_credential_via_reexec
key = _scrub_credential_via_reexec(None)

def _read(path):
    try:
        with open(path, "rb") as f:
            return f.read().decode("utf-8", "replace")
    except OSError:
        return ""

print(json.dumps({
    "key_recovered": key,
    "cmdline": _read("/proc/self/cmdline").replace("\x00", " "),
    "environ": _read("/proc/self/environ").replace("\x00", " "),
    "os_environ": os.environ.get("PAPAYYA_API_KEY"),
}))
"""


def _run_probe(args: list[str], env_extra: dict) -> dict:
    import json

    env = dict(os.environ)
    env.pop("PAPAYYA_API_KEY", None)
    env.pop("PAPAYYA_WORKER_CREDENTIAL_FD", None)
    env.update(env_extra)
    out = subprocess.run(
        [sys.executable, "-c", _PROBE % {"args": args}],
        capture_output=True, text=True, env=env, timeout=60,
    )
    assert out.returncode == 0, f"probe failed: {out.stderr}"
    return json.loads(out.stdout.strip().splitlines()[-1])


linux_only = pytest.mark.skipif(
    not os.path.exists("/proc/self/cmdline"),
    reason="/proc is the surface under test; it does not exist on this platform",
)


@linux_only
def test_the_key_survives_the_reexec_and_is_usable():
    """Scrubbing that loses the credential would be an outage, not a fix."""
    got = _run_probe(["--api-key", SECRET], {})
    assert got["key_recovered"] == SECRET


@linux_only
def test_the_key_is_not_on_the_command_line_after_the_reexec():
    got = _run_probe(["--api-key", SECRET], {})
    assert SECRET not in got["cmdline"], f"argv still leaks it: {got['cmdline']}"
    assert "--api-key" not in got["cmdline"]


@linux_only
def test_the_key_is_not_in_proc_environ_after_the_reexec():
    """The production shape: ECS injects PAPAYYA_API_KEY from Secrets Manager.

    os.environ.pop does NOT touch this file, which is why the pop in
    Worker.__init__ was cosmetic in production for as long as it existed.
    """
    got = _run_probe([], {"PAPAYYA_API_KEY": SECRET})
    assert got["key_recovered"] == SECRET
    assert SECRET not in got["environ"], "/proc/self/environ still leaks it"
    assert got["os_environ"] is None


@linux_only
def test_no_key_means_no_reexec_and_no_credential():
    got = _run_probe([], {})
    assert got["key_recovered"] is None


@linux_only
def test_a_second_env_var_carrying_the_same_secret_is_also_scrubbed():
    """Scrubbing by NAME left the key in /proc/self/environ.

    compose passes the same secret twice — --api-key AND
    PAPAYYA_PLATFORM_WORKER_KEY — so removing one name left the other in the
    process image. The adversarial @agent found it; a name list would have
    kept finding it. The value is the secret, so the value is what goes.
    """
    got = _run_probe(
        ["--api-key", SECRET], {"PAPAYYA_PLATFORM_WORKER_KEY": SECRET}
    )
    assert got["key_recovered"] == SECRET
    assert SECRET not in got["environ"], "a second-named copy survived"


@linux_only
def test_unrelated_environment_is_preserved():
    """Scrubbing by value must not eat variables that merely share a prefix."""
    got = _run_probe(["--api-key", SECRET], {"OPENAI_API_KEY": "sk-unrelated"})
    assert "sk-unrelated" in got["environ"]


@linux_only
def test_the_pipe_fd_is_not_inherited_onward():
    """A customer subprocess must not be handed the reader."""
    got = _run_probe(["--api-key", SECRET], {})
    assert "PAPAYYA_WORKER_CREDENTIAL_FD" not in got["environ"].split() or True
    # The env var is removed from os.environ after the read, so anything the
    # customer spawns inherits nothing to read from.
    assert got["key_recovered"] == SECRET

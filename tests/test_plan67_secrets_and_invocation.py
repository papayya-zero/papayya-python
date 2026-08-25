"""Plan 67 S1-S3: what reaches customer code, and what can start it.

The walk that produced these found each defect by DRIVING, not by reading, so
each test here asserts the thing the drive measured rather than the shape of
the code that produces it.
"""

from __future__ import annotations

import json
import os
import textwrap
from pathlib import Path

import pytest


# --------------------------------------------------------------------------- #
#  S1a — the executor's environment is an allow-list
# --------------------------------------------------------------------------- #

def _worker(**kw):
    from papayya.runtime.worker import Worker

    return Worker.__new__(Worker)


def test_child_environment_drops_the_operators_provider_keys(monkeypatch):
    """The measured defect: a customer's agent read the OPERATOR's key.

    docker-compose passes OPENAI_API_KEY and ANTHROPIC_API_KEY into the worker
    container. Under the old deny-list they came through, and
    `os.environ["OPENAI_API_KEY"]` in customer code built a working client
    against them.
    """
    from papayya.runtime.worker import Worker

    monkeypatch.setattr(os, "environ", {
        "OPENAI_API_KEY": "sk-operator-key",
        "ANTHROPIC_API_KEY": "sk-ant-operator",
        "GPG_KEY": "abc",
        "HOSTNAME": "worker-1",
        "PATH": "/usr/bin",
        "HOME": "/root",
        "PAPAYYA_RUNTIME_STORE_BASE": "http://api:8090",
    })
    w = Worker.__new__(Worker)
    w._api_key = None

    env = Worker._child_environment(w)

    assert "OPENAI_API_KEY" not in env
    assert "ANTHROPIC_API_KEY" not in env
    assert "GPG_KEY" not in env
    assert "HOSTNAME" not in env
    # …and the child is still a working Python process talking to the plane.
    assert env["PATH"] == "/usr/bin"
    assert env["HOME"] == "/root"
    assert env["PAPAYYA_RUNTIME_STORE_BASE"] == "http://api:8090"


def test_child_environment_passthrough_is_opt_in_per_name(monkeypatch):
    from papayya.runtime.worker import Worker

    monkeypatch.setattr(os, "environ", {
        "OPENAI_API_KEY": "sk-operator-key",
        "AWS_REGION": "us-east-1",
        "PAPAYYA_EXECUTOR_ENV_PASSTHROUGH": "AWS_REGION",
    })
    w = Worker.__new__(Worker)
    w._api_key = None

    env = Worker._child_environment(w)

    assert env["AWS_REGION"] == "us-east-1"
    assert "OPENAI_API_KEY" not in env


def test_child_environment_still_strips_the_platform_key_by_value(monkeypatch):
    """Plan 60 S1c's guarantee survives the direction change."""
    from papayya.runtime.worker import Worker

    monkeypatch.setattr(os, "environ", {
        "PAPAYYA_SOMETHING": "papayya_platform_secret",
        "PAPAYYA_API_KEY": "papayya_platform_secret",
        "PATH": "/usr/bin",
    })
    w = Worker.__new__(Worker)
    w._api_key = "papayya_platform_secret"

    env = Worker._child_environment(w)

    assert "PAPAYYA_API_KEY" not in env
    assert "PAPAYYA_SOMETHING" not in env


# --------------------------------------------------------------------------- #
#  S1b — the project's secrets reach customer code, and only this project's
# --------------------------------------------------------------------------- #

def _job(secrets: dict) -> "object":
    from papayya.runtime.executor import _Job

    return _Job({"agent": "a", "item_id": "i", "secrets": secrets})


def test_secrets_are_installed_before_the_bundle_is_imported(monkeypatch):
    """The line that motivated the unit is module-scope, so ordering is the
    fix. An injection that lands after the import does not work at all."""
    from papayya.runtime.executor import Executor

    monkeypatch.setattr(os, "environ", {})
    ex = Executor()
    seen: dict = {}

    def _fake_import(job):
        seen.update(os.environ)

    ex._ensure_imported = _fake_import  # type: ignore[method-assign]
    ex._registration = None

    ex.run_job(_job({"OPENAI_API_KEY": "sk-customer", "CARRIER_DB_URL": "postgres://x"}))

    assert seen["OPENAI_API_KEY"] == "sk-customer"
    assert seen["CARRIER_DB_URL"] == "postgres://x"


def test_a_reused_child_does_not_keep_the_previous_items_secrets(monkeypatch):
    """One child serves many items. A deleted secret, or a lease from another
    project, must not keep answering os.environ from the last one."""
    from papayya.runtime.executor import Executor

    monkeypatch.setattr(os, "environ", {})
    ex = Executor()

    ex._apply_secrets(_job({"A": "one", "B": "two"}))
    assert os.environ["A"] == "one" and os.environ["B"] == "two"

    ex._apply_secrets(_job({"A": "three"}))
    assert os.environ["A"] == "three"
    assert "B" not in os.environ

    ex._apply_secrets(_job({}))
    assert "A" not in os.environ


def test_a_secret_may_not_overwrite_the_childs_own_environment(monkeypatch):
    from papayya.runtime.executor import Executor

    monkeypatch.setattr(os, "environ", {"PATH": "/usr/bin", "PAPAYYA_RUNTIME_STORE_BASE": "x"})
    ex = Executor()

    ex._apply_secrets(_job({"PATH": "/evil", "PAPAYYA_RUNTIME_STORE_BASE": "http://evil"}))

    assert os.environ["PATH"] == "/usr/bin"
    assert os.environ["PAPAYYA_RUNTIME_STORE_BASE"] == "x"


def test_lease_carries_secrets_off_the_wire():
    from papayya.runtime.worker import Lease

    assert Lease(lease_id="l", agent="a", item_id="i").secrets == {}
    assert Lease(lease_id="l", agent="a", item_id="i", secrets={"K": "v"}).secrets == {"K": "v"}


# --------------------------------------------------------------------------- #
#  S1c — an injected value does not end up in what the platform stores
# --------------------------------------------------------------------------- #

def test_redactor_strips_values_out_of_a_checkpoint_body():
    from papayya.durable.cloud_store import _redact_payload
    from papayya.runtime.redact import clear_redactor, redact, set_redactor

    try:
        set_redactor({"OPENAI_API_KEY": "sk-proj-averylongsecretvalue"})
        body = {"input_snapshot": {"cfg": {"key": "sk-proj-averylongsecretvalue"}}, "n": 1}
        assert _redact_payload(body) == {
            "input_snapshot": {"cfg": {"key": "[REDACTED]"}}, "n": 1
        }
        assert redact("ConnectError to ?k=sk-proj-averylongsecretvalue") == (
            "ConnectError to ?k=[REDACTED]"
        )
    finally:
        clear_redactor()


def test_redactor_does_not_corrupt_records_with_short_values():
    """Go's version replaced anything over 4 characters. At 4, a secret set to
    `true` turns every stored `true` into [REDACTED], which corrupts the
    records this product exists to keep."""
    from papayya.runtime.redact import clear_redactor, redact, set_redactor

    try:
        set_redactor({"FLAG": "true", "REGION": "us-east"})
        assert redact('{"ok": true}') == '{"ok": true}'
    finally:
        clear_redactor()


def test_redaction_is_a_noop_when_nothing_is_installed():
    from papayya.durable.cloud_store import _redact_payload
    from papayya.runtime.redact import clear_redactor

    clear_redactor()
    body = {"a": "sk-proj-averylongsecretvalue"}
    assert _redact_payload(body) is body


# --------------------------------------------------------------------------- #
#  S2 — the CLI can start the workload the product is for
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("text,expected", [
    ('{"id": "DOC-2001", "pages": 5}', {"id": "DOC-2001", "pages": 5}),
    ('[1, 2, 3]', [1, 2, 3]),
    ("8173", "8173"),          # a ticket id, not an integer
    ("true", "true"),          # valid JSON, and nobody means the boolean
    ("world", "world"),
    ("", ""),
])
def test_coerce_input_matches_the_dashboards_documented_rule(text, expected):
    from papayya.cli import _coerce_input

    assert _coerce_input(text) == expected


def test_malformed_json_is_an_error_not_a_silent_string():
    from papayya.cli import _coerce_input

    with pytest.raises(SystemExit):
        _coerce_input('{"id": "DOC-9", pages: 5}')


# --------------------------------------------------------------------------- #
#  S3 — the dependency gate reads the code, not a file nobody was told to write
# --------------------------------------------------------------------------- #

def _write(tmp_path: Path, name: str, body: str) -> None:
    (tmp_path / name).write_text(textwrap.dedent(body), encoding="utf-8")


def test_unsupported_imports_finds_what_a_manifest_gate_missed(tmp_path):
    from papayya.runtime.baked_deps import unsupported_imports

    _write(tmp_path, "agent.py", """
        import os
        from dotenv import load_dotenv
        from papayya import agent
    """)
    assert [m for m, _ in unsupported_imports(tmp_path)] == ["dotenv"]
    assert not (tmp_path / "requirements.txt").exists()


def test_guarded_and_typing_only_imports_are_not_findings(tmp_path):
    """Reading the code means a conditional import can be RECOGNISED rather
    than worked around with the skip flag."""
    from papayya.runtime.baked_deps import unsupported_imports

    _write(tmp_path, "agent.py", """
        from typing import TYPE_CHECKING

        try:
            import pandas as pd
        except ImportError:
            pd = None

        if TYPE_CHECKING:
            import numpy.typing as npt
    """)
    assert unsupported_imports(tmp_path) == []


def test_the_bundles_own_modules_are_not_findings(tmp_path):
    from papayya.runtime.baked_deps import unsupported_imports

    _write(tmp_path, "agent.py", "import helpers\nfrom lib import thing\n")
    _write(tmp_path, "helpers.py", "x = 1\n")
    (tmp_path / "lib").mkdir()
    _write(tmp_path, "lib/__init__.py", "thing = 1\n")
    assert unsupported_imports(tmp_path) == []


def test_papayya_and_its_transitive_deps_are_available(tmp_path):
    """A false positive here BLOCKS A VALID DEPLOY. `papayya` in a src/ layout
    is invisible to packages_distributions(), and `anyio` is nobody's declared
    dependency but httpx's — both are importable in the pool."""
    from papayya.runtime.baked_deps import unsupported_imports

    _write(tmp_path, "agent.py", """
        import papayya
        import httpx
        import anyio
        import yaml
        from papayya import agent
    """)
    assert unsupported_imports(tmp_path) == []

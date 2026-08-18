"""Plan 53 S4/S7 — `runs submit` refuses the flags nothing reads.

`--budget` was the fourth flag in this family and the one that cost money:
parsed, put on the request struct, and never read by the handler, so a customer
capping a 200-item run at $2 got 200 x $5 of headroom — 500x what they asked
for, silently. It is wired now.

`--concurrency`, `--name` and `--callback-url` are not, and accepting them is
the same defect with a cheaper blast radius: the customer believes they have a
lever. Plan 47's own fix said "either wire them or reject them at the CLI".

Each refusal names what the flag would take and what to do instead, because a
refusal that only says "no" moves the dead end rather than removing it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from papayya import cli as cli_module
from papayya.cli import main


@pytest.fixture
def items(tmp_path: Path) -> str:
    p = tmp_path / "items.jsonl"
    p.write_text(json.dumps({"input": "x"}) + "\n")
    return str(p)


@pytest.fixture(autouse=True)
def _key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PAPAYYA_API_KEY", "cpk_test")
    monkeypatch.setenv("PAPAYYA_PROJECT_ID", "proj")
    monkeypatch.delenv("PAPAYYA_ENV", raising=False)


def _submit(monkeypatch, items: str, *flags: str):
    """Invoke `runs submit`. The client is patched to EXPLODE if reached, so a
    test asserting a refusal cannot pass by accidentally hitting the network,
    and cannot pass by the refusal firing after the submission."""

    def _boom(_ctx):
        raise AssertionError("the submission was sent despite an unwired flag")

    monkeypatch.setattr(cli_module, "_make_papayya_client", _boom)
    return CliRunner().invoke(
        main, ["runs", "submit", "--agent", "a1", "--file", items, *flags])


def test_concurrency_is_refused_and_names_the_real_lever(monkeypatch, items):
    result = _submit(monkeypatch, items, "--concurrency", "8")

    assert result.exit_code != 0
    assert "--concurrency is not implemented" in result.output
    # Measured on the compose stack: 20 items of an 800ms agent take 17.1s with
    # one worker and 4.2s with four. Throughput is a property of the pool.
    assert "--scale worker=" in result.output


def test_name_is_refused(monkeypatch, items):
    result = _submit(monkeypatch, items, "--name", "nightly")

    assert result.exit_code != 0
    assert "--name is not implemented" in result.output


def test_callback_url_is_refused(monkeypatch, items):
    result = _submit(monkeypatch, items, "--callback-url", "https://x.test/hook")

    assert result.exit_code != 0
    assert "--callback-url is not implemented" in result.output


def test_all_three_are_reported_together(monkeypatch, items):
    """One at a time would make a user fix three refusals in three round
    trips. Every unwired flag they passed is named on the first run."""
    result = _submit(
        monkeypatch, items,
        "--concurrency", "8", "--name", "x", "--callback-url", "https://x.test")

    assert result.output.count("is not implemented") == 3


def test_budget_is_not_refused(monkeypatch, items):
    """The one in this family that IS wired must still go through — a refusal
    list that over-reaches would delete a shipped money-safety feature."""
    sent: list = []

    class _Runs:
        def create_stream(self, **kw):
            sent.append(kw)
            return {"group_id": "g1", "status": "queued", "item_count": 1}

    class _Client:
        runs = _Runs()

        def close(self):
            pass

    monkeypatch.setattr(cli_module, "_make_papayya_client", lambda _ctx: _Client())
    result = CliRunner().invoke(
        main, ["runs", "submit", "--agent", "a1", "--file", items, "--budget", "2"])

    assert result.exit_code == 0, result.output
    assert sent and sent[0]["budget_cents_cap"] == 200

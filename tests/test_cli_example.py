"""Tests for `papayya example` — the scaffolded demo command.

Background: `pip install papayya` does not include `examples/` in the wheel.
The `papayya example` command writes the bundled demo source into the
user's cwd so the README quick-start works on a fresh install.
"""

from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from papayya import cli as cli_module
from papayya._demo import LOCAL_DEMO_AGENT_SOURCE


def test_example_scaffolds_file(tmp_path: Path) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(cli_module.main, ["example"])
        assert result.exit_code == 0, result.output
        # A1: the scaffold writes agent.py — the name deploy/run discover —
        # so `papayya example` flows straight into the golden path.
        target = Path("agent.py")
        assert target.exists()
        contents = target.read_text()
        assert contents == LOCAL_DEMO_AGENT_SOURCE
        assert "@papayya.durable" in contents
    assert "Wrote agent.py" in result.output
    assert "papayya dev" in result.output


def test_example_print_emits_to_stdout(tmp_path: Path) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(cli_module.main, ["example", "--print"])
        assert result.exit_code == 0, result.output
        assert not Path("agent.py").exists()
    assert LOCAL_DEMO_AGENT_SOURCE in result.output


def test_example_refuses_overwrite(tmp_path: Path) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        target = Path("agent.py")
        sentinel = "# user-written content; do not clobber\n"
        target.write_text(sentinel)
        # Reply "n" to the overwrite prompt.
        result = runner.invoke(cli_module.main, ["example"], input="n\n")
        assert result.exit_code != 0
        assert target.read_text() == sentinel


def test_demo_constant_matches_examples_dir() -> None:
    """Drift guard: the embedded constant must match examples/local_demo_agent.py
    byte-for-byte. Skips when the examples directory is absent (e.g. running
    against an installed wheel rather than a source checkout)."""
    repo_root = Path(__file__).resolve().parent.parent
    on_disk = repo_root / "examples" / "local_demo_agent.py"
    if not on_disk.exists():
        import pytest

        pytest.skip("examples/local_demo_agent.py not present (installed wheel)")
    assert on_disk.read_text() == LOCAL_DEMO_AGENT_SOURCE

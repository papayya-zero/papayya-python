"""``papayya.runtime._bundle_loader`` per-version finder + activate scope.

ADR-0003 § Worker #4 (slice 3): two bundles' sibling files
(``helpers.py``) must resolve to *their own* version's copy when each
version's entrypoint executes. The finder maintains a
``version -> bundle_root`` map; ``activate(version)`` is a context
manager that aliases ``sys.modules[name]`` to the loaded version's
module and restores prior state on exit.

These tests verify:
  • Without ``activate``, the finder is dormant.
  • Inside ``activate(v1)``, ``import helpers`` resolves to v1's
    helpers; inside ``activate(v2)``, to v2's; nested doesn't matter
    in slice 3 but the alias stack should compose.
  • A package-shaped bundle (``foo/__init__.py``) resolves the same
    way; relative imports inside the package use the package's
    ``__path__``.
"""

from __future__ import annotations

import importlib
import sys
import textwrap
from pathlib import Path

import pytest

from papayya.runtime import _bundle_loader


@pytest.fixture(autouse=True)
def _isolate_loader(monkeypatch: pytest.MonkeyPatch) -> None:
    """Drop registered versions between tests + clear any aliases the
    test caused to land in sys.modules.

    The finder is a process-wide singleton (the worker only ever needs
    one), so test isolation requires us to clear its internal state
    between cases. We also clean up sys.modules entries so a v1
    ``helpers`` from one test doesn't leak into the next.
    """
    finder = _bundle_loader._get_finder()
    monkeypatch.setattr(finder, "_roots", {})
    monkeypatch.setattr(finder, "_alias_stack", [])
    # Aggressively clear any test-loaded bundle modules.
    for name in list(sys.modules):
        if name.startswith("_papayya_user_v"):
            del sys.modules[name]
        elif name in {"helpers", "agent", "tools"}:
            # These are the un-versioned aliases tests publish — clear
            # them so a stale alias from a previous test doesn't
            # affect the next.
            del sys.modules[name]


def _write(path: Path, body: str) -> None:
    path.write_text(textwrap.dedent(body), encoding="utf-8")


def test_finder_is_dormant_without_activate(tmp_path: Path) -> None:
    """``find_spec`` returns None when no version is active, even if
    the version was registered."""
    bundle = tmp_path / "v1"
    bundle.mkdir()
    _write(bundle / "helpers.py", 'MARKER = "v1"\n')
    _bundle_loader.register_bundle("1", bundle)

    finder = _bundle_loader._get_finder()
    assert finder.find_spec("helpers", None, None) is None


def test_two_versions_resolve_to_their_own_helpers(tmp_path: Path) -> None:
    """Inside ``activate(v1)`` ``import helpers`` returns v1's module;
    inside ``activate(v2)`` it returns v2's. The two coexist in
    sys.modules under versioned names, with the un-versioned alias
    swapped per-frame."""
    v1 = tmp_path / "v1"
    v1.mkdir()
    _write(v1 / "helpers.py", 'MARKER = "v1"\n')

    v2 = tmp_path / "v2"
    v2.mkdir()
    _write(v2 / "helpers.py", 'MARKER = "v2"\n')

    _bundle_loader.register_bundle("1", v1)
    _bundle_loader.register_bundle("2", v2)

    with _bundle_loader.activate("1"):
        helpers_v1 = importlib.import_module("helpers")
        assert helpers_v1.MARKER == "v1"
    # Outside the scope, the alias is gone.
    assert "helpers" not in sys.modules

    with _bundle_loader.activate("2"):
        helpers_v2 = importlib.import_module("helpers")
        assert helpers_v2.MARKER == "v2"
    assert "helpers" not in sys.modules

    # Both versioned modules are still resident.
    assert "_papayya_user_v1__helpers" in sys.modules
    assert "_papayya_user_v2__helpers" in sys.modules
    assert sys.modules["_papayya_user_v1__helpers"].MARKER == "v1"
    assert sys.modules["_papayya_user_v2__helpers"].MARKER == "v2"


def test_activate_round_trip_returns_to_v1_view(tmp_path: Path) -> None:
    """After running v1, then v2, then v1 again, the active alias
    points at v1's module. Verifies that the alias stack restores
    cleanly across multiple activations."""
    v1 = tmp_path / "v1"
    v1.mkdir()
    _write(v1 / "helpers.py", 'MARKER = "v1"\n')

    v2 = tmp_path / "v2"
    v2.mkdir()
    _write(v2 / "helpers.py", 'MARKER = "v2"\n')

    _bundle_loader.register_bundle("1", v1)
    _bundle_loader.register_bundle("2", v2)

    with _bundle_loader.activate("1"):
        importlib.import_module("helpers")
    with _bundle_loader.activate("2"):
        importlib.import_module("helpers")
    with _bundle_loader.activate("1"):
        # Should re-resolve to v1; the cached _papayya_user_v1__helpers
        # is reused, but the un-versioned alias is freshly published.
        helpers_v1 = importlib.import_module("helpers")
        assert helpers_v1.MARKER == "v1"


def test_activate_none_is_noop(tmp_path: Path) -> None:
    """``activate(None)`` is a no-op so the worker can wrap every
    lease unconditionally — local-dev (no agent_version) hits this
    branch."""
    bundle = tmp_path / "v1"
    bundle.mkdir()
    _write(bundle / "helpers.py", 'MARKER = "v1"\n')
    _bundle_loader.register_bundle("1", bundle)

    with _bundle_loader.activate(None):
        # Finder should still be dormant — None doesn't activate anything.
        finder = _bundle_loader._get_finder()
        assert finder.find_spec("helpers", None, None) is None


def test_package_shaped_bundle_resolves(tmp_path: Path) -> None:
    """A bundle sibling can be a package directory (``foo/__init__.py``).
    The finder treats it as ``is_package=True`` and adds the
    package's location to ``submodule_search_locations`` so internal
    relative imports work."""
    bundle = tmp_path / "v1"
    pkg = bundle / "tools"
    pkg.mkdir(parents=True)
    _write(pkg / "__init__.py", "FROM_PACKAGE = True\n")

    _bundle_loader.register_bundle("1", bundle)

    with _bundle_loader.activate("1"):
        tools = importlib.import_module("tools")
        assert getattr(tools, "FROM_PACKAGE", False) is True


def test_unrelated_imports_pass_through(tmp_path: Path) -> None:
    """Names that aren't in the bundle root must fall through to the
    default finders. ``activate`` doesn't shadow the rest of Python."""
    bundle = tmp_path / "v1"
    bundle.mkdir()
    _write(bundle / "helpers.py", 'MARKER = "v1"\n')
    _bundle_loader.register_bundle("1", bundle)

    with _bundle_loader.activate("1"):
        # ``json`` is in the standard library; the finder must NOT
        # claim it. If it does, the import would either fail or
        # return a wrong module.
        import json  # noqa: F401 — exercising the import path
        assert sys.modules["json"].__name__ == "json"

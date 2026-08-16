"""Process-global state that leaks between tests, contained in one place.

Two kinds of leak were failing 27 tests in this suite, both invisible when the
affected file was run on its own:

1. **The developer's own ~/.papayya/config.json.** ``_config.CONFIG_DIR`` is
   ``Path.home() / ".papayya"`` evaluated at IMPORT time, so the
   ``monkeypatch.setenv("HOME", tmp_path)`` that several files use to isolate
   themselves cannot redirect it — the constant is already bound. The real key
   was then found by ``_resolve_durable_credentials``, which put CloudStore
   ahead of SQLiteStore in ``Papayya._auto_store()``, and the tests dialled
   api.getpapayya.com. Redirecting the constants (what the CLI test files
   already do by hand) is the thing that actually works, so it happens here for
   everyone.

2. **``os.environ`` written as a side effect of construction.**
   ``Worker.__init__`` sets PAPAYYA_RUNTIME_STORE_BASE / _PLATFORM_WORKER_KEY /
   _LOCAL_DB_PATH on the process (worker.py ~L421) — that is how it hands the
   store choice to in-process @agent code. ``test_worker_store_seam.py`` knew
   and wrapped its Worker in a save/restore; ``test_pause_releases_lease.py``
   did not, so every test after it resolved to a runtime store pointed at
   ``http://localhost:1`` and got ECONNREFUSED. monkeypatch cannot undo a write
   it did not make, so the snapshot below is explicit.

Both are contained rather than fixed at the source: the import-time constants
are monkeypatched by five test files already (changing them to lazy lookups
would break those), and making the Worker stop writing to os.environ is a real
refactor of how the store seam is passed to customer code, not a test fix.
"""

from __future__ import annotations

import os

import pytest


@pytest.fixture(autouse=True)
def _isolate_papayya_globals(monkeypatch, tmp_path):
    """Give every test its own config file and its own PAPAYYA_* environment.

    Autouse, so a test cannot forget. A file that wants a *populated* config
    still requests its own fixture and overwrites these — its monkeypatch runs
    after this one and wins.
    """
    from papayya import _config as cfg_module

    config_dir = tmp_path / "_papayya_home" / ".papayya"
    monkeypatch.setattr(cfg_module, "CONFIG_DIR", config_dir)
    monkeypatch.setattr(cfg_module, "CONFIG_FILE", config_dir / "config.json")
    # cli.py binds its own alias at import (`CONFIG_FILE as _CONFIG_FILE`), so
    # the module constant alone is not enough. Imported lazily and guarded:
    # most tests here never touch the CLI.
    try:
        from papayya import cli as cli_module
    except Exception:  # pragma: no cover - CLI extras absent
        pass
    else:
        monkeypatch.setattr(cli_module, "_CONFIG_FILE", config_dir / "config.json")
        # Resolving a project id from a key is a network call the CLI memoises
        # for the life of the process; a stale entry would answer for a later
        # test's key.
        cli_module._PROJECT_ID_BY_KEY.clear()

    # HOME too, so anything that reads it at CALL time (rather than at import,
    # which is the bug above) also lands in the sandbox.
    monkeypatch.setenv("HOME", str(tmp_path / "_papayya_home"))

    saved = {k: v for k, v in os.environ.items() if k.startswith("PAPAYYA_")}
    for name in saved:
        os.environ.pop(name, None)

    yield

    for name in [k for k in os.environ if k.startswith("PAPAYYA_")]:
        os.environ.pop(name, None)
    os.environ.update(saved)

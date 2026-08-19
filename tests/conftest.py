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
        # Resolving a key's projects is a network call the CLI memoises for the
        # life of the process; a stale entry would answer for a later test's
        # key. Cleared AND stubbed: since plan 57 D8 an explicit
        # PAPAYYA_API_KEY makes the CLI ask the control plane which project
        # that key can reach, so without a stub most of this suite would dial
        # api.getpapayya.com through five retries.
        #
        # None is the honest stub — it is what the real function returns when
        # it cannot ask, and callers must treat that as "unknown", never as a
        # refusal. A test that cares about the lookup monkeypatches this
        # itself; its patch runs after and wins.
        cli_module._PROJECTS_BY_KEY.clear()
        monkeypatch.setattr(cli_module, "_projects_for_api_key",
                            lambda api_key, base_url: None)

    # Plan 58 R — DO NOT SERVE THE RETRY BACKOFF IN TESTS.
    #
    # `run.step` now retries a raising step four times with a 1/2/4/8s ladder.
    # Six existing tests raise on purpose and have no reason to care, and they
    # took the suite from 57s to 142s the moment the ladder landed. A suite
    # that slow stops being run, which costs more than the coverage it buys.
    #
    # THE CONSTANTS, NOT `time.sleep`. The first version of this patched
    # `run_module._time.sleep` — and `run.py` does `import time as _time`, so
    # `_time` IS the time module and that setattr replaced `time.sleep` FOR THE
    # WHOLE PROCESS. Three unrelated tests broke on it: the heartbeat harness
    # recorded no beats, a duration came back 0ms, and an llm-judge timeout
    # stopped containing anything. Worse, the damage depended on test ORDER, so
    # a shuffled run passed and `-p no:randomly` failed. That is precisely the
    # leak class this file exists to contain, introduced by the file itself.
    #
    # Zeroing the ladder keeps the RETRY and removes only the waiting, so call
    # counts, emission counts and propagation stay production-identical. A test
    # that wants the real ladder restores these itself; its monkeypatch runs
    # after this one and wins (test_step_retry.py).
    try:
        from papayya.durable import run as _run_module
    except Exception:  # pragma: no cover - durable extras absent
        pass
    else:
        monkeypatch.setattr(_run_module, "STEP_RETRY_BASE_SECONDS", 0.0)
        monkeypatch.setattr(_run_module, "STEP_RETRY_MAX_SECONDS", 0.0)

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

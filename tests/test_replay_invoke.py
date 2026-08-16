"""How a captured input snapshot is turned back into a call to the agent.

These are the surviving tests of the old ``test_replay_cli.py``. That file
drove ``papayya replay --db <sqlite>`` — the local dev-loop replay that Plan 37
(``2a410a3``, "repoint `papayya replay` to hosted replay") deleted. Eight tests
kept invoking a flag the command no longer has; seven failed with
"No such option: --db" and the eighth
(``test_replay_rejects_non_failed_run``) asserted only ``exit_code != 0``, so
the usage error kept it green while it tested nothing.

``_replay_invoke`` itself is untouched by that cutover — it is the binding
rule, not the transport — so its tests move here intact.
"""

from __future__ import annotations

from papayya.durable._replay import _replay_invoke


def test_replay_invoke_unpacks_dict_when_keys_bind() -> None:
    def fn(item_id, retries=0):
        return (item_id, retries)

    assert _replay_invoke(fn, {"item_id": "x"}) == ("x", 0)
    assert _replay_invoke(fn, {"item_id": "x", "retries": 3}) == ("x", 3)


def test_replay_invoke_falls_back_to_positional_when_keys_dont_bind() -> None:
    """A dict whose keys don't match the fn's params is passed as one
    positional argument — the agent receives the whole dict as before.
    """
    def fn(payload):
        return payload

    snap = {"unrelated_key": "x"}
    assert _replay_invoke(fn, snap) == snap


def test_replay_invoke_passes_non_dict_positionally() -> None:
    def fn(payload):
        return payload

    assert _replay_invoke(fn, "raw-string") == "raw-string"
    assert _replay_invoke(fn, [1, 2, 3]) == [1, 2, 3]

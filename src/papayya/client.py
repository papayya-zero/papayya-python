"""Deprecated ``papayya.Client`` shim — folded into ``Papayya`` (Plan 34).

BREAKING in 0.3.0: the old ``Client`` class (``client.run(agent_id, input)``
/ ``run_sync`` / ``get_status`` / ``get_steps`` against the v1 trigger
endpoints) is gone — it was deprecated with a removal notice in the
previous release, and the v1 endpoints it spoke retired with the v1 DROP.

``Client`` is now an alias of :class:`papayya.Papayya`:

* per-item reads:  ``Papayya().items.get(...)`` / ``.list()`` / ``.steps(...)``
* submissions:     ``Papayya().runs.create(agent_id, items)``
* durable local:   ``Papayya().item(agent="...")``

``RunResult`` is kept importable for callers that type-annotated against
it; nothing constructs it anymore.
"""

from __future__ import annotations

from papayya.papayya import Papayya

# Alias, not a subclass: isinstance(client, Papayya) and
# isinstance(client, Client) are the same check.
Client = Papayya


class RunResult(str):
    """Legacy result type of the removed ``Client.run_sync``.

    Behaves like a string (the output) but also exposes ``run_id`` and
    ``status``. Kept importable for old type annotations; the SDK no
    longer returns it.
    """

    def __new__(cls, output: str, run_id: str, status: str):
        instance = super().__new__(cls, output)
        instance.output = output
        instance.run_id = run_id
        instance.status = status
        return instance


__all__ = ["Client", "RunResult"]

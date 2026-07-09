"""Deprecated alias module (Plan 34): batch → run.

``Batches`` forwards to the new :class:`papayya.resources.runs.Runs`
invocation surface — same methods, same frozen HTTP wire. New code should
use ``Papayya().runs``. Scheduled for removal one release after 0.3.0.
"""

from __future__ import annotations

from papayya.resources.runs import Runs


class Batches(Runs):
    """Deprecated alias of :class:`Runs` (invocations). Use ``.runs``."""


__all__ = ["Batches"]

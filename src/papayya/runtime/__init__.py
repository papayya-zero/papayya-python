"""Long-running worker runtime.

This package is the *hosted* execution model: a long-lived process that
imports the customer agent module once on boot, then pulls items from a
dispatcher in a loop. It replaces the per-item-container shim that lives
in `tribe-agents/runtime-images/python/papayya_shim/` for the new worker
pool architecture (see tribe-agents/RUNTIME_VISION.md).

Entry point: ``python -m papayya.runtime --agent-module PATH ...``
"""

from .worker import Worker

__all__ = ["Worker"]

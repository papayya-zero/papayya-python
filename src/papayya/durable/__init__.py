"""Durable execution wrapper for AI agents."""

from ._replay import ReplayError, replay, replay_slice
from .client import PapayyaClient, PapayyaClientConfig, papayya
from .cloud_store import CloudStore, CloudStoreConfig
from .run import Item, PapayyaRun
from .sqlite_store import SQLiteStore
from .store import FileStore, MemoryStore
from .types import (
    CheckpointStore,
    DurableRunConfig,
    DurableRunResult,
    RunCheckpoint,
    TaskEntry,
)

__all__ = [
    "papayya",
    "PapayyaClient",
    "PapayyaClientConfig",
    "Item",
    "PapayyaRun",
    "MemoryStore",
    "FileStore",
    "SQLiteStore",
    "CloudStore",
    "CloudStoreConfig",
    "CheckpointStore",
    "DurableRunConfig",
    "DurableRunResult",
    "RunCheckpoint",
    "TaskEntry",
    "replay",
    "replay_slice",
    "ReplayError",
]


# ``papayya.durable`` serves double duty: the durable-execution subpackage
# (``from papayya.durable import papayya`` etc., above) AND the public lead
# decorator ``@papayya.durable``. Make the module object itself callable so
# both work without a name collision — the decorator delegates to
# ``papayya.agent.durable`` (which is ``@agent`` with the signature freed and
# the name optional). Assigning a module's ``__class__`` to a ModuleType
# subclass is supported since Python 3.5.
import sys as _sys
import types as _types


class _DurableModule(_types.ModuleType):
    def __call__(self, fn=None, *, name=None, **kwargs):
        from papayya.agent import durable as _durable

        return _durable(fn, name=name, **kwargs)


_sys.modules[__name__].__class__ = _DurableModule

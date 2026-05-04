"""Per-version module finder for hosted-worker bundles.

ADR-0003 § Worker #4 mandates that a worker hold v1 and v2 of the same
agent slug resident at once. Slice 2 punted on namespacing: it inserted
the bundle root onto ``sys.path`` so the entrypoint's
``from helpers import x`` resolved. With two versions resident, two
bundles' ``helpers.py`` files would collide in ``sys.modules`` —
whichever was imported first wins forever.

This module fixes that by routing top-level imports made *during* a
bundle's execution through a custom ``MetaPathFinder``. The finder is
keyed by version; activation is scoped via a ContextVar so the worker
controls when a particular bundle's imports take effect (during
``exec_module`` of the entrypoint, and during ``_handle_lease`` calls
on the registered fn).

Concurrency contract:
    The worker is single-threaded for lease execution. The heartbeat
    thread doesn't import bundle modules. The contextvar is only read
    inside ``find_spec``, which only fires inside an ``activate`` scope
    on the same thread. Multi-thread safety would require additional
    work; not needed for slice 3.

Limitations (inherited from slice 2):
    - Customer code that mutates *process-global* state at import time
      (singletons, module-level file handles, third-party module
      state) runs once per version. The two versions share that
      global state. True isolation would require subprocess workers
      (Phase 4 ``@agent(isolated=True)``).
    - Native (``.so``) extensions live in C-level state; if two
      versions of a C extension were extracted under different bundle
      roots, the first-imported wins. ``papayya.bundler`` already
      excludes ``.so`` from extraction, so this is moot in practice.

No in-process eviction is performed — when a deploy ships a new dep
graph, the worker recycles to a fresh process (ADR-0003 § Worker #6,
implemented in ``worker.py``). Slice E will revisit if pooled long-
lived workers need LRU eviction across versions (ADR-0003 Q5).
"""

from __future__ import annotations

import importlib.abc
import importlib.util
import logging
import sys
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Iterator

log = logging.getLogger("papayya.runtime.bundle_loader")


# Sentinel used to track "this name had no prior sys.modules entry"
# — distinguished from "prior entry was None" (legal Python state).
_MISSING: Any = object()


class _BundleFinder(importlib.abc.MetaPathFinder):
    """Resolves top-level imports inside an active versioned bundle.

    Registered once on ``sys.meta_path`` via
    :func:`install_finder`. It maintains a map of
    ``version -> bundle_root`` and a contextvar holding the currently-
    active version. ``find_spec`` only handles top-level absolute
    imports; relative imports inside packages flow through Python's
    default ``__path__``-driven resolution untouched.
    """

    def __init__(self) -> None:
        self._roots: dict[str, Path] = {}
        # ``None`` = no active version → finder is dormant. Set by
        # :func:`activate` before exec_module / lease invocation.
        self._active_version: ContextVar[str | None] = ContextVar(
            "papayya_active_bundle_version", default=None
        )
        # Aliases pushed during the current activate() scope so we can
        # restore prior sys.modules state on exit. Maps the un-versioned
        # name to whatever was in sys.modules before our load (or
        # _MISSING if the slot was empty).
        #
        # Using a stack lets nested activate() calls compose (loading
        # v2 while v1's frame is still on the stack restores v1's view
        # cleanly on v2 exit). Slice 3 doesn't actually nest, but the
        # stack avoids surprising behaviour if a future change does.
        self._alias_stack: list[dict[str, Any]] = []

    # -- registration ------------------------------------------------- #

    def register_bundle(self, version: str, root: Path) -> None:
        """Make a bundle's root searchable while ``version`` is active.

        Called from ``Worker._import_bundle_module`` after extraction.
        Re-registering the same version replaces the path silently —
        the bundle cache guarantees the path is canonical.
        """
        self._roots[version] = root.resolve()

    def unregister_bundle(self, version: str) -> None:
        """Drop a version's bundle root.

        Slice 3 doesn't call this — recycling is a fresh process, so
        there's nothing to evict in-process. Exposed for tests and for
        a future LRU path (Slice E).
        """
        self._roots.pop(version, None)

    # -- contextvar scope -------------------------------------------- #

    @contextmanager
    def activate(self, version: str | None) -> Iterator[None]:
        """Make ``version`` the active bundle for the duration of the block.

        ``version is None`` is a no-op — local-dev / LocalDispatcher
        leases hit this branch and the finder stays dormant. The worker
        wraps each bundle's ``exec_module`` and each ``_handle_lease``
        invocation in this so module-level *and* function-body imports
        of bundle siblings resolve to the right version.
        """
        if version is None:
            yield
            return
        token = self._active_version.set(version)
        self._alias_stack.append({})
        try:
            yield
        finally:
            aliases = self._alias_stack.pop()
            for name, prior in aliases.items():
                if prior is _MISSING:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = prior
            self._active_version.reset(token)

    # -- MetaPathFinder protocol ------------------------------------- #

    def find_spec(
        self,
        fullname: str,
        path: list[str] | None,
        target: Any = None,
    ) -> Any:
        """Locate ``fullname`` inside the active bundle's root.

        Returns ``None`` (defer to default finders) when:
          - No version is active.
          - The import is non-top-level (``path is not None``) — those
            are handled by the package's own ``__path__``.
          - The active version's bundle root has no matching file or
            package directory.
        """
        version = self._active_version.get()
        if version is None or path is not None:
            return None
        # Don't shadow papayya itself or other top-level packages
        # already known to Python — only act when the name is a literal
        # candidate inside the bundle root.
        root = self._roots.get(version)
        if root is None:
            return None
        candidate_file = root / f"{fullname}.py"
        candidate_pkg = root / fullname / "__init__.py"
        if candidate_file.is_file():
            origin = candidate_file
            is_package = False
        elif candidate_pkg.is_file():
            origin = candidate_pkg
            is_package = True
        else:
            return None

        # We use a versioned name for the spec so sys.modules holds
        # distinct entries per version; the un-versioned alias is
        # written by ``_BundleLoader.exec_module`` for the duration of
        # the active scope, which is what makes ``from helpers import
        # x`` resolve from the customer's code.
        versioned_name = f"_papayya_user_v{version}__{fullname}"
        loader = _BundleLoader(
            finder=self,
            unversioned_name=fullname,
            versioned_name=versioned_name,
            origin=origin,
            is_package=is_package,
        )
        spec = importlib.util.spec_from_loader(
            versioned_name,
            loader,
            origin=str(origin),
            is_package=is_package,
        )
        if spec is None:
            return None
        # For packages, expose the bundle root as ``__path__`` so
        # relative imports (``from .helpers import x``) flow through
        # Python's default machinery against the same root.
        if is_package and spec.submodule_search_locations is not None:
            spec.submodule_search_locations.append(str(origin.parent))
        return spec


class _BundleLoader(importlib.abc.Loader):
    """Loader that publishes the loaded module under both the versioned
    and the un-versioned sys.modules name.

    The un-versioned name is what ``from helpers import x`` looks up
    after ``__import__`` returns — without the alias, the customer's
    bundle code can't find its own siblings. The alias is scoped to
    the current ``activate(version)`` block; on exit the prior
    sys.modules entry (if any) is restored so the next activate scope
    starts fresh.
    """

    def __init__(
        self,
        *,
        finder: _BundleFinder,
        unversioned_name: str,
        versioned_name: str,
        origin: Path,
        is_package: bool,
    ) -> None:
        self._finder = finder
        self._unversioned_name = unversioned_name
        self._versioned_name = versioned_name
        self._origin = origin
        self._is_package = is_package

    def create_module(self, spec: Any) -> Any:
        return None  # default module creation

    def exec_module(self, module: Any) -> None:
        # Read the source and exec it into the module's namespace.
        # Using SourceFileLoader directly would also work, but routing
        # through here lets us perform the alias bookkeeping in one
        # place.
        from importlib.machinery import SourceFileLoader

        source_loader = SourceFileLoader(self._versioned_name, str(self._origin))
        source_loader.exec_module(module)

        # Record the prior sys.modules entry (if any) for the
        # unversioned name in the current activate-frame's alias dict
        # so we can restore it on scope exit. _MISSING distinguishes
        # "no prior entry" from "prior entry was None".
        if not self._finder._alias_stack:
            # Defensive: should never happen — find_spec only runs when
            # a frame is active. If it does, skip the alias to avoid
            # corrupting sys.modules.
            log.warning(
                "bundle loader fired without an active scope; skipping alias for %s",
                self._unversioned_name,
            )
            return
        frame = self._finder._alias_stack[-1]
        if self._unversioned_name not in frame:
            # Capture the original entry the first time we shadow this
            # name in this frame; subsequent loads in the same frame
            # don't re-capture (they'd record their own un-restored
            # value).
            frame[self._unversioned_name] = sys.modules.get(
                self._unversioned_name, _MISSING
            )
        sys.modules[self._unversioned_name] = module


# -- module-level singleton ----------------------------------------- #

_FINDER: _BundleFinder | None = None


def _get_finder() -> _BundleFinder:
    global _FINDER
    if _FINDER is None:
        _FINDER = _BundleFinder()
        # Insert at the front so we get first crack at top-level names
        # before the default finders (which might surface a stale
        # sys.modules entry from a prior version).
        sys.meta_path.insert(0, _FINDER)
    return _FINDER


def register_bundle(version: str, root: Path) -> None:
    """Public wrapper. Idempotent install + register."""
    _get_finder().register_bundle(version, root)


def unregister_bundle(version: str) -> None:
    """Drop a registered version. Idempotent."""
    if _FINDER is not None:
        _FINDER.unregister_bundle(version)


@contextmanager
def activate(version: str | None) -> Iterator[None]:
    """Activate a registered version for the duration of the block.

    See :meth:`_BundleFinder.activate` for semantics. ``version is
    None`` is a no-op so the worker can wrap *every* lease invocation
    unconditionally (LocalDispatcher leases pass None).
    """
    with _get_finder().activate(version):
        yield

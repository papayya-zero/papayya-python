"""What the hosted worker pool can import, and how `deploy` checks against it.

ADR 0010. The managed pool runs every customer's ``@agent`` body **in one
process**, off a fixed dependency set baked into the worker image. There is no
per-bundle ``pip install``: a venv only takes effect for a new interpreter, and
injecting one onto ``sys.path`` would install dependencies without isolating
them — ``import openai`` resolves once, process-globally, so the first account
to lease would decide the version every other account gets. That is plan 47 S1's
cross-tenant execution bug relocated into ``sys.modules``, where no cache key
reaches it. Per-bundle dependencies therefore wait for a subprocess executor.

Until then the honest thing is to say so at DEPLOY time. Plan 47 S2's failure
mode was ``ModuleNotFoundError`` on item 1 of a 200-item run, after the deploy
had reported success — a customer discovering our limitation as their outage.

**The set is not defined here.** It is ``papayya``'s own package metadata: the
core ``dependencies`` (always installed with the SDK) plus the ``runtime``
extra (installed only into the worker image, by the Dockerfile that also
installs the SDK). Reading it back from ``importlib.metadata`` rather than
restating it is deliberate — a duplicated list drifts, and the drift shows up as
a deploy that passes preflight and then fails at runtime, which is the exact
failure this module exists to prevent.
"""

from __future__ import annotations

import ast
import re
import sys
from importlib import metadata
from pathlib import Path

__all__ = [
    "baked_distributions",
    "unsupported_requirements",
    "unsupported_imports",
    "pool_modules",
    "normalize",
]

# PEP 503 normalization: names compare case-insensitively with -, _ and .
# interchangeable, so `Pillow`, `pillow` and `PIL_LOW` do not read as three
# different packages. (`PIL` really is a different package from `Pillow`; the
# import name is not the distribution name, which §"Known limits" covers.)
_NORMALIZE = re.compile(r"[-_.]+")

# A requirement line is a distribution name followed by any of extras, version
# specifiers, an environment marker, or a URL. We want the name only — the
# preflight deliberately does NOT try to satisfy a pin. Claiming
# `openai>=2.0` is satisfied by whatever the image happens to carry would be a
# worse lie than the one this module removes.
_REQ_NAME = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)")

# Lines a requirements.txt can carry that are not requirements. `-r other.txt`
# is not followed: a bundle that splits its manifest gets the conservative
# answer (we check what we can see) rather than a wrong one.
_NON_REQUIREMENT_PREFIXES = ("-", "--", "#")

# Directories the bundler drops, mirrored here so the gate reads what will
# actually ship rather than what happens to be on disk (bundler.EXCLUDE_DIRS).
_SKIP_DIRS = frozenset({
    ".git", "__pycache__", ".venv", "venv", "node_modules", ".mypy_cache",
    ".pytest_cache", "dist", "build",
})


def normalize(name: str) -> str:
    """PEP 503 normalized distribution name."""
    return _NORMALIZE.sub("-", name.strip()).lower()


def _requirement_name(line: str) -> str | None:
    """Distribution name from one requirement line, or None if there isn't one.

    Handles the shapes a real ``requirements.txt`` carries: bare names, pins,
    extras (``uvicorn[standard]``), environment markers, inline comments, and
    blank/flag/comment lines.
    """
    text = line.split("#", 1)[0].strip()
    if not text or text.startswith(_NON_REQUIREMENT_PREFIXES):
        return None
    # A direct URL/VCS requirement (`pkg @ git+https://...`) still leads with
    # its name; `git+https://...` with no name does not, and _REQ_NAME's
    # leading-alnum anchor rejects it. Unnamed URLs are reported as
    # unsupported, which is the truthful answer — the pool cannot fetch them.
    match = _REQ_NAME.match(text)
    if match is None:
        return None
    return match.group(1)


def baked_distributions() -> set[str] | None:
    """Normalized names the worker image carries, or None if undeterminable.

    Core ``dependencies`` plus the ``runtime`` extra, read from installed
    metadata — see the module docstring for why it is read rather than listed.
    ``papayya`` itself is included; the worker imports it by definition.

    Returns ``None`` when the metadata cannot be read (an unusual install
    layout, a vendored copy). Callers must treat that as "cannot check" and
    skip the preflight. Blocking a deploy because our own introspection failed
    would be a worse outcome than the runtime error we are preventing.
    """
    try:
        requires = metadata.requires("papayya")
    except metadata.PackageNotFoundError:
        return None
    if requires is None:
        return None

    available = {normalize("papayya")}
    for raw in requires:
        # Metadata requirement lines carry markers after ';' —
        #   openai>=1.0; extra == "runtime"
        # Core deps have no marker; extras have one naming the extra. We take
        # core plus `runtime` and skip the rest (test/dev/otel are not in the
        # image), so the set is exactly what the Dockerfile installs.
        head, _, marker = raw.partition(";")
        if marker.strip() and 'extra == "runtime"' not in marker and "extra == 'runtime'" not in marker:
            continue
        name = _requirement_name(head)
        if name:
            available.add(normalize(name))
    return available


def unsupported_requirements(requirements_text: str) -> list[str]:
    """Requirement lines from a bundle that the hosted pool cannot import.

    Returns the original lines (stripped), not just names, so the message a
    customer reads quotes their own file back at them. Empty list means every
    declared dependency is available — or that the check could not run, which
    :func:`baked_distributions` reports separately as ``None``.
    """
    available = baked_distributions()
    if available is None:
        return []

    unsupported: list[str] = []
    for line in requirements_text.splitlines():
        name = _requirement_name(line)
        if name is None:
            continue
        if normalize(name) not in available:
            unsupported.append(line.split("#", 1)[0].strip())
    return unsupported


# --------------------------------------------------------------------------- #
#  Reading the CODE instead of the manifest (plan 67 S3).
#
#  The manifest check above answers "does requirements.txt name something the
#  pool lacks", and returns immediately when there is no requirements.txt.
#  Onboarding step 4 says "Save as agent.py" and never mentions a manifest, so
#  a first user has one file and no reason to invent a second — which means the
#  gate did not run for exactly the person it was built for. Measured:
#
#      $ ls
#      agent.py                          # with `from dotenv import load_dotenv`
#      $ papayya deploy
#        Deployed dotenvdoc → 11eeed82-…
#      $ papayya run dotenvdoc "DOC-3001"
#        0 step(s) — failed
#            ModuleNotFoundError: No module named 'dotenv'
#
#  Plan 47 S2, through the hole in the gate built to stop plan 47 S2.
#
#  So read what the customer actually wrote. The manifest stays as an
#  additional signal — a bundle that declares `pandas` and does not import it
#  yet is still worth failing — but it is no longer the trigger.
# --------------------------------------------------------------------------- #


def _closure(names: set[str]) -> set[str]:
    """Every distribution the image ends up with, given these direct ones.

    The worker image runs ``pip install papayya[runtime]``, and pip installs
    the transitive graph — so ``anyio`` is importable in the pool even though
    it is nobody's declared dependency but ``httpx``'s. Checking against the
    ten direct names alone would fail deploys for imports that work.

    Walks ``metadata.requires`` and takes CORE requirements only at each hop:
    an extra of a dependency is not installed unless something asked for it,
    and nothing does.
    """
    seen: set[str] = set()
    queue = list(names)
    while queue:
        name = normalize(queue.pop())
        if name in seen:
            continue
        seen.add(name)
        try:
            requires = metadata.requires(name)
        except metadata.PackageNotFoundError:
            continue
        for raw in requires or ():
            head, _, marker = raw.partition(";")
            # Extras are opt-in; `extra == "..."` marks a requirement that pip
            # did not install. Environment markers without an extra (python
            # version, platform) are conservatively followed — over-including
            # here costs a missed warning, under-including costs a false one,
            # and the false one blocks a valid deploy.
            if "extra ==" in marker or "extra==" in marker:
                continue
            dep = _requirement_name(head)
            if dep:
                queue.append(dep)
    return seen


def pool_modules() -> set[str] | None:
    """Top-level module names importable inside the hosted worker, or None.

    A distribution name is not an import name — ``python-dotenv`` is imported
    as ``dotenv``, ``pyyaml`` as ``yaml`` — so the manifest check's vocabulary
    cannot answer a question asked about ``import`` statements.
    ``packages_distributions()`` is the standard-library map between the two.

    Read from the LOCAL environment on purpose: the SDK the CLI is running
    from and the SDK the worker image installs are the same distribution with
    the same dependency graph, so the local closure is the image's closure.
    That is the same argument :func:`baked_distributions` makes for reading
    metadata rather than restating a list.

    Returns None when the baked set cannot be determined, which callers must
    treat as "cannot check" — blocking a deploy because our own introspection
    failed is worse than the runtime error it would have prevented.
    """
    baked = baked_distributions()
    if baked is None:
        return None

    installed = _closure(baked)
    modules = set(sys.stdlib_module_names)
    try:
        mapping = metadata.packages_distributions()
    except Exception:  # noqa: BLE001 — an exotic install layout is "cannot check"
        return None
    mapped: set[str] = set()
    for module, dists in mapping.items():
        for dist in dists:
            if normalize(dist) in installed:
                modules.add(module)
                mapped.add(normalize(dist))
    # A distribution packages_distributions() cannot see. An editable/src
    # layout with no top_level.txt is the common case, and `papayya` itself is
    # one — which made the check report `import papayya` as unavailable in the
    # pool that is defined as carrying it. A false positive here BLOCKS A VALID
    # DEPLOY, so every distribution in the closure gets an import name one way
    # or another: its own top_level.txt, else the PEP 503 name with dashes
    # turned into underscores, which is what a package following the usual
    # convention is imported as.
    for dist in installed - mapped:
        top_level = None
        try:
            top_level = metadata.distribution(dist).read_text("top_level.txt")
        except Exception:  # noqa: BLE001 — absent metadata is the normal case
            top_level = None
        if top_level:
            modules.update(line.strip() for line in top_level.splitlines() if line.strip())
        else:
            modules.add(dist.replace("-", "_"))
    return modules


def _module_roots(tree: "ast.AST") -> set[str]:
    """Top-level module names this source imports, excluding guarded ones.

    SKIPS ANYTHING UNDER ``try:`` OR ``if TYPE_CHECKING:``. ADR 0010's original
    comment argued the escape hatch existed because "we are reading a manifest,
    not the code" — reading the code means the conditional import can be
    recognised instead of worked around. ``try: import pandas / except
    ImportError:`` is a customer saying out loud that the import is optional,
    and a gate that fails on it is a gate people learn to skip.

    Relative imports (``from . import x``) are the bundle's own and are never
    reported.
    """
    guarded: set[int] = set()
    for node in ast.walk(tree):
        blocks = []
        if isinstance(node, ast.Try):
            blocks = [node.body]
        elif isinstance(node, ast.If):
            test = node.test
            names = {
                t.id for t in ast.walk(test) if isinstance(t, ast.Name)
            } | {
                t.attr for t in ast.walk(test) if isinstance(t, ast.Attribute)
            }
            if "TYPE_CHECKING" in names:
                blocks = [node.body]
        for block in blocks:
            for stmt in block:
                for inner in ast.walk(stmt):
                    guarded.add(id(inner))

    roots: set[str] = set()
    for node in ast.walk(tree):
        if id(node) in guarded:
            continue
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots.add(alias.name.split(".", 1)[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # relative — the bundle's own
                continue
            if node.module:
                roots.add(node.module.split(".", 1)[0])
    return roots


def unsupported_imports(project_dir: "str | Path") -> list[tuple[str, str]]:
    """``(module, file)`` pairs the bundle imports and the pool cannot provide.

    Sorted, one entry per module, naming the first file that imports it so the
    message a customer reads points at a line they can go and look at.

    Empty when everything resolves, when the check cannot run, or when the
    directory has no Python in it. Modules the bundle itself provides — a
    sibling ``helpers.py``, a ``lib/`` package — are resolved against the
    directory first and never reported.
    """
    available = pool_modules()
    if available is None:
        return []

    root = Path(project_dir)
    sources = [
        f for f in sorted(root.rglob("*.py"))
        if not any(part in _SKIP_DIRS or part.endswith(".egg-info") for part in f.parts)
    ]
    if not sources:
        return []

    # The bundle's own modules, by the name an `import` would use for them.
    local = {f.stem for f in sources} | {
        d.name for d in root.rglob("*") if d.is_dir() and (d / "__init__.py").exists()
    }

    found: dict[str, str] = {}
    for path in sources:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, SyntaxError, ValueError):
            # A file we cannot parse is the bundler's problem, not the gate's.
            continue
        for module in _module_roots(tree):
            if module in available or module in local or module in found:
                continue
            found[module] = str(path.relative_to(root))
    return sorted(found.items())

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

import re
from importlib import metadata

__all__ = ["baked_distributions", "unsupported_requirements", "normalize"]

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

"""Keep injected secret values out of what the platform stores.

Plan 67 S1. The control plane has carried a ``secrets.Redactor`` — constructed
from a name→value map, replacing every value with ``[REDACTED]``, with a
passing test file — since secrets were added, and it had **zero callers**, the
same as the ``GetDecrypted`` it was written to pair with. Both halves of secret
delivery were built and wired to nothing.

Now that a value reaches the customer's process, it can reach three places the
customer did not intend:

  * a **traceback**. ``httpx`` and most provider SDKs put the request URL in
    the exception, and a key in a query string or a basic-auth netloc goes with
    it. That string becomes ``durable_runs.error`` and is rendered, verbatim,
    in the red banner on the item page.
  * a **checkpoint**. ``run.step`` snapshots the step's arguments, and a step
    that takes a client or a config dict carries the key into
    ``checkpoints.input_snapshot`` in clear text.
  * a **completion**. Same for what the step returns.

WHY THIS IS IN THE EXECUTOR AND NOT IN GO. The values are known here, and here
is upstream of every write: checkpoints are written by the SDK from inside this
process, so one filter on the outbound body covers all three cases above. The
control plane would have to decrypt a project's secrets on every checkpoint
POST to do the same job later and worse.

WHAT THIS IS NOT. It is a defence against accident, not against the customer:
code running in this process can obviously print its own secret, and the plan
that removes that is the one where their code and their credential are not in
the same address space. What it removes is the case where a customer sets a
key, never handles it, and finds it in a dashboard.

Values shorter than 8 characters are skipped. Go's version used 4; at 4 a
secret set to ``true`` or a two-letter region code turns every occurrence of
that substring in every stored value into ``[REDACTED]``, which corrupts the
records this product exists to keep. A real credential is longer than 8.
"""

from __future__ import annotations

import threading

__all__ = ["set_redactor", "redact", "clear_redactor"]

# The minimum length at which replacing a value is more likely to be redaction
# than corruption. See the module docstring.
_MIN_LENGTH = 8

_lock = threading.Lock()
_values: tuple[str, ...] = ()


def set_redactor(secrets: dict[str, str]) -> None:
    """Install the values to strip. Replaces any previous set.

    Called once per item by the executor, from the same place that installs the
    secrets into ``os.environ`` — so the filter and the environment can never
    disagree about which project's values are live.

    Longest first, so that a secret which is a prefix of another does not
    truncate the replacement of the longer one.
    """
    global _values
    live = sorted(
        {v for v in secrets.values() if isinstance(v, str) and len(v) >= _MIN_LENGTH},
        key=len,
        reverse=True,
    )
    with _lock:
        _values = tuple(live)


def clear_redactor() -> None:
    """Forget every value. Used when a child is repurposed and by tests."""
    global _values
    with _lock:
        _values = ()


def redact(text: str) -> str:
    """Replace every known secret value in ``text`` with ``[REDACTED]``.

    A no-op — the same object back — when nothing is installed, which is every
    local run and every hosted run for a project with no secrets. That is the
    common path and it must not cost anything.
    """
    with _lock:
        values = _values
    if not values or not text:
        return text
    for value in values:
        if value in text:
            text = text.replace(value, "[REDACTED]")
    return text

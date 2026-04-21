"""Canonical serialization of user-provided data at storage and HTTP
boundaries.

BYOF agents hand us provider-specific response objects — OpenAI
``ChatCompletion`` (Pydantic v2), Anthropic ``Message`` (Pydantic v2),
Gemini SDK objects (custom classes), ``SimpleNamespace``, dataclasses,
plain classes — none of which are JSON-serializable by default.
``json.dumps`` on any of these raises ``TypeError`` and tears the step
record apart before it can land.

``encode_user_value`` runs a shape ladder so the record always survives:

    1. ``json.dumps(value)``                    — JSON-native fast path
    2. pydantic v2 ``.model_dump()``             — then retry JSON
    3. pydantic v1 ``.dict()``                   — then retry JSON
    4. ``dataclasses.asdict(value)``             — then retry JSON
    5. ``vars(value)`` (``__dict__``)            — then retry JSON
    6. ``json.dumps(repr(value))``               — always valid JSON

The output is always valid JSON text, which ``SQLiteStore.load()`` and
replay paths depend on (they call ``json.loads`` on the stored column).

Nested objects inside a dict/list are handled by the same ladder via
``json.dumps(..., default=...)``, so a dict containing a Pydantic model
still serializes cleanly.

For snapshot-style data (explicit lineage, user curates the value)
``encode_user_value(value, strict=True)`` matches the legacy
``_encode_snapshot`` behavior: raises ``ValueError`` with a message
pointing the user at serializable alternatives.
"""

from __future__ import annotations

import dataclasses
import json
from typing import Any


def _coerce(obj: Any) -> Any:
    """Shape-ladder fallback for :func:`json.dumps` ``default=``.

    Returns a JSON-friendly replacement — dict, list, primitive, or
    string — for objects ``json.dumps`` can't natively encode. Never
    raises; the final tier is ``repr(obj)`` so callers always get
    *something* storable.
    """
    model_dump = getattr(obj, "model_dump", None)
    if callable(model_dump):
        try:
            return model_dump()
        except Exception:
            pass

    v1_dict = getattr(obj, "dict", None)
    if callable(v1_dict) and not isinstance(obj, type):
        try:
            result = v1_dict()
            if isinstance(result, dict):
                return result
        except Exception:
            pass

    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        try:
            return dataclasses.asdict(obj)
        except Exception:
            pass

    if hasattr(obj, "__dict__"):
        try:
            d = vars(obj)
            if isinstance(d, dict) and d:
                return d
        except Exception:
            pass

    return repr(obj)


def encode_user_value(value: Any, *, strict: bool = False) -> str:
    """Serialize ``value`` for durable storage or HTTP transport.

    Returns valid JSON text. ``strict=True`` preserves the snapshot
    contract: raise ``ValueError`` instead of coercing, so lineage data
    never silently degrades to a ``repr`` string.
    """
    try:
        return json.dumps(value)
    except (TypeError, ValueError) as exc:
        if strict:
            raise ValueError(
                "value must be JSON-encodable. Pass a dict/list/primitive, "
                "or store a reference (e.g. an S3 key) instead of the object. "
                f"Original error: {exc}"
            ) from exc
    try:
        return json.dumps(value, default=_coerce)
    except (TypeError, ValueError):
        # Circular refs / pathological default=_coerce returns reach
        # here. Last-resort: stringify the top-level value so the step
        # record still lands as valid JSON.
        return json.dumps(repr(value))

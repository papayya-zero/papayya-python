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
                                                   (skipped under strict=True)

The output is always valid JSON text, which ``SQLiteStore.load()`` and
replay paths depend on (they call ``json.loads`` on the stored column).

Nested objects inside a dict/list are handled by the same ladder via
``json.dumps(..., default=...)``, so a dict containing a Pydantic model
still serializes cleanly.

For snapshot-style data (explicit lineage, user curates the value)
``encode_user_value(value, strict=True)`` raises ``ValueError`` rather than
storing something useless.

STRICT MEANS "NEVER DEGRADE TO A REPR", NOT "ONLY ACCEPT JSON-NATIVE"
(plan 53 S5). It used to mean the second, and the cost was that the shape
this SDK's own ``@papayya.llm`` docstring publishes —

    @papayya.llm
    def call_model(prompt: str) -> dict:
        return client.messages.create(...)

— raised ``ValueError: value must be JSON-encodable ... Object of type
ChatCompletion is not JSON serializable``, from ``save_task``'s
``output_snapshot``, AFTER THE MODEL CALL HAD BEEN PAID FOR. On the decorator
whose entire job is capturing that spend. Plan 47 S4 watched 100 items fail
this way before the cause was clear.

The guarantee worth keeping is that lineage never silently becomes
``"<ChatCompletion object at 0x10f...>"``, which nothing can replay or verify
against. Tiers 2-5 do not do that — a ``model_dump()`` of a provider response
is at least as faithful as the dict a customer would have hand-written, and is
the provider's own canonical form. Only tier 6 degrades. So strict runs the
same ladder and refuses only the last rung, which is exactly the promise the
old spelling was trying to make.
"""

from __future__ import annotations

import dataclasses
import inspect
import json
from typing import Any


class NotEncodable(Exception):
    """A value reached the ladder's repr tier under ``strict=True``.

    Carried out through ``json.dumps``'s ``default=`` hook, which propagates
    whatever the callable raises. Internal — ``encode_user_value`` converts it
    to the ``ValueError`` its callers already handle.
    """

    def __init__(self, obj: Any):
        super().__init__(type(obj).__name__)
        self.obj = obj


def _coerce(obj: Any, *, allow_repr: bool = True) -> Any:
    """Shape-ladder fallback for :func:`json.dumps` ``default=``.

    Returns a JSON-friendly replacement — dict, list, primitive, or
    string — for objects ``json.dumps`` can't natively encode.

    ``allow_repr=False`` is the strict mode: every structured tier still
    applies, and only the final ``repr`` rung raises :class:`NotEncodable`.
    That is the difference between "we could not store this faithfully" and
    "this was not already a dict", and the second is not worth failing a paid
    model call over.
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

    if not allow_repr:
        raise NotEncodable(obj)
    return repr(obj)


def _strict_coerce(obj: Any) -> Any:
    """``_coerce`` with the repr rung removed. Named rather than a lambda so
    the traceback of a refused snapshot says which hook refused it."""
    return _coerce(obj, allow_repr=False)


def encode_user_value(value: Any, *, strict: bool = False) -> str:
    """Serialize ``value`` for durable storage or HTTP transport.

    Returns valid JSON text. ``strict=True`` preserves the snapshot
    contract — lineage data never silently degrades to a ``repr`` string —
    while still running the structured tiers of the ladder, so a provider
    response object stores as its own ``model_dump()`` rather than raising.
    """
    try:
        return json.dumps(value)
    except (TypeError, ValueError):
        pass

    if strict:
        try:
            return json.dumps(value, default=_strict_coerce)
        except NotEncodable as exc:
            raise ValueError(
                f"could not store a {type(exc.obj).__name__} faithfully: it is "
                "not JSON-encodable and exposes no model_dump() / dict() / "
                "dataclass / __dict__ shape to read one from. Pass a "
                "dict/list/primitive, or store a reference (e.g. an S3 key) "
                "instead of the object."
            ) from exc
        except (TypeError, ValueError) as exc:
            # Circular references and pathological containers. Refused rather
            # than repr'd for the reason strict exists at all.
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


def bind_arguments(
    sig: inspect.Signature | None,
    args: tuple,
    kwargs: dict,
) -> dict | None:
    """Bind args/kwargs against ``sig`` and return the LIVE argument values.

    Sibling of :func:`build_input_snapshot`, which encodes the same binding
    for storage. Conservation contracts (Plan 40 Unit 1) need the real
    objects — counting rows off a JSON round-trip would be both wasteful
    and wrong for inputs that aren't JSON-encodable. Returns ``None``,
    never raises, when the signature is unavailable or ``bind()`` rejects
    the call.
    """
    if sig is None:
        return None
    try:
        bound = sig.bind(*args, **kwargs)
    except TypeError:
        return None
    bound.apply_defaults()
    return dict(bound.arguments)


def build_input_snapshot(
    sig: inspect.Signature | None,
    args: tuple,
    kwargs: dict,
) -> Any:
    """Bind args/kwargs against ``sig``, apply defaults, return a JSON-encodable dict.

    Returns ``None`` — never raises — if the signature is unavailable, ``bind()``
    rejects the call, or the bound args aren't JSON-encodable. Used by the
    ``@agent`` decorator and ``run.step`` auto-capture to record the input
    state of a callable invocation as lineage data.
    """
    if sig is None:
        return None
    try:
        bound = sig.bind(*args, **kwargs)
    except TypeError:
        return None
    bound.apply_defaults()
    snap = dict(bound.arguments)
    try:
        encode_user_value(snap, strict=True)
    except (TypeError, ValueError):
        return None
    return snap

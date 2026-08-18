"""Regression tests for ``papayya._serialize.encode_user_value``.

Covers the shape ladder — JSON-native, Pydantic v2, Pydantic v1,
dataclass, ``__dict__`` objects, last-resort repr — and confirms every
branch returns text that ``json.loads`` can round-trip (the replay path
depends on that invariant).
"""
from __future__ import annotations

import dataclasses
import json
from types import SimpleNamespace

import pytest

from papayya._serialize import encode_user_value


def _roundtrip(value):
    encoded = encode_user_value(value)
    return encoded, json.loads(encoded)


def test_json_native_primitives_pass_through():
    assert _roundtrip("hello") == ('"hello"', "hello")
    assert _roundtrip(42) == ("42", 42)
    assert _roundtrip(None) == ("null", None)
    assert _roundtrip([1, 2, 3])[1] == [1, 2, 3]
    assert _roundtrip({"a": 1})[1] == {"a": 1}


def test_simple_namespace_falls_through_to_dict():
    ns = SimpleNamespace(model="gemini-2.0-flash", tokens=50)
    _, decoded = _roundtrip(ns)
    assert decoded == {"model": "gemini-2.0-flash", "tokens": 50}


def test_dataclass_uses_asdict():
    @dataclasses.dataclass
    class Response:
        model: str
        prompt_tokens: int
        completion_tokens: int

    resp = Response(model="claude-sonnet", prompt_tokens=40, completion_tokens=10)
    _, decoded = _roundtrip(resp)
    assert decoded == {
        "model": "claude-sonnet",
        "prompt_tokens": 40,
        "completion_tokens": 10,
    }


def test_nested_dict_with_simple_namespace():
    outer = {
        "wrapper": True,
        "response": SimpleNamespace(model="gpt-4o", finish="stop"),
    }
    _, decoded = _roundtrip(outer)
    assert decoded == {
        "wrapper": True,
        "response": {"model": "gpt-4o", "finish": "stop"},
    }


def test_nested_list_with_mixed_shapes():
    @dataclasses.dataclass
    class Item:
        id: int

    payload = [SimpleNamespace(tag="a"), Item(id=7), "literal"]
    _, decoded = _roundtrip(payload)
    assert decoded == [{"tag": "a"}, {"id": 7}, "literal"]


def test_pydantic_v2_model_dump(monkeypatch):
    class FakePydanticV2:
        def __init__(self, **data):
            self._data = data

        def model_dump(self):
            return dict(self._data)

    resp = FakePydanticV2(model="gpt-4o-mini", tokens=12)
    _, decoded = _roundtrip(resp)
    assert decoded == {"model": "gpt-4o-mini", "tokens": 12}


def test_pydantic_v1_dict_method():
    class FakePydanticV1:
        def __init__(self, **data):
            self._data = data

        # No model_dump; older pydantic used .dict()
        def dict(self):
            return dict(self._data)

    resp = FakePydanticV1(finish="length", reason="max_tokens")
    _, decoded = _roundtrip(resp)
    assert decoded == {"finish": "length", "reason": "max_tokens"}


def test_opaque_class_falls_through_to_repr():
    class Opaque:
        __slots__ = ()

        def __repr__(self):
            return "<Opaque custom>"

    _, decoded = _roundtrip(Opaque())
    assert decoded == "<Opaque custom>"


def test_raising_model_dump_does_not_break_encoding():
    class Flaky:
        def model_dump(self):
            raise RuntimeError("boom")

        def __repr__(self):
            return "Flaky()"

    _, decoded = _roundtrip(Flaky())
    # Falls through to __dict__ (empty) -> repr
    assert decoded == "Flaky()"


def test_strict_mode_raises_only_when_it_would_have_to_repr():
    """Plan 53 S5. This asserted that `SimpleNamespace(x=1)` was REJECTED —
    pinning the defect, because a SimpleNamespace has a __dict__ and stores as
    {"x": 1}, which is not a degradation at all.

    Strict's guarantee is that lineage never silently becomes
    "<Thing object at 0x...>". Only an object the whole ladder cannot read a
    shape from can do that."""

    class Opaque:
        __slots__ = ()  # no model_dump, no dict(), not a dataclass, no __dict__

    with pytest.raises(ValueError, match="faithfully"):
        encode_user_value(Opaque(), strict=True)


def test_strict_mode_stores_a_structured_object_rather_than_refusing_it():
    assert encode_user_value(SimpleNamespace(x=1), strict=True) == '{"x": 1}'


def test_strict_mode_stores_a_provider_response_as_its_model_dump():
    """THE CASE THIS COST. `@papayya.llm`'s own docstring publishes
    `return client.messages.create(...)`, and strict raised on it from
    save_task's output_snapshot — AFTER the model call had been paid for, on
    the decorator whose job is capturing that spend."""

    class ChatCompletion:
        def model_dump(self):
            return {"id": "chatcmpl-x", "choices": [{"message": {"content": "hi"}}]}

    encoded = encode_user_value(ChatCompletion(), strict=True)
    assert json.loads(encoded)["choices"][0]["message"]["content"] == "hi"


def test_strict_mode_still_refuses_a_circular_reference():
    circular: dict = {}
    circular["self"] = circular
    with pytest.raises(ValueError):
        encode_user_value(circular, strict=True)


def test_strict_mode_passes_through_serializable():
    assert encode_user_value({"ok": True}, strict=True) == '{"ok": true}'


def test_output_is_always_valid_json():
    # Replay path depends on json.loads always succeeding on the stored
    # text. Exhaustively verify each ladder tier produces valid JSON.
    @dataclasses.dataclass
    class D:
        x: int

    class P:
        def model_dump(self):
            return {"a": 1}

    class V1:
        def dict(self):
            return {"b": 2}

    class Opaque:
        pass

    cases = [
        {"native": "ok"},
        SimpleNamespace(x=1),
        D(x=3),
        P(),
        V1(),
        Opaque(),
        [SimpleNamespace(y=2), D(x=4)],
    ]
    for case in cases:
        text = encode_user_value(case)
        json.loads(text)  # must not raise


def test_circular_reference_falls_through_gracefully():
    a = {}
    a["self"] = a
    # json.dumps raises ValueError on circular refs; our fallback's
    # default=_coerce re-invokes json.dumps which will also hit the
    # circularity. Expect the final repr path to catch it.
    text = encode_user_value(a)
    # Must be valid JSON (a string) and not raise.
    json.loads(text)

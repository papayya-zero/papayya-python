"""Plan 57 D5 — `papayya items get DOC-2001` 404'd on the id it is named after.

    $ papayya items get DOC-2001
    Error: HTTP 404: durable run not found

The argument is `ITEM_ID`. The help said "Fetch one hosted item by id". It
resolved a run uuid and nothing else — so the id the triage feed had just
handed the operator was the one id this command could not take.
"""

from __future__ import annotations

from typing import Any

import pytest

from papayya.api import PapayyaAPIError
from papayya.resources.items import Items, _looks_like_uuid


class _FakeAPI:
    """Records every request so the ORDER of lookups is testable — the whole
    design here is which round trip is spent first."""

    def __init__(self, *, run_rows: dict[str, dict], by_item: dict[str, list[dict]]):
        self.run_rows = run_rows
        self.by_item = by_item
        self.calls: list[str] = []

    def _request(self, method: str, path: str, **_: Any) -> Any:
        self.calls.append(path)
        if path.startswith("/v1/durable/runs?"):
            from urllib.parse import parse_qs, urlparse
            q = parse_qs(urlparse(path).query)
            key = q.get("item_id", [""])[0]
            return list(self.by_item.get(key, []))
        run_id = path.rsplit("/", 1)[-1]
        if run_id in self.run_rows:
            return self.run_rows[run_id]
        raise PapayyaAPIError(404, "durable run not found")


UUID_A = "0ac9eb00-1111-4111-8111-111111111111"
UUID_B = "9be12e39-2222-4222-8222-222222222222"


def _doc(run_id: str, created: str, status: str = "completed") -> dict:
    return {"run_id": run_id, "item_id": "DOC-2001", "status": status,
            "created_at": created}


def test_get_resolves_the_customers_own_key() -> None:
    api = _FakeAPI(run_rows={}, by_item={"DOC-2001": [_doc(UUID_A, "2026-08-19T10:00:00Z")]})
    row = Items(api).get("DOC-2001")
    assert row["run_id"] == UUID_A
    # Not a uuid, so no round trip is spent asking whether it is one.
    assert api.calls == ["/v1/durable/runs?item_id=DOC-2001"]


def test_get_still_takes_a_run_id_first() -> None:
    api = _FakeAPI(run_rows={UUID_A: _doc(UUID_A, "2026-08-19T10:00:00Z")}, by_item={})
    row = Items(api).get(UUID_A)
    assert row["run_id"] == UUID_A
    assert api.calls == [f"/v1/durable/runs/{UUID_A}"]


def test_a_uuid_shaped_customer_key_still_resolves() -> None:
    """Order numbers from an upstream system are often uuids. The shape decides
    which lookup goes first, never which one is authoritative."""
    api = _FakeAPI(run_rows={}, by_item={UUID_B: [_doc(UUID_B, "2026-08-19T10:00:00Z")]})
    row = Items(api).get(UUID_B)
    assert row["run_id"] == UUID_B
    assert api.calls == [f"/v1/durable/runs/{UUID_B}",
                         f"/v1/durable/runs?item_id={UUID_B}"]


def test_a_re_driven_document_returns_the_newest_and_resolve_returns_both() -> None:
    """THE case. Replay mints a new run id under the same key, so after a
    recovery the document lives under two ids and nothing joined them."""
    old = _doc(UUID_A, "2026-08-19T10:00:00Z", status="failed")
    new = _doc(UUID_B, "2026-08-19T10:04:00Z")
    api = _FakeAPI(run_rows={}, by_item={"DOC-2001": [old, new]})
    items = Items(api)

    assert items.get("DOC-2001")["run_id"] == UUID_B, "newest wins"
    assert [r["run_id"] for r in items.resolve("DOC-2001")] == [UUID_B, UUID_A]


def test_unknown_key_raises_a_404_that_says_both_things_were_tried() -> None:
    api = _FakeAPI(run_rows={}, by_item={})
    with pytest.raises(PapayyaAPIError) as exc:
        Items(api).get("DOC-9999")
    assert exc.value.status == 404
    assert "DOC-9999" in str(exc.value)
    assert "item_id" in str(exc.value)


def test_a_non_404_from_the_run_lookup_is_not_swallowed() -> None:
    """A 403 means the caller has a real problem with THAT run; retrying it as
    a customer key would replace a clear error with 'not found'."""

    class Forbidden(_FakeAPI):
        def _request(self, method: str, path: str, **kw: Any) -> Any:
            if not path.startswith("/v1/durable/runs?"):
                raise PapayyaAPIError(403, "insufficient permissions")
            return super()._request(method, path, **kw)

    with pytest.raises(PapayyaAPIError) as exc:
        Items(Forbidden(run_rows={}, by_item={})).get(UUID_A)
    assert exc.value.status == 403


def test_looks_like_uuid_is_shape_only() -> None:
    assert _looks_like_uuid(UUID_A)
    assert not _looks_like_uuid("DOC-2001")
    assert not _looks_like_uuid("")
    assert not _looks_like_uuid(None)  # type: ignore[arg-type]

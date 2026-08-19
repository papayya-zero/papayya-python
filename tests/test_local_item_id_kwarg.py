"""Plan 56 F8 — the local path can declare an item_id.

Plan 54 N1, re-confirmed by plan 55 with the wire captured:

    "item_id": {"id": "DOC-1001", "carrier": "northwind", ...}

the whole document, where the server wants a string, so `python agent.py`
400s for ANY dict input. Onboarding step 4 passes the literal "world", which
is why it kept passing — and a document pipeline is inherently dict-input, so
the entire customer shape was dead at step 4 with no workaround.

Root cause: plan 43 B2a changed the calling convention from fn(item_id) to
fn(input), and the fix shipped only where a dispatcher could declare the id
(the worker injects it into a contextvar). Run locally there is no dispatcher,
so _resolve_item_id fell through to args[0] — now the input object.

The channel already existed for the field beside it: _extract_partition_key
has read kwargs since it was written, two functions away.
"""

from __future__ import annotations

from papayya.agent import (
    _resolve_item_id,
    reset_bootstrap_item_id,
    set_bootstrap_item_id,
)


def test_kwarg_wins_over_a_dict_input():
    # The exact plan 55 shape.
    doc = {"id": "DOC-1001", "carrier": "northwind", "pages": 150}

    assert _resolve_item_id((doc,), {"item_id": "DOC-1001"}) == "DOC-1001"


def test_without_the_kwarg_the_dict_still_comes_through():
    # Unchanged behaviour, and it is what 400s server-side. Pinned so the fix
    # is visibly a NEW channel rather than a change to the old one — any
    # customer relying on positional string ids keeps working.
    doc = {"id": "DOC-1001"}
    assert _resolve_item_id((doc,), {}) == doc
    assert _resolve_item_id(("co_42",), {}) == "co_42"


def test_partition_key_style_call_shape_is_the_precedent():
    # partition_key has always been read out of kwargs. item_id now is too,
    # with the same call shape, so there is one convention rather than two.
    from papayya.agent import _extract_partition_key

    kwargs = {"item_id": "DOC-1001", "partition_key": "northwind"}
    assert _resolve_item_id(({"id": "x"},), kwargs) == "DOC-1001"
    assert _extract_partition_key(kwargs) == "northwind"


def test_the_hosted_declaration_still_wins():
    # plan 43 B2b C11: a hosted invocation NEVER falls through to the
    # customer's own arguments. The worker's declared id must outrank a kwarg,
    # or a customer passing item_id= could rename someone else's record.
    token = set_bootstrap_item_id("LEASE-DECLARED")
    try:
        got = _resolve_item_id(({"id": "DOC-1001"},), {"item_id": "CUSTOMER-SAID"})
    finally:
        reset_bootstrap_item_id(token)
    assert got == "LEASE-DECLARED"


def test_an_explicit_none_kwarg_is_respected():
    # `item_id=None` is a declaration that this record has no id — not an
    # absent kwarg. Falling through to args[0] here would re-derive an id from
    # the input for the customer who explicitly declined to name one, which is
    # what deleting dispatch.InputToItemID stopped doing one layer down.
    assert _resolve_item_id(({"id": "DOC-1001"},), {"item_id": None}) is None


# --- N2: the server's own explanation reaches the customer ----------------

def test_http_error_carries_the_servers_message():
    """Plan 54 N2 / plan 56 F8.

    The failure surfaced as a raw traceback ending in

        httpx.HTTPStatusError: Client error '400 Bad Request' for url ...

    with the sentence that names the problem sitting unread in the response
    body. raise_for_status throws it away.
    """
    import httpx
    import pytest

    from papayya.durable.cloud_store import _raise_with_server_message

    request = httpx.Request("POST", "http://cp.invalid/v1/runtime/runs/r1/checkpoints")
    resp = httpx.Response(
        400, json={"error": {"message": "invalid request body"}}, request=request
    )

    with pytest.raises(httpx.HTTPStatusError) as exc:
        _raise_with_server_message(resp)

    assert "invalid request body" in str(exc.value)
    # Same exception type with the response attached, so _is_transient and
    # every other caller inspecting .response keep working.
    assert exc.value.response.status_code == 400


def test_http_error_falls_back_to_the_raw_body():
    # An HTML error page from a proxy is still more use than a bare status.
    import httpx
    import pytest

    from papayya.durable.cloud_store import _raise_with_server_message

    request = httpx.Request("POST", "http://cp.invalid/x")
    resp = httpx.Response(502, text="<html>upstream gone</html>", request=request)

    with pytest.raises(httpx.HTTPStatusError) as exc:
        _raise_with_server_message(resp)
    assert "upstream gone" in str(exc.value)


def test_success_does_not_raise():
    import httpx

    from papayya.durable.cloud_store import _raise_with_server_message

    request = httpx.Request("POST", "http://cp.invalid/x")
    _raise_with_server_message(httpx.Response(201, json={}, request=request))


def test_http_error_includes_the_request_id():
    # Plan 56 F7 echoes X-Request-Id on every response. Surfacing it here is
    # what makes a 500 findable in a log the operator does not control.
    import httpx
    import pytest

    from papayya.durable.cloud_store import _raise_with_server_message

    request = httpx.Request("POST", "http://cp.invalid/v1/agents/x/deployments")
    resp = httpx.Response(
        500,
        json={"error": {"message": "failed to upload artifact"}},
        headers={"X-Request-Id": "abc123/xyz-000042"},
        request=request,
    )
    with pytest.raises(httpx.HTTPStatusError) as exc:
        _raise_with_server_message(resp)

    assert "failed to upload artifact" in str(exc.value)
    assert "abc123/xyz-000042" in str(exc.value)

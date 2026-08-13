"""What the customer's @agent function is called with — plan 43 B2a step 1.

The hosted lease has carried the submitted `input` since the v2 cutover and the
worker ignored it, calling `fn(item_id)` — where `item_id` is
`dispatch.InputToItemID(input)`, i.e. the input's compact JSON text for any
shape that is not a JSON string. So a customer submitting `{"order_id": …}` had
their function handed a string to re-parse while the parsed object sat in the
same lease, and `def triage(ticket: dict)` — the signature in the published
quickstart — could not have worked.

The headline test is `test_falsy_inputs_are_still_the_input`. The obvious
spelling of this rule, `payload.get("input") or item_id`, is wrong for six
legitimate inputs, and every one of them fails by silently handing the function
a stringified id where it expected an empty value.
"""

import pytest

from papayya.runtime.worker import Lease


def lease(payload):
    return Lease(lease_id="l-1", agent="summarizer", item_id="the-item-id",
                 payload=payload)


@pytest.mark.parametrize("value", [
    {"order_id": "co_42", "text": "summarise me"},   # the case that motivated it
    ["a", "b"],
    "co_42",
    42,
    3.14,
    True,
])
def test_the_submitted_input_is_the_argument(value):
    assert lease({"input": value}).agent_argument == value


# THE test. All six are falsy in Python, all six are legitimate submissions, and
# `.get("input") or item_id` sends the id for every one of them.
@pytest.mark.parametrize("value", [None, "", 0, False, {}, []])
def test_falsy_inputs_are_still_the_input(value):
    got = lease({"input": value}).agent_argument
    assert got == value and got != "the-item-id"
    # `0` == `False` in Python, so equality alone would not catch a swap
    # between them; the type is part of what the function receives.
    assert type(got) is type(value)


def test_a_lease_with_no_input_key_falls_back_to_the_item_id():
    # LocalDispatcher leases, and any hosted path that records no input.
    assert lease({"run_id": "r-1"}).agent_argument == "the-item-id"
    assert lease(None).agent_argument == "the-item-id"


def test_an_object_input_is_not_a_string():
    # The regression this change exists to prevent: `fn` used to receive
    # '{"order_id":"co_42"}' as text and had to parse it itself.
    got = lease({"input": {"order_id": "co_42"}}).agent_argument
    assert isinstance(got, dict)
    assert got["order_id"] == "co_42"

# Source for `papayya example`. Bundled in the wheel so a fresh
# pip install can scaffold the demo without examples/ on disk.
# Kept byte-for-byte in sync with examples/local_demo_agent.py via
# tests/test_cli_example.py::test_demo_constant_matches_examples_dir.
#
# Design constraints (Plan 34 Unit 4):
# - Beginner-facing: reads like a quickstart, not a runtime feel-test.
# - Writes to the default ledger (.papayya/local.db) — NO db-path env
#   var, so `python agent.py` → `papayya dev` just works.
# - A couple of items come back DEGRADED so the dashboard shows the
#   ran-vs-worked wedge on the very first run.

LOCAL_DEMO_AGENT_SOURCE = '''"""Your first Papayya run — no API key, no network.

Run it:

    python agent.py

Then open the local dashboard:

    papayya dev

You'll see ONE RUN of six items — and two of them flagged **degraded**.
Those two "worked" as far as any status code can tell: the (fake) model
returned a 200. But it returned a refusal with no content, and Papayya
inspects what came back, not just whether the call returned. That
ran-vs-worked verdict, per item, per tenant, is the point.

Swap the canned data and the fake model call for your own and the same
three lines of Papayya keep doing the same job.
"""

from __future__ import annotations

import time

import papayya

# A canned support queue across two tenants — stands in for your real
# data. The two "refund" tickets are the ones the model will fumble.
TICKETS = [
    {"id": "tkt_101", "tenant": "acme", "text": "Card was charged twice this month"},
    {"id": "tkt_102", "tenant": "acme", "text": "Refund never arrived"},
    {"id": "tkt_103", "tenant": "acme", "text": "Export button throws a 500"},
    {"id": "tkt_201", "tenant": "globex", "text": "Invoice total looks wrong"},
    {"id": "tkt_202", "tenant": "globex", "text": "Please refund the annual plan"},
    {"id": "tkt_203", "tenant": "globex", "text": "Password reset email never arrives"},
]


@papayya.llm
def classify(text: str) -> dict:
    """Stands in for your real LLM call — same response shape, no network.

    The one Papayya line is the decorator. It records this as an LLM step
    (model, tokens, timing) and runs the ran-vs-worked inspectors on the
    response: a refusal or empty result flips the item to *degraded*,
    with no check written anywhere.
    """
    time.sleep(0.05)  # tiny simulated latency so timings aren't all zeros
    if "refund" in text.lower():
        # The degraded slice: a 200 that didn't work. No exception, no
        # error status — just a refusal with no usable content.
        return {
            "model": "demo-model",
            "stop_reason": "refusal",
            "usage": {"input_tokens": 42, "output_tokens": 2},
            "content": "",
        }
    return {
        "model": "demo-model",
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 42, "output_tokens": 12},
        "content": "billing" if "charge" in text.lower() or "invoice" in text.lower() else "bug",
    }


def triage(ticket: dict) -> str:
    """Ordinary business logic — no Papayya in the signature."""
    return classify(ticket["text"])["content"]  # "" when the model refused


if __name__ == "__main__":
    # One map() call is ONE RUN; each ticket is one item, recorded and
    # outcome-inspected in .papayya/local.db — where `papayya dev` reads.
    for label in papayya.map(
        triage,
        TICKETS,
        agent="triage-ticket",
        item_id=lambda t: t["id"],
        partition_key=lambda t: t["tenant"],
    ):
        print(label or "(refused)")
    print()
    print("Now run: papayya dev")
'''

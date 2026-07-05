# Source for `papayya example`. Bundled in the wheel so a fresh
# pip install can scaffold the demo without examples/ on disk.
# Kept byte-for-byte in sync with examples/local_demo_agent.py via
# tests/test_cli_example.py::test_demo_constant_matches_examples_dir.

LOCAL_DEMO_AGENT_SOURCE = '''"""Local demo workload — feel the durable loop without a provider key.

Run it directly:

    python agent.py

Then open the dashboard to see each item's run, its steps, and the
ran-vs-worked outcome Papayya recorded for it:

    papayya dev

Or exercise the full worker runtime (dispatcher + worker + dashboard):

    # terminal 1 — dispatcher with a few items
    python -m papayya.runtime.dispatcher --port 8765 --enqueue enrich:co_42,co_43,co_44

    # terminal 2 — a worker
    python -m papayya.runtime --agent-module $PWD/agent.py --dispatcher http://127.0.0.1:8765 --store /tmp/papayya-feel.db

    # terminal 3 — the dashboard
    PAPAYYA_LOCAL_DB_PATH=/tmp/papayya-feel.db papayya dev

The point is to feel the iteration loop: edit this file, re-run, watch new
lineage appear in seconds. Two journaled steps, a fake LLM call, no network.
"""

from __future__ import annotations

import os
import time

import papayya


# Canned data. Stand in for a real fetch (Clearbit, scraper, etc.) so the
# demo runs offline. Add domains here freely.
SNIPPETS = {
    "co_42": ("Stripe", "Financial infrastructure platform. Founded 2010, San Francisco."),
    "co_43": ("Anthropic", "AI safety company. Founded 2021, San Francisco. Makes Claude."),
    "co_44": ("Vercel", "Frontend cloud. Founded 2015, San Francisco. Makers of Next.js."),
    "co_99": ("Acme", "Roadrunner-trap manufacturer. Founded eternally."),
}


@papayya.step
def fetch_snippet(item_id: str) -> dict:
    name, snippet = SNIPPETS.get(item_id, (f"Unknown ({item_id})", "no snippet"))
    return {"name": name, "snippet": snippet}


@papayya.llm
def fake_extract(name: str, snippet: str) -> dict:
    """Stand in for an LLM call. @papayya.llm records it as an LLM step —
    usage, timing, and the ran-vs-worked verdict — even though we don't make
    a real network call."""
    # Tiny simulated latency so the dashboard timing isn't all zeros.
    time.sleep(0.05)
    words = snippet.split()
    return {
        "name": name,
        "summary": " ".join(words[:8]),
        "word_count": len(words),
    }


@papayya.durable
def enrich(item_id: str) -> dict:
    fetched = fetch_snippet(item_id)
    extracted = fake_extract(fetched["name"], fetched["snippet"])
    return {"id": item_id, **fetched, **extracted}


if __name__ == "__main__":
    # Run the whole batch durably — one run per item, each recorded and
    # outcome-inspected. `papayya dev` shows the lineage.
    os.environ.setdefault("PAPAYYA_LOCAL_DB_PATH", "/tmp/papayya-feel.db")
    for result in papayya.map(
        enrich,
        list(SNIPPETS),
        item_id=lambda item_id: item_id,
        partition_key=lambda item_id: "demo",
    ):
        print(result)
'''

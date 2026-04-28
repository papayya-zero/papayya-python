"""Local demo agent for the worker-runtime feel-test.

Use this with the local dispatcher to exercise the full loop without
needing an OpenAI key:

    # terminal 1 — start dispatcher with a few items
    python -m papayya.runtime.dispatcher --port 8765 \\
        --enqueue enrich:co_42,co_43,co_44

    # terminal 2 — start a worker
    python -m papayya.runtime \\
        --agent-module $PWD/examples/local_demo_agent.py \\
        --dispatcher http://127.0.0.1:8765 \\
        --store /tmp/papayya-feel.db

    # terminal 3 — open the dashboard
    PAPAYYA_LOCAL_DB_PATH=/tmp/papayya-feel.db papayya dev

    # iterate: edit this file, restart the worker, enqueue more items
    curl -X POST http://127.0.0.1:8765/enqueue \\
        -H 'Content-Type: application/json' \\
        -d '{"agent":"enrich","item_id":"co_99"}'

The point of this script is to feel the iteration loop. Two-step
durable run, fake LLM step, no network calls. Change the prompt /
schema / step structure, restart the worker, watch new lineage
appear in seconds.
"""

from __future__ import annotations

import os
import time

from papayya import agent
from papayya.durable import papayya


# Canned data. Stand in for a real fetch (Clearbit, scraper, etc.) so
# the demo runs offline. Add domains here freely.
SNIPPETS = {
    "co_42": ("Stripe", "Financial infrastructure platform. Founded 2010, San Francisco."),
    "co_43": ("Anthropic", "AI safety company. Founded 2021, San Francisco. Makes Claude."),
    "co_44": ("Vercel", "Frontend cloud. Founded 2015, San Francisco. Makers of Next.js."),
    "co_99": ("Acme", "Roadrunner-trap manufacturer. Founded eternally."),
}


def fetch_snippet(item_id: str) -> dict:
    name, snippet = SNIPPETS.get(item_id, (f"Unknown ({item_id})", "no snippet"))
    return {"name": name, "snippet": snippet}


def fake_extract(name: str, snippet: str) -> dict:
    """Stand in for an LLM call. kind='llm' on the step still records intent
    even though we don't make a real network call."""
    # Tiny simulated latency so the dashboard timing isn't all zeros.
    time.sleep(0.05)
    words = snippet.split()
    return {
        "name": name,
        "summary": " ".join(words[:8]),
        "word_count": len(words),
    }


@agent(name="enrich")
def enrich(item_id: str) -> dict:
    run = papayya().run("enrich", item_id=item_id)

    fetch = run.step("fetch_snippet", fetch_snippet)
    extract = run.step("fake_extract", fake_extract, kind="llm")

    fetched = fetch(item_id)
    extracted = extract(fetched["name"], fetched["snippet"])

    result = {"id": item_id, **fetched, **extracted}
    run.complete(result)
    return result


if __name__ == "__main__":
    # Allow running directly to verify the agent logic without a worker:
    #   python examples/local_demo_agent.py
    os.environ.setdefault("PAPAYYA_LOCAL_DB_PATH", "/tmp/papayya-feel.db")
    for item_id in SNIPPETS:
        print(enrich(item_id))

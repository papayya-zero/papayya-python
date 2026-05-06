"""Enrich a list of companies, one durable run per row.

Each row becomes a run tagged with ``item_id=<company_id>``. Open
``papayya dev`` → http://localhost:8585/item?id=co_42 to see the full
timeline for a single company: every step that touched it, input/output
snapshots, and a field-level diff.

Requires:
    pip install papayya openai
    export OPENAI_API_KEY=sk-...

Run:
    python examples/enrich_companies.py              # uses inline sample
    python examples/enrich_companies.py --csv my.csv # id,name,domain columns
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys
from typing import Any

from openai import OpenAI
from papayya.durable import papayya

SAMPLE_CSV = """\
id,name,domain
co_42,Stripe,stripe.com
co_43,Anthropic,anthropic.com
co_44,Vercel,vercel.com
"""

# Canned snippets stand in for a real fetch (Clearbit, scraper, etc.) so the
# example runs with just an OpenAI key. Swap this for requests/httpx when wiring
# to a real source.
SNIPPETS: dict[str, str] = {
    "stripe.com": "Stripe is a financial infrastructure platform. Founded 2010 in San Francisco. Payments, billing, issuing.",
    "anthropic.com": "Anthropic is an AI safety company founded in 2021. Headquartered in San Francisco. Makes Claude.",
    "vercel.com": "Vercel is the frontend cloud. Founded 2015 in San Francisco. Makers of Next.js.",
}


def fetch_snippet(domain: str) -> str:
    return SNIPPETS.get(domain, f"No snippet available for {domain}.")


def extract_fields(name: str, snippet: str) -> dict[str, Any]:
    client = OpenAI()
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "Extract {hq, founded_year, category} from the snippet. Return JSON."},
            {"role": "user", "content": f"Company: {name}\n\nSnippet: {snippet}"},
        ],
    )
    return json.loads(resp.choices[0].message.content or "{}")


def enrich_one(company_id: str, name: str, domain: str) -> dict[str, Any]:
    run = papayya().run("enrich-companies", item_id=company_id)

    fetch = run.step("fetch_snippet", fetch_snippet)
    extract = run.step("extract_fields", extract_fields, kind="llm")

    snippet = fetch(domain)
    fields = extract(name, snippet)

    result = {"id": company_id, "name": name, "domain": domain, **fields}
    run.complete(result)
    return result


def load_rows(path: str | None) -> list[dict[str, str]]:
    source = open(path, newline="") if path else io.StringIO(SAMPLE_CSV)
    with source as f:
        return list(csv.DictReader(f))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", help="CSV with id,name,domain columns")
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("Set OPENAI_API_KEY before running.")

    rows = load_rows(args.csv)
    print(f"Enriching {len(rows)} companies...\n")

    for row in rows:
        result = enrich_one(row["id"], row["name"], row["domain"])
        print(f"  {row['id']}: {result}")
        print(f"    → http://localhost:8585/item?id={row['id']}\n")

    print("Done. Run `papayya dev` (if it isn't already) and click any URL above.")


if __name__ == "__main__":
    main()

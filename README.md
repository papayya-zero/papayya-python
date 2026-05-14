# Papayya

**Run your AI agents on hundreds of items without losing money on failures you can't see.**

For periodic and batch LLM jobs — KB ingestion, nightly evals, lead enrichment, document extraction, conversation post-processing, codemod runs. The pipelines that need to actually finish, even when 3% of items quietly fail.

Without Papayya: *container exited 1.*
With Papayya: *agent spent $4.20 thrashing on step 7 because the tool returned malformed JSON — and 3% of items in this batch hit the same pattern.*

## Install

```bash
pip install papayya
```

## Try it in 30 seconds (no LLM key needed)

```bash
papayya init                                    # writes papayya.yaml
papayya example                                 # scaffolds local_demo_agent.py
python local_demo_agent.py                      # one keyless durable run
papayya dev                                     # open the local dashboard
```

The demo agent runs a two-step durable workflow against canned data — no provider key, no network. Open `papayya dev` to see the run, the per-step input/output, and the lineage. That's the iteration loop.

## Quick start with your own LLM

### Define an agent

```python
from papayya import agent, tool

@tool
def search_web(query: str) -> str:
    """Search the web for information."""
    # Your implementation here
    return "..."

@agent(name="research-bot", model="gpt-4o-mini", budget_usd=1.0)
def research_bot(input_data):
    from openai import OpenAI
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": input_data}],
    )
    return response.choices[0].message.content
```

`budget_usd` on the `@agent` decorator is metadata the cloud runtime uses to cap per-run spend. It is **not** enforced when you call the function directly on your laptop.

### Run a pile of items, durably

Wrap long-running work in checkpoint-able steps. Each item gets its own run, tagged so you can find it later. If a run crashes, it resumes from the last checkpoint instead of re-executing completed steps.

```python
from papayya.durable import papayya

for company in companies:
    run = papayya().run("enrich-company", item_id=company.id)

    fetch    = run.step("fetch",    fetch_company_snippet)
    extract  = run.step("extract",  extract_fields, kind="llm")

    snippet = fetch(company.domain)        # cached on replay
    fields  = extract(company.name, snippet)  # cached on replay

    run.complete({**company.dict(), **fields})
```

`step()` and `task()` are aliases. Use whichever reads better. `kind="llm"` records the step as an LLM call so the dashboard renders tokens, model, and cost beside it — without wrapping any provider SDK.

When `PAPAYYA_API_KEY` is set, checkpoints round-trip through the cloud control plane. Without a key, the SDK writes to a local SQLite at `.papayya/local.db` — the same database `papayya dev` reads from.

### Budget enforcement (cloud only)

Budget caps and cost tracking are enforced by the runtime shim when your agent runs in a Papayya container. Set a cap on the `@agent` decorator (shown above) or pass `budget_cents` when triggering a run via the API. There is no local budget API — local `PapayyaRun` is durability-only.

### Deploy

```bash
papayya login
papayya deploy
```

## What you get

- **Per-item visibility.** Tag each run with `item_id=…`; the dashboard groups by item so you can answer *"why didn't X happen?"* without grep.
- **Failure clustering.** When 47 of 1,000 items fail, see them grouped into 3 patterns — not 47 separate stack traces.
- **Per-item cost outliers.** Find the one bad input that ate $4 in retries before your bill does.
- **Replay from any step.** Re-run a single item against a new prompt without re-running the whole batch.
- **Budget enforcement.** Pause-on-cap, not just alert-on-cap. Cloud runtime only.
- **Bring your own framework.** OpenAI, Anthropic, Bedrock, raw HTTP — Papayya never touches your LLM client.

## Key concepts

- **BYOF (Bring Your Own Function)** — Papayya doesn't wrap your LLM calls. You use any SDK directly inside your agent function.
- **`@agent` decorator** — Registers your function for deployment. The function stays callable locally.
- **`@tool` decorator** — Defines tools your agent can call, with automatic JSON Schema generation from type hints.
- **Durable runs** — Checkpoint-and-replay execution. Steps are cached so replayed runs skip completed work.
- **Local dashboard** — `papayya dev` reads from the same SQLite the SDK writes to. No control plane required.
- **Budgets** — Cloud-only. The runtime shim reserves cost before each LLM call and pauses the run when the cap is hit.

## Examples

Runnable starter agents at [github.com/papayya-zero/examples](https://github.com/papayya-zero/examples) — lead enrichment, document extraction, eval harness. Each one clones-and-runs in under 60 seconds.

## Requirements

- Python 3.10+

## License

MIT

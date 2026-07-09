# Papayya

**Run your AI agents on hundreds of items without losing money on failures you can't see.**

For periodic and batch LLM jobs — KB ingestion, nightly evals, lead enrichment, document extraction, conversation post-processing, codemod runs. The pipelines that need to actually finish, even when 3% of items quietly fail.

Without Papayya: *container exited 1.*
With Papayya: *agent spent $4.20 thrashing on step 7 because the tool returned malformed JSON — and 3% of items in this run hit the same pattern.*

## Install

```bash
pip install papayya
```


```bash
papayya init                                    # writes papayya.yaml
papayya example                                 # scaffolds agent.py
python agent.py                                 # one keyless durable run
papayya dev                                     # open the local dashboard
```

The demo agent triages a canned support queue across two tenants — no provider key, no network. Two of its six items come back **degraded**: the fake model returns a refusal on a clean 200, and Papayya flags it without you writing a check. Open `papayya dev` and the run shows *2 of 6 degraded* before you click anything. That's the loop, and that's the wedge.

## The vocabulary (30 seconds)

An **agent** is loaded once by the worker pool; each **run** processes **items**; every item shows what it did and what it cost, and you can **replay** the ones that didn't work. A **step** is one node in an item's trace; a **trigger** invokes an agent from an HTTP call. The SDK, the CLI, `papayya dev`, and the cloud dashboard all speak this vocabulary the same way.

## Your first agent

Two entrypoints, and you touch both: `@papayya.durable` marks the function that owns one item, and `papayya.map` fans it out. Nothing threads a `run` through your signature.

```python
import papayya

@papayya.llm
def extract_fields(name, snippet):
    ...                     # your LLM call — recorded and inspected

@papayya.durable
def enrich(company):
    snippet = fetch_company_snippet(company.domain)
    fields  = extract_fields(company.name, snippet)
    return {**company.dict(), **fields}

papayya.map(enrich, companies,
            item_id=lambda c: c.id,
            partition_key=lambda c: c.tenant)
```

One `map()` call is one **run**; each element is one **item** in it, running in its own durable isolate. The `@papayya.llm` call inside the body is journaled *and* inspected — Papayya records not just that it ran, but whether it **worked** (a refusal, an empty result, or a degenerate stop-reason flips the item to `degraded`, even on a 200). `item_id` and `partition_key` tag every row, so the dashboard groups by item and tenant. No step bookkeeping, no handles threaded through signatures — your function stays ordinary Python.

`@papayya.durable` is also deployable: `papayya deploy` discovers it and the cloud runtime runs the same function per item.

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

### Finer control: explicit steps (the escape hatch)

`@papayya.durable` + `papayya.map` cover the common case. When you need to draw the checkpoint seams yourself — non-deterministic side-effecting code that isn't an LLM/tool call, or a workflow you'd rather orchestrate by hand — drop to explicit steps. Each element gets its own item record; if it crashes, replay feeds cached results back instead of re-executing completed steps.

```python
from papayya.durable import papayya

for company in companies:
    item = papayya().item("enrich-company", item_id=company.id)

    fetch    = item.step("fetch",    fetch_company_snippet)
    extract  = item.llm_step("extract", extract_fields)

    snippet = fetch(company.domain)        # cached on replay
    fields  = extract(company.name, snippet)  # cached on replay

    item.complete({**company.dict(), **fields})
```

`item.llm_step(...)` records the step as an LLM call so the dashboard renders tokens, model, and cost beside it — and runs the ran-vs-worked inspectors on the result — without wrapping any provider SDK.

When `PAPAYYA_API_KEY` is set, checkpoints round-trip through the cloud control plane. Without a key, the SDK writes to a local SQLite at `.papayya/local.db` — the same database `papayya dev` reads from.

### Budget enforcement (cloud only)

Budget caps and cost tracking are enforced by the runtime shim when your agent runs in a Papayya container. Set a cap on the `@agent` decorator (shown above) or pass `budget_cents` when triggering a run via the API. There is no local budget API — the local `Item` handle is durability-only.

### Deploy

```bash
papayya login
papayya deploy
```

## What you get

- **Per-item visibility.** Tag each item with `item_id=…`; the dashboard groups by item so you can answer *"why didn't X happen?"* without grep.
- **Failure clustering.** When 47 of 1,000 items fail, see them grouped into 3 patterns — not 47 separate stack traces.
- **Per-item cost outliers.** Find the one bad input that ate $4 in retries before your bill does.
- **Replay from any step.** Re-run a single item against a new prompt — or `papayya replay --run <id>` to re-drive just the run's not-ok slice — without re-running everything that already worked.
- **Budget enforcement.** Pause-on-cap, not just alert-on-cap. Cloud runtime only.
- **Bring your own framework.** OpenAI, Anthropic, Bedrock, raw HTTP — Papayya never touches your LLM client.

## Key concepts

- **BYOF (Bring Your Own Function)** — Papayya doesn't wrap your LLM calls. You use any SDK directly inside your agent function.
- **`@agent` decorator** — Registers your function for deployment. The function stays callable locally.
- **`@tool` decorator** — Defines tools your agent can call, with automatic JSON Schema generation from type hints.
- **Durable items** — Checkpoint-and-replay execution. Steps are cached so a replayed item skips completed work.
- **Local dashboard** — `papayya dev` reads from the same SQLite the SDK writes to. No control plane required.
- **Budgets** — Cloud-only. The runtime shim reserves cost before each LLM call and pauses the run when the cap is hit.

## Examples

Runnable starter agents at [github.com/papayya-zero/examples](https://github.com/papayya-zero/examples) — lead enrichment, document extraction, eval harness. Each one clones-and-runs in under 60 seconds.

## Requirements

- Python 3.10+

## License

MIT

# Papayya

Durable background jobs for AI agents. Bring your own LLM — Papayya handles execution, checkpointing, and deployment. Budget enforcement and cost tracking live on the cloud runtime.

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

## Quick Start with your own LLM

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

### Durable execution

Wrap long-running work in checkpoint-able steps. If a run crashes, it resumes from the last checkpoint instead of re-executing completed steps.

```python
from papayya import papayya

run = papayya().run("my-agent")

fetch    = run.step("fetch",     fetch_data)
analyse  = run.step("analyse",   analyse_results)

data    = fetch(query)        # cached on replay
summary = analyse(data)       # cached on replay

run.complete(summary)
```

`step()` and `task()` are aliases. Use whichever reads better in context.

When `PAPAYYA_API_KEY` is set, checkpoints round-trip through the cloud control plane. Without a key, the SDK writes to a local SQLite at `.papayya/local.db` — the same database `papayya dev` reads from.

### Budget enforcement (cloud only)

Budget caps and cost tracking are enforced by the runtime shim when your agent runs in a Papayya container. Set a cap on the `@agent` decorator (shown above) or pass `budget_cents` when triggering a run via the API. There is no local budget API — local `PapayyaRun` is durability-only.

### Deploy

```bash
papayya login
papayya deploy
```

## Key Concepts

- **BYOF (Bring Your Own Function)** — Papayya doesn't wrap your LLM calls. You use any SDK (OpenAI, Anthropic, Bedrock, etc.) directly inside your agent function.
- **`@agent` decorator** — Registers your function for deployment. The function stays callable locally.
- **`@tool` decorator** — Defines tools your agent can call, with automatic JSON Schema generation from type hints.
- **Durable runs** — Checkpoint-and-replay execution. Steps are cached so replayed runs skip completed work.
- **Local dashboard** — `papayya dev` reads from the same SQLite the SDK writes to. No control plane required.
- **Budgets** — Cloud-only. The runtime shim reserves cost before each LLM call and pauses the run when the cap is hit.

## Requirements

- Python 3.10+

## License

MIT

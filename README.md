# Papayya

Durable background jobs for AI agents. Bring your own LLM — Papayya handles execution, checkpointing, and deployment. Budget enforcement and cost tracking live on the cloud runtime.

## Install

```bash
pip install papayya
```

## Quick Start

### Define an agent

```python
from papayya import agent, tool

@tool
def search_web(query: str) -> str:
    """Search the web for information."""
    # Your implementation here
    return results

@agent(name="research-bot", model="gpt-4o-mini", budget_usd=1.0)
def research_bot(input_data):
    from openai import OpenAI
    client = OpenAI()
    # Your agent logic — call your LLM directly
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": input_data}],
    )
    return response.choices[0].message.content
```

`budget_usd` on the `@agent` decorator is metadata the cloud runtime uses to cap per-run spend. It is **not** enforced when you call the function directly on your laptop.

### Durable execution

Wrap long-running work in checkpoint-able tasks. If a run crashes, it resumes from the last checkpoint instead of re-executing completed steps.

```python
from papayya import papayya

run = papayya(agent="my-agent")

search = run.task("search", search_web)
summarize = run.task("summarize", summarize_results)

results = search(query)        # cached on replay
summary = summarize(results)   # cached on replay

run.complete(summary)
```

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
- **Durable runs** — Checkpoint-and-replay execution. Tasks are cached so replayed runs skip completed work.
- **Budgets** — Cloud-only. The runtime shim reserves cost before each LLM call and pauses the run when the cap is hit.

## Requirements

- Python 3.10+

## License

MIT

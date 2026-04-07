"""Manual usage tracking for durable runs.

Papayya does not monkey-patch or adapt LLM provider SDKs — that approach
broke every time a provider shipped an SDK update and risked corrupting
customer runs. Instead, this module exposes a `UsageTracker` that callers
feed explicitly via `tracker.record(...)` after each LLM call. For durable
execution, prefer `papayya.durable.run.record_cost(...)` at the call site.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class UsageTracker:
    """Collects token usage and cost from explicit calls.

    The tracker does NOT observe LLM calls automatically. Callers must invoke
    ``record(...)`` after every LLM response — typically from inside a
    ``papayya.durable.run.task(...)`` block — passing the token counts and the
    cost they computed themselves (based on their own pricing).
    """

    steps: list[dict[str, Any]] = field(default_factory=list)
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_cents: float = 0.0
    model: str = "unknown"

    def record(
        self,
        *,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float = 0.0,
        model: str = "unknown",
        duration_ms: int = 0,
        tool_calls: list[dict[str, Any]] | None = None,
        response_text: str | None = None,
    ) -> None:
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_cost_cents += cost_usd * 100
        self.model = model

        step: dict[str, Any] = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "model": model,
            "cost_usd": cost_usd,
            "duration_ms": duration_ms,
        }
        if tool_calls:
            step["tool_calls"] = tool_calls
        if response_text:
            step["response"] = response_text

        self.steps.append(step)

    def get_totals(self) -> dict[str, Any]:
        return {
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
            "model": self.model,
        }

    def get_total_cost_cents(self) -> float:
        return self.total_cost_cents

    def get_steps(self) -> list[dict[str, Any]]:
        return list(self.steps)

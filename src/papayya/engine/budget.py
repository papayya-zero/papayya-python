"""Provider-agnostic budget tracking.

Papayya does not ship per-provider rate cards — LLM pricing changes too
often and varies per customer contract. Callers supply their own pricing
(USD per million input/output tokens) to the BudgetTracker; if none is
provided a neutral placeholder is used so `budget_usd` still acts as a
rough ceiling, but accurate cost numbers should come from the caller.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TokenPricing:
    input_per_million: float
    output_per_million: float


DEFAULT_PRICING = TokenPricing(input_per_million=3.0, output_per_million=15.0)


class BudgetTracker:
    def __init__(
        self,
        limit_usd: float | None = None,
        *,
        pricing: TokenPricing | None = None,
    ) -> None:
        self._limit = limit_usd
        self._pricing = pricing or DEFAULT_PRICING
        self.consumed_usd: float = 0.0

    def record(self, *, input_tokens: int, output_tokens: int) -> None:
        self.consumed_usd += (input_tokens / 1_000_000) * self._pricing.input_per_million
        self.consumed_usd += (output_tokens / 1_000_000) * self._pricing.output_per_million

    def exceeded(self) -> bool:
        if self._limit is None:
            return False
        return self.consumed_usd >= self._limit

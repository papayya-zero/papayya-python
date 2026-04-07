"""High-level client for backend integration.

Usage:
    from papayya import Client

    client = Client()

    # Fire-and-forget (async)
    run = client.run(agent_id="research-agent", input="Stripe")
    print(run["id"])

    # Simple blocking call
    result = client.run_sync(agent_id="research-agent", input="Stripe")
    print(result)
"""

from __future__ import annotations

import time
from typing import Any

from papayya.api import APIClient, PapayyaAPIError, resolve_config


class RunResult(str):
    """Result of a synchronous run. Behaves like a string (the output) but
    also exposes run_id and status for programmatic access.

        result = client.run_sync(...)
        print(result)           # prints the output
        print(result.run_id)    # the run ID
        steps = client.get_steps(result.run_id)
    """

    def __new__(cls, output: str, run_id: str, status: str):
        instance = super().__new__(cls, output)
        instance.output = output
        instance.run_id = run_id
        instance.status = status
        return instance


class Client:
    """Developer-facing client for triggering and monitoring runs."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        config = resolve_config(api_key=api_key, base_url=base_url)
        self._api = APIClient(config)

    def run(
        self,
        agent_id: str,
        input: Any,
        *,
        model: str = "gpt-4o-mini",
        system_prompt: str = "You are a helpful assistant.",
        max_steps: int = 50,
        budget_cents: int = 500,
    ) -> dict[str, Any]:
        """Trigger a run and return the run object."""
        return self._api.trigger_run(
            agent_id=agent_id,
            model=model,
            system_prompt=system_prompt,
            input_data={"message": input} if isinstance(input, str) else input,
            max_steps=max_steps,
            budget_cents=budget_cents,
        )

    def get_status(self, run_id: str) -> dict[str, Any]:
        return self._api.get_run(run_id)

    def get_steps(self, run_id: str) -> list[dict[str, Any]]:
        return self._api.get_steps(run_id)

    def run_sync(
        self,
        agent_id: str,
        input: Any,
        *,
        model: str = "gpt-4o-mini",
        system_prompt: str = "You are a helpful assistant.",
        max_steps: int = 50,
        budget_cents: int = 500,
        timeout: float = 300,
        poll_interval: float = 2,
    ) -> RunResult:
        """Trigger a run and block until it completes.

        Returns a RunResult that behaves like a string (the output) but also
        exposes .run_id and .status for programmatic access.

        Raises TimeoutError if the run doesn't finish within `timeout` seconds.
        Raises PapayyaAPIError if the run fails.
        """
        run = self.run(
            agent_id=agent_id,
            input=input,
            model=model,
            system_prompt=system_prompt,
            max_steps=max_steps,
            budget_cents=budget_cents,
        )
        run_id = run["id"]
        deadline = time.monotonic() + timeout

        while time.monotonic() < deadline:
            status = self.get_status(run_id)

            if status["status"] == "completed":
                return RunResult(
                    output=status.get("output", ""),
                    run_id=run_id,
                    status="completed",
                )

            if status["status"] in ("failed", "cancelled"):
                error = status.get("error_message", status["status"])
                raise PapayyaAPIError(400, f"Run {status['status']}: {error}")

            time.sleep(poll_interval)

        raise TimeoutError(f"Run {run_id} did not complete within {timeout}s")

    def cancel(self, run_id: str) -> dict[str, Any]:
        return self._api.cancel_run(run_id)

    def close(self) -> None:
        self._api.close()

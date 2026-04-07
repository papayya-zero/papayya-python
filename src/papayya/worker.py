"""Tool worker — polls for pending tool calls and executes them locally."""

from __future__ import annotations

import json
import signal
import sys
import time
from typing import Any

from papayya.agent import Agent
from papayya.api import APIClient, APIConfig


def run_worker(
    agent: Agent,
    api: APIClient,
    poll_interval: float = 2.0,
) -> None:
    """Run the tool worker loop. Blocks until interrupted."""
    tool_map = agent.tool_map

    if not tool_map:
        print("Error: Agent has no tools defined")
        sys.exit(1)

    print(f"Tool worker started with {len(tool_map)} tools: {', '.join(tool_map.keys())}")
    print(f"Polling every {poll_interval}s...")

    running = True

    def shutdown(signum: int, frame: Any) -> None:
        nonlocal running
        print("\nShutting down worker...")
        running = False

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    while running:
        try:
            pending = api.poll_tool_calls()

            for call in pending:
                if not running:
                    break

                tool_name = call["tool_name"]
                tool = tool_map.get(tool_name)

                if tool is None:
                    print(f"  Warning: Unknown tool '{tool_name}', skipping")
                    continue

                run_id = call["run_id"]
                call_id = call["id"]
                print(f"  Executing '{tool_name}' for run {run_id[:8]}...")

                try:
                    # Parse input
                    raw_input = call.get("tool_input", {})
                    if isinstance(raw_input, str):
                        raw_input = json.loads(raw_input)

                    result = tool.execute(raw_input)
                    print(f"  '{tool_name}' completed")

                    response = api.resolve_tool_call(call_id, result)

                    if response.get("all_resolved"):
                        print(f"  All tools resolved for run {run_id[:8]} — run will continue")

                except Exception as e:
                    print(f"  '{tool_name}' failed: {e}")
                    try:
                        api.resolve_tool_call(call_id, {"error": str(e)})
                    except Exception as resolve_err:
                        print(f"  Failed to report error: {resolve_err}")

        except Exception as e:
            if running:
                print(f"  Poll error: {e}")

        if running:
            time.sleep(poll_interval)

    print("Worker stopped")

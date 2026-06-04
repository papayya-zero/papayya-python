"""HTTP client for the Papayya control plane API."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import httpx

from papayya._defaults import DEFAULT_BASE_URL


class PapayyaAPIError(Exception):
    def __init__(self, status: int, message: str):
        self.status = status
        super().__init__(f"HTTP {status}: {message}")


@dataclass
class APIConfig:
    api_key: str
    base_url: str = DEFAULT_BASE_URL
    timeout: float = 30.0


def resolve_config(
    api_key: str | None = None,
    base_url: str | None = None,
) -> APIConfig:
    key = api_key or os.environ.get("PAPAYYA_API_KEY")
    if not key:
        raise PapayyaAPIError(401, "No API key. Set PAPAYYA_API_KEY or pass --api-key.")

    url = base_url or os.environ.get("PAPAYYA_BASE_URL", DEFAULT_BASE_URL)
    return APIConfig(api_key=key, base_url=url)


class APIClient:
    """Thin wrapper around the control plane REST API."""

    def __init__(self, config: APIConfig) -> None:
        self._config = config
        headers: dict[str, str] = {
            "Accept": "application/json",
        }
        # API keys (cpk_...) use X-Api-Key header; JWTs use Authorization: Bearer
        if config.api_key.startswith("cpk_"):
            headers["X-Api-Key"] = config.api_key
        else:
            headers["Authorization"] = f"Bearer {config.api_key}"

        self._http = httpx.Client(
            base_url=config.base_url,
            timeout=config.timeout,
            headers=headers,
        )

    def close(self) -> None:
        self._http.close()

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        resp = self._http.request(method, path, **kwargs)
        if not resp.is_success:
            raise PapayyaAPIError(resp.status_code, resp.text)
        return resp.json()

    # -- Auth ----------------------------------------------------------------

    def login(self, email: str, password: str) -> dict[str, Any]:
        return self._request("POST", "/v1/auth/login", json={"email": email, "password": password})

    def register(self, email: str, password: str, name: str) -> dict[str, Any]:
        return self._request("POST", "/v1/auth/register", json={"email": email, "password": password, "name": name})

    # -- Projects ------------------------------------------------------------

    def create_project(self, name: str, slug: str) -> dict[str, Any]:
        return self._request("POST", "/v1/projects", json={"name": name, "slug": slug})

    def list_projects(self) -> list[dict[str, Any]]:
        return self._request("GET", "/v1/projects")

    # -- API Keys ------------------------------------------------------------

    def create_api_key(self, project_id: str, name: str) -> dict[str, Any]:
        return self._request("POST", f"/v1/projects/{project_id}/api-keys", json={"name": name})

    # -- Agents --------------------------------------------------------------

    def deploy_agent(self, agent_def: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", "/v1/agents", json=agent_def)

    def create_agent(self, project_id: str, name: str, slug: str, config: dict[str, Any] | None = None) -> dict[str, Any]:
        body: dict[str, Any] = {"project_id": project_id, "name": name, "slug": slug}
        if config:
            body["config"] = config
        return self._request("POST", "/v1/agents", json=body)

    def list_agents(self, project_id: str | None = None) -> list[dict[str, Any]]:
        params = {}
        if project_id:
            params["project_id"] = project_id
        return self._request("GET", "/v1/agents", params=params)

    def get_agent(self, agent_id: str) -> dict[str, Any]:
        return self._request("GET", f"/v1/agents/{agent_id}")

    # -- Runs ----------------------------------------------------------------

    def trigger_run(
        self,
        agent_id: str,
        *,
        model: str,
        system_prompt: str,
        input_data: Any,
        max_steps: int = 50,
        budget_cents: int = 500,
    ) -> dict[str, Any]:
        return self._request("POST", "/v1/runs", json={
            "agent_id": agent_id,
            "model": model,
            "system_prompt": system_prompt,
            "input": input_data,
            "max_steps": max_steps,
            "budget_cents": budget_cents,
        })

    def get_run(self, run_id: str) -> dict[str, Any]:
        # v1→v2 cutover: a triggered run is now a durable_run. The v1
        # /v1/runs/{id} surface reads the (now-unfed) runs table, so poll
        # the durable run instead. Response carries status + checkpoints.
        return self._request("GET", f"/v1/durable/runs/{run_id}")

    def get_steps(self, run_id: str) -> list[dict[str, Any]]:
        # Steps are durable checkpoints after the cutover. Each item is
        # {label, result, cost_usd, duration_ms, ...} — not the v1
        # {step_number, step_type, output} shape.
        return self._request("GET", f"/v1/durable/runs/{run_id}/checkpoints")

    # v1→v2 cutover: cancel_run AND the tool-call worker bridge
    # (poll_tool_calls/resolve_tool_call → /v1/tool-calls/*) retired with the
    # v1 DROP. Those routes and the tool_calls table are gone server-side
    # (control-pane slice 1 + migration 063); durable runs have no cancel verb
    # (use the quarantine→discard lifecycle instead).

    # -- Deployments ---------------------------------------------------------

    def upload_deployment(
        self,
        agent_id: str,
        tarball: bytes,
        runtime: str = "python",
        entrypoint: str = "agent.py",
    ) -> dict[str, Any]:
        """Upload a deployment artifact (multipart)."""
        import io
        resp = self._http.post(
            f"/v1/agents/{agent_id}/deploy",
            files={"file": ("artifact.tar.gz", io.BytesIO(tarball), "application/gzip")},
            data={"runtime": runtime, "entrypoint": entrypoint},
        )
        if not resp.is_success:
            raise PapayyaAPIError(resp.status_code, resp.text)
        return resp.json()

    def get_deployment(self, deployment_id: str) -> dict[str, Any]:
        return self._request("GET", f"/v1/deployments/{deployment_id}")

    def list_deployments(self, agent_id: str) -> list[dict[str, Any]]:
        return self._request("GET", f"/v1/agents/{agent_id}/deployments")

    # -- Secrets -------------------------------------------------------------

    def set_secret(self, project_id: str, name: str, value: str) -> dict[str, Any]:
        return self._request("POST", f"/v1/projects/{project_id}/secrets", json={"name": name, "value": value})

    def list_secrets(self, project_id: str) -> list[dict[str, Any]]:
        return self._request("GET", f"/v1/projects/{project_id}/secrets")

    def delete_secret(self, project_id: str, name: str) -> None:
        resp = self._http.request("DELETE", f"/v1/projects/{project_id}/secrets/{name}")
        if not resp.is_success:
            raise PapayyaAPIError(resp.status_code, resp.text)

    # -- Schedules -----------------------------------------------------------

    def list_schedules(self, agent_id: str) -> list[dict[str, Any]]:
        return self._request("GET", f"/v1/agents/{agent_id}/schedules")

    def create_schedule(
        self,
        agent_id: str,
        cron_expression: str,
        timezone: str = "UTC",
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            f"/v1/agents/{agent_id}/schedules",
            json={"cron_expression": cron_expression, "timezone": timezone},
        )

    def delete_schedule(self, schedule_id: str) -> None:
        resp = self._http.request("DELETE", f"/v1/schedules/{schedule_id}")
        if not resp.is_success:
            raise PapayyaAPIError(resp.status_code, resp.text)

    def put_schedules(
        self,
        agent_id: str,
        schedules: list[dict[str, Any]],
        *,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Replace all code-managed schedules for an agent in one call.

        ``schedules`` is a list of dicts each shaped like the POST body
        (``cron_expression`` + optional ``timezone`` / ``input`` /
        ``max_steps`` / ``budget_cents``). Each item carries
        ``managed_by='code'`` on the wire so the server scopes its
        full-replace to code-managed rows only — ``managed_by='api'``
        rows (dashboard / direct-POST) are invisible to this call.

        ``dry_run=True`` flips the server into preview mode: the same
        diff is computed against current ``managed_by='code'`` rows but
        no rows are mutated. The response shape changes to the diff
        envelope (``managed_by``, ``create``, ``update``, ``delete``,
        ``unmanaged_skipped``) — see Plan 13 for the consumer.

        Returns the server's apply-mode ``{items, summary}`` envelope by
        default, or the dry-run diff envelope when ``dry_run=True``.
        """
        body = {
            "items": [{**item, "managed_by": "code"} for item in schedules],
        }
        path = f"/v1/agents/{agent_id}/schedules"
        if dry_run:
            path = f"{path}?dry_run=true"
        return self._request("PUT", path, json=body)

    # -- Webhooks ------------------------------------------------------------

    def list_webhooks(self, agent_id: str) -> list[dict[str, Any]]:
        return self._request("GET", f"/v1/agents/{agent_id}/webhooks")

    def create_webhook(self, agent_id: str, name: str) -> dict[str, Any]:
        """Create a webhook. Response `secret` + `trigger_url` are only visible here."""
        return self._request(
            "POST",
            f"/v1/agents/{agent_id}/webhooks",
            json={"name": name},
        )

    def delete_webhook(self, webhook_id: str) -> None:
        resp = self._http.request("DELETE", f"/v1/webhooks/{webhook_id}")
        if not resp.is_success:
            raise PapayyaAPIError(resp.status_code, resp.text)

    def put_webhooks(
        self,
        agent_id: str,
        webhooks: list[dict[str, Any]],
        *,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Replace all code-managed webhooks for an agent in one call.

        ``webhooks`` is a list of dicts each shaped like the POST body
        (``name`` + optional ``description``). Each item carries
        ``managed_by='code'`` on the wire — ``managed_by='api'`` rows
        are not touched.

        ``dry_run=True`` flips the server into preview mode: the proposed
        diff is computed and returned without generating any new webhook
        secrets and without mutating any row. Use Plan 13's CLI renderer
        to surface the diff to the operator.

        Returns the server's apply-mode ``{items, summary}`` envelope by
        default (newly-created rows carry ``secret`` + ``trigger_url``
        exactly once), or the dry-run diff envelope when
        ``dry_run=True`` (no secret in the response).
        """
        body = {
            "items": [{**item, "managed_by": "code"} for item in webhooks],
        }
        path = f"/v1/agents/{agent_id}/webhooks"
        if dry_run:
            path = f"{path}?dry_run=true"
        return self._request("PUT", path, json=body)

    # -- Rate card -----------------------------------------------------------

    def get_rate_card(self, project_id: str) -> dict[str, Any]:
        """Return the project's per-model rate card. Empty dict if unset."""
        return self._request("GET", f"/v1/projects/{project_id}/rate-card")

    def set_rate_card(self, project_id: str, rate_card: dict[str, Any]) -> dict[str, Any]:
        """Replace the project's rate card wholesale."""
        return self._request("PUT", f"/v1/projects/{project_id}/rate-card", json=rate_card)

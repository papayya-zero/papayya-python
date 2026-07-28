"""Papayya config surfaces.

Two concerns live here, kept in one module so the CLI has one import:

1. Deploy specs (`ScheduleSpec` / `WebhookSpec` / `AgentSpec` / `EnvSpec`) —
   the data structures the reconcile path diffs against the control plane.
   Synthesized from `@schedule` / `@trigger` decorators (code-first); no
   `papayya.yaml` is read anywhere.
2. `~/.papayya/config.json` — CLI session state (api keys, current env).
   JSON, mutable, backwards-compatible with the legacy flat format.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)


# ---------------------------------------------------------------------------
# Deploy specs (synthesized from decorators)
# ---------------------------------------------------------------------------


class _Strict(BaseModel):
    """Forbid unknown fields so typos fail loud rather than disappearing."""

    model_config = ConfigDict(extra="forbid")


class ScheduleSpec(_Strict):
    cron: str = Field(..., description="Cron expression, UTC only in v1.")
    timezone: str = Field(
        default="UTC",
        description="IANA timezone for the cron expression. Defaults to UTC.",
    )


class WebhookSpec(_Strict):
    name: str = Field(..., description="Stable identifier — URL derives from this.")
    secret_env: str = Field(
        ..., description="Process env var holding the HMAC shared secret."
    )


class AgentSpec(_Strict):
    schedules: list[ScheduleSpec] = Field(default_factory=list)
    # Plan 34: `triggers:` is the documented key (a trigger is an inbound
    # invocation hook; webhook survives as the transport detail).
    # `webhooks:` is the pre-0.3.0 spelling, accepted as an alias — both
    # keys merge into `webhooks`, which downstream reconcile code reads.
    webhooks: list[WebhookSpec] = Field(default_factory=list)
    triggers: list[WebhookSpec] = Field(default_factory=list)

    @model_validator(mode="after")
    def _merge_triggers_into_webhooks(self) -> "AgentSpec":
        if self.triggers:
            names = {w.name for w in self.webhooks}
            dupes = sorted(names & {t.name for t in self.triggers})
            if dupes:
                raise ValueError(
                    f"trigger name(s) {dupes} declared under both `webhooks:` "
                    "and `triggers:` — use one key per trigger"
                )
            self.webhooks = [*self.webhooks, *self.triggers]
        return self


class EnvSpec(_Strict):
    agents: dict[str, AgentSpec] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# ~/.papayya/config.json — CLI session state
# ---------------------------------------------------------------------------
#
# v1 shape (legacy):
#   {"api_key": "...", "base_url": "...", "project_id": "...", "email": "..."}
#
# v2 shape (current):
#   {
#     "version": 2,
#     "current_env": "dev",
#     "envs": {
#       "dev": {"api_key": "...", "base_url": "...", "project_id": "...", "email": "..."}
#     },
#     "auth": {"jwt": "...", "email": "..."}   # account-level, optional
#   }
#
# Migration is transparent: load_cli_config() wraps legacy data under
# envs.dev. The flag `_migrated_from_v1` is set so the CLI main callback
# can print a one-time notice.

CONFIG_DIR = Path.home() / ".papayya"
CONFIG_FILE = CONFIG_DIR / "config.json"

DEFAULT_ENV = "dev"
CONFIG_SCHEMA_VERSION = 2

_LEGACY_ENV_KEYS = {"api_key", "base_url", "project_id", "email"}


def load_cli_config() -> dict[str, Any]:
    """Load the persisted CLI config, migrating legacy flat format on the fly."""
    try:
        raw = CONFIG_FILE.read_text()
    except FileNotFoundError:
        return {}
    except OSError:
        return {}
    try:
        data = json.loads(raw)
    except ValueError:
        return {}
    if not isinstance(data, dict):
        return {}
    return _migrate_config(data)


def save_cli_config(data: dict[str, Any]) -> None:
    """Persist config in v2 shape. Strips transient markers like _migrated_from_v1."""
    to_write = {k: v for k, v in data.items() if not k.startswith("_")}
    to_write.setdefault("version", CONFIG_SCHEMA_VERSION)
    to_write.setdefault("current_env", DEFAULT_ENV)
    to_write.setdefault("envs", {})
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE.write_text(json.dumps(to_write, indent=2) + "\n")


def _migrate_config(data: dict[str, Any]) -> dict[str, Any]:
    """Upgrade legacy flat config to v2 envs structure. Idempotent."""
    if data.get("version") == CONFIG_SCHEMA_VERSION:
        return data

    # Extract account-level auth (jwt/email) that survived from v1 saves.
    auth: dict[str, Any] = {}
    if data.get("jwt"):
        auth["jwt"] = data["jwt"]
    if data.get("email"):
        auth["email"] = data["email"]

    # Pull legacy per-project fields under envs.dev.
    env_block = {k: v for k, v in data.items() if k in _LEGACY_ENV_KEYS and v is not None}

    migrated: dict[str, Any] = {
        "version": CONFIG_SCHEMA_VERSION,
        "current_env": DEFAULT_ENV,
        "envs": {DEFAULT_ENV: env_block} if env_block else {},
    }
    if auth:
        migrated["auth"] = auth
    migrated["_migrated_from_v1"] = True
    return migrated


def current_env(data: dict[str, Any]) -> str:
    return str(data.get("current_env") or DEFAULT_ENV)


def env_config(data: dict[str, Any], env: str | None = None) -> dict[str, Any]:
    """Return a copy of the env's config dict (empty if env doesn't exist)."""
    env_name = env or current_env(data)
    envs = data.get("envs") or {}
    return dict(envs.get(env_name) or {})


def set_env_config(data: dict[str, Any], env: str, patch: dict[str, Any]) -> None:
    """Merge-write a single env's config into the loaded dict (mutates in place)."""
    envs = data.setdefault("envs", {})
    existing = dict(envs.get(env) or {})
    existing.update(patch)
    envs[env] = existing


def list_envs(data: dict[str, Any]) -> list[str]:
    return sorted((data.get("envs") or {}).keys())

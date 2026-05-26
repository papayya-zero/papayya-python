"""Papayya config surfaces.

Two concerns live here, kept in one module so the CLI has one import:

1. `papayya.yaml` — declarative project config (envs, schedules, webhooks).
   Pydantic-validated, read-only from the CLI's perspective.
2. `~/.papayya/config.json` — CLI session state (api keys, current env).
   JSON, mutable, backwards-compatible with the legacy flat format.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator


# ---------------------------------------------------------------------------
# papayya.yaml schema
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
    webhooks: list[WebhookSpec] = Field(default_factory=list)


class EnvSpec(_Strict):
    agents: dict[str, AgentSpec] = Field(default_factory=dict)


class PapayyaYaml(_Strict):
    version: Literal[1]
    envs: dict[str, EnvSpec] = Field(default_factory=dict)
    # v9 partition-key convention: name the metadata key whose value is
    # the partitioning axis for this project (most often: a tenant /
    # organization identifier). When set, every run() call must include
    # this key in its metadata; the SDK extracts it into the indexed
    # partition_key column so dashboards and downstream layers
    # (per-partition budgets, rate-limit pools, fairness) can filter
    # and aggregate without joining. Leave unset for single-partition
    # projects.
    partition_key: str | None = Field(
        default=None,
        description="Metadata key whose value identifies the partition (often a tenant) for this project.",
    )

    @field_validator("partition_key")
    @classmethod
    def _partition_key_nonempty(cls, value: str | None) -> str | None:
        # An empty-string partition_key is almost certainly a mistake —
        # pydantic would otherwise accept it and the SDK would extract
        # empty values without complaint. Fail loud at parse time instead.
        if value is not None and value.strip() == "":
            raise ValueError("partition_key must be a non-empty string when set")
        return value


class PapayyaYamlError(Exception):
    """Any problem loading or validating papayya.yaml."""


def load_yaml(path: str | Path) -> PapayyaYaml:
    """Load and validate a papayya.yaml file.

    Raises PapayyaYamlError with a human-readable message on any failure
    (missing file, malformed yaml, unknown version, schema violation).
    """
    p = Path(path)
    try:
        raw = p.read_text()
    except FileNotFoundError as exc:
        raise PapayyaYamlError(f"papayya.yaml not found at {p}") from exc
    except OSError as exc:
        raise PapayyaYamlError(f"Could not read {p}: {exc}") from exc

    try:
        data = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise PapayyaYamlError(f"Malformed yaml in {p}: {exc}") from exc

    if data is None:
        raise PapayyaYamlError(f"{p} is empty.")
    if not isinstance(data, dict):
        raise PapayyaYamlError(f"{p} must be a mapping at the top level.")

    # Unknown version gets a dedicated message before generic schema errors
    # (pydantic's Literal[1] mismatch otherwise reads as a cryptic type error).
    version = data.get("version")
    if version is not None and version != 1:
        raise PapayyaYamlError(
            f"papayya.yaml version {version!r} is not supported. "
            f"This CLI understands `version: 1`. Upgrade papayya or change the file."
        )

    try:
        return PapayyaYaml.model_validate(data)
    except ValidationError as exc:
        raise PapayyaYamlError(_format_validation_error(p, exc)) from exc


def _format_validation_error(path: Path, exc: ValidationError) -> str:
    lines = [f"Invalid papayya.yaml ({path}):"]
    for err in exc.errors():
        loc = ".".join(str(x) for x in err["loc"]) or "<root>"
        lines.append(f"  - {loc}: {err['msg']}")
    return "\n".join(lines)


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

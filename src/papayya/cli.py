"""Papayya CLI."""

from __future__ import annotations

import importlib.util
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import click

from papayya._cli_errors import SafeGroup
from papayya._config import (
    CONFIG_FILE as _CONFIG_FILE,
    DEFAULT_ENV as _DEFAULT_ENV,
    current_env as _current_env,
    env_config as _env_config,
    list_envs as _list_envs,
    load_cli_config as _load_cli_config,
    save_cli_config as _save_cli_config,
    set_env_config as _set_env_config,
)
from papayya._defaults import DEFAULT_BASE_URL, DEFAULT_DASHBOARD_URL
from papayya.api import APIClient, APIConfig, PapayyaAPIError, resolve_config


def _load_agent_from_file(path: str) -> Any:
    """Import a Python file and return the `agent` variable (legacy)."""
    filepath = Path(path).resolve()
    if not filepath.exists():
        click.echo(f"Error: File not found: {filepath}", err=True)
        sys.exit(1)

    spec = importlib.util.spec_from_file_location("_agent_module", filepath)
    if spec is None or spec.loader is None:
        click.echo(f"Error: Cannot load module from: {filepath}", err=True)
        sys.exit(1)

    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    agent = getattr(mod, "agent", None)
    if agent is None:
        click.echo("Error: Agent file must define an `agent` variable", err=True)
        sys.exit(1)

    return agent


def _discover_agents(path: str) -> list:
    """Import a Python file and return all @agent-registered functions.

    Falls back to the legacy `agent` variable if no decorators found.
    Returns a list of AgentRegistration objects.
    """
    from papayya.agent import get_registry, _registry, AgentRegistration

    # Clear registry before importing so we only get this file's agents
    _registry.clear()

    filepath = Path(path).resolve()
    if not filepath.exists():
        click.echo(f"Error: File not found: {filepath}", err=True)
        sys.exit(1)

    spec = importlib.util.spec_from_file_location("_agent_module", filepath)
    if spec is None or spec.loader is None:
        click.echo(f"Error: Cannot load module from: {filepath}", err=True)
        sys.exit(1)

    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    agents = list(get_registry().values())

    # Fallback: legacy agent variable + main()
    if not agents:
        agent_obj = getattr(mod, "agent", None)
        main_fn = getattr(mod, "main", None)
        if agent_obj is not None and main_fn is not None and callable(main_fn):
            agents.append(AgentRegistration(
                name=getattr(agent_obj, "name", "unknown"),
                model=getattr(agent_obj, "model", ""),
                instructions=getattr(agent_obj, "instructions", ""),
                fn=main_fn,
                tools=getattr(agent_obj, "tools", []),
                max_steps=getattr(agent_obj, "max_steps", 50),
                budget_usd=getattr(agent_obj, "budget_usd", None),
            ))

    if not agents:
        click.echo(
            "Error: No agents found. Use @agent decorator:\n\n"
            "    from papayya import agent\n\n"
            "    @agent(name='my-agent', model='gpt-4o-mini')\n"
            "    def my_agent(input_data):\n"
            "        ...\n",
            err=True,
        )
        sys.exit(1)

    return agents


def _distinct_step_count(checkpoints: Any) -> int:
    """How many STEPS a run has completed, from its checkpoint list.

    Not ``len(checkpoints)`` (Plan 41 R3 §T9). A step can be recorded more
    than once — a re-execution is its own checkpoint row — so the raw
    length counts executions and would report progress past the agent's
    own step count, which reads as a bug rather than as information.
    """
    return len({c.get("label") for c in (checkpoints or [])})


# Memo for _project_id_from_api_key, keyed by (api_key, base_url). _env_scope
# is called more than once per command by some code paths, and the lookup is a
# network round trip.
_PROJECT_ID_BY_KEY: dict[tuple[str, str], str | None] = {}


def _project_id_from_api_key(api_key: str | None, base_url: str) -> str | None:
    """Ask the control plane which project an API key should operate on.

    Onboarding tells a new user to export PAPAYYA_API_KEY and nothing else, so
    for their first `papayya deploy` the key is the ONLY thing they have. Every
    other source of a project id — PAPAYYA_PROJECT_ID, the saved env config —
    requires a step onboarding never mentions, so without this the documented
    path dead-ends on itself.

    Same rule `papayya login` already applies: exactly one reachable project
    means there is nothing to choose. Note the listing is ACCOUNT-scoped, not
    key-scoped — a project key still sees its siblings — so more than one
    project is genuinely ambiguous here and has to be resolved by the user.

    Best effort by construction: any unclear answer returns None so the
    caller's existing "no project id" error is still what the user sees. It
    never raises, because this runs on paths that would otherwise have failed
    anyway and must not turn a clear error into a traceback.
    """
    if not api_key:
        return None

    memo_key = (api_key, base_url)
    if memo_key in _PROJECT_ID_BY_KEY:
        return _PROJECT_ID_BY_KEY[memo_key]

    project_id: str | None = None
    try:
        api = APIClient(APIConfig(api_key=api_key, base_url=base_url))
        try:
            projects = api.list_projects()
        finally:
            api.close()
        if len(projects) == 1:
            project_id = projects[0].get("id")
    except Exception:
        project_id = None

    _PROJECT_ID_BY_KEY[memo_key] = project_id
    return project_id


# _resolve_project_id IS GONE, and so is _resolve_api_key below it (plan 53
# S8). Both were one-line delegates kept for `deploy`, the last command that
# had not been routed through _env_scope — and having them made deploy LOOK
# like it resolved things the same way as everything else while it read
# base_url straight off the flag. A wrapper that hides which of three values
# a command actually agrees with the rest of the CLI about is worse than no
# wrapper. `_env_scope` is the one door.


_DEPLOY_POLL_SECONDS = 1
_DEPLOY_TIMEOUT_SECONDS = 60


def _await_ready(api: APIClient, deployment_id: str) -> tuple[str, str]:
    """Read a deployment's terminal state. Returns ``(state, detail)``.

    A deployment is terminal the moment the row exists: the server writes
    ``ready`` at creation, after the artifact upload has already returned, and
    ``ready`` now means "the bundle endpoint can serve this version" — which is
    the only property the worker actually tests at lease time.

    This was ``_await_build``, polling every 3s for up to 15 minutes while an
    async ``docker build`` ran server-side. That build's base image had been
    deleted on purpose, so it always failed, every row said ``failed``, and
    deploy exited 1 saying "was NOT deployed" about a bundle that was live
    (plan 48 W2). Plan 49 deleted the build.

    The loop is kept, short, because ``failed`` is still a legal value on rows
    written before that change, and a first read can still race a slow write.
    """
    deadline = time.monotonic() + _DEPLOY_TIMEOUT_SECONDS
    while True:
        status = api.get_deployment(deployment_id)
        state = status.get("status", "unknown")
        if state == "ready":
            return "ready", f"v{status.get('version', '?')}"
        if state == "failed":
            return "failed", status.get("error_message", "unknown error")
        if time.monotonic() >= deadline:
            return "timed out", (
                f"still {state!r} after {_DEPLOY_TIMEOUT_SECONDS}s — "
                f"deployment {deployment_id}"
            )
        time.sleep(_DEPLOY_POLL_SECONDS)


def _preflight_dependencies(project_dir: str) -> None:
    """Fail the deploy if the bundle needs something the pool cannot import.

    ADR 0010. The managed worker pool carries a fixed dependency set and never
    pip-installs a bundle, so a `requirements.txt` naming anything outside that
    set is a run that will die on `import` — which is what plan 47 S2 recorded:
    ModuleNotFoundError on item 1 of 200, after deploy had said success.

    Deploy-time and named beats runtime and mysterious. Escape hatch for an
    import that is genuinely conditional (`try: import pandas except: ...`),
    because we are reading a manifest, not the code.
    """
    import os

    from papayya.runtime.baked_deps import baked_distributions, unsupported_requirements

    manifest = Path(project_dir) / "requirements.txt"
    if not manifest.is_file():
        return
    try:
        unsupported = unsupported_requirements(manifest.read_text(encoding="utf-8"))
    except OSError:
        return  # unreadable manifest is the bundler's problem, not the gate's
    if not unsupported:
        return

    if os.getenv("PAPAYYA_SKIP_DEP_PREFLIGHT") == "1":
        click.echo(
            "  Warning: requirements the hosted runtime does not carry "
            f"({', '.join(unsupported)}) — preflight skipped, imports may fail at run time.",
            err=True,
        )
        return

    available = baked_distributions() or set()
    click.echo("", err=True)
    click.echo(
        "Error: this bundle declares dependencies the hosted runtime does not carry:",
        err=True,
    )
    for line in unsupported:
        click.echo(f"  {line}", err=True)
    click.echo(
        "\nThe managed worker pool ships: "
        + ", ".join(sorted(available))
        + "\n(plus the Python standard library). Per-bundle dependency"
        "\ninstallation is not available yet — see ADR 0010."
        "\n\nIf the import is conditional, re-run with PAPAYYA_SKIP_DEP_PREFLIGHT=1.",
        err=True,
    )
    sys.exit(1)


def _find_or_create_agent(api: APIClient, project_id: str, reg) -> str:
    """Look up an agent by slug in the project, or create it. Returns agent ID."""
    slug = reg.name.lower().replace(" ", "-")

    # List agents and find by slug
    agents = api.list_agents(project_id)
    for a in agents:
        if a.get("slug") == slug:
            click.echo(f"  Found existing agent: {a['id']} ({slug})")
            return a["id"]

    # Create new agent
    click.echo(f"  Creating agent: {slug}")
    config: dict[str, Any] = {
        "model": reg.model,
        "max_steps": reg.max_steps,
    }
    if reg.budget_usd is not None:
        config["budget_usd"] = reg.budget_usd
    if reg.concurrency_per_key is not None:
        config["concurrency_per_key"] = reg.concurrency_per_key
    if reg.rate_limit_per_min is not None:
        config["rate_limit_per_min"] = reg.rate_limit_per_min

    result = api.create_agent(
        project_id=project_id,
        name=reg.name,
        slug=slug,
        config=config,
    )
    click.echo(f"  Created agent: {result['id']} ({slug})")
    return result["id"]


@dataclass(frozen=True)
class _EnvScope:
    """Resolved (env, api_key, project_id, base_url) for a server-hitting command."""

    env: str
    api_key: str | None
    project_id: str | None
    base_url: str


def _require_known_env(cfg: dict, requested: str | None) -> None:
    """Fail when --env names an env that does not exist.

    ``env_config`` returns ``{}`` for an unknown name, so an explicit
    ``--env staging`` on a machine with no `staging` silently fell through to
    the environment variables and ran — against whatever PAPAYYA_BASE_URL
    happened to say. A typo was invisible, and `--env prod` on a box that has
    only `dev` aimed at prod and hit dev.

    Only an EXPLICIT name is checked, and DEFAULT_ENV is always allowed: a user
    who has never run `papayya login` has no envs at all, and `papayya deploy`
    prints `papayya --env dev logs <run-id>` to exactly that person.
    """
    if not requested or requested == _DEFAULT_ENV:
        return
    known = _list_envs(cfg)
    if requested in known:
        return
    known_str = ", ".join(known) if known else "none configured"
    raise click.ClickException(
        f"no env named '{requested}' (known: {known_str}). "
        "Add one with `papayya login --env " + requested + "`."
    )


def _env_scope(ctx_obj: dict) -> _EnvScope:
    """Resolve env + credentials + base_url from the current CLI context.

    Precedence:
      - api_key: ctx flag / PAPAYYA_API_KEY env > envs[env].api_key
      - project_id: PAPAYYA_PROJECT_ID env > envs[env].project_id > the key's
        own project (see _project_id_from_api_key)
      - base_url: explicit --base-url / PAPAYYA_BASE_URL > envs[env].base_url > DEFAULT_BASE_URL

    The key-derived project id is LAST, below the saved config, so a config
    that names a project still wins — resolving from the key is the floor for
    someone who has only ever done what onboarding told them to, not an
    override of a choice the user made explicitly.
    """
    cfg = _load_cli_config()
    env_name = ctx_obj.get("env") or _current_env(cfg)
    _require_known_env(cfg, ctx_obj.get("env"))
    env_cfg = _env_config(cfg, env_name)

    api_key = (
        ctx_obj.get("api_key")
        or os.environ.get("PAPAYYA_API_KEY")
        or env_cfg.get("api_key")
    )
    project_id = (
        os.environ.get("PAPAYYA_PROJECT_ID")
        or env_cfg.get("project_id")
    )

    # Explicit --base-url (flag or PAPAYYA_BASE_URL) beats the env's stored
    # base_url; env-stored wins over DEFAULT_BASE_URL when the flag defaults.
    explicit = ctx_obj.get("base_url_source") in {"COMMANDLINE", "ENVIRONMENT"}
    ctx_base = ctx_obj.get("base_url") or DEFAULT_BASE_URL
    base_url = ctx_base if explicit else (env_cfg.get("base_url") or ctx_base)

    # Resolved last, and only when nothing else answered — it needs base_url,
    # and it costs a round trip on a path that would otherwise hard-error.
    if not project_id:
        project_id = _project_id_from_api_key(api_key, base_url)

    return _EnvScope(env=env_name, api_key=api_key, project_id=project_id, base_url=base_url)


def _require_api_key(scope: _EnvScope) -> str:
    if not scope.api_key:
        click.echo(
            # Capitalised to match its sibling ("Error: No project ID ...").
            "Error: No API key. Run `papayya login` to paste one, or set "
            f"PAPAYYA_API_KEY (mint a key in the dashboard at {DEFAULT_DASHBOARD_URL}).",
            err=True,
        )
        sys.exit(1)
    return scope.api_key


def _require_project_id(scope: _EnvScope) -> str:
    if not scope.project_id:
        # Reaching here with a key in hand means resolving the project FROM the
        # key did not produce a single answer — usually the key's account has
        # several projects. Don't assert the cause (the lookup is best-effort
        # and stays quiet when the control plane is unreachable); name the ways
        # out, since the old message asked for a project id without saying
        # where to get one.
        click.echo(
            f"Error: No project ID for env '{scope.env}'. Pick one explicitly:\n"
            "  papayya login --key <key> --project-id <id>\n"
            f"  papayya envs link {scope.env} --project-id ...\n"
            "  or set PAPAYYA_PROJECT_ID.",
            err=True,
        )
        sys.exit(1)
    return scope.project_id


@click.group(cls=SafeGroup)
@click.version_option(package_name="papayya", prog_name="papayya")
@click.option("--api-key", envvar="PAPAYYA_API_KEY", help="API key")
@click.option("--base-url", envvar="PAPAYYA_BASE_URL", default=DEFAULT_BASE_URL, help="Control plane URL")
@click.option("--env", "env", envvar="PAPAYYA_ENV", default=None,
              help="Override the current env (defaults to envs.current_env in ~/.papayya/config.json)")
@click.pass_context
def main(ctx: click.Context, api_key: str | None, base_url: str, env: str | None) -> None:
    """Papayya — durable background jobs for AI agents."""
    ctx.ensure_object(dict)
    ctx.obj["api_key"] = api_key
    ctx.obj["base_url"] = base_url
    ctx.obj["env"] = env
    # Stash where `base_url` came from so `_env_scope` can tell whether the
    # user passed an explicit override vs. landed on the default.
    source = ctx.get_parameter_source("base_url")
    ctx.obj["base_url_source"] = source.name if source is not None else "DEFAULT"

    # One-time notice when the legacy flat config gets wrapped into envs.dev.
    cfg = _load_cli_config()
    if cfg.get("_migrated_from_v1"):
        click.echo(
            f"Notice: migrated your existing config into env 'dev' ({_CONFIG_FILE}). "
            f"Run `papayya envs list` to see it.",
            err=True,
        )
        _save_cli_config(cfg)  # strips the marker


# ---------------------------------------------------------------------------
# signup / login
# ---------------------------------------------------------------------------

@main.command()
def signup() -> None:
    """Create your Papayya account (in the dashboard)."""
    # Account creation and key issuance live in the dashboard, not the CLI —
    # the CLI only consumes an API key you mint there. See `papayya login`.
    click.echo("Create your Papayya account in the dashboard:")
    click.echo(f"  {DEFAULT_DASHBOARD_URL}/register")
    click.echo("")
    click.echo("Then create a project + API key there and connect the CLI:")
    click.echo("  papayya login            # paste your API key")
    click.echo("  # or: export PAPAYYA_API_KEY=pk_...")


@main.command()
@click.option("--key", "key", default=None,
              help="API key from the dashboard. Prompted (hidden) if omitted.")
@click.option("--project-id", "project_id_opt", default=None,
              help="Project the key belongs to. Auto-resolved when the account has one project.")
@click.pass_context
def login(ctx: click.Context, key: str | None, project_id_opt: str | None) -> None:
    """Connect the CLI by pasting an API key from the dashboard.

    Sign up and mint a key at the dashboard (Project → API keys), then run
    `papayya login` and paste it — account creation lives in the dashboard,
    not the CLI. Setting PAPAYYA_API_KEY works too and skips this entirely.
    Pass `--env <name>` (global flag) to connect a specific env.
    """
    # NOT _env_scope: login is the command that WRITES an env, so it cannot
    # read its credentials from the env it is about to create. It reads the
    # flag — but re-logging in to an env that already exists must not silently
    # repoint it. `papayya --env dev login` on a dev configured for localhost
    # would otherwise move it to the public default just because a key was
    # re-pasted, which is S8's defect wearing a different hat: the flag's
    # DEFAULT is not a user's choice, only its explicit value is.
    cfg_now = _load_cli_config()
    target_env_guess = ctx.obj.get("env") or (
        _current_env(cfg_now) if cfg_now.get("envs") else "dev"
    )
    explicit_base = ctx.obj.get("base_url_source") in {"COMMANDLINE", "ENVIRONMENT"}
    base_url = ctx.obj["base_url"]
    if not explicit_base:
        base_url = _env_config(cfg_now, target_env_guess).get("base_url") or base_url

    api_key = (key or click.prompt("Paste your Papayya API key", hide_input=True) or "").strip()
    if not api_key:
        click.echo(
            f"Error: no API key provided. Create one at {DEFAULT_DASHBOARD_URL} "
            "(Project → API keys).",
            err=True,
        )
        sys.exit(1)

    api = APIClient(APIConfig(api_key=api_key, base_url=base_url))
    try:
        # Validate the key and discover which project(s) it can reach.
        try:
            projects = api.list_projects()
        except PapayyaAPIError as e:
            if e.status in (401, 403):
                click.echo(
                    "Error: that API key was rejected. Copy a current key from "
                    f"{DEFAULT_DASHBOARD_URL} (Project → API keys) and try again.",
                    err=True,
                )
                sys.exit(1)
            raise

        if not projects:
            click.echo(
                "Error: this key can't see any projects. Create a project in the "
                f"dashboard at {DEFAULT_DASHBOARD_URL} first.",
                err=True,
            )
            sys.exit(1)

        project_ids = [p["id"] for p in projects]
        if project_id_opt:
            if project_id_opt not in project_ids:
                click.echo(
                    f"Error: project '{project_id_opt}' isn't reachable with this key.\n"
                    f"  Reachable: {', '.join(project_ids)}",
                    err=True,
                )
                sys.exit(1)
            project_id = project_id_opt
        elif len(projects) == 1:
            project_id = project_ids[0]
        else:
            click.echo("This key reaches multiple projects — re-run with --project-id <id>:", err=True)
            for p in projects:
                click.echo(f"  {p['id']}  {p.get('name', '')}", err=True)
            sys.exit(1)

        cfg = _load_cli_config()
        # Honor a global --env override; otherwise write the current env (or dev).
        target_env = ctx.obj.get("env") or (_current_env(cfg) if cfg.get("envs") else "dev")
        _set_env_config(cfg, target_env, {
            "api_key": api_key,
            "base_url": base_url,
            "project_id": project_id,
        })
        cfg["current_env"] = target_env
        _save_cli_config(cfg)
        click.echo(f"✓ Connected! Config saved to {_CONFIG_FILE}")
        click.echo(f"  Env: {target_env}")
        click.echo(f"  Project: {project_id}")
        click.echo("\nNext: papayya deploy")
    finally:
        api.close()


@main.command()
def logout() -> None:
    """Remove the saved CLI config (~/.papayya/config.json)."""
    if not _CONFIG_FILE.exists():
        click.echo("Not signed in — no config to remove.")
        return
    existing = _load_cli_config()
    email = (existing.get("auth") or {}).get("email") or _env_config(existing).get("email")
    _CONFIG_FILE.unlink()
    who = f" ({email})" if email else ""
    click.echo(f"✓ Logged out{who}. Removed {_CONFIG_FILE}.")


# ---------------------------------------------------------------------------
# envs
# ---------------------------------------------------------------------------


@main.group()
def envs() -> None:
    """Manage papayya environments (each env maps to its own project + API key)."""


@envs.command("list")
def envs_list() -> None:
    """List all configured envs, marking the current one with an asterisk."""
    cfg = _load_cli_config()
    names = _list_envs(cfg)
    if not names:
        click.echo(
            "No envs configured yet.\n"
            "  Run `papayya signup` to create your first env,\n"
            "  or `papayya envs link <name> --project-id ... --api-key ...` "
            "to link an existing project."
        )
        return
    current = _current_env(cfg)
    for name in names:
        env_block = _env_config(cfg, name)
        project = env_block.get("project_id") or "<no project>"
        marker = "*" if name == current else " "
        click.echo(f" {marker} {name}  (project: {project})")


@envs.command("use")
@click.argument("name")
def envs_use(name: str) -> None:
    """Switch the current env. Subsequent commands use this env's credentials."""
    cfg = _load_cli_config()
    if name not in _list_envs(cfg):
        configured = ", ".join(_list_envs(cfg)) or "(none)"
        click.echo(
            f"Error: env '{name}' is not configured. Configured envs: {configured}",
            err=True,
        )
        sys.exit(1)
    cfg["current_env"] = name
    _save_cli_config(cfg)
    click.echo(f"✓ Current env: {name}")


@envs.command("link")
@click.argument("name")
@click.option("--project-id", required=True, help="Existing project ID (from the dashboard)")
@click.option("--api-key", "api_key", required=True, help="Project-scoped API key (cpk_...)")
@click.option("--base-url", default=None, help="Override the control plane URL for this env")
@click.pass_context
def envs_link(ctx: click.Context, name: str, project_id: str, api_key: str, base_url: str | None) -> None:
    """Link an existing project + API key into a named env."""
    if not name or not name.strip():
        click.echo("Error: env name must be non-empty.", err=True)
        sys.exit(1)
    cfg = _load_cli_config()
    _set_env_config(cfg, name, {
        "api_key": api_key,
        "base_url": base_url or ctx.obj["base_url"],
        "project_id": project_id,
    })
    if _current_env(cfg) not in _list_envs(cfg):
        cfg["current_env"] = name
    _save_cli_config(cfg)
    click.echo(f"✓ Linked env '{name}' to project {project_id}.")
    click.echo(f"  Switch to it with: papayya envs use {name}")


@envs.command("create")
@click.argument("name")
@click.pass_context
def envs_create(ctx: click.Context, name: str) -> None:
    """Set up a new env from a dashboard project + API key.

    Projects and keys are minted in the dashboard, then linked here — the CLI
    doesn't create accounts or projects. See `papayya envs link`.
    """
    if not name or not name.strip():
        click.echo("Error: env name must be non-empty.", err=True)
        sys.exit(1)

    cfg = _load_cli_config()
    if name in _list_envs(cfg):
        click.echo(
            f"Error: env '{name}' already exists. Use `papayya envs use {name}` "
            f"to switch, or pick a different name.",
            err=True,
        )
        sys.exit(1)

    click.echo(f"Create a project + API key in the dashboard at {DEFAULT_DASHBOARD_URL},")
    click.echo(f"then link them into env '{name}':")
    click.echo("")
    click.echo(f"  papayya envs link {name} --project-id <id> --api-key <key>")
    click.echo("")
    click.echo(f"Or paste the key interactively: papayya --env {name} login")


# ---------------------------------------------------------------------------
# deploy
# ---------------------------------------------------------------------------

@main.command()
@click.argument("file", required=False, default=None)
@click.option("--agent-id", default=None, help="Agent ID (overrides auto-discovery)")
@click.option("--project-id", default=None, envvar="PAPAYYA_PROJECT_ID", help="Project ID")
@click.option("--runtime", default="python", type=click.Choice(["python", "node"]), help="Runtime type")
@click.option("--entrypoint", default=None, help="Entrypoint file (default: auto-detected)")
@click.option("--dry-run", "dry_run", is_flag=True,
              help="Show planned trigger changes without applying them.")
@click.pass_context
def deploy(
    ctx: click.Context,
    file: str | None,
    agent_id: str | None,
    project_id: str | None,
    runtime: str,
    entrypoint: str | None,
    dry_run: bool,
) -> None:
    """Deploy agent code to the control plane.

    \b
    Usage:
      papayya deploy              # auto-discover agent.py in cwd
      papayya deploy agents.py    # explicit file
      papayya deploy --dry-run    # preview trigger reconciliation

    Schedules and webhooks declared via `@schedule` / `@trigger` decorators on
    the deployed `@agent` functions are reconciled against the selected env's
    project after the bundle upload.
    """
    from papayya.bundler import bundle_project
    from papayya import _reconcile

    # Auto-discover file
    if file is None:
        if Path("agent.py").exists():
            file = "agent.py"
        else:
            click.echo("Error: No agent.py found in current directory. Specify a file:\n  papayya deploy my_agents.py", err=True)
            sys.exit(1)

    # Env selection is code-first: --env / PAPAYYA_ENV / current_env in
    # ~/.papayya/config.json (all folded into ctx.obj["env"] by the main
    # callback). There is no papayya.yaml env block.
    #
    # THROUGH _env_scope, like every other server-hitting command (plan 47 S8).
    # This command used to resolve the same three things by hand — key via
    # _resolve_api_key, project via _resolve_project_id, and base_url straight
    # off ctx.obj["base_url"] — and the third one is what broke it: ctx.obj
    # holds the FLAG, whose default is DEFAULT_BASE_URL, so deploy dialled
    # api.getpapayya.com on a machine whose env said localhost and died on
    # DNS. Every deploy in plans 47 and 48 needed PAPAYYA_BASE_URL= in front
    # of it, on the third command of the quickstart.
    #
    # Driving it turned up worse than the walk recorded. _env_scope is also
    # where plan 52 E1 put the unknown-env check, so `papayya --env <typo>
    # deploy` skipped it: without a project id it reported "No API key found.
    # Run `papayya signup`" — a wrong diagnosis pointing at a wrong remedy —
    # and WITH --project-id it validated nothing at all. It uploaded a bundle,
    # printed `Env: nope-not-real`, and signed off telling the user to run
    # `papayya --env nope-not-real logs <run-id>`, which the same binary
    # refuses. The product printed an instruction it had already decided was
    # invalid. That is plan 52's whole shape, on the command plan 52 quoted.
    scope = _env_scope(ctx.obj)
    env_name: str | None = ctx.obj.get("env")

    resolved_key = scope.api_key
    if not resolved_key:
        click.echo(
            "Error: No API key found.\n"
            "  Run `papayya signup` first, or set PAPAYYA_API_KEY.",
            err=True,
        )
        sys.exit(1)

    config = APIConfig(api_key=resolved_key, base_url=scope.base_url)
    api = APIClient(config)

    if entrypoint is None:
        entrypoint = Path(file).name

    try:
        # Discover @agent functions
        agents = _discover_agents(file)
        click.echo(f"Found {len(agents)} agent(s): {', '.join(a.name for a in agents)}")

        # Bundle the project (one bundle for all agents — they share code)
        project_dir = str(Path(file).resolve().parent)
        click.echo(f"Bundling project from {project_dir}...")
        tarball, sha256 = bundle_project(project_dir, entrypoint=entrypoint)
        click.echo(f"  Archive: {len(tarball)} bytes (SHA256: {sha256[:16]}...)")

        _preflight_dependencies(project_dir)

        # Resolve project ID for agent lookup/create. The --project-id flag
        # still wins; the scope is the floor, and it already applied the
        # PAPAYYA_PROJECT_ID > envs[env].project_id > derive-from-key
        # precedence this used to re-implement.
        if not project_id:
            project_id = scope.project_id
        if not project_id and not agent_id:
            # Not `papayya signup` — that only opens the dashboard, which is
            # where the user just came from.
            click.echo(
                "Error: No project ID. Run `papayya login` to connect the CLI, "
                "or set PAPAYYA_PROJECT_ID.",
                err=True,
            )
            sys.exit(1)

        # Deploy each agent; track slug -> agent_id for the reconciler.
        deployed: dict[str, str] = {}
        failed_deploys: list[tuple[str, str]] = []
        for reg in agents:
            click.echo(f"\nDeploying {reg.name}...")

            # Resolve agent ID
            if agent_id and len(agents) == 1:
                resolved_agent_id = agent_id
            elif project_id:
                resolved_agent_id = _find_or_create_agent(api, project_id, reg)
            else:
                click.echo(f"  Error: Cannot resolve agent ID for '{reg.name}'. Pass --agent-id or --project-id.", err=True)
                continue

            # Upload
            click.echo("  Uploading deployment...")
            result = api.upload_deployment(
                agent_id=resolved_agent_id,
                tarball=tarball,
                runtime=runtime,
                entrypoint=entrypoint,
            )
            deployment_id = result.get("id", "unknown")
            click.echo(f"  Deployment ID: {deployment_id}")
            click.echo(f"  Version: {result.get('version', '?')}")

            state, detail = _await_ready(api, deployment_id)
            if state != "ready":
                # NOT deployed: skip the success line and keep it out of
                # `deployed`, so nothing downstream treats it as live.
                click.echo(f"  Deployment {state}: {detail}", err=True)
                failed_deploys.append((reg.name, detail))
                continue

            slug = reg.name.lower().replace(" ", "-")
            deployed[slug] = resolved_agent_id
            click.echo(f"  Deployed {slug} → {resolved_agent_id}")

        # Stop here, BEFORE reconciling triggers. A failed build used to fall
        # through to the success line, the "Next: papayya run ..." nudge and
        # exit 0 — so the command told the user to invoke something that was
        # never built, and the very next command contradicted it. Exiting
        # before the reconcile also avoids compounding a failed deploy by
        # mutating schedules and webhooks for a half-deployed bundle.
        if failed_deploys:
            click.echo("", err=True)
            for name, why in failed_deploys:
                # First line only. The full detail already went out above;
                # repeating it whole buries the one sentence that matters.
                headline = why.strip().splitlines()[0] if why.strip() else "unknown error"
                click.echo(f"Error: {name} was NOT deployed — {headline}", err=True)
            sys.exit(1)

        # Reconcile triggers. Source = @schedule / @trigger decorators
        # attached to @agent functions — populated in the module-level
        # registry by _discover_agents above, harvested via
        # _decorator_synthesis.
        #
        # The synthesis helper is imported lazily because it transitively
        # pulls papayya.decorators (croniter + zoneinfo). Eager import at
        # cli-module load time changes module-init ordering enough to mask
        # cross-process SQLite WAL writes in the worker subprocess test
        # (see Plan 11's __init__.py __getattr__ fix for context).
        from papayya.agent import get_registry
        from papayya._decorator_synthesis import env_spec_from_registry
        env_spec = env_spec_from_registry(get_registry())
        has_triggers = any(
            a.schedules or a.webhooks for a in env_spec.agents.values()
        )
        if has_triggers:
            label = f" for env '{env_name}'" if env_name else ""
            click.echo(f"\nReconciling triggers{label}...")
            try:
                plan = _reconcile.diff_env(env_spec, deployed, api)
            except _reconcile.ReconcileError as e:
                click.echo(f"Error: {e}", err=True)
                sys.exit(1)

            _print_reconcile_plan(plan, api_base_url=scope.base_url)

            # Plan 13 — PUT-dry-run preview for managed_by='code'. Probes
            # the same PUT endpoints apply_plan would call, so the preview
            # is byte-faithful to what the next deploy would do. Runs on
            # every deploy that hits reconcile (both --dry-run and apply)
            # so the operator sees the same diff before the apply attempt.
            try:
                managed_diffs = _collect_managed_diff(env_spec, deployed, api)
            except PapayyaAPIError as e:
                click.echo(f"Error (managed_by preview): {e}", err=True)
                sys.exit(1)
            _print_managed_diff(managed_diffs)

            if dry_run:
                click.echo("\nDry run — no changes applied.")
                return

            if plan.is_noop:
                click.echo("\nNo changes to apply.")
            else:
                result = _reconcile.apply_plan(plan, api)
                _print_apply_result(result, api_base_url=scope.base_url)
                if result.error is not None:
                    sys.exit(1)

        # Per-env next-step nudge
        if deployed:
            current = env_name or _current_env(_load_cli_config())
            first_slug = next(iter(deployed))
            click.echo(f"\nEnv: {current}")
            click.echo("\nNext:")
            click.echo(f'  papayya run {first_slug} "your input"')
            click.echo(f"  papayya --env {current} logs <run-id>")

    finally:
        api.close()


def _print_reconcile_plan(plan, *, api_base_url: str) -> None:
    """Render the plan to stdout before apply (and as the dry-run output)."""
    for agent_plan in plan.agents:
        click.echo(f"\nagent: {agent_plan.slug} ({agent_plan.agent_id})")
        if agent_plan.is_noop:
            click.echo("  (no changes)")
            continue
        # Schedules
        for op in agent_plan.schedule_ops:
            prefix = "+" if op.kind == "create" else "-"
            suffix = "create" if op.kind == "create" else "delete (not in yaml)"
            click.echo(f"  {prefix} schedule {op.cron:<22} {suffix}")
        # Webhooks — surface rotation warning before the delete/create pair.
        rotating_names = {
            op.name for op in agent_plan.webhook_ops
            if op.kind == "create" and op.reason == "rename"
        }
        if rotating_names:
            for name in sorted(rotating_names):
                click.echo(
                    f"  WARNING: rotating webhook '{name}' — downstream senders "
                    "must update URL and secret"
                )
        for op in agent_plan.webhook_ops:
            prefix = "+" if op.kind == "create" else "-"
            if op.kind == "create":
                tag = "create (rename)" if op.reason == "rename" else "create"
            else:
                tag = "delete (rename)" if op.reason == "removed" else "delete (not in yaml)"
            click.echo(f"  {prefix} webhook  {op.name:<22} {tag}")
            if op.kind == "create" and op.secret_env:
                if not os.environ.get(op.secret_env):
                    click.echo(
                        f"      note: $ {op.secret_env} is not set locally — "
                        "you'll need the secret printed after create"
                    )


def _collect_managed_diff(
    env_spec,
    deployed: dict[str, str],
    api,
) -> list[tuple[str, str, dict]]:
    """Probe the PUT endpoints with ?dry_run=true for each agent's
    schedules + webhooks. Returns ordered (slug, resource, diff) tuples
    so the renderer prints per agent with schedules before webhooks.

    Probes both endpoints unconditionally even when the yaml declares
    zero of a given resource type — an empty desired set is still a
    valid desired state, and the operator needs to see "this deploy
    would delete the N leftover managed_by='code' rows" before it
    happens.
    """
    out: list[tuple[str, str, dict]] = []
    for slug, agent_spec in env_spec.agents.items():
        agent_id = deployed[slug]
        sched_payload = [
            {"cron_expression": s.cron, "timezone": getattr(s, "timezone", "UTC")}
            for s in agent_spec.schedules
        ]
        # `secret_env` is NOT sent (plan 53). It is a LOCAL declaration —
        # which env var the customer reads the signing secret from — and the
        # control plane has no such concept anywhere: zero occurrences outside
        # tests. The server's item struct rejects it, so this probe 400'd on
        # every deploy carrying a @trigger.
        #
        # The apply path (_reconcile.apply_plan) has always sent `{"name": ...}`
        # alone. That divergence is the whole defect: this probe's own
        # docstring promises it is "byte-faithful to what the next deploy would
        # do", and it was sending a body the deploy never sends and the server
        # never accepted.
        wh_payload = [{"name": w.name} for w in agent_spec.webhooks]
        sched_diff = api.put_schedules(agent_id, sched_payload, dry_run=True)
        wh_diff = api.put_webhooks(agent_id, wh_payload, dry_run=True)
        out.append((slug, "schedule", sched_diff))
        out.append((slug, "webhook", wh_diff))
    return out


def _summarize_field_changes(before: dict, after: dict) -> str:
    """`"k: 'old' → 'new', k2: 'old2' → 'new2'"` for an update line.
    Unchanged keys are omitted. Capped at three keys to keep the line
    scannable; truncation marker is `…`.
    """
    parts: list[str] = []
    for key in sorted(set(before) | set(after)):
        if before.get(key) != after.get(key):
            parts.append(f"{key}: {before.get(key)!r} → {after.get(key)!r}")
        if len(parts) >= 3:
            parts.append("…")
            break
    return ", ".join(parts)


def _print_managed_diff(diffs: list[tuple[str, str, dict]]) -> None:
    """Render the managed_by='code' diff produced by the PUT dry-run
    probes. Composes below the legacy reconcile-plan output (additive)."""
    if not diffs:
        return
    click.echo("\nmanaged_by='code' diff (PUT-replace preview):")
    current_slug: str | None = None
    for slug, resource, diff in diffs:
        if slug != current_slug:
            click.echo(f"\nagent: {slug}")
            current_slug = slug
        creates = diff.get("create") or []
        updates = diff.get("update") or []
        deletes = diff.get("delete") or []
        n_unmanaged = diff.get("unmanaged_skipped", 0)
        click.echo(
            f"  managed_by='code' {resource}s: "
            f"{len(creates)} to create, {len(updates)} to update, "
            f"{len(deletes)} to delete "
            f"({n_unmanaged} unmanaged rows untouched)"
        )
        for op in creates:
            label = op.get("cron_expression") or op.get("name") or "?"
            click.echo(f"    + {resource} {label:<22} create")
        for op in updates:
            before = op.get("before") or {}
            after = op.get("after") or {}
            label = (
                after.get("cron_expression")
                or after.get("name")
                or op.get("id")
                or "?"
            )
            changes = _summarize_field_changes(before, after)
            suffix = f" ({changes})" if changes else ""
            click.echo(f"    ~ {resource} {label:<22} update{suffix}")
        for op in deletes:
            label = op.get("cron_expression") or op.get("name") or "?"
            click.echo(f"    - {resource} {label:<22} delete")


def _print_apply_result(result, *, api_base_url: str) -> None:
    """Render apply output (webhook secrets/URLs + final summary)."""
    for created in result.created_webhooks:
        name = created.get("name", "?")
        secret = created.get("secret", "")
        trigger_url = created.get("trigger_url") or ""
        if trigger_url and not trigger_url.startswith("http"):
            trigger_url = f"{api_base_url.rstrip('/')}{trigger_url}"
        secret_env = created.get("secret_env")
        click.echo(f"\nWebhook '{name}' created:")
        click.echo(f"  URL:    {trigger_url}")
        if secret_env:
            unset_note = " (not set locally)" if not os.environ.get(secret_env) else ""
            click.echo(
                f"  secret: {secret}  "
                f"(store in ${secret_env} — only shown once{unset_note})"
            )
        else:
            click.echo(f"  secret: {secret}  (only shown once)")

    if result.error is not None:
        click.echo(f"\nError: {result.error}", err=True)
        click.echo(
            f"Applied {result.applied} of {result.total} operations.",
            err=True,
        )
    else:
        click.echo(f"\nApplied {result.applied} of {result.total} operations.")


# ---------------------------------------------------------------------------
# triage — unified Needs Attention feed across DLQ + quarantine
#
# v1→v2 cutover: the read feed is durable-backed. Both lanes are live —
# quarantine actions (release/discard) shipped with the cutover; the DLQ-lane
# dispositions (replayed/skipped/acknowledged) came back in migration 070
# (Plan 41 R1), which is what lets the feed drain instead of accumulating
# every degraded run forever.
# ---------------------------------------------------------------------------

@main.group()
def triage() -> None:
    """Unified triage feed: list runs needing attention; retry/dismiss/
    acknowledge dispatch to the right per-state endpoint client-side."""


@triage.command("list")
@click.option("--partition-key", "partition_key", default=None,
              help="Filter by the durable run's partition key")
@click.option("--tenant", default=None, help="Alias for --partition-key")
@click.option(
    "--kind",
    type=click.Choice(["all", "dlq", "quarantine"]),
    default="all",
    show_default=True,
)
@click.option(
    "--limit",
    type=int,
    default=50,
    show_default=True,
    help="Page size (server-clamped to [1,200])",
)
@click.pass_context
def triage_list(
    ctx: click.Context,
    partition_key: str | None,
    tenant: str | None,
    kind: str,
    limit: int,
) -> None:
    """List runs awaiting triage (NDJSON, auto-paginates)."""
    client = _make_papayya_client(ctx)
    try:
        for row in client.triage.iter(
            partition_key=partition_key,
            tenant=tenant,
            kind=kind,
            page_size=limit,
        ):
            click.echo(json.dumps(row))
    except PapayyaAPIError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    finally:
        client.close()


# Each verb dispatches on the row's lane, because the two lanes reach the same
# operator intent through different endpoints: a quarantined run is non-terminal
# and its verbs move it (release resumes, discard abandons), while a dlq row is
# already terminal and its verbs only record what the operator decided. Keeping
# the dispatch here rather than in the server preserves `available_actions` as
# the single description of what a row supports.


@triage.command("retry")
@click.argument("run_id")
@click.pass_context
def triage_retry(ctx: click.Context, run_id: str) -> None:
    """Retry a triage row.

    Quarantined: resumes the run in-place (``/release``). Degraded/failed:
    mints a new run that re-drives the captured item (``/replay``) and marks
    the source ``dlq_disposition='replayed'`` so it leaves the feed.
    """
    client = _make_papayya_client(ctx)
    try:
        run = client.items.get(run_id)
        if run.get("status") == "quarantine":
            out = client.items.release(run_id)
        else:
            out = client.items.replay(run_id)
        click.echo(json.dumps(out, indent=2))
    except PapayyaAPIError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    finally:
        client.close()


@triage.command("dismiss")
@click.argument("run_id")
@click.pass_context
def triage_dismiss(ctx: click.Context, run_id: str) -> None:
    """Dismiss a triage row.

    Quarantined: abandons the run (``/discard``). Degraded/failed: drains it
    from the feed without re-driving (``dlq_disposition='skipped'``).
    """
    client = _make_papayya_client(ctx)
    try:
        run = client.items.get(run_id)
        if run.get("status") == "quarantine":
            out = client.items.discard(run_id)
        else:
            out = client.items.dismiss(run_id)
        click.echo(json.dumps(out, indent=2))
    except PapayyaAPIError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    finally:
        client.close()


@triage.command("acknowledge")
@click.argument("run_id")
@click.pass_context
def triage_acknowledge(ctx: click.Context, run_id: str) -> None:
    """Acknowledge a degraded/failed triage row.

    Records that it was seen and drains it from the feed
    (``dlq_disposition='acknowledged'``) without re-driving or declining it.
    Quarantine rows don't offer this — see `available_actions` on the row.
    """
    client = _make_papayya_client(ctx)
    try:
        out = client.items.acknowledge(run_id)
        click.echo(json.dumps(out, indent=2))
    except PapayyaAPIError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    finally:
        client.close()


@main.command("replay")
@click.argument("run_positional", required=False)
@click.option("--run", "run_id", default=None,
              help="Run ID to replay. Mints a new run that re-drives the "
                   "original run's captured item.")
@click.option("--tenant", "tenant", default=None,
              help="Replay only the run's slice whose partition_key "
                   "(tenant) matches this value.")
@click.option(
    "--latest",
    "latest",
    is_flag=True,
    default=False,
    help=(
        "Replay on the agent's current version instead of the version "
        "captured on the original run (ADR-0002 #7). Pre-#7 runs whose "
        "agent_version is NULL always replay on latest."
    ),
)
@click.option(
    "--force",
    "force",
    is_flag=True,
    default=False,
    help=(
        "Re-drive a clean run too. By default only a run that didn't work "
        "(worst outcome != ok, or a failed/quarantined run) is replayable."
    ),
)
@click.option("--wait/--no-wait", "wait", default=True,
              help="Poll the new run to completion (default) or return "
                   "immediately after triggering.")
@click.pass_context
def replay_cmd(
    ctx: click.Context,
    run_positional: str | None,
    run_id: str | None,
    tenant: str | None,
    latest: bool,
    force: bool,
    wait: bool,
) -> None:
    """Replay work that didn't work, in the cloud.

    \b
    Usage:
      papayya replay <run_id>                 # replay the run
      papayya replay --run <run_id>           # same, explicit flag
      papayya replay <run_id> --tenant acme   # one partition slice only
      papayya replay <run_id> --latest        # on the agent's current version
      papayya replay <run_id> --force         # re-drive even a clean run

    Hosted replay (Plan 37 Unit R): mints a NEW run linked to the original
    via replayed_from and re-drives its captured item through the worker
    pool. Only a terminal run (completed/failed/quarantined) replays; by
    default only a run that didn't work is eligible (--force overrides).

    Local single-item / --from-step replay ran off the SQLite ledger, which
    is deactivated (Plan 37) — replay is a cloud operation now.
    """
    # Resolve the run id (positional wins over --run).
    if run_positional is not None and run_id is not None and run_positional != run_id:
        click.echo("Error: run id given twice (positional and --run). Pick one.", err=True)
        sys.exit(1)
    target = run_positional if run_positional is not None else run_id
    if not target:
        click.echo('Error: run id required.\n  papayya replay <run_id>', err=True)
        sys.exit(1)

    scope = _env_scope(ctx.obj)
    resolved_key = _require_api_key(scope)
    config = APIConfig(api_key=resolved_key, base_url=scope.base_url)
    api = APIClient(config)

    try:
        try:
            result = api.replay_run(target, tenant=tenant, latest=latest, force=force)
        except PapayyaAPIError as exc:
            click.echo(f"Error: replay failed ({exc.status}): {exc}", err=True)
            sys.exit(1)

        new_run_id = result.get("run_id", "unknown")
        click.echo(f"Replay triggered: {new_run_id}")
        click.echo(f"  Replayed from: {result.get('replayed_from', target)}")
        click.echo(f"  Status: {result.get('status', 'unknown')}")

        if not wait:
            return

        click.echo("Waiting for completion...")
        while True:
            time.sleep(2)
            status_resp = api.get_run(new_run_id)
            state = status_resp.get("status", "unknown")
            done = _distinct_step_count(status_resp.get("checkpoints"))
            click.echo(f"  {done} step(s) — {state}")
            if state in ("completed", "failed", "quarantined", "paused"):
                click.echo(f"\nFinal status: {state}")
                break
    finally:
        api.close()


@main.command("resume")
@click.argument("run_id")
@click.pass_context
def resume_cmd(ctx: click.Context, run_id: str) -> None:
    """Resume a run a fence stopped.

    \b
      papayya resume <run_id>

    The verb for a run something DECIDED to stop, as distinct from `replay`,
    which is the verb for a run that ENDED badly. The server refuses each on
    the other's states, and until this command existed a fenced run had no way
    out through the CLI at all: `replay` answers 409 "run is paused; only a
    terminal run can be replayed", and there was nothing else to type.

    Resuming clears the pause and re-queues the run's parked item. Two numbers
    come back and they are worth reading together — see the output.
    """
    scope = _env_scope(ctx.obj)
    config = APIConfig(api_key=_require_api_key(scope), base_url=scope.base_url)
    api = APIClient(config)
    try:
        try:
            result = api.resume_run(run_id)
        except PapayyaAPIError as exc:
            # The 409 names the other verb, because being told "this is not
            # resumable" without being told what IS is how a dead end at 2am
            # gets built.
            if getattr(exc, "status", None) == 409:
                click.echo(
                    f"Error: {exc}\n"
                    f"  A run that ENDED badly is replayed, not resumed:\n"
                    f"    papayya replay {run_id}",
                    err=True,
                )
                sys.exit(1)
            click.echo(f"Error: resume failed ({exc.status}): {exc}", err=True)
            sys.exit(1)

        click.echo(f"Resumed: {result.get('run_id', run_id)}")
        click.echo(f"  Status: {result.get('status', 'unknown')}")

        # THE TWO NUMBERS, and why both are printed rather than a cheerful
        # "resumed". `reexecuting=0` means this will produce exactly what it
        # produced before, which is almost never what the person resuming
        # wants; `redriven=false` means the pause is cleared but nothing was
        # re-queued, so the run will not move at all.
        reexecuting = result.get("reexecuting", 0)
        if result.get("redriven"):
            click.echo(f"  Re-executing: {reexecuting} step(s) the fence objected to")
            if not reexecuting:
                click.echo(
                    "  Nothing will be re-executed, so this will produce what it "
                    "produced before.\n"
                    f"  If you changed the code, replay it instead: papayya replay {run_id}"
                )
        else:
            click.echo(
                "  Nothing was re-queued: this run's lease was already completed, "
                "so there is no parked item to hand back to a worker.\n"
                f"  Re-drive it instead: papayya replay {run_id}"
            )
    finally:
        api.close()


# ---------------------------------------------------------------------------
# pull — materialize a production incident as local fixtures
#
# Plan 41 R4; ADR 0009 D7b. The cohort's records already carry input, the real
# bad output, verdict, tenant and timestamp — which IS a fixture set. Written
# to disk, the production failure becomes reproducible locally, offline and
# deterministically: fix against it, verify against it before spending a cent
# of inference, then re-drive only that cohort.
# ---------------------------------------------------------------------------

@main.command("pull")
@click.option("--like", "like_record", default=None,
              help="SEARCH BY EXAMPLE: paste one record id you know is bad and "
                   "see what selects everything like it, with the blast radius "
                   "of each. Writes nothing — pick one and pass --probe.")
@click.option("--probe", "probe_id", default=None,
              help="Pull the cohort a derived predicate selects (from --like).")
@click.option("--drift-episode", "drift_episode", default=None,
              help="Pull the records a detected change moved.")
@click.option("--agent", default=None, help="Narrow the cohort to one agent slug.")
@click.option("--tenant", default=None,
              help="Narrow to one partition key (the customer's own tenant id).")
@click.option("--run", "run_id", default=None,
              help="Narrow to one run's records. One predicate term among "
                   "several, not the addressing scheme.")
@click.option("--outcome", default=None,
              type=click.Choice(["not_ok", "any", "degraded", "failed"]),
              help="Verdict axis. Default not_ok — everything that didn't work.")
@click.option("--flagged", is_flag=True, default=False,
              help="Only records a human said were wrong — a thumbs-down, a "
                   "ticket, a refund. IMPLIES --outcome any unless you pass "
                   "one: what people flag usually completed fine.")
@click.option("--since", default=None, help="Window start (RFC3339).")
@click.option("--until", default=None, help="Window end (RFC3339).")
@click.option("--include-triaged", is_flag=True, default=False,
              help="Re-admit records someone already dispositioned. Off by "
                   "default so a pull doesn't resurrect handled work.")
@click.option("--limit", type=int, default=None,
              help="Cap the number of fixtures written.")
@click.option("--out", "out_dir", default="./fixtures",
              help="Directory to write fixtures into (default ./fixtures).")
@click.option("--dry-run", is_flag=True, default=False,
              help="Show what the predicate selects without writing anything.")
@click.pass_context
def pull_cmd(
    ctx: click.Context,
    like_record: str | None,
    probe_id: str | None,
    drift_episode: str | None,
    agent: str | None,
    tenant: str | None,
    run_id: str | None,
    outcome: str | None,
    flagged: bool,
    since: str | None,
    until: str | None,
    include_triaged: bool,
    limit: int | None,
    out_dir: str,
    dry_run: bool,
) -> None:
    """Pull a cohort of failed records to disk as reproducible fixtures.

    \b
    Describe the cohort yourself:
      papayya pull --agent enrich --tenant acme --since 2026-08-01T00:00:00Z

    \b
    Or point at one bad record and let the predicate be derived:
      papayya pull --like 9c21e8b4-...        # what selects everything like it?
      papayya pull --probe bf88-...           # pull the one you picked
      papayya verify --fixtures ./fixtures --agent-module app

    \b
    Or take what the world complained about, and compose it with either:
      papayya pull --agent enrich --flagged
      papayya pull --probe bf88-... --flagged   # like this one AND flagged
    """
    from papayya.cohort_diff import record_verdict
    from papayya.fixtures import fixture_from_record, write_fixtures

    scope = _env_scope(ctx.obj)
    config = APIConfig(api_key=_require_api_key(scope), base_url=scope.base_url)
    api = APIClient(config)

    try:
        # SEARCH BY EXAMPLE (ADR 0009 §4 step 3). This branch DERIVES and
        # PRINTS; it never writes a fixture, for the same reason --dry-run is
        # the documented default workflow — see the blast radius, then decide.
        #
        # It deliberately does not auto-pick the largest proposal and pull it.
        # The proposals are different definitions of "like this" (the SDK's own
        # verdict token, a hollow field, a short output), they routinely nest,
        # and choosing between them is the operator's judgement about their own
        # domain. Picking for them and then re-driving what we picked is the
        # failure this product exists to catch, wearing the shape of
        # convenience.
        if like_record:
            if probe_id or drift_episode:
                click.echo("Error: --like derives a predicate; --probe and "
                           "--drift-episode use one. Pass one at a time.", err=True)
                sys.exit(1)
            _echo_probe_proposals(api, like_record, since, until)
            return

        # The fixture's record of what it was pulled by. `outcome` states the
        # EFFECTIVE axis, not the flag the operator typed: --flagged implies
        # `any` server-side, and a fixture claiming it came from a not_ok
        # cohort while holding records that completed ok would be a small lie
        # in the one artifact meant to be re-read weeks later.
        predicate = {
            "probe": probe_id, "drift_episode": drift_episode,
            "agent": agent, "tenant": tenant, "run_id": run_id,
            "outcome": outcome or ("any" if flagged else "not_ok"),
            "flagged": flagged, "since": since, "until": until,
            "include_triaged": include_triaged,
        }
        if flagged and outcome is None:
            click.echo("Selecting flagged records at --outcome any: what a "
                       "human flags usually completed fine.", err=True)
        try:
            resp = api.get_cohort(
                probe=probe_id, drift_episode=drift_episode,
                agent=agent, tenant=tenant, run_id=run_id, outcome=outcome,
                since=since, until=until, flagged=flagged,
                include_triaged=include_triaged, limit=limit,
            )
        except ValueError as exc:
            # The mutually-exclusive-doors rule, refused client-side before it
            # travels. The server refuses it too; this is the earlier copy.
            click.echo(f"Error: {exc}", err=True)
            sys.exit(1)
        except PapayyaAPIError as exc:
            click.echo(f"Error: cohort selection failed ({exc.status}): {exc}", err=True)
            sys.exit(1)
        members = resp.get("members", [])
        total = resp.get("total", len(members))

        if not members:
            click.echo("No records matched. Nothing to pull.")
            # The advice has to match the door. On a derived predicate the
            # window is frozen server-side and --outcome is refused outright,
            # so telling an operator to widen or loosen it names two flags that
            # cannot help — and an empty cohort in a silent-failure product is
            # exactly the moment to not send someone down a dead end.
            if probe_id or drift_episode:
                click.echo("  This predicate selects its own window. Derive a "
                           "fresh one with --like <record>, or widen it there.")
            elif flagged:
                # --outcome any is already applied, so naming it here would be
                # a dead end — the same class of wrong advice the two lines
                # above exist to avoid.
                click.echo("  Nobody has flagged a record matching this yet. "
                           "Drop --flagged, or widen the window with --since.")
            else:
                click.echo("  Widen the window with --since, or pass --outcome any.")
            return

        # Say the truncation out loud. An operator who reads "wrote 100
        # fixtures" and acts as though that is the incident has been misled
        # by us, not by the data.
        if resp.get("truncated"):
            click.echo(
                f"NOTE: cohort is {total} records; pulling {len(members)}. "
                f"Raise --limit to take the rest.",
                err=True,
            )

        if dry_run:
            click.echo(f"Would pull {len(members)} of {total} record(s) into {out_dir}:")
            for m in members[:20]:
                # Both columns, same collapse as `release` prints — the
                # selection response carries `status` beside
                # `worst_outcome_status`. Printing the step verdict alone
                # labelled every crashed record `ok` in the listing of a
                # cohort selected as "everything that didn't work".
                verdict = record_verdict(
                    m.get("status"), m.get("worst_outcome_status")
                )
                click.echo(f"  {m.get('item_id') or m.get('id')}  {verdict}")
            if len(members) > 20:
                click.echo(f"  … and {len(members) - 20} more")
            return

        fixtures = []
        no_input = 0
        for m in members:
            try:
                steps = api.get_steps(m["id"])
            except Exception as exc:  # noqa: BLE001
                # One unreadable record must not cost the operator the
                # cohort — the whole point is to get the incident on disk.
                click.echo(f"  ! {m.get('item_id') or m['id']}: steps unavailable ({exc})", err=True)
                steps = []
            fx = fixture_from_record(m, steps, cohort=predicate)
            if fx.input is None:
                no_input += 1
            fixtures.append(fx)

        paths = write_fixtures(fixtures, out_dir)
        click.echo(f"Pulled {len(paths)} of {total} record(s) into {out_dir}/")
        if no_input:
            click.echo(
                f"  {no_input} fixture(s) have no recorded input and cannot be "
                f"re-run by `papayya verify` — they are kept for their trace.",
                err=True,
            )
        click.echo(f"\nNext: papayya verify --fixtures {out_dir} --agent-module <module>")
    finally:
        api.close()


def _echo_probe_proposals(
    api: Any, record: str, since: str | None, until: str | None,
) -> None:
    """Render `papayya pull --like <record>` — ADR 0009 §4 step 3.

    The output IS the deliverable here, the same way the cohort report is for
    a pull. An operator at 2am has one bad record and needs to know what it is
    an example OF, and how much of it there is.
    """
    try:
        d = api.derive_probes(record, since=since, until=until)
    except PapayyaAPIError as exc:
        click.echo(f"Error: could not read that record ({exc.status}): {exc}", err=True)
        sys.exit(1)

    proposals = d.get("proposals", [])
    refusals = d.get("refusals", [])

    # THE SCOPE, NOT JUST THE AGENT. A count means a different incident
    # depending on whether it spans one tenant or all of them, and a NULL
    # partition means no filter at all — agent-wide — rather than "the records
    # with no tenant". Saying "all tenants" out loud beats omitting the line and
    # letting the operator assume whichever they were already thinking about.
    scope = (f"tenant {d['partition_key']}" if d.get("partition_key")
             else "all tenants")
    click.echo(f"Records like {record} — agent {d.get('agent')}, {scope}")
    click.echo(f"  window {d.get('from')} → {d.get('to')}\n")

    if not proposals:
        # A REAL ANSWER, NOT AN ERROR. It means the platform's vocabulary has
        # no word for what is wrong with this output — which is worth saying
        # plainly rather than dressing up as a failure to try.
        click.echo("No predicate fits this record.")
        click.echo("  Nothing about it is expressible in the vocabulary this "
                   "platform will commit to — how it broke (its error, or the "
                   "category of it), or what one of its steps did (missing, "
                   "empty, short, truncated, or a verdict something already "
                   "made).")
    for p in proposals:
        click.echo(f"  {p.get('count'):>7,}  {p.get('why')}")
        click.echo(f"           papayya pull --probe {p.get('probe')}")

    # Refusals are printed, not swallowed. "Your baseline is still forming" and
    # "this record looks fine on that axis" are completely different answers,
    # and only one of them means come back later.
    if refusals:
        click.echo("\nNot available for this record:")
        for r in refusals:
            click.echo(f"  {r.get('condition')}: {r.get('reason')}")

    if note := d.get("window_note"):
        click.echo(f"\n({note})")
    if proposals:
        click.echo("\nProposals can overlap — a hollow field and the verdict "
                   "that flagged it often select the same records.")


# ---------------------------------------------------------------------------
# verify — run the patched function against pulled fixtures, offline
#
# Plan 41 R4 C8. The middle verb of the recovery loop: pull the incident,
# fix, PROVE the fix against the real records, then release. No control-plane
# call, no API key, no store — see papayya/verify.py for what that costs and
# what it deliberately does not claim about API spend.
# ---------------------------------------------------------------------------

@main.command("verify")
@click.option("--fixtures", "fixtures_path", default="./fixtures",
              help="Fixture file or directory (default ./fixtures).")
@click.option("--agent-module", "agent_module", default=None,
              help="Module your @agent lives in. Defaults to ./agent.py.")
@click.option("--strict", is_flag=True, default=False,
              help="Fail when a fixture could not be verified at all (no "
                   "recorded input, unknown agent, input that doesn't fit "
                   "the signature). The CI setting.")
@click.option("--json", "as_json", is_flag=True, default=False,
              help="Emit the full result as JSON instead of a report.")
def verify_cmd(
    fixtures_path: str,
    agent_module: str | None,
    strict: bool,
    as_json: bool,
) -> None:
    """Verify a fix against pulled fixtures — offline, no re-drive.

    \b
      papayya pull   --agent enrich --tenant acme --since 2026-08-01T00:00:00Z
      papayya verify --fixtures ./fixtures
      papayya verify --fixtures ./fixtures --strict     # in CI

    Runs your function over each fixture's recorded input in this process and
    re-derives the verdict with the same inspectors production used, so
    "fixed" means what "ok" means in the dashboard. Nothing is sent, nothing
    is stored, no record is re-driven. Exits non-zero when any fixture is
    still not ok (or, under --strict, when any could not be verified).

    \b
    NOTE ON API SPEND: verify stops Papayya spending and stops the re-drive
    from touching production. It does NOT stop YOUR function from calling
    your LLM provider — that is your code, your keys, your process. Fixtures
    are free to verify only when the failure reproduces without the provider.
    """
    from papayya.verify import (
        FIXED, NEWLY_BROKEN, STILL_NOT_OK, STILL_OK,
        VerifyError, verify as run_verify,
    )

    try:
        summary = run_verify(
            fixtures_path, agent_module=agent_module, strict=strict
        )
    except VerifyError as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)

    if as_json:
        click.echo(json.dumps(summary.to_dict(), indent=2, default=str))
        sys.exit(0 if summary.ok else 1)

    _MARK = {FIXED: "FIXED", STILL_NOT_OK: "STILL NOT OK",
             NEWLY_BROKEN: "NEWLY BROKEN", STILL_OK: "ok"}
    for r in summary.results:
        label = r.item_id or r.record_id
        mark = _MARK.get(r.verdict, r.verdict.replace("_", " ").upper())
        line = f"  {mark:<13} {label}  [{r.recorded_status} -> {r.new_status or '-'}]"
        if r.new_reason:
            line += f" {r.new_reason}"
        click.echo(line)
        if r.raised:
            click.echo(f"                raised {r.raised}")
        if r.verdict == STILL_NOT_OK and r.output_changed is False:
            # Says more than the verdict does: the fix never reached this path.
            click.echo("                output is UNCHANGED — the fix did not "
                       "touch this record's path")
        if r.note:
            click.echo(f"                {r.note}")
        if (r.current_agent_version and r.recorded_agent_version
                and r.current_agent_version != r.recorded_agent_version):
            # Reported, never refused: verify exists to run changed code.
            click.echo(f"                version {r.recorded_agent_version} "
                       f"-> {r.current_agent_version}")

    counts = summary.counts
    click.echo(
        f"\n{len(summary.results)} fixture(s): "
        f"{counts.get(FIXED, 0)} fixed, "
        f"{counts.get(STILL_NOT_OK, 0)} still not ok, "
        f"{counts.get(NEWLY_BROKEN, 0)} newly broken, "
        f"{counts.get(STILL_OK, 0)} still ok"
    )
    if summary.unanswered:
        click.echo(
            f"{summary.unanswered} fixture(s) could not be verified and are "
            f"NOT counted as passing"
            + ("." if strict else " (pass --strict to fail on these)."),
            err=True,
        )
    if summary.answered == 0:
        # Never print "Verified" over a run that verified nothing.
        click.echo(
            "\nNothing was verified — every fixture is in the list above with "
            "the reason it could not run.",
            err=True,
        )
    elif summary.ok:
        click.echo("\nVerified. Re-drive the cohort with `papayya release`.")
    sys.exit(0 if summary.ok else 1)


# ---------------------------------------------------------------------------
# release — re-drive the cohort, and show the diff
#
# Plan 41 R4 C5/C6/C7. The last verb of pull → verify → release, and the one
# that answers "did the fix work, on which records, at what cost" — including
# the column that earns the diff: how many records the fix NEWLY BROKE.
# ---------------------------------------------------------------------------

@main.command("release")
@click.option("--probe", "probe_id", default=None,
              help="Re-drive the cohort a derived predicate selects "
                   "(from `papayya pull --like`). Excludes records that are "
                   "themselves re-drives, so this is slightly narrower than "
                   "the same probe's pull.")
@click.option("--drift-episode", "drift_episode", default=None,
              help="Re-drive the records a detected change moved.")
@click.option("--agent", default=None, help="Narrow the cohort to one agent slug.")
@click.option("--tenant", default=None,
              help="Narrow to one partition key (the customer's own tenant id).")
@click.option("--run", "run_id", default=None,
              help="Narrow to one run's records. One predicate term among "
                   "several, not the addressing scheme.")
@click.option("--outcome", default=None,
              type=click.Choice(["not_ok", "any", "degraded", "failed"]),
              help="Verdict axis. Default not_ok — everything that didn't work.")
@click.option("--flagged", is_flag=True, default=False,
              help="Only records a human said were wrong. On a hand-written "
                   "predicate this REQUIRES --outcome: unlike `pull`, a "
                   "re-drive will not widen its own selection for you.")
@click.option("--since", default=None, help="Window start (RFC3339).")
@click.option("--until", default=None, help="Window end (RFC3339).")
@click.option("--include-triaged", is_flag=True, default=False,
              help="Re-admit records someone already dispositioned.")
@click.option("--latest", is_flag=True, default=False,
              help="Re-drive the whole cohort on the agent's CURRENT version. "
                   "This is the 'I shipped the fix' flag. Without it each "
                   "record replays on the version it originally ran.")
@click.option("--wait/--no-wait", "wait", default=True,
              help="Poll the re-driven records to completion and print the "
                   "diff (default), or return the manifest immediately.")
@click.option("--timeout", "timeout_s", type=int, default=600,
              help="Seconds to wait for the cohort to finish (default 600).")
@click.option("--yes", "-y", is_flag=True, default=False,
              help="Skip the confirmation prompt.")
@click.option("--json", "as_json", is_flag=True, default=False,
              help="Emit the diff as JSON.")
@click.pass_context
def release_cmd(
    ctx: click.Context,
    probe_id: str | None,
    drift_episode: str | None,
    agent: str | None,
    tenant: str | None,
    run_id: str | None,
    outcome: str | None,
    flagged: bool,
    since: str | None,
    until: str | None,
    include_triaged: bool,
    latest: bool,
    wait: bool,
    timeout_s: int,
    yes: bool,
    as_json: bool,
) -> None:
    """Re-drive a cohort of failed records, then show what changed.

    \b
      papayya pull    --agent enrich --tenant acme --since 2026-08-01T00:00:00Z
      papayya verify  --fixtures ./fixtures
      papayya release --agent enrich --tenant acme --since 2026-08-01T00:00:00Z --latest

    \b
    Or re-drive a derived predicate end to end:
      papayya pull    --like 9c21e8b4-...
      papayya pull    --probe bf88-...
      papayya verify  --fixtures ./fixtures
      papayya release --probe bf88-... --latest

    The cohort is the same predicate `pull` and the dashboard use, so what
    you previewed is what you re-drive. Quota is reserved for the WHOLE
    cohort up front: if it doesn't fit, NOTHING is released and you're told
    how far the quota went — a half-released cohort mid-incident is worse
    than none.
    """
    from papayya.cohort_diff import (
        NEWLY_BROKEN, PENDING, RECOVERED, STILL_NOT_OK, STILL_OK,
        diff_from_release, merge_runs, pending_run_ids,
    )

    scope = _env_scope(ctx.obj)
    config = APIConfig(api_key=_require_api_key(scope), base_url=scope.base_url)
    api = APIClient(config)

    try:
        # REFUSED BEFORE THE PREVIEW, not at the release call. `pull --flagged`
        # implies --outcome any; a re-drive will not, because the records people
        # flag usually completed fine and the flag would silently widen what
        # gets re-executed. Raising later would show the operator a count, take
        # their confirmation, and only then refuse — which teaches them a number
        # that was never releasable.
        if flagged and outcome is None and not (probe_id or drift_episode):
            click.echo(
                "Error: --flagged selects records that completed ok, so a "
                "re-drive would act on records the default predicate excludes. "
                "Pass --outcome any to re-drive them, or --outcome not_ok for "
                "the flagged records that also failed.", err=True)
            sys.exit(1)

        # Preview before acting. A re-drive is the one verb here with a
        # production side effect, so the operator sees the size of it first.
        try:
            preview = api.get_cohort(
                probe=probe_id, drift_episode=drift_episode,
                agent=agent, tenant=tenant, run_id=run_id, outcome=outcome,
                since=since, until=until, flagged=flagged,
                include_triaged=include_triaged, limit=0,
            )
        except ValueError as exc:
            click.echo(f"Error: {exc}", err=True)
            sys.exit(1)
        except PapayyaAPIError as exc:
            click.echo(f"Error: cohort selection failed ({exc.status}): {exc}", err=True)
            sys.exit(1)
        total = preview.get("total", 0)
        if not total:
            click.echo("No records matched. Nothing to release.")
            return
        if not yes:
            version_note = ("the agent's CURRENT version" if latest
                            else "the version each record originally ran")
            # ON THE PROBE PATH THIS COUNT IS AN UPPER BOUND, and saying so is
            # not pedantry — it is the one place the preview and the release
            # legitimately disagree. A by-example preview admits records that
            # are themselves re-drives, because an operator asking "what else
            # looks like this" wants their whole history; release excludes them
            # so nobody re-drives a re-drive. Without this line the prompt
            # names a number the release will not honour, which is precisely
            # the "approve one cohort, re-drive another" failure the shared
            # predicate exists to prevent.
            if probe_id:
                click.echo(
                    f"  {total} record(s) match this probe. Records that are "
                    f"themselves re-drives are excluded from a release, so "
                    f"fewer may go."
                )
                ask = f"Re-drive up to {total} record(s) on {version_note}?"
            else:
                ask = f"Re-drive {total} record(s) on {version_note}?"
            click.confirm(ask, abort=True)

        try:
            release = api.release_cohort(
                probe=probe_id, drift_episode=drift_episode,
                agent=agent, tenant=tenant, run_id=run_id, outcome=outcome,
                since=since, until=until, flagged=flagged,
                include_triaged=include_triaged, latest=latest,
            )
        except ValueError as exc:
            click.echo(f"Error: {exc}", err=True)
            sys.exit(1)
        except PapayyaAPIError as exc:
            click.echo(f"Error: release failed ({exc.status}): {exc}", err=True)
            sys.exit(1)

        diff = diff_from_release(release)
        if not as_json:
            # --json emits the diff and NOTHING else on stdout: every fact in
            # these lines is in the payload, and a progress line ahead of it
            # makes the output unparseable for the caller who asked for JSON.
            click.echo(f"Released {len(diff.records)} of {diff.cohort_total} record(s).")
        _echo_release_skips(diff)

        if wait:
            deadline = time.time() + timeout_s
            while True:
                pending = pending_run_ids(diff)
                if not pending:
                    break
                fetched = []
                for rid in pending:
                    try:
                        fetched.append((rid, api.get_run(rid)))
                    except PapayyaAPIError:
                        # One unreadable run must not cost the operator the
                        # diff; it stays PENDING and is counted as such.
                        continue
                merge_runs(diff, fetched)
                # Read BEFORE sleeping — a short cohort on a warm worker pool
                # can already be terminal by the time the manifest returns,
                # and sleeping first would charge every release two seconds
                # it did not need.
                if not pending_run_ids(diff):
                    break
                if time.time() >= deadline:
                    click.echo(
                        f"\nTimed out with {len(pending_run_ids(diff))} record(s) "
                        f"still running. They are re-driving; re-read them in "
                        f"the dashboard or raise --timeout.",
                        err=True,
                    )
                    break
                time.sleep(2)

        if as_json:
            click.echo(json.dumps(diff.to_dict(), indent=2, default=str))
            return

        _MARK = {RECOVERED: "RECOVERED", STILL_NOT_OK: "STILL NOT OK",
                 NEWLY_BROKEN: "NEWLY BROKEN", STILL_OK: "ok", PENDING: "…running"}
        for rec in diff.records:
            label = rec.item_id or rec.source_id
            click.echo(
                f"  {_MARK.get(rec.verdict, rec.verdict):<13} {label}  "
                f"[{rec.before_status} -> {rec.after_status or '-'}]  "
                f"{_fmt_usd(rec.before_cost_usd)} -> {_fmt_usd(rec.after_cost_usd)}"
            )

        counts = diff.counts
        click.echo(
            f"\n{len(diff.records)} record(s): "
            f"{counts.get(RECOVERED, 0)} recovered, "
            f"{counts.get(STILL_NOT_OK, 0)} still not ok, "
            f"{counts.get(NEWLY_BROKEN, 0)} newly broken, "
            f"{counts.get(STILL_OK, 0)} still ok"
            + (f", {counts[PENDING]} still running" if counts.get(PENDING) else "")
        )
        click.echo(
            f"Cost of the re-drive: {_fmt_usd(diff.after_cost_usd)} "
            f"(the originals cost {_fmt_usd(diff.before_cost_usd)})"
        )
        if diff.cost_note:
            click.echo(f"  {diff.cost_note}")
        if counts.get(NEWLY_BROKEN):
            # Loudest line in the output, deliberately. It is the fact an
            # operator is least likely to look for and most needs.
            click.echo(
                f"\n{counts[NEWLY_BROKEN]} record(s) were fine before this "
                f"re-drive and are not now.",
                err=True,
            )
    finally:
        api.close()


def _echo_release_skips(diff: Any) -> None:
    """Say what the release did NOT touch. Named, never silent."""
    if diff.skipped_not_terminal:
        click.echo(
            f"  {diff.skipped_not_terminal} record(s) are still running and "
            f"were not re-driven — they stay in the cohort, so release again "
            f"once they land.",
            err=True,
        )
    if diff.skipped_agent_missing:
        click.echo(
            f"  {diff.skipped_agent_missing} record(s) have no agent to route "
            f"to and were not re-driven.",
            err=True,
        )


# ---------------------------------------------------------------------------
# runs — run (invocation) ops, and items — per-item inspection
#
# Plural is deliberate. Top-level `papayya run` (workflow: create+wait+tail)
# stays unchanged. BREAKING (0.3.0, Plan 34): `papayya runs` used to be the
# hosted per-ITEM surface; a run is now one INVOCATION (one map() call, one
# cron fire, one submitted file of items). Per-item verbs moved to
# `papayya items` — the old name persists with new semantics, so it could
# not be aliased.
# ---------------------------------------------------------------------------

@main.group()
def runs() -> None:
    """Runs — one invocation each (submit hosted).

    A run is one invocation of an agent: one map() call, one cron fire,
    one submitted batch of items. `submit` sends a JSONL file to the
    hosted control plane.

    To see what has already run, use `papayya items list` (one record per
    line) or the dashboard. This group's `list` read the local SQLite
    ledger and was deactivated with it; the help kept advertising it, so
    `papayya runs list` answered "No such command" to anyone who read the
    line above it.

    BREAKING (0.3.0): this group used to operate on per-item records;
    per-item inspection moved to `papayya items`.
    """


# The local-ledger `papayya runs list` that used to sit here is GONE, not
# deactivated. It read `.papayya/local.db`, which `papayya dev` stopped
# creating; it was never registered on the `runs` group, so no user could reach
# it; and its own comment called the cloud equivalent "the follow-up wiring".
# That wiring is now `runs list` below, over GET /v2/runs. Keeping a retained
# body for a discontinued surface meant two functions named runs_list, one
# shadowing the other, and two tests exercising the unreachable one.

# ---------------------------------------------------------------------------
# items — hosted per-item inspection (the pre-0.3.0 `runs` verbs, renamed)
# ---------------------------------------------------------------------------

@main.group()
def items() -> None:
    """Inspect hosted items — one record each (list, get, stream)."""


@items.command("list")
@click.option("--run", "run_id", default=None,
              help="Only items belonging to this run (the id `papayya run` printed).")
@click.option("--agent", default=None, help="Only items for this agent.")
@click.option("--tenant", "partition_key", default=None,
              help="Only items for this partition key.")
@click.pass_context
def items_list(
    ctx: click.Context,
    run_id: str | None,
    agent: str | None,
    partition_key: str | None,
) -> None:
    """List hosted items (NDJSON, one item per line).

    ``--run`` is the one `papayya status` has been pointing at all along: it
    prints "Items in this run: papayya items list", and until now that command
    could not be scoped to a run.
    """
    client = _make_papayya_client(ctx)
    try:
        rows = client.items.list(
            run_id=run_id, agent=agent, partition_key=partition_key
        )
    except PapayyaAPIError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    finally:
        client.close()

    for row in rows:
        click.echo(json.dumps(row))


@items.command("get")
@click.argument("item_id")
@click.pass_context
def items_get(ctx: click.Context, item_id: str) -> None:
    """Fetch one hosted item by id."""
    client = _make_papayya_client(ctx)
    try:
        row = client.items.get(item_id)
    except PapayyaAPIError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    finally:
        client.close()

    click.echo(json.dumps(row, indent=2))


@items.command("stream")
@click.argument("item_id")
@click.option(
    "--from-step",
    type=int,
    default=None,
    help="Resume from this step (use the highest step number already observed)",
)
@click.pass_context
def items_stream(ctx: click.Context, item_id: str, from_step: int | None) -> None:
    """Tail steps for a hosted item via SSE (NDJSON output).

    Yields one JSON object per server-sent event: ``{"event": "step" |
    "terminal" | "error", "data": {...}, "id": <step_number>}``. The
    stream exits when the item reaches a terminal status.
    """
    client = _make_papayya_client(ctx)
    try:
        for event in client.items.stream(item_id, from_step=from_step):
            click.echo(json.dumps(event))
            sys.stdout.flush()
    except PapayyaAPIError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    finally:
        client.close()


# ---------------------------------------------------------------------------
# agents — hosted agent CRUD
# ---------------------------------------------------------------------------

def _parse_json_option(raw: str | None, flag_name: str) -> Any:
    """Parse a JSON-string CLI option into a Python value, or None.

    Used by commands that accept a structured payload (config, kwargs)
    as a single flag. Errors out cleanly on invalid JSON.
    """
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        click.echo(f"Error: {flag_name} must be valid JSON: {exc}", err=True)
        sys.exit(1)


@main.group()
def agents() -> None:
    """Hosted agent CRUD (create, list, get, update)."""


@agents.command("create")
@click.option("--name", required=True, help="Display name")
@click.option("--slug", required=True, help="URL-safe slug (unique per project)")
@click.option("--project-id", required=True, help="Project the agent belongs to")
@click.option("--description", default=None, help="Optional description")
@click.option("--config", "config_raw", default=None,
              help="Optional JSON config object")
@click.pass_context
def agents_create(
    ctx: click.Context,
    name: str,
    slug: str,
    project_id: str,
    description: str | None,
    config_raw: str | None,
) -> None:
    """Create a hosted agent."""
    config = _parse_json_option(config_raw, "--config")
    client = _make_papayya_client(ctx)
    try:
        agent = client.agents.create(
            name, slug, project_id, config=config, description=description
        )
    except PapayyaAPIError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    finally:
        client.close()

    click.echo(json.dumps(agent, indent=2))


@agents.command("list")
@click.option("--project-id", default=None,
              help="Filter by project (defaults to listing all agents the caller can see)")
@click.pass_context
def agents_list(ctx: click.Context, project_id: str | None) -> None:
    """List agents (NDJSON)."""
    client = _make_papayya_client(ctx)
    try:
        items = client.agents.list(project_id)
    except PapayyaAPIError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    finally:
        client.close()

    for agent in items:
        click.echo(json.dumps(agent))


@agents.command("get")
@click.argument("agent_id")
@click.pass_context
def agents_get(ctx: click.Context, agent_id: str) -> None:
    """Fetch one agent by ID."""
    client = _make_papayya_client(ctx)
    try:
        agent = client.agents.get(agent_id)
    except PapayyaAPIError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    finally:
        client.close()

    click.echo(json.dumps(agent, indent=2))


@agents.command("update")
@click.argument("agent_id")
@click.option("--name", default=None, help="New display name")
@click.option("--description", default=None, help="New description")
@click.option("--config", "config_raw", default=None,
              help="Replacement JSON config object")
@click.pass_context
def agents_update(
    ctx: click.Context,
    agent_id: str,
    name: str | None,
    description: str | None,
    config_raw: str | None,
) -> None:
    """Patch fields on an existing agent.

    Only the flags you pass are sent; omitted fields stay untouched.
    """
    patch: dict[str, Any] = {}
    if name is not None:
        patch["name"] = name
    if description is not None:
        patch["description"] = description
    if config_raw is not None:
        patch["config"] = _parse_json_option(config_raw, "--config")

    if not patch:
        click.echo("Error: pass at least one of --name / --description / --config", err=True)
        sys.exit(1)

    client = _make_papayya_client(ctx)
    try:
        agent = client.agents.update(agent_id, **patch)
    except PapayyaAPIError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    finally:
        client.close()

    click.echo(json.dumps(agent, indent=2))


# ---------------------------------------------------------------------------
# schedules — hosted cron schedules over agents
# ---------------------------------------------------------------------------

def _dollars_to_cents(dollars: float | None) -> int | None:
    """$X.YZ → cents (int). None passes through."""
    return None if dollars is None else int(round(dollars * 100))


@main.group()
def schedules() -> None:
    """Hosted cron schedules (create, list, get, update, enable/disable, delete)."""


@schedules.command("create")
@click.option("--agent", "agent_id", required=True, help="Agent the schedule fires against")
@click.option("--cron", required=True, help="Cron expression (e.g. '0 */6 * * *')")
@click.option("--timezone", default=None, help="IANA timezone (e.g. 'America/Toronto')")
@click.option("--input", "input_str", default=None, help="Run input passed each time the schedule fires")
@click.option("--budget", type=float, default=None, help="Per-run budget cap in whole dollars")
@click.pass_context
def schedules_create(
    ctx: click.Context,
    agent_id: str,
    cron: str,
    timezone: str | None,
    input_str: str | None,
    budget: float | None,
) -> None:
    """Create a cron schedule on an agent.

    No --max-steps: the step ceiling belongs to the deployed bundle,
    ``@agent(name=..., max_steps=N)``. The flag existed, was stored, and was
    never applied; the server now rejects the field.
    """
    client = _make_papayya_client(ctx)
    try:
        sched = client.schedules.create(
            agent_id,
            cron,
            timezone=timezone,
            input=input_str,
            budget_cents=_dollars_to_cents(budget),
        )
    except PapayyaAPIError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    finally:
        client.close()

    click.echo(json.dumps(sched, indent=2))


@schedules.command("list")
@click.option("--agent", "agent_id", default=None,
              help="Filter to one agent's schedules (defaults to all)")
@click.pass_context
def schedules_list(ctx: click.Context, agent_id: str | None) -> None:
    """List schedules (NDJSON)."""
    client = _make_papayya_client(ctx)
    try:
        items = client.schedules.list(agent_id)
    except PapayyaAPIError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    finally:
        client.close()

    for sched in items:
        click.echo(json.dumps(sched))


@schedules.command("get")
@click.argument("schedule_id")
@click.pass_context
def schedules_get(ctx: click.Context, schedule_id: str) -> None:
    """Fetch one schedule by ID."""
    client = _make_papayya_client(ctx)
    try:
        sched = client.schedules.get(schedule_id)
    except PapayyaAPIError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    finally:
        client.close()

    click.echo(json.dumps(sched, indent=2))


@schedules.command("update")
@click.argument("schedule_id")
@click.option("--cron", default=None, help="New cron expression")
@click.option("--timezone", default=None, help="New timezone")
@click.option("--input", "input_str", default=None, help="New run input")
@click.option("--budget", type=float, default=None, help="New per-run budget cap (whole dollars)")
@click.pass_context
def schedules_update(
    ctx: click.Context,
    schedule_id: str,
    cron: str | None,
    timezone: str | None,
    input_str: str | None,
    budget: float | None,
) -> None:
    """Patch fields on an existing schedule (only fields you pass are sent).

    No --max-steps — see `schedules create`.
    """
    patch: dict[str, Any] = {}
    if cron is not None:
        patch["cron"] = cron
    if timezone is not None:
        patch["timezone"] = timezone
    if input_str is not None:
        patch["input"] = input_str
    if budget is not None:
        patch["budget_cents"] = _dollars_to_cents(budget)

    if not patch:
        click.echo("Error: pass at least one of --cron / --timezone / --input / --budget", err=True)
        sys.exit(1)

    client = _make_papayya_client(ctx)
    try:
        sched = client.schedules.update(schedule_id, **patch)
    except PapayyaAPIError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    finally:
        client.close()

    click.echo(json.dumps(sched, indent=2))


@schedules.command("delete")
@click.argument("schedule_id")
@click.pass_context
def schedules_delete(ctx: click.Context, schedule_id: str) -> None:
    """Delete a schedule."""
    client = _make_papayya_client(ctx)
    try:
        client.schedules.delete(schedule_id)
    except PapayyaAPIError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    finally:
        client.close()

    click.echo(f"Schedule {schedule_id} deleted")


@schedules.command("enable")
@click.argument("schedule_id")
@click.pass_context
def schedules_enable(ctx: click.Context, schedule_id: str) -> None:
    """Enable a paused schedule."""
    client = _make_papayya_client(ctx)
    try:
        sched = client.schedules.enable(schedule_id)
    except PapayyaAPIError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    finally:
        client.close()

    click.echo(json.dumps(sched, indent=2))


@schedules.command("disable")
@click.argument("schedule_id")
@click.pass_context
def schedules_disable(ctx: click.Context, schedule_id: str) -> None:
    """Pause a schedule (it stays in the DB but won't fire)."""
    client = _make_papayya_client(ctx)
    try:
        sched = client.schedules.disable(schedule_id)
    except PapayyaAPIError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    finally:
        client.close()

    click.echo(json.dumps(sched, indent=2))


# ---------------------------------------------------------------------------
# triggers — inbound invocation hooks per agent (Plan 34: webhook → trigger;
# webhook survives only as the transport detail). The SDK resource stays
# `client.webhooks` until the control-pane rename (Unit 5).
# ---------------------------------------------------------------------------

@main.group()
def triggers() -> None:
    """Manage triggers — inbound invocation hooks (create, list, delete).

    A trigger fires a run of an agent from an HTTP call (transport:
    signed webhook). Pre-0.3.0 spelling: `papayya webhooks` — kept as a
    hidden alias.
    """


@triggers.command("create")
@click.option("--agent", "agent_id", required=True, help="Agent the trigger belongs to")
@click.option("--name", required=True, help="Display name")
@click.option("--description", default=None, help="Optional description")
@click.pass_context
def triggers_create(
    ctx: click.Context, agent_id: str, name: str, description: str | None
) -> None:
    """Create a trigger on an agent. Returns the trigger with its signing secret."""
    client = _make_papayya_client(ctx)
    try:
        hook = client.webhooks.create(agent_id, name, description=description)
    except PapayyaAPIError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    finally:
        client.close()

    click.echo(json.dumps(hook, indent=2))


@triggers.command("list")
@click.argument("agent_id")
@click.pass_context
def triggers_list(ctx: click.Context, agent_id: str) -> None:
    """List triggers on an agent (NDJSON)."""
    client = _make_papayya_client(ctx)
    try:
        hooks = client.webhooks.list(agent_id)
    except PapayyaAPIError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    finally:
        client.close()

    for hook in hooks:
        click.echo(json.dumps(hook))


@triggers.command("delete")
@click.argument("trigger_id")
@click.pass_context
def triggers_delete(ctx: click.Context, trigger_id: str) -> None:
    """Delete a trigger."""
    client = _make_papayya_client(ctx)
    try:
        client.webhooks.delete(trigger_id)
    except PapayyaAPIError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    finally:
        client.close()

    click.echo(f"Trigger {trigger_id} deleted")


# Deprecated alias (webhook → trigger): shared command objects, hidden
# from --help, removed one release after 0.3.0.
webhooks = click.Group(
    "webhooks",
    hidden=True,
    help="Deprecated alias of `papayya triggers` (webhook → trigger in 0.3.0).",
)
for _name, _cmd in triggers.commands.items():
    webhooks.add_command(_cmd, _name)
main.add_command(webhooks)


# ---------------------------------------------------------------------------
# projects — hosted project resource (plural, distinct from local `project`)
#
# `papayya project` (singular) handles the LOCAL SQLite history (export /
# import). `papayya projects` (plural) is the hosted CRUD surface.
# ---------------------------------------------------------------------------

@main.group()
def projects() -> None:
    """Hosted projects (list, get, update, delete). Create via `papayya envs create`."""


@projects.command("list")
@click.pass_context
def projects_list(ctx: click.Context) -> None:
    """List hosted projects (NDJSON)."""
    client = _make_papayya_client(ctx)
    try:
        items = client.projects.list()
    except PapayyaAPIError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    finally:
        client.close()

    for proj in items:
        click.echo(json.dumps(proj))


@projects.command("get")
@click.argument("project_id")
@click.pass_context
def projects_get(ctx: click.Context, project_id: str) -> None:
    """Fetch one project by ID."""
    client = _make_papayya_client(ctx)
    try:
        proj = client.projects.get(project_id)
    except PapayyaAPIError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    finally:
        client.close()

    click.echo(json.dumps(proj, indent=2))


@projects.command("update")
@click.argument("project_id")
@click.option("--name", default=None, help="New display name")
@click.option("--slug", default=None, help="New slug")
@click.pass_context
def projects_update(
    ctx: click.Context, project_id: str, name: str | None, slug: str | None
) -> None:
    """Patch fields on a hosted project."""
    patch: dict[str, Any] = {}
    if name is not None:
        patch["name"] = name
    if slug is not None:
        patch["slug"] = slug

    if not patch:
        click.echo("Error: pass at least one of --name / --slug", err=True)
        sys.exit(1)

    client = _make_papayya_client(ctx)
    try:
        proj = client.projects.update(project_id, **patch)
    except PapayyaAPIError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    finally:
        client.close()

    click.echo(json.dumps(proj, indent=2))


@projects.command("delete")
@click.argument("project_id")
@click.confirmation_option(
    prompt="Deleting a project is irreversible and removes all its agents, runs, and history. Continue?",
)
@click.pass_context
def projects_delete(ctx: click.Context, project_id: str) -> None:
    """Delete a hosted project (irreversible)."""
    client = _make_papayya_client(ctx)
    try:
        client.projects.delete(project_id)
    except PapayyaAPIError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    finally:
        client.close()

    click.echo(f"Project {project_id} deleted")


# ---------------------------------------------------------------------------
# deployments — inspect hosted deployments (create stays as `papayya deploy`)
# ---------------------------------------------------------------------------

@main.group()
def deployments() -> None:
    """Inspect hosted deployments. Create via `papayya deploy`."""


@deployments.command("list")
@click.argument("agent_id")
@click.pass_context
def deployments_list(ctx: click.Context, agent_id: str) -> None:
    """List deployments for an agent (NDJSON, newest first)."""
    client = _make_papayya_client(ctx)
    try:
        items = client.deployments.list(agent_id)
    except PapayyaAPIError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    finally:
        client.close()

    for dep in items:
        click.echo(json.dumps(dep))


@deployments.command("get")
@click.argument("deployment_id")
@click.pass_context
def deployments_get(ctx: click.Context, deployment_id: str) -> None:
    """Fetch one deployment by ID."""
    client = _make_papayya_client(ctx)
    try:
        dep = client.deployments.get(deployment_id)
    except PapayyaAPIError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    finally:
        client.close()

    click.echo(json.dumps(dep, indent=2))


# ---------------------------------------------------------------------------
# api-keys — inspect and revoke API keys per project
# ---------------------------------------------------------------------------

@main.group("api-keys")
def api_keys() -> None:
    """Inspect and revoke project API keys. Create via `papayya envs create`."""


@api_keys.command("list")
@click.option("--project-id", required=True, help="Project whose keys to list")
@click.pass_context
def api_keys_list(ctx: click.Context, project_id: str) -> None:
    """List API keys for a project (NDJSON). Key prefixes only — secrets are write-once."""
    client = _make_papayya_client(ctx)
    try:
        items = client.api_keys.list(project_id)
    except PapayyaAPIError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    finally:
        client.close()

    for key in items:
        click.echo(json.dumps(key))


@api_keys.command("revoke")
@click.argument("key_id")
@click.option("--project-id", required=True, help="Project the key belongs to")
@click.confirmation_option(
    prompt="Revoking an API key is immediate and will break any service still using it. Continue?",
)
@click.pass_context
def api_keys_revoke(ctx: click.Context, key_id: str, project_id: str) -> None:
    """Revoke an API key (irreversible)."""
    client = _make_papayya_client(ctx)
    try:
        client.api_keys.revoke(project_id, key_id)
    except PapayyaAPIError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    finally:
        client.close()

    click.echo(f"API key {key_id} revoked")


# ---------------------------------------------------------------------------
# usage — usage rollups (summary, breakdown)
# ---------------------------------------------------------------------------

@main.group()
def usage() -> None:
    """Usage rollups (summary, breakdown)."""


@usage.command("summary")
@click.option("--from", "from_date", default=None,
              help="Start of window (ISO date or RFC3339 timestamp)")
@click.option("--to", "to_date", default=None,
              help="End of window (ISO date or RFC3339 timestamp)")
@click.pass_context
def usage_summary(ctx: click.Context, from_date: str | None, to_date: str | None) -> None:
    """Aggregate usage over an optional date range."""
    client = _make_papayya_client(ctx)
    try:
        summary = client.usage.summary(from_date=from_date, to_date=to_date)
    except PapayyaAPIError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    finally:
        client.close()

    click.echo(json.dumps(summary, indent=2))


@usage.command("breakdown")
@click.option("--from", "from_date", default=None,
              help="Start of window (ISO date or RFC3339 timestamp)")
@click.option("--to", "to_date", default=None,
              help="End of window (ISO date or RFC3339 timestamp)")
@click.pass_context
def usage_breakdown(ctx: click.Context, from_date: str | None, to_date: str | None) -> None:
    """Per-dimension usage breakdown (NDJSON)."""
    client = _make_papayya_client(ctx)
    try:
        rows = client.usage.breakdown(from_date=from_date, to_date=to_date)
    except PapayyaAPIError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    finally:
        client.close()

    for row in rows:
        click.echo(json.dumps(row))


# Plan 37: the `project` group manages the LOCAL project history (export /
# import of the SQLite ledger) — DEACTIVATED. Not registered on `main`; the
# group + its export/import subcommands are retained for revival.
@click.group()
def project() -> None:
    """Manage local project history (export, import)."""


@project.command("export")
@click.option("--out", required=True, help="Output JSONL file path")
@click.option("--db", default=".papayya/local.db", envvar="PAPAYYA_LOCAL_DB_PATH",
              help="Path to SQLite database (also honors PAPAYYA_LOCAL_DB_PATH)")
@click.option(
    "--include-response-text",
    is_flag=True,
    default=False,
    help="Include raw LLM response text in the export. OFF by default — "
         "response text may contain PII, customer data, or proprietary "
         "prompts. Only enable if you've reviewed the data.",
)
def project_export(out: str, db: str, include_response_text: bool) -> None:
    """Export local history (runs, items, steps) to a JSONL file.

    Intended for uploading to Papayya Cloud after signup so your local
    dashboard's history comes with you. Until the hosted import endpoint
    lands, this command is local-only — the output file is saved to
    disk, not sent anywhere.
    """
    import json as _json
    import sqlite3 as _sqlite
    from pathlib import Path as _Path

    db_path = _Path(db)
    if not db_path.exists():
        click.echo(f"No local database at {db_path.resolve()}", err=True)
        raise click.exceptions.Exit(1)

    # The export reads the local ledger, so make sure it's at the current
    # schema (a DB written by an older SDK may predate the v12 noun
    # consolidation: batches/runs/tasks -> runs/items/steps).
    from papayya.durable.sqlite_store import ensure_migrated as _ensure_migrated
    _ensure_migrated(db_path)

    conn = _sqlite.connect(db_path)
    conn.row_factory = _sqlite.Row

    out_path = _Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = {"runs": 0, "items": 0, "steps": 0}

    with out_path.open("w", encoding="utf-8") as fh:
        for row in conn.execute("SELECT * FROM runs"):
            fh.write(_json.dumps({"type": "run", "data": dict(row)}) + "\n")
            written["runs"] += 1

        for row in conn.execute("SELECT * FROM items"):
            fh.write(_json.dumps({"type": "item", "data": dict(row)}) + "\n")
            written["items"] += 1

        for row in conn.execute("SELECT * FROM steps"):
            data = dict(row)
            if not include_response_text:
                # Step results and snapshots can carry raw LLM output /
                # customer payloads — same privacy posture as the old
                # response_text column.
                data.pop("result", None)
                data.pop("input_snapshot", None)
                data.pop("output_snapshot", None)
            fh.write(_json.dumps({"type": "step", "data": data}) + "\n")
            written["steps"] += 1

    conn.close()
    click.echo(
        f"Exported {written['runs']} runs, {written['items']} items, "
        f"{written['steps']} steps to {out_path}"
    )
    if not include_response_text:
        click.echo(
            "Note: step results/snapshots excluded by default. Re-run with "
            "--include-response-text to include them."
        )


@project.command("import")
@click.argument("file")
def project_import(file: str) -> None:
    """Import a previously-exported JSONL into Papayya Cloud.

    Stub for now — the hosted import endpoint is not yet live. The
    command validates the file shape and prints what would be uploaded.
    """
    import json as _json
    from pathlib import Path as _Path

    path = _Path(file)
    if not path.exists():
        click.echo(f"File not found: {path}", err=True)
        raise click.exceptions.Exit(1)

    counts: dict[str, int] = {}
    for i, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            obj = _json.loads(line)
        except _json.JSONDecodeError as e:
            click.echo(f"Line {i}: invalid JSON ({e})", err=True)
            raise click.exceptions.Exit(1)
        kind = obj.get("type")
        # v12 exports emit run/item/step; pre-v12 exports emitted
        # batch/run/step (with the old per-item meaning of "run"). Accept
        # both so previously-exported files still validate.
        if kind not in ("run", "item", "step", "batch"):
            click.echo(f"Line {i}: unknown record type {kind!r}", err=True)
            raise click.exceptions.Exit(1)
        counts[kind] = counts.get(kind, 0) + 1

    click.echo("Validated import file:")
    plurals = {"run": "runs", "item": "items", "step": "steps", "batch": "batches (pre-0.3.0 export)"}
    for kind in ("run", "item", "step", "batch"):
        if kind == "batch" and counts.get(kind, 0) == 0:
            continue
        click.echo(f"  {plurals[kind]}: {counts.get(kind, 0)}")
    click.echo(
        "\nHosted import endpoint is not yet live. "
        "Signup at https://app.getpapayya.com to be notified."
    )


_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)


def _resolve_agent_id(
    positional: str | None,
    agent_id_flag: str | None,
    ctx_obj: dict,
) -> str:
    """Turn a slug-or-uuid positional (or --agent-id flag) into an agent UUID.

    --agent-id wins when both are supplied. A uuid-shaped positional passes
    through without an API call. Otherwise the positional is treated as a
    slug and resolved against the selected env's project via list_agents.
    Fails loud with available slugs on a miss.
    """
    if agent_id_flag:
        return agent_id_flag
    if not positional:
        click.echo(
            "Error: agent required. Pass a slug or UUID:\n"
            '  papayya run my-agent "input"',
            err=True,
        )
        sys.exit(1)

    if _UUID_RE.match(positional):
        return positional

    scope = _env_scope(ctx_obj)
    resolved_key = _require_api_key(scope)
    project_id = _require_project_id(scope)
    api = APIClient(APIConfig(api_key=resolved_key, base_url=scope.base_url))
    try:
        agents = api.list_agents(project_id)
    finally:
        api.close()

    for a in agents:
        if a.get("slug") == positional:
            return a["id"]

    slugs = sorted(a["slug"] for a in agents if a.get("slug"))
    available = ", ".join(slugs) if slugs else "(none deployed)"
    click.echo(
        f"Error: no agent '{positional}' in env '{scope.env}'. "
        f"Available: {available}",
        err=True,
    )
    sys.exit(1)


@main.command()
@click.argument("agent", required=False)
@click.argument("input_positional", required=False)
@click.option("--file", default=None, help="Path to agent definition file (default: agent.py in cwd)")
@click.option("--input", "input_flag", default=None, help="Input for the agent (alt to positional)")
@click.option("--agent-id", default=None, help="Agent UUID (escape hatch; wins over positional)")
@click.option("--name", "agent_name", default=None, help="Agent name (required when file declares multiple @agent functions)")
@click.pass_context
def run(
    ctx: click.Context,
    agent: str | None,
    input_positional: str | None,
    file: str | None,
    input_flag: str | None,
    agent_id: str | None,
    agent_name: str | None,
) -> None:
    """Trigger a cloud run.

    \b
    Usage:
      papayya run my-agent "hello"              # slug + positional input
      papayya run my-agent "hello" --file a.py  # explicit file
      papayya run <uuid> "hello"                # UUID also works

    To run locally without the cloud, execute your file directly:
      python agent.py
    """
    # Resolve input: positional wins; fall back to --input flag.
    if input_positional is not None and input_flag is not None:
        click.echo(
            "Error: input provided twice (positional and --input). Pick one.",
            err=True,
        )
        sys.exit(1)
    input_text = input_positional if input_positional is not None else input_flag
    if not input_text:
        click.echo(
            'Error: input required.\n  papayya run <agent> "your input"',
            err=True,
        )
        sys.exit(1)

    # Resolve file: --file wins; else auto-discover agent.py in cwd.
    resolved_file = file
    if resolved_file is None:
        if Path("agent.py").exists():
            resolved_file = "agent.py"
        else:
            click.echo(
                "Error: --file required (or place agent.py in the current directory).",
                err=True,
            )
            sys.exit(1)

    registrations = _discover_agents(resolved_file)
    if len(registrations) == 1:
        reg = registrations[0]
    else:
        if not agent_name:
            names = ", ".join(r.name for r in registrations)
            click.echo(
                f"Error: {resolved_file} declares {len(registrations)} agents ({names}).\n"
                "  Pass --name <agent-name> to pick one.",
                err=True,
            )
            sys.exit(1)
        matches = [r for r in registrations if r.name == agent_name]
        if not matches:
            names = ", ".join(r.name for r in registrations)
            click.echo(
                f"Error: no @agent named '{agent_name}' in {resolved_file}. Available: {names}",
                err=True,
            )
            sys.exit(1)
        reg = matches[0]

    resolved_agent_id = _resolve_agent_id(agent, agent_id, ctx.obj)
    _run_cloud(ctx, reg, resolved_file, input_text, resolved_agent_id)


def _run_cloud(ctx: click.Context, reg: Any, file: str, input_text: str, agent_id: str) -> None:
    """Trigger a cloud run.

    ``reg`` is an ``AgentRegistration`` produced by ``_discover_agents``;
    ``agent_id`` has already been resolved (slug → uuid) by the caller.
    """
    scope = _env_scope(ctx.obj)
    resolved_key = _require_api_key(scope)

    budget_cents = int(reg.budget_usd * 100) if reg.budget_usd is not None else 500

    config = APIConfig(api_key=resolved_key, base_url=scope.base_url)
    api = APIClient(config)

    try:
        result = api.trigger_run(
            agent_id=agent_id,
            model=reg.model,
            system_prompt=reg.instructions,
            # The SUBMITTED INPUT is what the customer's function is called
            # with — Lease.agent_argument returns payload["input"] verbatim.
            # Wrapping it in {"message": ...} made that envelope the argument,
            # so `papayya run hello "world"` returned
            # "Hello, {'message': 'world'}!" where the same agent run locally
            # returned "Hello, world!". The two execution paths have to agree.
            input_data=input_text,
            max_steps=reg.max_steps,
            budget_cents=budget_cents,
        )
        run_id = result["id"]
        click.echo(f"Run triggered: {run_id}")
        click.echo(f"  Status: {result.get('status', 'unknown')}")
        click.echo(f"  Model: {reg.model}")

        # Poll until complete. The run is a durable_run now: it starts
        # 'queued', flips to 'running' when a worker leases it, then to a
        # terminal state. Progress = number of checkpoints written so far.
        click.echo("Waiting for completion...")
        while True:
            time.sleep(2)
            status_resp = api.get_run(run_id)
            state = status_resp.get("status", "unknown")
            done = _distinct_step_count(status_resp.get("checkpoints"))
            click.echo(f"  {done} step(s) — {state}")

            if state in _RUN_WAIT_STOP_STATES:
                break

        # Show final result. Durable checkpoints carry {label, result};
        # there's no v1 step_number/step_type/output shape any more.
        click.echo(f"\nFinal status: {state}")
        checkpoints = api.get_steps(run_id)
        for c in checkpoints:
            label = c.get("label", "?")
            # Mark re-executions, or two rows with the same label read as
            # a duplicate rather than as the second attempt (Plan 41 R3).
            attempt = c.get("attempt", 1)
            if attempt > 1:
                label = f"{label} (attempt {attempt})"
            result = c.get("result")
            content = (
                result.get("content", "")
                if isinstance(result, dict)
                else "" if result is None else str(result)
            )
            click.echo(f"  {label}: {str(content)[:200]}")

        # AFTER the steps, not before them. `paused` and `quarantine` are
        # terminal FOR THIS COMMAND without being terminal for the run, and the
        # word alone strands the user — something stopped this and nothing will
        # move it until a human acts. Printed last because it is the thing to do
        # next, and a next-step buried above the output is one nobody reads.
        #
        # The loop above used to spin on both forever. It never came up because,
        # until the fence guard landed, a fenced run reported `completed`.
        if state in _RUN_WAIT_OPERATOR_STATES:
            reason = status_resp.get("pause_reason") or status_resp.get("quarantine_reason")
            click.echo("")
            click.echo(f"Stopped by: {reason}" if reason
                       else f"Stopped, and {state} is not a state that clears itself.")
            click.echo("Nothing will move this run until you act on it:")
            # `resume`, NOT `replay`. Typing the string this command had first
            # printed answered 409 — "run is paused; only a terminal run can be
            # replayed" — which is plan 52's shape reproduced by the fix for it,
            # in the same session. A paused run is resumed; a terminal one is
            # replayed; the two verbs are not interchangeable and the message
            # has to know which state it is looking at.
            click.echo(f"  papayya resume {run_id}   # clear the fence and re-drive")

    except PapayyaAPIError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    finally:
        api.close()


# States `papayya run --wait` stops polling on. NOT the same as "terminal".
#
# `paused` and `quarantine` are non-terminal on the run row — an operator can
# resume or release them — but nothing in the system will move them without one,
# so a client waiting for them waits forever. That is not hypothetical: a fence
# raises WorkloadPaused at the start of the NEXT step, so a fence that trips on
# a run's last step leaves the run paused with the body already returned, and
# this command printed "4 step(s) — paused" every two seconds until killed.
_RUN_WAIT_OPERATOR_STATES = ("paused", "quarantine")
_RUN_WAIT_STOP_STATES = (
    "completed", "failed", "cancelled", "budget_exceeded",
) + _RUN_WAIT_OPERATOR_STATES


def _fmt_usd(amount: float) -> str:
    """Render a dollar amount without rounding a real charge to zero.

    ``${:.4f}`` is what the dashboard shipped until plan 51 had to remove it:
    a real $0.000034 item rendered ``$0.0000``, which is this product's own
    cost column telling a customer their run was free. Sub-cent amounts get
    the precision they need; everything else stays readable.
    """
    if amount == 0:
        return "$0.00"
    if abs(amount) < 0.01:
        return f"${amount:.6f}"
    return f"${amount:,.2f}"


def _fmt_cost(record: dict[str, Any], amount: float) -> str:
    """Cost for one record, or the reason there isn't one.

    ``cost_priced`` is false when the amount is 0 only because nothing could
    price it — a project with an empty rate card prices every record at 0, and
    printing ``$0.00`` for that is the silent-wrong-answer this product exists
    to catch (plan 51 / ``model.PricingFor``). The server already ships the
    verdict and a sentence to go with it; both are rendered rather than
    re-derived, so the CLI cannot disagree with the dashboard.
    """
    if record.get("cost_priced") is False:
        reason = record.get("cost_unpriced_reason") or "unpriced"
        return f"—  ({reason})"
    return _fmt_usd(amount)


def _fence_objection(run: dict[str, Any]) -> str | None:
    """The fence's objection, when the run finished anyway (plan 53).

    A fence raises WorkloadPaused at the START OF THE NEXT STEP, so one that
    trips on a run's LAST step never raises: the body returns, the worker
    completes, and the run reaches terminal carrying `paused_at` and
    `pause_reason` from a pause nothing acted on. The row is not lying — the
    work did finish AND a fence objected, and those are two facts on two
    columns — but until now no surface read the second one, so a customer who
    had just been alerted "auto-paused" opened the run and saw `completed`.

    Returns None while the pause is LIVE (a non-terminal run) — there the
    status word already says it — and None once an operator has resolved it.
    """
    if run.get("status") not in ("completed", "failed"):
        return None
    if not run.get("paused_at") or run.get("pause_resolved_at"):
        return None
    return run.get("pause_reason") or "a fence objected to this run"


def _echo_run_verdict(run: dict[str, Any]) -> None:
    """Say how the run ENDED, under a list of how its steps went.

    A step list is not a verdict. The step that raised never checkpoints, so a
    failed run's `papayya logs` is a clean list of the steps that worked and no
    mention of the one that didn't — the command named "logs" being the one
    place the failure is invisible. Silent on a healthy run: the list already
    said it.
    """
    status = run.get("status")
    if status in ("failed", "paused", "cancelled"):
        line = f"Run {status}"
        if run.get("error"):
            line += f": {run['error']}"
            if run.get("error_category"):
                line += f"  [{run['error_category']}]"
        click.echo(line)
        return
    worst = run.get("worst_outcome_status")
    if worst and worst != "ok":
        click.echo(f"Run {status}, but its worst step outcome was {worst}.")
    if objection := _fence_objection(run):
        click.echo(f"A fence objected while it ran: {objection}")


def _echo_submission_status(api: "APIClient", run_id: str,
                            *, original: Exception) -> None:
    """Render a SUBMISSION's rollup for an id the item surface did not know.

    Re-raises the original 404 when this surface does not know it either: an id
    that is neither an item nor a submission is a typo, and inventing a second
    error for it would bury the first.
    """
    run = api.get_run_v2(run_id)
    if run is None:
        raise original

    items = f"{run.get('item_count', 0)}"
    failed = run.get("failed_count") or 0
    click.echo(f"Run:     {run.get('id') or run_id}")
    click.echo(f"Agent:   {run.get('agent') or '?'}")
    click.echo(f"Status:  {run.get('status', 'unknown')}"
               + (f"  ({failed} of {items} item(s) failed)" if failed else ""))

    worst = run.get("worst_outcome_status")
    degraded = run.get("degraded_count") or 0
    if (worst and worst != "ok") or degraded:
        click.echo(f"Outcome: {worst or 'ok'}"
                   + (f" ({degraded} degraded step(s))" if degraded else ""))

    click.echo(f"Items:   {items}"
               + (f" — {run.get('queued_count', 0)} queued, "
                  f"{run.get('running_count', 0)} running, "
                  f"{run.get('completed_count', 0)} completed"
                  if run.get("status") not in ("completed",) else ""))
    click.echo(f"Cost:    {_fmt_run_cost(run)}")
    click.echo(f"\nItems in this run:  papayya items list --run {run_id}")


def _fmt_run_cost(run: dict[str, Any]) -> str:
    """A RUN's cost, which is a SUM and so has its own unpriced rule.

    ``cost_priced`` is a property of ONE record; a run is many, and its
    run-grain equivalent is ``unpriced_item_count``. The reading is asymmetric,
    and matches ``formatRunCost`` in the dashboard so the two surfaces cannot
    disagree about the same run: with SOME items unpriced the sum is a real
    lower bound worth showing beside what it is missing, and with EVERY item
    unpriced it bounds nothing, so leading with a number would be the
    free-vs-unpriced confusion one level up (plan 51).
    """
    unpriced = run.get("unpriced_item_count") or 0
    total = run.get("item_count") or 0
    if unpriced and unpriced >= total:
        return "—"
    rendered = _fmt_usd(run.get("cost_usd") or 0)
    return f"{rendered}*" if unpriced else rendered


@main.command()
@click.argument("run_id")
@click.pass_context
def status(ctx: click.Context, run_id: str) -> None:
    """Check the status of a run."""
    scope = _env_scope(ctx.obj)
    resolved_key = _require_api_key(scope)
    config = APIConfig(api_key=resolved_key, base_url=scope.base_url)
    api = APIClient(config)

    # Every field is read with .get() and a fallback. This command used to
    # index `result['id']` and crashed with `KeyError: 'id'` on every hosted
    # run there has ever been — the durable API returns `run_id` (plan 48 W6).
    # Two of its other three lines read `current_step` and `total_cost_cents`,
    # which no response has ever carried either, so they printed a confident
    # `Step: 0` / `Cost: 0 cents` for four months.
    try:
        try:
            result = api.get_run(run_id)
        except PapayyaAPIError as e:
            # An id the customer holds is just an id. `runs submit` returns a
            # GROUP id and `run` returns an ITEM id, and they live on different
            # surfaces — /v2/runs/{id} and /v1/durable/runs/{id} — so this
            # command used to 404 on half the ids the product had handed out
            # (plan 48 W6's second half). Nobody should have to know which noun
            # they were given to ask how it went.
            if getattr(e, "status", None) != 404:
                raise
            _echo_submission_status(api, run_id, original=e)
            return
        agent = result.get("agent") or "?"
        version = result.get("agent_version")
        click.echo(f"Run:     {result.get('run_id') or run_id}")
        click.echo(f"Agent:   {agent}" + (f" (v{version})" if version else ""))
        run_status = result.get("status", "unknown")
        click.echo(f"Status:  {run_status}")

        # HOW LONG it has been non-terminal, which is the whole of the question
        # on a run that is not finished (plan 53 W3). `queued` on its own is a
        # true statement that answers nothing: queued for four seconds is a
        # healthy submission and queued for three days is a dead worker pool,
        # and the word is identical in both. The server now keeps the status
        # honest — a run whose item is waiting in the queue says `queued`
        # rather than `running` — so this line is what turns that honesty into
        # something a customer can act on.
        #
        # Terminal runs are excluded deliberately: "failed 3d ago" is a fact
        # about the past that `runs list` already carries, and putting an age
        # under every status would bury the one case where it is the signal.
        if run_status in ("queued", "running", "paused"):
            waited = _ago(result.get("created_at"))
            click.echo(f"Waiting: {waited.removesuffix(' ago')} in {run_status}")

        # Why it failed, on the command whose whole job is to say how the run
        # is doing. The columns exist since plan 50; nothing was reading them.
        if result.get("error"):
            category = result.get("error_category")
            click.echo(
                f"Error:   {result['error']}"
                + (f"  [{category}]" if category else "")
            )

        # A fence objected and the run finished anyway. Two facts, two columns,
        # and until plan 53 the second one reached nobody — so a customer with
        # an "auto-paused" alert in their inbox opened the run and read
        # `completed`. Printed above the outcome line because it is the stronger
        # statement: the platform did not merely observe a bad outcome, it
        # decided this should stop.
        if objection := _fence_objection(result):
            click.echo(f"Fenced:  {objection}")
            click.echo("         (it finished before the fence could stop it; "
                       "replay it if you want the objected steps re-run)")

        # The wedge: a run can be `completed` and still not have worked. Say so
        # here rather than leaving the operator to notice it on the dashboard.
        worst = result.get("worst_outcome_status")
        degraded = result.get("degraded_count") or 0
        if (worst and worst != "ok") or degraded:
            click.echo(
                f"Outcome: {worst or 'ok'}"
                + (f" ({degraded} degraded step(s))" if degraded else "")
            )

        spent = result.get("budget_consumed_usd") or 0
        limit = result.get("budget_limit_usd")
        click.echo(
            f"Cost:    {_fmt_cost(result, spent)}"
            + (f" of {_fmt_usd(limit)} budget" if limit else "")
        )
    finally:
        api.close()


def _fmt_step_time(raw: object) -> str:
    """Wall-clock HH:MM:SS for a step, or "" when there is nothing to show.

    `papayya logs` printed durations and no timestamps, so "where did it stop"
    meant reading to the end and inferring (plan 55 D6.1). A duration says how
    long a step took; only a timestamp says when the run went quiet.

    Never raises on a shape it does not recognise — a log command that dies on
    an unexpected date format is worse than one that omits a column.
    """
    if not isinstance(raw, str) or not raw:
        return ""
    try:
        from datetime import datetime

        # Server sends RFC3339 with a Z; fromisoformat wants +00:00 before 3.11.
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).strftime("%H:%M:%S")
    except Exception:  # noqa: BLE001 — see docstring
        return ""


@main.command()
@click.argument("run_id")
@click.option(
    "--tail", "-n", type=int, default=None,
    help="Show only the last N steps. A wedged run is found at the END.",
)
@click.pass_context
def logs(ctx: click.Context, run_id: str, tail: int | None) -> None:
    """Show step-by-step logs for a run.

    Prints the customer's own ``item_id`` per step, not just the step label —
    "where did it stop" is a question about the record, and a 150-page
    document dumps 150 entries without ``--tail``.
    """
    scope = _env_scope(ctx.obj)
    resolved_key = _require_api_key(scope)
    config = APIConfig(api_key=resolved_key, base_url=scope.base_url)
    api = APIClient(config)

    # Durable checkpoints carry {label, result, outcome_status, attempt, seq,
    # cost_usd, duration_ms, llm_*_tokens}. This command read step_number,
    # step_type, status, input_tokens, output_tokens and output — six names,
    # none of which the API has returned since the v1 cutover, so it raised
    # `KeyError: 'step_number'` on every hosted run with a step (plan 48 W4).
    # `run --wait` calls the same api.get_steps sixty lines above and WAS
    # migrated; the comment there declaring the v1 shape gone sat in this file
    # the whole time.
    try:
        # The run first, then its steps. "No steps found." meant three
        # different things and exited 0 for all of them: this run recorded no
        # steps, this run does not exist (a typo'd id), or the command had just
        # crashed. Fetching the run makes a bad id a 404 instead of a healthy
        # empty result, and gives us the verdict for the trailer below.
        run = api.get_run(run_id)
        steps = api.get_steps(run_id)
        total = len(steps)
        if tail is not None and tail > 0 and total > tail:
            # Say what was hidden. A silently truncated list reads as a
            # complete one, and the operator counting steps would be counting
            # the wrong number.
            steps = steps[-tail:]
            click.echo(f"(showing the last {tail} of {total} steps)\n")
        if not steps:
            click.echo(
                f"Run {run.get('run_id') or run_id} is "
                f"{run.get('status', 'unknown')} and recorded no steps."
            )
            click.echo(
                "  Steps come from run.step(...) calls inside the agent; "
                "an agent that never calls one has nothing to show here."
            )
            _echo_run_verdict(run)
            return

        for i, s in enumerate(steps, 1):
            label = s.get("label", "?")
            # Mark re-executions, matching `run --wait`: two rows with the same
            # label otherwise read as a duplicate rather than as attempt two.
            attempt = s.get("attempt", 1)
            if attempt > 1:
                label = f"{label} (attempt {attempt})"
            outcome = s.get("outcome_status") or "?"
            duration = s.get("duration_ms") or 0

            # THE CUSTOMER'S OWN ID, which this command has never printed
            # (plan 55 D6.1). It printed the step LABEL, and the `#72` in
            # `read-photo#72` is the auto-suffix on a repeated label — not the
            # id the customer set. They coincided in plan 55 only because that
            # loop happened to be 1:1 with pages, which is exactly the kind of
            # coincidence that makes a missing field look present.
            #
            # The renderer's field list predates item_id on the checkpoint
            # model; nobody re-derived it when the model grew.
            item = s.get("item_id")
            when = _fmt_step_time(s.get("created_at") or s.get("completed_at"))
            head = f"{i}. {label} — {outcome}  ({duration}ms)"
            if when:
                head += f"  [{when}]"
            click.echo(head)
            if item:
                click.echo(f"   Item: {item}")

            # Tokens and cost only when the step actually made a model call.
            # llm_*_tokens are omitempty server-side, so defaulting them to 0
            # would print "Tokens: 0 in / 0 out" for every non-LLM step — a
            # measurement where there was no measurement.
            tokens_in = s.get("llm_prompt_tokens")
            tokens_out = s.get("llm_completion_tokens")
            if tokens_in is not None or tokens_out is not None:
                click.echo(
                    f"   Tokens: {tokens_in or 0} in / {tokens_out or 0} out"
                    f" | Cost: {_fmt_cost(s, s.get('cost_usd') or 0)}"
                )

            reason = s.get("outcome_reason")
            if reason:
                click.echo(f"   Reason: {reason}")

            result = s.get("result")
            if result is not None:
                rendered = result if isinstance(result, str) else json.dumps(result)
                click.echo(f"   Result: {rendered[:300]}")

            click.echo()

        _echo_run_verdict(run)
    finally:
        api.close()


def _secrets_scope(ctx: click.Context, project_id_override: str | None) -> tuple[APIClient, str]:
    """Resolve (APIClient, project_id) for a secrets command.

    --project-id (flag) wins; otherwise fall back to the env's project_id.
    Fixes a pre-Phase-1 bug where secrets read the legacy flat
    `project_id` key and silently broke for migrated accounts.
    """
    scope = _env_scope(ctx.obj)
    resolved_key = _require_api_key(scope)
    project_id = project_id_override or _require_project_id(scope)
    return APIClient(APIConfig(api_key=resolved_key, base_url=scope.base_url)), project_id


@main.group()
@click.pass_context
def secrets(ctx: click.Context) -> None:
    """Manage project secrets."""
    pass


@secrets.command("set")
@click.argument("name")
@click.argument("value")
@click.option("--project-id", required=False, default=None, help="Project ID (overrides env config)")
@click.pass_context
def secrets_set(ctx: click.Context, name: str, value: str, project_id: str | None) -> None:
    """Set a secret for a project."""
    api, project_id = _secrets_scope(ctx, project_id)
    try:
        api.set_secret(project_id, name, value)
        click.echo(f"Secret '{name}' set successfully.")
    finally:
        api.close()


@secrets.command("list")
@click.option("--project-id", required=False, default=None, help="Project ID (overrides env config)")
@click.pass_context
def secrets_list(ctx: click.Context, project_id: str | None) -> None:
    """List secrets for a project (names only)."""
    api, project_id = _secrets_scope(ctx, project_id)
    try:
        result = api.list_secrets(project_id)
        if not result:
            click.echo("No secrets found.")
            return
        for s in result:
            click.echo(f"  {s['name']}  (updated: {s.get('updated_at', '?')})")
    finally:
        api.close()


@secrets.command("delete")
@click.argument("name")
@click.option("--project-id", required=False, default=None, help="Project ID (overrides env config)")
@click.pass_context
def secrets_delete(ctx: click.Context, name: str, project_id: str | None) -> None:
    """Delete a secret."""
    api, project_id = _secrets_scope(ctx, project_id)
    try:
        api.delete_secret(project_id, name)
        click.echo(f"Secret '{name}' deleted.")
    finally:
        api.close()


# ---------------------------------------------------------------------------
# rate-card — per-project per-model token pricing for dashboard $ estimates.
# Customer provides their own rates; Papayya doesn't ship a pricing table.
# ---------------------------------------------------------------------------


def _dollars_per_million_to_cents(amount: float) -> int:
    """Convert the dollar-per-million-tokens amount humans type from a
    pricing page into the integer cents stored internally. Rounds to the
    nearest cent — providers don't publish fractional-cent rates."""
    return int(round(amount * 100))


def _cents_per_million_to_dollars(cents: int) -> float:
    return cents / 100.0


def _require_rate_card_context(ctx: click.Context) -> tuple[APIClient, str]:
    """Resolve API key + project id, build an APIClient. Exits on missing auth."""
    scope = _env_scope(ctx.obj)
    resolved_key = _require_api_key(scope)
    project_id = _require_project_id(scope)
    api = APIClient(APIConfig(api_key=resolved_key, base_url=scope.base_url))
    return api, project_id


@main.group("rate-card")
@click.pass_context
def rate_card(ctx: click.Context) -> None:
    """Manage per-model token pricing for dashboard $ estimates.

    Papayya doesn't ship a built-in pricing table — you bring your own
    rates. Token counts are always recorded; rate cards turn them into
    dollar estimates only where you've configured pricing.
    """


@rate_card.command("show")
@click.pass_context
def rate_card_show(ctx: click.Context) -> None:
    """Print the current rate card as JSON (cents per million tokens)."""
    api, project_id = _require_rate_card_context(ctx)
    try:
        result = api.get_rate_card(project_id)
    except PapayyaAPIError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    finally:
        api.close()
    click.echo(json.dumps(result, indent=2, sort_keys=True))


@rate_card.command("set")
@click.argument("model")
@click.option("--input-per-million", type=float, required=True, help="Dollars per million input tokens (e.g. 3.00)")
@click.option("--output-per-million", type=float, required=True, help="Dollars per million output tokens (e.g. 15.00)")
@click.pass_context
def rate_card_set(ctx: click.Context, model: str, input_per_million: float, output_per_million: float) -> None:
    """Add or update pricing for a single model. Dollars in, cents stored."""
    if input_per_million < 0 or output_per_million < 0:
        click.echo("Error: prices must be non-negative.", err=True)
        sys.exit(1)

    api, project_id = _require_rate_card_context(ctx)
    try:
        current = api.get_rate_card(project_id)
        current[model] = {
            "input_cents_per_million":  _dollars_per_million_to_cents(input_per_million),
            "output_cents_per_million": _dollars_per_million_to_cents(output_per_million),
        }
        api.set_rate_card(project_id, current)
    except PapayyaAPIError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    finally:
        api.close()

    click.echo(f"Rate card updated for {model}: ${input_per_million:.2f}/1M in, ${output_per_million:.2f}/1M out")


@rate_card.command("remove")
@click.argument("model")
@click.pass_context
def rate_card_remove(ctx: click.Context, model: str) -> None:
    """Remove pricing for a single model."""
    api, project_id = _require_rate_card_context(ctx)
    try:
        current = api.get_rate_card(project_id)
        if model not in current:
            click.echo(f"Model {model} not in rate card (nothing to remove).")
            return
        del current[model]
        api.set_rate_card(project_id, current)
    except PapayyaAPIError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    finally:
        api.close()
    click.echo(f"Removed rate card entry for {model}.")


@rate_card.command("import")
@click.option("--file", "file_path", required=True, type=click.Path(exists=True), help="JSON file (cents per million tokens)")
@click.pass_context
def rate_card_import(ctx: click.Context, file_path: str) -> None:
    """Bulk-replace the rate card from a JSON file.

    The file's shape must match `papayya rate-card show` output — a JSON
    object mapping model_id → {input_cents_per_million, output_cents_per_million}.
    """
    try:
        with open(file_path) as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        click.echo(f"Error reading {file_path}: {e}", err=True)
        sys.exit(1)

    if not isinstance(payload, dict):
        click.echo("Error: rate card file must contain a JSON object.", err=True)
        sys.exit(1)

    api, project_id = _require_rate_card_context(ctx)
    try:
        api.set_rate_card(project_id, payload)
    except PapayyaAPIError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    finally:
        api.close()
    click.echo(f"Rate card imported ({len(payload)} models).")


@rate_card.command("edit")
@click.pass_context
def rate_card_edit(ctx: click.Context) -> None:
    """Open the current rate card in $EDITOR and write back on save."""
    api, project_id = _require_rate_card_context(ctx)
    try:
        current = api.get_rate_card(project_id)
    except PapayyaAPIError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    edited_raw = click.edit(json.dumps(current, indent=2, sort_keys=True) + "\n", extension=".json")
    if edited_raw is None:
        click.echo("No changes (editor exited without saving).")
        api.close()
        return

    try:
        edited = json.loads(edited_raw)
    except json.JSONDecodeError as e:
        click.echo(f"Error: edited content is not valid JSON: {e}", err=True)
        api.close()
        sys.exit(1)
    if not isinstance(edited, dict):
        click.echo("Error: rate card must be a JSON object.", err=True)
        api.close()
        sys.exit(1)

    try:
        api.set_rate_card(project_id, edited)
    except PapayyaAPIError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    finally:
        api.close()
    click.echo(f"Rate card saved ({len(edited)} models).")


# ---------------------------------------------------------------------------
# batch — submit / inspect / cancel / retry batches
# ---------------------------------------------------------------------------

def _make_papayya_client(ctx: click.Context) -> Any:
    """Resolve auth and return a Papayya client, exiting with a friendly
    error if no API key is configured. Callers are responsible for
    ``client.close()`` in a finally block."""
    from papayya import Papayya

    scope = _env_scope(ctx.obj)
    resolved_key = _require_api_key(scope)
    return Papayya(api_key=resolved_key, base_url=scope.base_url)


def _iter_jsonl_items(path: str) -> Iterator[dict[str, Any]]:
    """Yield one dict per non-blank line of a JSONL file.

    Each line must be a JSON object — we don't reshape it. The SDK accepts
    ``{"input": ..., "metadata"?: ...}`` and the backend enforces the
    schema, so bad rows surface as a 400 from the stream endpoint rather
    than here.
    """
    filepath = Path(path)
    if not filepath.exists():
        click.echo(f"Error: File not found: {filepath}", err=True)
        sys.exit(1)

    with filepath.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                click.echo(f"Error: {filepath}:{lineno}: invalid JSON ({e})", err=True)
                sys.exit(1)


def _ago(iso: str | None) -> str:
    """A timestamp as "3m ago", the way a person reads a list.

    Absolute times are the right thing on ONE record's page, where the question
    is "when exactly"; in a list the question is "which of these is recent",
    and 14 identical date prefixes answer it worse than a relative offset does.
    """
    if not iso:
        return "—"
    from datetime import datetime, timezone

    try:
        when = datetime.fromisoformat(iso.replace("Z", "+00:00"))
    except ValueError:
        return iso
    seconds = (datetime.now(timezone.utc) - when).total_seconds()
    if seconds < 0:
        return "just now"
    for unit, size in (("d", 86400), ("h", 3600), ("m", 60)):
        if seconds >= size:
            return f"{int(seconds // size)}{unit} ago"
    return f"{int(seconds)}s ago"


@runs.command("list")
@click.option("--agent", "agent_slug", default=None,
              help="Only this agent's runs (slug, e.g. `triage`).")
@click.option("--limit", type=int, default=20, show_default=True,
              help="How many runs to show, newest first.")
@click.option("--json", "as_json", is_flag=True, default=False,
              help="NDJSON instead of a table — one run per line, for jq.")
@click.pass_context
def runs_list(ctx: click.Context, agent_slug: str | None, limit: int,
              as_json: bool) -> None:
    """List your runs, newest first.

    The answer to "what have I run?". Until now the CLI had none: `items list`
    prints per-record NDJSON, and `status` / `logs` both need a run id you had
    to keep — so losing the id `papayya run` printed left no way back to your
    own work (plan 52 G1).
    """
    client = _make_papayya_client(ctx)
    try:
        runs_ = client.runs.list(agent=agent_slug, limit=limit)
    except PapayyaAPIError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    finally:
        client.close()

    if as_json:
        for r in runs_:
            click.echo(json.dumps(r))
        return

    if not runs_:
        click.echo(
            f"No runs yet{f' for agent {agent_slug}' if agent_slug else ''}."
        )
        click.echo("  Submit one with `papayya run <agent> \"<input>\"` or "
                   "`papayya runs submit --agent <id> --file items.jsonl`.")
        return

    click.echo(f"{'RUN':<38} {'AGENT':<16} {'STATUS':<10} "
               f"{'ITEMS':>12} {'COST':>12}  STARTED")
    for r in runs_:
        items = f"{r.get('item_count', 0)}"
        failed = r.get("failed_count") or 0
        if failed:
            items += f" ({failed} failed)"
        click.echo(
            f"{r.get('id', '?'):<38} {(r.get('agent') or '?'):<16} "
            f"{(r.get('status') or '?'):<10} {items:>12} "
            f"{_fmt_run_cost(r):>12}  {_ago(r.get('created_at'))}"
        )

    # A list that silently stops at its cap answers "what have I run" with a
    # subset and looks like the whole. Say so, and say what to type.
    if len(runs_) >= limit:
        click.echo(
            f"\n(showing the newest {limit} — there may be more; "
            f"pass --limit)",
            err=True,
        )


def _refuse_unwired_submit_flags(
    *, concurrency_cap: int | None, name: str | None, callback_url: str | None,
) -> None:
    """Refuse the three `runs submit` flags nothing reads (plan 53 S4).

    Each message says what the flag would need and what to do instead, because
    a refusal that only says "no" moves the dead end rather than removing it.
    """
    problems: list[str] = []
    if concurrency_cap is not None:
        # MEASURED, not asserted. 20 items of an 800ms agent on the compose
        # stack: 17.1s with one worker, 4.2s with four — linear, because
        # LeaseRuntimeItem is FOR UPDATE SKIP LOCKED and N workers never
        # collide. Throughput is a property of the pool, not of the run.
        #
        # The per-key cap machinery this flag looks like it should use is real
        # (runtime_pending.concurrency_per_key, enforced by the dispatcher's
        # CheckAndReserve with deferral). It buckets on the PARTITION KEY,
        # falling back to the account — so wiring `--concurrency 8` to it on a
        # tenant-keyed run means 8 PER TENANT, which is `--budget`'s $2-becomes-
        # $1000 defect in a different column. It needs a run-grain bucket first.
        problems.append(
            "--concurrency is not implemented. Nothing reads concurrency_cap: "
            "each worker runs one item at a time, and throughput comes from "
            "the number of workers (`docker compose up -d --scale worker=N` "
            "locally; the hosted pool autoscales). A per-run cap needs a "
            "run-grain bucket in the dispatcher's limiter, which today buckets "
            "per tenant."
        )
    if name is not None:
        problems.append(
            "--name is not implemented. There is no column to put a run label "
            "in — `invocations` has no `name` — which is why the run list shows "
            "bare UUIDs. Use --idempotency-key if you need your own handle on a "
            "submission."
        )
    if callback_url is not None:
        problems.append(
            "--callback-url is not implemented on the v2 run path. The delivery "
            "machinery exists and is wired into the handler, and the handler "
            "never calls it. Poll `papayya status <run-id>`, or use a @trigger "
            "webhook."
        )
    if not problems:
        return
    for line in problems:
        click.echo(f"Error: {line}", err=True)
    sys.exit(2)


@runs.command("submit")
@click.option("--agent", "agent_id", required=True, help="Agent ID to run each item against")
@click.option("--file", "file_path", required=True, type=click.Path(exists=False), help="JSONL file — one item per line, e.g. {\"input\": ..., \"metadata\"?: ...}")
@click.option("--budget", "budget_dollars", type=float, default=None, help="Total run budget in whole dollars (converted to cents)")
@click.option("--concurrency", "concurrency_cap", type=int, default=None,
              help="NOT IMPLEMENTED — refused rather than silently dropped. "
                   "Throughput comes from the size of the worker pool.")
@click.option("--name", "name", default=None,
              help="NOT IMPLEMENTED — refused rather than silently dropped. "
                   "There is no column to store a run label in yet.")
@click.option("--callback-url", "callback_url", default=None,
              help="NOT IMPLEMENTED — refused rather than silently dropped. "
                   "Nothing delivers callbacks on the v2 run path.")
@click.option("--idempotency-key", "idempotency_key", default=None, help="Client-supplied key to dedupe duplicate submissions")
@click.pass_context
def runs_submit(
    ctx: click.Context,
    agent_id: str,
    file_path: str,
    budget_dollars: float | None,
    concurrency_cap: int | None,
    name: str | None,
    callback_url: str | None,
    idempotency_key: str | None,
) -> None:
    """Submit a hosted run over the items in a JSONL file.

    Always uses the NDJSON streaming path — no item ceiling, only a 1 GiB
    byte guard enforced by the backend. Prints the run ID on success.
    (Pre-0.3.0 spelling: `papayya batch submit` — kept as a hidden alias.)
    """
    # REFUSED, NOT DROPPED (plan 47 S4/S7, closed in plan 53).
    #
    # `--budget` was the fourth flag in this family and the one that cost
    # money: parsed, put on the request struct, and never read by the handler,
    # so a customer capping a 200-item run at $2 got 200 x $5 of headroom. It
    # is wired now. These three are not, and accepting them is the same defect
    # with a cheaper blast radius — the customer believes they have a lever.
    #
    # Refusing is plan 47's own stated fallback ("either wire them or reject
    # them at the CLI"), and the message names what each would actually take,
    # because "not implemented" without a way forward is a dead end.
    _refuse_unwired_submit_flags(
        concurrency_cap=concurrency_cap, name=name, callback_url=callback_url)

    budget_cents_cap = int(round(budget_dollars * 100)) if budget_dollars is not None else None

    client = _make_papayya_client(ctx)
    try:
        result = client.runs.create_stream(
            agent_id=agent_id,
            items=_iter_jsonl_items(file_path),
            name=name,
            budget_cents_cap=budget_cents_cap,
            concurrency_cap=concurrency_cap,
            callback_url=callback_url,
            idempotency_key=idempotency_key,
        )
    except PapayyaAPIError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    finally:
        client.close()

    # `group_id`, which is what POST /v1/batches returns and what every read
    # surface addresses this submission by — GET /v2/runs/{id}, `papayya
    # status`, `papayya logs`. Reading `id` with a '?' default printed
    # "Run submitted: ?" on every successful submission there has ever been,
    # destroying the only handle to the work at the moment it was created
    # (plan 47 S6). The fallbacks are ordered oldest-wire-last, and there is no
    # '?': a submission with no id is a bug worth seeing, not a shrug.
    run_id = result.get("group_id") or result.get("run_id") or result.get("id")
    click.echo(f"Run submitted: {run_id}")
    status_val = result.get("status")
    if status_val:
        click.echo(f"  Status: {status_val}")
    total = result.get("total_items")
    if total is not None:
        click.echo(f"  Items:  {total}")
    if run_id:
        click.echo(f"\nNext:\n  papayya status {run_id}\n  papayya runs list")


# Deprecated alias (Plan 34: batch → run): `papayya batch submit` keeps
# working, hidden from --help. The command object is shared, so behavior
# can never drift from `runs submit`.
batch = click.Group(
    "batch",
    hidden=True,
    help="Deprecated alias of `papayya runs` (batch → run in 0.3.0).",
)
batch.add_command(runs_submit, "submit")
main.add_command(batch)


# ---------------------------------------------------------------------------
# Tiered --help (Plan 34 Unit 4). Rung-0 — the free local loop — reads
# first, as a quickstart; hosted/ops groups sit below. Anything new that
# isn't registered here lands in a trailing "Other commands" bucket
# (see TieredGroup) rather than disappearing from help.
# ---------------------------------------------------------------------------

# Plan 37: `example`, `dev`, and `project` (local ledger export/import) are
# deactivated local surfaces — dropped from the help tiers.
main.sections = [
    ("Getting started", ["deploy", "replay", "login"]),
    # `pull` and `verify` sit beside `triage` and `replay` deliberately: they
    # are the recovery loop's verbs (Plan 41 R4 / ADR 0009 D7b), and an
    # operator looking for "what do I do about these failures" should meet
    # them in the same section they found the failures in.
    # `resume` sits beside `replay` for the same reason `pull` sits beside
    # `triage`: they are the two halves of one question. A run that ENDED badly
    # is replayed; a run something DECIDED to stop is resumed; the server
    # refuses each on the other's states, and finding only one of them in the
    # help is how an operator meets a 409 instead of a verb.
    ("Run agents & inspect results",
     ["run", "runs", "items", "status", "logs", "agents", "schedules",
      "triggers", "triage", "pull", "verify", "release", "resume"]),
    ("Account & platform ops",
     ["signup", "logout", "envs", "secrets", "projects",
      "deployments", "api-keys", "usage", "rate-card"]),
]

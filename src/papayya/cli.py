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
    PapayyaYaml,
    PapayyaYamlError,
    current_env as _current_env,
    env_config as _env_config,
    list_envs as _list_envs,
    load_cli_config as _load_cli_config,
    load_yaml as _load_yaml,
    save_cli_config as _save_cli_config,
    set_env_config as _set_env_config,
)
from papayya._defaults import DEFAULT_BASE_URL
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


def _resolve_project_id(ctx_obj: dict) -> str | None:
    """Resolve project ID from flag, env, or the current env's saved config."""
    pid = os.environ.get("PAPAYYA_PROJECT_ID")
    if pid:
        return pid
    cfg = _load_cli_config()
    env_name = ctx_obj.get("env") or _current_env(cfg)
    return _env_config(cfg, env_name).get("project_id")


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


def _resolve_api_key(ctx_key: str | None, env: str | None = None) -> str | None:
    """Resolve API key from CLI flag, env var, or the current env's saved config."""
    key = ctx_key or os.environ.get("PAPAYYA_API_KEY")
    if key:
        return key
    cfg = _load_cli_config()
    return _env_config(cfg, env or _current_env(cfg)).get("api_key")


@dataclass(frozen=True)
class _EnvScope:
    """Resolved (env, api_key, project_id, base_url) for a server-hitting command."""

    env: str
    api_key: str | None
    project_id: str | None
    base_url: str


def _env_scope(ctx_obj: dict) -> _EnvScope:
    """Resolve env + credentials + base_url from the current CLI context.

    Precedence:
      - api_key: ctx flag / PAPAYYA_API_KEY env > envs[env].api_key
      - project_id: PAPAYYA_PROJECT_ID env > envs[env].project_id
      - base_url: explicit --base-url / PAPAYYA_BASE_URL > envs[env].base_url > DEFAULT_BASE_URL
    """
    cfg = _load_cli_config()
    env_name = ctx_obj.get("env") or _current_env(cfg)
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

    return _EnvScope(env=env_name, api_key=api_key, project_id=project_id, base_url=base_url)


def _require_api_key(scope: _EnvScope) -> str:
    if not scope.api_key:
        click.echo(
            "Error: No API key. Run `papayya signup` first, or set PAPAYYA_API_KEY.",
            err=True,
        )
        sys.exit(1)
    return scope.api_key


def _require_project_id(scope: _EnvScope) -> str:
    if not scope.project_id:
        click.echo(
            f"Error: No project ID for env '{scope.env}'. "
            f"Run `papayya envs link {scope.env} --project-id ...` "
            "or set PAPAYYA_PROJECT_ID.",
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
# init
# ---------------------------------------------------------------------------

@main.command()
def init() -> None:
    """Scaffold a minimal papayya.yaml in the current directory."""
    cwd = Path.cwd()
    target = cwd / "papayya.yaml"

    if target.exists():
        click.confirm(
            "papayya.yaml already exists. Overwrite?",
            default=False,
            abort=True,
        )

    # A starter env so the first `papayya deploy` doesn't fail
    # `_pick_yaml_env` with "no `envs:` block." `agents: {}` is valid —
    # deploy discovers the @agent-decorated function from agent.py; the
    # yaml block only carries per-agent schedules/webhooks.
    target.write_text(
        "version: 1\n"
        "envs:\n"
        "  dev:\n"
        "    agents: {}\n"
    )
    click.echo(f"✓ Created papayya.yaml in {cwd}")
    click.echo("")
    click.echo("Next: scaffold a runnable demo to feel the loop:")
    click.echo("")
    click.echo("    papayya example     # writes agent.py")
    click.echo("    python agent.py")
    click.echo("    papayya dev         # open the dashboard")
    click.echo("")
    click.echo("Or write your own agent — see https://docs.getpapayya.com")


# ---------------------------------------------------------------------------
# example
# ---------------------------------------------------------------------------

@main.command()
@click.option(
    "--print",
    "print_only",
    is_flag=True,
    default=False,
    help="Print the demo to stdout instead of writing a file.",
)
def example(print_only: bool) -> None:
    """Scaffold agent.py — a keyless durable run you can execute immediately."""
    from papayya._demo import LOCAL_DEMO_AGENT_SOURCE

    if print_only:
        click.echo(LOCAL_DEMO_AGENT_SOURCE, nl=False)
        return

    cwd = Path.cwd()
    # Write agent.py — the same name `deploy` / `run` auto-discover, so the
    # scaffold flows straight into the golden path instead of failing with
    # "No agent.py found."
    target = cwd / "agent.py"

    if target.exists():
        click.confirm(
            "agent.py already exists. Overwrite?",
            default=False,
            abort=True,
        )

    target.write_text(LOCAL_DEMO_AGENT_SOURCE)
    click.echo(f"✓ Wrote agent.py to {cwd}")
    click.echo("")
    click.echo("Run it:")
    click.echo("")
    click.echo("    python agent.py")
    click.echo("")
    click.echo("Then open the dashboard:")
    click.echo("")
    click.echo("    papayya dev")


# ---------------------------------------------------------------------------
# signup / login
# ---------------------------------------------------------------------------

@main.command()
@click.option("--email", prompt=True, help="Your email address")
@click.option("--password", prompt=True, hide_input=True, confirmation_prompt=True, help="Password (min 8 characters)")
@click.option("--name", prompt="Account name", help="Your name or org name")
@click.option("--force", is_flag=True, default=False, help="Overwrite an existing ~/.papayya/config.json.")
@click.pass_context
def signup(ctx: click.Context, email: str, password: str, name: str, force: bool) -> None:
    """Create a Papayya account and get an API key."""
    base_url = ctx.obj["base_url"]

    existing = _load_cli_config()
    existing_dev = _env_config(existing, "dev")
    if existing_dev.get("api_key") and not force:
        current_email = existing_dev.get("email") or (existing.get("auth") or {}).get("email", "<unknown email>")
        click.echo(
            f"Error: already signed in as {current_email} ({_CONFIG_FILE}).\n"
            "  Run `papayya logout` to clear the current config, or pass --force to overwrite.\n"
            "  (Creating a new account clobbers your existing API key.)",
            err=True,
        )
        sys.exit(1)

    if len(password) < 8:
        click.echo("Error: Password must be at least 8 characters.", err=True)
        sys.exit(1)

    api = APIClient(APIConfig(api_key="none", base_url=base_url))
    try:
        # 1. Register
        click.echo("Creating account...")
        reg = api.register(email, password, name)
        click.echo(f"  ✓ Account created ({reg.get('email', email)})")

        # 2. Login to get JWT
        click.echo("Logging in...")
        login_resp = api.login(email, password)
        jwt = login_resp["access_token"]

        # Re-create client with JWT auth
        api.close()
        api = APIClient(APIConfig(api_key=jwt, base_url=base_url))

        # 3. Create project
        click.echo("Setting up project...")
        slug = name.lower().replace(" ", "-")[:30]
        project = api.create_project(name=f"{name}'s project", slug=f"{slug}-project")
        project_id = project["id"]
        click.echo(f"  ✓ Project created ({project_id})")

        # 4. Create API key
        click.echo("Generating API key...")
        key_resp = api.create_api_key(project_id, name="default")
        api_key = key_resp.get("key") or key_resp.get("api_key") or key_resp.get("raw_key", "")
        click.echo(f"  ✓ API key generated")

        # 5. Persist config — dev env holds the project-scoped key; JWT lives
        # at the top level under `auth` for commands like `envs create` that
        # need account-level credentials.
        cfg = _load_cli_config()
        _set_env_config(cfg, "dev", {
            "api_key": api_key,
            "base_url": base_url,
            "project_id": project_id,
            "email": email,
        })
        cfg["current_env"] = "dev"
        cfg["auth"] = {"jwt": jwt, "email": email}
        _save_cli_config(cfg)
        click.echo(f"\n✓ All set! Config saved to {_CONFIG_FILE}")
        click.echo(f"  Env: dev")
        click.echo(f"  API key: {api_key[:12]}...")
        click.echo(f"  Project: {project_id}")
        click.echo("\nNext: papayya init")
        click.echo("Tip: pass --env <name> on any command to target a non-default env.")

    except PapayyaAPIError as e:
        if e.status == 409:
            click.echo("Error: An account with that email already exists. Try `papayya login`.", err=True)
            sys.exit(1)
        raise
    finally:
        api.close()


@main.command()
@click.option("--email", prompt=True, help="Your email address")
@click.option("--password", prompt=True, hide_input=True, help="Password")
@click.pass_context
def login(ctx: click.Context, email: str, password: str) -> None:
    """Log in to an existing Papayya account."""
    base_url = ctx.obj["base_url"]
    api = APIClient(APIConfig(api_key="none", base_url=base_url))

    try:
        click.echo("Logging in...")
        login_resp = api.login(email, password)
        jwt = login_resp["access_token"]

        # Re-create client with JWT auth
        api.close()
        api = APIClient(APIConfig(api_key=jwt, base_url=base_url))

        # Find existing projects and API keys
        projects = api.list_projects()
        if not projects:
            click.echo("Error: No projects found. Run `papayya signup` to create one.", err=True)
            sys.exit(1)

        project_id = projects[0]["id"]

        # Try to get existing API key or create one
        try:
            key_resp = api.create_api_key(project_id, name="cli-login")
            api_key = key_resp.get("key") or key_resp.get("api_key") or key_resp.get("raw_key", "")
        except PapayyaAPIError:
            # Use JWT as fallback
            api_key = jwt

        cfg = _load_cli_config()
        # Default to writing into dev unless another env is already current.
        target_env = _current_env(cfg) if cfg.get("envs") else "dev"
        _set_env_config(cfg, target_env, {
            "api_key": api_key,
            "base_url": base_url,
            "project_id": project_id,
            "email": email,
        })
        cfg["current_env"] = target_env
        cfg["auth"] = {"jwt": jwt, "email": email}
        _save_cli_config(cfg)
        click.echo(f"✓ Logged in! Config saved to {_CONFIG_FILE}")
        click.echo(f"  Env: {target_env}")
        click.echo(f"  Project: {project_id}")

    except PapayyaAPIError as e:
        if e.status == 401:
            click.echo("Error: Invalid email or password.", err=True)
            sys.exit(1)
        raise
    finally:
        api.close()


@main.command()
def logout() -> None:
    """Remove the saved CLI config (~/.papayya/config.json)."""
    if not _CONFIG_FILE.exists():
        click.echo("Not signed in — no config to remove.")
        return
    existing = _load_cli_config()
    email = (existing.get("auth") or {}).get("email") or _env_config(existing).get("email", "<unknown>")
    _CONFIG_FILE.unlink()
    click.echo(f"✓ Logged out ({email}). Removed {_CONFIG_FILE}.")


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
    """Provision a new project + API key and persist it as an env.

    Requires an account-level session (JWT) in ~/.papayya/config.json.
    Run `papayya login` if the command rejects your stored credentials.
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

    jwt = (cfg.get("auth") or {}).get("jwt")
    if not jwt:
        click.echo(
            "Error: no account session found. Run `papayya login` first — "
            "`envs create` needs account-level credentials to create projects.",
            err=True,
        )
        sys.exit(1)

    base_url = ctx.obj["base_url"]
    api = APIClient(APIConfig(api_key=jwt, base_url=base_url))
    try:
        slug = f"env-{name.lower()}"[:60]
        click.echo(f"Creating project for env '{name}'...")
        try:
            project = api.create_project(name=f"papayya env {name}", slug=slug)
        except PapayyaAPIError as exc:
            if exc.status in (401, 403):
                click.echo(
                    "Error: stored credentials were rejected. "
                    "Run `papayya login` to refresh, then retry.",
                    err=True,
                )
                sys.exit(1)
            raise
        project_id = project["id"]
        click.echo(f"  ✓ Project created ({project_id})")

        click.echo("Generating API key...")
        key_resp = api.create_api_key(project_id, name=f"cli-env-{name}")
        api_key = key_resp.get("key") or key_resp.get("api_key") or key_resp.get("raw_key", "")
        click.echo(f"  ✓ API key generated")

        _set_env_config(cfg, name, {
            "api_key": api_key,
            "base_url": base_url,
            "project_id": project_id,
        })
        cfg["current_env"] = name
        _save_cli_config(cfg)
        click.echo(f"\n✓ Env '{name}' is ready and selected as current.")
        click.echo(f"  Project: {project_id}")
        click.echo(f"  API key: {api_key[:12]}...")
    finally:
        api.close()


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

    If a `papayya.yaml` is present, schedules and webhooks declared in it are
    reconciled against the selected env's project after the bundle upload.
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

    # Detect optional papayya.yaml
    yaml_path = Path("papayya.yaml")
    spec: PapayyaYaml | None = None
    env_name: str | None = None
    if yaml_path.exists():
        try:
            spec = _load_yaml(yaml_path)
        except PapayyaYamlError as e:
            click.echo(f"Error: {e}", err=True)
            sys.exit(1)
        env_name = _pick_yaml_env(spec, ctx.obj.get("env"))

    # Resolve auth
    resolved_key = _resolve_api_key(ctx.obj["api_key"], env=env_name)
    if not resolved_key:
        click.echo(
            "Error: No API key found.\n"
            "  Run `papayya signup` first, or set PAPAYYA_API_KEY.",
            err=True,
        )
        sys.exit(1)

    config = APIConfig(api_key=resolved_key, base_url=ctx.obj["base_url"])
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

        # Resolve project ID for agent lookup/create
        if not project_id:
            project_id = _resolve_project_id({**ctx.obj, "env": env_name or ctx.obj.get("env")})
        if not project_id and not agent_id:
            click.echo("Error: No project ID. Set PAPAYYA_PROJECT_ID or run `papayya signup`.", err=True)
            sys.exit(1)

        if spec is not None:
            click.echo(f"Using env '{env_name}' (project {project_id})")

        # Deploy each agent; track slug -> agent_id for the reconciler.
        deployed: dict[str, str] = {}
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

            # Poll until build completes
            click.echo("  Building container image...")
            while True:
                time.sleep(3)
                status = api.get_deployment(deployment_id)
                state = status.get("status", "unknown")

                if state == "ready":
                    click.echo(f"  Build complete! Image: {status.get('image_ref', '?')}")
                    break
                elif state == "failed":
                    click.echo(f"  Build failed: {status.get('error_message', 'unknown error')}", err=True)
                    break
                else:
                    click.echo(f"  Status: {state}...")

            slug = reg.name.lower().replace(" ", "-")
            deployed[slug] = resolved_agent_id
            click.echo(f"  Deployed {slug} → {resolved_agent_id}")

        # Reconcile triggers. Sources are:
        #   1. papayya.yaml (if present) — yaml_env below.
        #   2. @schedule / @trigger decorators attached to @agent functions
        #      — populated in the module-level registry by _discover_agents
        #      above, harvested via _decorator_synthesis.
        #
        # The synthesis helper is imported lazily because it transitively
        # pulls papayya.decorators (croniter + zoneinfo). Eager import at
        # cli-module load time changes module-init ordering enough to mask
        # cross-process SQLite WAL writes in the worker subprocess test
        # (see Plan 11's __init__.py __getattr__ fix for context).
        yaml_env = spec.envs[env_name] if (spec is not None and env_name is not None) else None
        from papayya.agent import get_registry
        from papayya._decorator_synthesis import env_spec_from_registry_and_yaml
        env_spec = env_spec_from_registry_and_yaml(yaml_env, get_registry())
        has_triggers = any(
            a.schedules or a.webhooks for a in env_spec.agents.values()
        )
        if spec is not None and env_name is not None and not has_triggers:
            click.echo("\nNo triggers declared.")
        elif has_triggers:
            label = f" for env '{env_name}'" if env_name else ""
            click.echo(f"\nReconciling triggers{label}...")
            try:
                plan = _reconcile.diff_env(env_spec, deployed, api)
            except _reconcile.ReconcileError as e:
                click.echo(f"Error: {e}", err=True)
                sys.exit(1)

            _print_reconcile_plan(plan, api_base_url=ctx.obj["base_url"])

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
                _print_apply_result(result, api_base_url=ctx.obj["base_url"])
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


def _pick_yaml_env(spec: PapayyaYaml, cli_env: str | None) -> str:
    """Choose the env to reconcile against, or fail loud."""
    envs = sorted(spec.envs.keys())
    if not envs:
        click.echo("Error: papayya.yaml has no `envs:` block.", err=True)
        sys.exit(1)
    if cli_env is not None:
        if cli_env not in spec.envs:
            click.echo(
                f"Error: env '{cli_env}' not defined in papayya.yaml. Available: {envs}.",
                err=True,
            )
            sys.exit(1)
        return cli_env
    if len(envs) == 1:
        return envs[0]
    click.echo(
        f"Error: papayya.yaml defines multiple envs {envs}. Pass --env NAME "
        "(or set PAPAYYA_ENV).",
        err=True,
    )
    sys.exit(1)


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
        wh_payload = [
            {"name": w.name, "secret_env": w.secret_env}
            for w in agent_spec.webhooks
        ]
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


@main.command()
@click.option("--port", default=8585, help="Port for the dev dashboard")
@click.option("--host", default="127.0.0.1", help="Host to bind to")
@click.option("--db", default=".papayya/local.db", envvar="PAPAYYA_LOCAL_DB_PATH",
              help="Path to SQLite database (also honors PAPAYYA_LOCAL_DB_PATH)")
def dev(port: int, host: str, db: str) -> None:
    """Launch the local development dashboard."""
    click.echo(f"Starting Papayya Dev Dashboard...")
    from papayya.dev.server import serve
    serve(host=host, port=port, db_path=db)


# ---------------------------------------------------------------------------
# triage — unified Needs Attention feed across DLQ + quarantine
#
# v1→v2 cutover: the read feed is durable-backed. Quarantine-lane actions
# (release/discard) are live; the DLQ-lane disposition actions
# (skip/acknowledge/replay) retired with the v1 DROP and their durable
# replacement (dlq_disposition) is a deferred follow-up.
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


# DLQ-lane disposition (skip/acknowledge/replay) retired with the v1 DROP;
# the durable replacement (dlq_disposition) is a deferred follow-up, so retry
# and dismiss act on the quarantine lane only. Failed/degraded rows surface in
# `triage list` but have no resolve action yet.
_DLQ_DEFERRED_MSG = (
    "DLQ disposition actions (retry/dismiss for failed/degraded runs) are not "
    "yet available on durable runs — pending the dlq_disposition follow-up. "
    "Only quarantined runs can be retried or dismissed today."
)


@triage.command("retry")
@click.argument("run_id")
@click.pass_context
def triage_retry(ctx: click.Context, run_id: str) -> None:
    """Retry a triage row.

    Resumes a quarantined run in-place (``/release``). Exits with code 2 on
    any other status — DLQ-lane retry is a deferred follow-up.
    """
    client = _make_papayya_client(ctx)
    try:
        run = client.runs.get(run_id)
        status = run.get("status")
        if status == "quarantine":
            out = client.runs.release(run_id)
        else:
            click.echo(f"Error: run {run_id} is status={status!r}; {_DLQ_DEFERRED_MSG}", err=True)
            sys.exit(2)
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

    Abandons a quarantined run (``/discard``). Exits with code 2 on any
    other status — DLQ-lane dismiss is a deferred follow-up.
    """
    client = _make_papayya_client(ctx)
    try:
        run = client.runs.get(run_id)
        status = run.get("status")
        if status == "quarantine":
            out = client.runs.discard(run_id)
        else:
            click.echo(f"Error: run {run_id} is status={status!r}; {_DLQ_DEFERRED_MSG}", err=True)
            sys.exit(2)
        click.echo(json.dumps(out, indent=2))
    except PapayyaAPIError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    finally:
        client.close()


@main.command("replay")
@click.option("--run", "run_id", required=True, help="Run ID to replay")
@click.option("--file", "file", default=None,
              help="Agent file (default: auto-discover agent.py in cwd)")
@click.option("--db", default=".papayya/local.db", envvar="PAPAYYA_LOCAL_DB_PATH",
              help="Path to SQLite database (also honors PAPAYYA_LOCAL_DB_PATH)")
@click.option(
    "--latest",
    "latest",
    is_flag=True,
    default=False,
    help=(
        "Replay on the agent's current code even if its agent_version "
        "differs from the one captured on the original run. Without this "
        "flag, a version mismatch aborts the replay (ADR-0002 #7). Pre-#7 "
        "runs whose agent_version is NULL replay freely."
    ),
)
@click.option(
    "--from-step",
    "from_step",
    default=None,
    help=(
        "Phase 3 step-level rewind: resume from this step. Accepts a "
        "step label (string) or a 1-indexed step number (integer). "
        "Steps before this one are seeded into the new run's cache "
        "from the original; this step and everything after re-execute "
        "fresh."
    ),
)
def replay_cmd(
    run_id: str,
    file: str | None,
    db: str,
    latest: bool,
    from_step: str | None,
) -> None:
    """Re-drive a failed local run using its captured input snapshot.

    \b
    Usage:
      papayya replay --run <run_id>
      papayya replay --run <run_id> --file my_agents.py
      papayya replay --run <run_id> --latest

    Reads the run's input_snapshot from the local DB, finds the matching
    @agent-decorated function in the agent file, and re-invokes it. When
    the snapshot is a dict whose keys bind to the agent's parameters
    (the format the @agent decorator captures), the dict is unpacked as
    kwargs. Otherwise the snapshot is passed as a single positional
    argument — back-compat for runs whose snapshot was hand-populated.

    Version gate (ADR-0002 #7): the run's captured agent_version is
    compared to the registration's current value. A mismatch aborts the
    replay unless --latest is passed; pre-#7 runs (NULL agent_version)
    replay without the gate.

    On any outcome (success or a new failure), the original run is marked
    with disposition='replayed'. If the replay also fails it shows up as a
    fresh dead letter, so the operator can see the pattern.
    """
    from papayya.durable import ReplayError
    from papayya.durable.client import replay as _sdk_replay

    # Click hands us a string from --from-step. Coerce to int when it
    # parses as a positive number — that's the "1-indexed step number"
    # form. Anything else stays a string and resolves as a label. None
    # passes through to mean "replay from the top" (Phase 1 behaviour).
    parsed_from_step: str | int | None = None
    if from_step is not None:
        try:
            parsed_from_step = int(from_step)
        except ValueError:
            parsed_from_step = from_step

    click.echo(f"Replaying run {run_id}...")
    try:
        result = _sdk_replay(
            run_id,
            agent_module=file,
            db_path=db,
            latest=latest,
            from_step=parsed_from_step,
        )
    except ReplayError as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)
    except Exception as exc:  # noqa: BLE001
        click.echo(f"Replay failed: {exc}", err=True)
        sys.exit(2)
    click.echo(f"Replay returned: {result!r}")


# ---------------------------------------------------------------------------
# runs — hosted run ops (list, stream)
#
# Plural is deliberate. Top-level `papayya run` (workflow: create+wait+tail)
# stays unchanged. `runs` is the resource group for read verbs that mirror
# client.runs.* one-for-one. v1→v2 cutover: cancel and replay-from-step
# retired with the v1 DROP (their durable endpoints are gone).
# ---------------------------------------------------------------------------

@main.group()
def runs() -> None:
    """Operate on hosted runs (list, stream)."""


@runs.command("list")
@click.pass_context
def runs_list(ctx: click.Context) -> None:
    """List hosted runs (NDJSON, one run per line)."""
    client = _make_papayya_client(ctx)
    try:
        items = client.runs.list()
    except PapayyaAPIError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    finally:
        client.close()

    for run in items:
        click.echo(json.dumps(run))


@runs.command("stream")
@click.argument("run_id")
@click.option(
    "--from-step",
    type=int,
    default=None,
    help="Resume from this step (use the highest step number already observed)",
)
@click.pass_context
def runs_stream(ctx: click.Context, run_id: str, from_step: int | None) -> None:
    """Tail steps for a hosted run via SSE (NDJSON output).

    Yields one JSON object per server-sent event: ``{"event": "step" |
    "terminal" | "error", "data": {...}, "id": <step_number>}``. The
    stream exits when the run reaches a terminal status.
    """
    client = _make_papayya_client(ctx)
    try:
        for event in client.runs.stream(run_id, from_step=from_step):
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
@click.option("--max-steps", type=int, default=None, help="Per-run max steps override")
@click.option("--budget", type=float, default=None, help="Per-run budget cap in whole dollars")
@click.pass_context
def schedules_create(
    ctx: click.Context,
    agent_id: str,
    cron: str,
    timezone: str | None,
    input_str: str | None,
    max_steps: int | None,
    budget: float | None,
) -> None:
    """Create a cron schedule on an agent."""
    client = _make_papayya_client(ctx)
    try:
        sched = client.schedules.create(
            agent_id,
            cron,
            timezone=timezone,
            input=input_str,
            max_steps=max_steps,
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
@click.option("--max-steps", type=int, default=None, help="New per-run max-steps")
@click.option("--budget", type=float, default=None, help="New per-run budget cap (whole dollars)")
@click.pass_context
def schedules_update(
    ctx: click.Context,
    schedule_id: str,
    cron: str | None,
    timezone: str | None,
    input_str: str | None,
    max_steps: int | None,
    budget: float | None,
) -> None:
    """Patch fields on an existing schedule (only fields you pass are sent)."""
    patch: dict[str, Any] = {}
    if cron is not None:
        patch["cron"] = cron
    if timezone is not None:
        patch["timezone"] = timezone
    if input_str is not None:
        patch["input"] = input_str
    if max_steps is not None:
        patch["max_steps"] = max_steps
    if budget is not None:
        patch["budget_cents"] = _dollars_to_cents(budget)

    if not patch:
        click.echo("Error: pass at least one of --cron / --timezone / --input / --max-steps / --budget", err=True)
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
# webhooks — outgoing webhooks per agent
# ---------------------------------------------------------------------------

@main.group()
def webhooks() -> None:
    """Manage outgoing webhooks (create, list, delete)."""


@webhooks.command("create")
@click.option("--agent", "agent_id", required=True, help="Agent the webhook belongs to")
@click.option("--name", required=True, help="Display name")
@click.option("--description", default=None, help="Optional description")
@click.pass_context
def webhooks_create(
    ctx: click.Context, agent_id: str, name: str, description: str | None
) -> None:
    """Create a webhook on an agent. Returns the webhook with its signing secret."""
    client = _make_papayya_client(ctx)
    try:
        hook = client.webhooks.create(agent_id, name, description=description)
    except PapayyaAPIError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    finally:
        client.close()

    click.echo(json.dumps(hook, indent=2))


@webhooks.command("list")
@click.argument("agent_id")
@click.pass_context
def webhooks_list(ctx: click.Context, agent_id: str) -> None:
    """List webhooks on an agent (NDJSON)."""
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


@webhooks.command("delete")
@click.argument("webhook_id")
@click.pass_context
def webhooks_delete(ctx: click.Context, webhook_id: str) -> None:
    """Delete a webhook."""
    client = _make_papayya_client(ctx)
    try:
        client.webhooks.delete(webhook_id)
    except PapayyaAPIError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    finally:
        client.close()

    click.echo(f"Webhook {webhook_id} deleted")


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


@main.group()
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
    """Export local history (batches, runs, steps) to a JSONL file.

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

    conn = _sqlite.connect(db_path)
    conn.row_factory = _sqlite.Row

    out_path = _Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = {"batches": 0, "runs": 0, "steps": 0}

    with out_path.open("w", encoding="utf-8") as fh:
        for row in conn.execute("SELECT * FROM batches"):
            fh.write(_json.dumps({"type": "batch", "data": dict(row)}) + "\n")
            written["batches"] += 1

        for row in conn.execute("SELECT * FROM runs"):
            fh.write(_json.dumps({"type": "run", "data": dict(row)}) + "\n")
            written["runs"] += 1

        for row in conn.execute("SELECT * FROM steps"):
            data = dict(row)
            if not include_response_text:
                data.pop("response_text", None)
            fh.write(_json.dumps({"type": "step", "data": data}) + "\n")
            written["steps"] += 1

    conn.close()
    click.echo(
        f"Exported {written['batches']} batches, {written['runs']} runs, "
        f"{written['steps']} steps to {out_path}"
    )
    if not include_response_text:
        click.echo(
            "Note: LLM response text excluded by default. Re-run with "
            "--include-response-text to include it."
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
        if kind not in ("batch", "run", "step"):
            click.echo(f"Line {i}: unknown record type {kind!r}", err=True)
            raise click.exceptions.Exit(1)
        counts[kind] = counts.get(kind, 0) + 1

    click.echo("Validated import file:")
    plurals = {"batch": "batches", "run": "runs", "step": "steps"}
    for kind in ("batch", "run", "step"):
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
            input_data={"message": input_text},
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
            done = len(status_resp.get("checkpoints", []) or [])
            click.echo(f"  {done} step(s) — {state}")

            if state in ("completed", "failed", "cancelled", "budget_exceeded"):
                break

        # Show final result. Durable checkpoints carry {label, result};
        # there's no v1 step_number/step_type/output shape any more.
        click.echo(f"\nFinal status: {state}")
        checkpoints = api.get_steps(run_id)
        for c in checkpoints:
            label = c.get("label", "?")
            result = c.get("result")
            content = (
                result.get("content", "")
                if isinstance(result, dict)
                else "" if result is None else str(result)
            )
            click.echo(f"  {label}: {str(content)[:200]}")

    except PapayyaAPIError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    finally:
        api.close()


@main.command()
@click.argument("run_id")
@click.pass_context
def status(ctx: click.Context, run_id: str) -> None:
    """Check the status of a run."""
    scope = _env_scope(ctx.obj)
    resolved_key = _require_api_key(scope)
    config = APIConfig(api_key=resolved_key, base_url=scope.base_url)
    api = APIClient(config)

    try:
        result = api.get_run(run_id)
        click.echo(f"Run:    {result['id']}")
        click.echo(f"Status: {result['status']}")
        click.echo(f"Step:   {result.get('current_step', 0)}")
        click.echo(f"Cost:   {result.get('total_cost_cents', 0)} cents")
    finally:
        api.close()


@main.command()
@click.argument("run_id")
@click.pass_context
def logs(ctx: click.Context, run_id: str) -> None:
    """Show step-by-step logs for a run."""
    scope = _env_scope(ctx.obj)
    resolved_key = _require_api_key(scope)
    config = APIConfig(api_key=resolved_key, base_url=scope.base_url)
    api = APIClient(config)

    try:
        steps = api.get_steps(run_id)
        if not steps:
            click.echo("No steps found.")
            return

        for s in steps:
            step_num = s["step_number"]
            step_type = s["step_type"]
            status = s["status"]
            tokens_in = s.get("input_tokens", 0)
            tokens_out = s.get("output_tokens", 0)
            duration = s.get("duration_ms", 0)

            click.echo(f"Step {step_num} [{step_type}] — {status}")
            click.echo(f"  Tokens: {tokens_in} in / {tokens_out} out | {duration}ms")

            output = s.get("output", {})
            if isinstance(output, dict):
                content = output.get("content", "")
                if content:
                    click.echo(f"  Output: {content[:300]}")

                tool_calls = output.get("tool_calls", [])
                for tc in tool_calls:
                    click.echo(f"  Tool: {tc.get('name', '?')}({json.dumps(tc.get('input', {}))})")

            click.echo()
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


@main.group()
def batch() -> None:
    """Submit and manage batches of runs."""


@batch.command("submit")
@click.option("--agent", "agent_id", required=True, help="Agent ID to run each item against")
@click.option("--file", "file_path", required=True, type=click.Path(exists=False), help="JSONL file — one item per line, e.g. {\"input\": ..., \"metadata\"?: ...}")
@click.option("--budget", "budget_dollars", type=float, default=None, help="Total batch budget in whole dollars (converted to cents)")
@click.option("--concurrency", "concurrency_cap", type=int, default=None, help="Max concurrent runs the dispatcher will launch")
@click.option("--name", "name", default=None, help="Human-readable batch label")
@click.option("--callback-url", "callback_url", default=None, help="Webhook URL invoked on terminal batch status")
@click.option("--idempotency-key", "idempotency_key", default=None, help="Client-supplied key to dedupe duplicate submissions")
@click.pass_context
def batch_submit(
    ctx: click.Context,
    agent_id: str,
    file_path: str,
    budget_dollars: float | None,
    concurrency_cap: int | None,
    name: str | None,
    callback_url: str | None,
    idempotency_key: str | None,
) -> None:
    """Submit a batch of runs from a JSONL file.

    Always uses the NDJSON streaming path — no item ceiling, only a 1 GiB
    byte guard enforced by the backend. Prints the batch ID on success.
    """
    budget_cents_cap = int(round(budget_dollars * 100)) if budget_dollars is not None else None

    client = _make_papayya_client(ctx)
    try:
        result = client.batches.create_stream(
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

    click.echo(f"Batch submitted: {result.get('id', '?')}")
    status_val = result.get("status")
    if status_val:
        click.echo(f"  Status: {status_val}")
    total = result.get("total_items")
    if total is not None:
        click.echo(f"  Items:  {total}")

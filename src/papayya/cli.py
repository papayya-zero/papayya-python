"""Papayya CLI."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterator

import click

from papayya._defaults import DEFAULT_BASE_URL
from papayya.api import APIClient, APIConfig, PapayyaAPIError, resolve_config


# ---------------------------------------------------------------------------
# Config persistence (~/.papayya/config.json)
# ---------------------------------------------------------------------------

_CONFIG_DIR = Path.home() / ".papayya"
_CONFIG_FILE = _CONFIG_DIR / "config.json"


def _load_cli_config() -> dict[str, Any]:
    """Load persisted CLI config (API key, base_url, project_id, etc.)."""
    try:
        return json.loads(_CONFIG_FILE.read_text())
    except Exception:
        return {}


def _save_cli_config(data: dict[str, Any]) -> None:
    _CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    _CONFIG_FILE.write_text(json.dumps(data, indent=2) + "\n")


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
    """Resolve project ID from flag, env, or saved config."""
    pid = os.environ.get("PAPAYYA_PROJECT_ID")
    if pid:
        return pid
    cfg = _load_cli_config()
    return cfg.get("project_id")


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

    result = api.create_agent(
        project_id=project_id,
        name=reg.name,
        slug=slug,
        config=config,
    )
    click.echo(f"  Created agent: {result['id']} ({slug})")
    return result["id"]


def _resolve_api_key(ctx_key: str | None) -> str | None:
    """Resolve API key from CLI flag, env, or saved config."""
    key = ctx_key or os.environ.get("PAPAYYA_API_KEY")
    if key:
        return key
    # Fall back to saved config
    cfg = _load_cli_config()
    return cfg.get("api_key")


@click.group()
@click.option("--api-key", envvar="PAPAYYA_API_KEY", help="API key")
@click.option("--base-url", envvar="PAPAYYA_BASE_URL", default=DEFAULT_BASE_URL, help="Control plane URL")
@click.pass_context
def main(ctx: click.Context, api_key: str | None, base_url: str) -> None:
    """Papayya — durable background jobs for AI agents."""
    ctx.ensure_object(dict)
    ctx.obj["api_key"] = api_key
    ctx.obj["base_url"] = base_url


# ---------------------------------------------------------------------------
# init
# ---------------------------------------------------------------------------

_AGENT_PY_TEMPLATE = '''\
"""My agent — edit the instructions, tools, and LLM call to fit your use-case.

Papayya is a durable execution runtime — it does NOT call LLMs on your
behalf. You bring your own LLM SDK (openai, anthropic, bedrock, ...) and
call it directly inside your @agent function.
"""

import json
from openai import OpenAI
from papayya import agent


# --- Tools (replace with your own) ---

def greet(name: str) -> str:
    return f"Hello, {{name}}! How can I help you today?"


def lookup_weather(location: str) -> str:
    return f"72°F and sunny in {{location}}"


TOOLS = [
    {{
        "type": "function",
        "function": {{
            "name": "greet",
            "description": "Return a friendly greeting.",
            "parameters": {{
                "type": "object",
                "properties": {{"name": {{"type": "string"}}}},
                "required": ["name"],
            }},
        }},
    }},
    {{
        "type": "function",
        "function": {{
            "name": "lookup_weather",
            "description": "Look up current weather for a location.",
            "parameters": {{
                "type": "object",
                "properties": {{"location": {{"type": "string"}}}},
                "required": ["location"],
            }},
        }},
    }},
]

TOOL_FNS = {{
    "greet": lambda args: greet(args["name"]),
    "lookup_weather": lambda args: lookup_weather(args["location"]),
}}

SYSTEM = (
    "You are a helpful assistant. "
    "Use the greet tool when the user says hello. "
    "Use the lookup_weather tool when asked about weather."
)


@agent(name="{name}", model="gpt-4o-mini", instructions=SYSTEM, max_steps=10, budget_usd=1.00)
def {name_underscore}(input_data):
    """Agent loop — calls OpenAI with tools. Replace with your own logic."""
    client = OpenAI()
    prompt = input_data if isinstance(input_data, str) else json.dumps(input_data)
    messages = [
        {{"role": "system", "content": SYSTEM}},
        {{"role": "user", "content": prompt}},
    ]

    for step in range(10):
        response = client.chat.completions.create(
            model="gpt-4o-mini", messages=messages, tools=TOOLS,
        )
        choice = response.choices[0]

        if not choice.message.tool_calls:
            return choice.message.content

        messages.append(choice.message)
        for tc in choice.message.tool_calls:
            args = json.loads(tc.function.arguments)
            fn = TOOL_FNS.get(tc.function.name)
            result = fn(args) if fn else f"Unknown tool: {{tc.function.name}}"
            messages.append({{"role": "tool", "tool_call_id": tc.id, "content": result}})

    return "Max steps reached."


# Run locally: python agent.py
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    print({name_underscore}("Hello! What\\'s the weather in Toronto?"))
'''

_REQUIREMENTS_TEMPLATE = """\
papayya>=0.1.0
openai>=1.0.0
python-dotenv>=1.0.0
# Install whichever LLM SDK you actually use — papayya does not depend on any.
# anthropic>=0.40.0
"""

_ENV_EXAMPLE_TEMPLATE = """\
# Papayya configuration
# Copy this file to .env and fill in your keys.

# LLM provider key — set whichever provider your agent code uses.
# Papayya does not read this; your LLM SDK does.
# ANTHROPIC_API_KEY=sk-ant-...
# OPENAI_API_KEY=sk-...

# Papayya platform key (for cloud deploys / runs)
# Run `papayya signup` to get one automatically.
PAPAYYA_API_KEY=

# Override the default control plane URL if needed
# PAPAYYA_BASE_URL=http://localhost:8090
"""


@main.command()
@click.option("--name", default="my-agent", help="Agent name")
def init(name: str) -> None:
    """Scaffold a new agent project in the current directory."""
    cwd = Path.cwd()
    click.echo(f'Initializing agent project "{name}" in {cwd}')

    name_underscore = name.replace("-", "_").replace(" ", "_")
    files = {
        "agent.py": _AGENT_PY_TEMPLATE.format(name=name, name_underscore=name_underscore),
        "requirements.txt": _REQUIREMENTS_TEMPLATE,
        ".env.example": _ENV_EXAMPLE_TEMPLATE,
    }

    created = 0
    for filename, content in files.items():
        target = cwd / filename
        if target.exists():
            click.echo(f"  ⚠ {filename} already exists — skipping")
            continue
        target.write_text(content)
        click.echo(f"  ✓ Created {filename}")
        created += 1

    if created > 0:
        click.echo("\nNext steps:")
        click.echo("  1. pip install -r requirements.txt")
        click.echo("  2. Edit agent.py to define your agent and tools")
        click.echo("  3. papayya run --local --file agent.py --input 'Hello'")


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
    if existing.get("api_key") and not force:
        current_email = existing.get("email", "<unknown email>")
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

        # 5. Persist config
        _save_cli_config({
            "api_key": api_key,
            "base_url": base_url,
            "project_id": project_id,
            "email": email,
        })
        click.echo(f"\n✓ All set! Config saved to {_CONFIG_FILE}")
        click.echo(f"  API key: {api_key[:12]}...")
        click.echo(f"  Project: {project_id}")
        click.echo("\nNext: papayya init --name my-agent")

    except PapayyaAPIError as e:
        if e.status == 409:
            click.echo("Error: An account with that email already exists. Try `papayya login`.", err=True)
        else:
            click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
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

        _save_cli_config({
            "api_key": api_key,
            "base_url": base_url,
            "project_id": project_id,
            "email": email,
        })
        click.echo(f"✓ Logged in! Config saved to {_CONFIG_FILE}")
        click.echo(f"  Project: {project_id}")

    except PapayyaAPIError as e:
        if e.status == 401:
            click.echo("Error: Invalid email or password.", err=True)
        else:
            click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    finally:
        api.close()


@main.command()
def logout() -> None:
    """Remove the saved CLI config (~/.papayya/config.json)."""
    if not _CONFIG_FILE.exists():
        click.echo("Not signed in — no config to remove.")
        return
    existing = _load_cli_config()
    email = existing.get("email", "<unknown>")
    _CONFIG_FILE.unlink()
    click.echo(f"✓ Logged out ({email}). Removed {_CONFIG_FILE}.")


# ---------------------------------------------------------------------------
# deploy
# ---------------------------------------------------------------------------

@main.command()
@click.argument("file", required=False, default=None)
@click.option("--agent-id", default=None, help="Agent ID (overrides auto-discovery)")
@click.option("--project-id", default=None, envvar="PAPAYYA_PROJECT_ID", help="Project ID")
@click.option("--runtime", default="python", type=click.Choice(["python", "node"]), help="Runtime type")
@click.option("--entrypoint", default=None, help="Entrypoint file (default: auto-detected)")
@click.pass_context
def deploy(ctx: click.Context, file: str | None, agent_id: str | None, project_id: str | None, runtime: str, entrypoint: str | None) -> None:
    """Deploy agent code to the control plane.

    \b
    Usage:
      papayya deploy              # auto-discover agent.py in cwd
      papayya deploy agents.py    # explicit file
    """
    from papayya.bundler import bundle_project

    # Auto-discover file
    if file is None:
        if Path("agent.py").exists():
            file = "agent.py"
        else:
            click.echo("Error: No agent.py found in current directory. Specify a file:\n  papayya deploy my_agents.py", err=True)
            sys.exit(1)

    # Resolve auth
    resolved_key = _resolve_api_key(ctx.obj["api_key"])
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
            project_id = _resolve_project_id(ctx.obj)
        if not project_id and not agent_id:
            click.echo("Error: No project ID. Set PAPAYYA_PROJECT_ID or run `papayya signup`.", err=True)
            sys.exit(1)

        # Deploy each agent
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

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    finally:
        api.close()


@main.command()
@click.option("--port", default=8585, help="Port for the dev dashboard")
@click.option("--host", default="127.0.0.1", help="Host to bind to")
@click.option("--db", default=".papayya/local.db", help="Path to SQLite database")
def dev(port: int, host: str, db: str) -> None:
    """Launch the local development dashboard."""
    click.echo(f"Starting Papayya Dev Dashboard...")
    from papayya.dev.server import serve
    serve(host=host, port=port, db_path=db)


@main.group()
def dlq() -> None:
    """Dead letter queue — triage failed runs from a batch."""


@dlq.command("replay")
@click.option("--run", "run_id", required=True, help="Run ID to replay")
@click.option("--file", "file", default=None,
              help="Agent file (default: auto-discover agent.py in cwd)")
@click.option("--db", default=".papayya/local.db", help="Path to SQLite database")
def dlq_replay(run_id: str, file: str | None, db: str) -> None:
    """Re-drive a failed run using its captured input snapshot.

    \b
    Usage:
      papayya dlq replay --run <run_id>
      papayya dlq replay --run <run_id> --file my_agents.py

    Reads the run's input_snapshot from the local DB, finds the matching
    @agent-decorated function in the agent file, and invokes it with the
    snapshot as its single positional argument. The agent is expected to
    take one argument — the same payload it received the first time.

    On any outcome (success or a new failure), the original run is marked
    with disposition='replayed'. If the replay also fails it shows up as a
    fresh dead letter, so the operator can see the pattern.
    """
    import sqlite3 as _sqlite
    import json as _json

    from papayya.durable import _schema

    # Locate DB
    db_path = Path(db)
    if not db_path.exists():
        click.echo(f"Error: No local database at {db_path.resolve()}", err=True)
        sys.exit(1)

    # Load run
    conn = _sqlite.connect(str(db_path))
    conn.row_factory = _sqlite.Row
    try:
        row = conn.execute(
            f"""SELECT run_id, agent, status,
                       {_schema.COL_RUN_DLQ_DISPOSITION} AS disp,
                       {_schema.COL_RUN_INPUT_SNAPSHOT} AS input_snapshot
                FROM {_schema.TBL_RUNS} WHERE run_id = ?""",
            (run_id,),
        ).fetchone()
    finally:
        conn.close()

    if row is None:
        click.echo(f"Error: Run '{run_id}' not found in {db_path}", err=True)
        sys.exit(1)
    if row["status"] != "failed":
        click.echo(
            f"Error: Run '{run_id}' has status {row['status']!r}, not 'failed'. "
            "Only failed runs can be replayed.", err=True,
        )
        sys.exit(1)
    if row["disp"] is not None:
        click.echo(
            f"Error: Run '{run_id}' is already resolved (disposition={row['disp']!r}).",
            err=True,
        )
        sys.exit(1)

    # Decode snapshot
    raw = row["input_snapshot"]
    if raw is None:
        click.echo(
            f"Error: Run '{run_id}' has no input_snapshot — cannot replay.\n"
            "Input must be captured at run creation time; older runs predate\n"
            "this feature and are not replayable.", err=True,
        )
        sys.exit(1)
    try:
        input_snapshot = _json.loads(raw)
    except (TypeError, ValueError):
        input_snapshot = raw

    agent_name = row["agent"]

    # Discover agent file
    if file is None:
        if Path("agent.py").exists():
            file = "agent.py"
        else:
            click.echo(
                "Error: No agent.py in cwd. Pass --file to point at the agent module.",
                err=True,
            )
            sys.exit(1)

    # Find matching registration
    registrations = _discover_agents(file)
    matching = next((r for r in registrations if r.name == agent_name), None)
    if matching is None:
        names = ", ".join(r.name for r in registrations) or "(none)"
        click.echo(
            f"Error: No @agent with name {agent_name!r} found in {file}.\n"
            f"Registered agents: {names}", err=True,
        )
        sys.exit(1)

    # Invoke
    click.echo(f"Replaying run {run_id} through agent {agent_name}...")
    replay_error: Exception | None = None
    try:
        result = matching.fn(input_snapshot)
        click.echo(f"Replay returned: {result!r}")
    except Exception as exc:  # noqa: BLE001
        replay_error = exc
        click.echo(f"Replay failed: {exc}", err=True)

    # Mark old run as replayed — attempted, regardless of outcome. A
    # failed replay becomes a fresh dead letter (the agent's own
    # `run.fail(...)` path creates it).
    from papayya.durable.sqlite_store import SQLiteStore
    store = SQLiteStore(str(db_path))
    try:
        store.mark_dlq_disposition(run_id, _schema.DLQ_REPLAYED)
    finally:
        store.close()

    if replay_error is not None:
        sys.exit(2)


@main.group()
def project() -> None:
    """Manage local project history (export, import)."""


@project.command("export")
@click.option("--out", required=True, help="Output JSONL file path")
@click.option("--db",  default=".papayya/local.db", help="Path to SQLite database")
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


@main.command()
@click.option("--file", required=True, help="Path to agent definition file")
@click.option("--input", "input_text", required=True, help="Input for the agent")
@click.option("--local", "use_local", is_flag=True, default=False, help="Run locally (no cloud needed)")
@click.option("--agent-id", default=None, help="Agent ID (required for cloud runs)")
@click.option("--name", "agent_name", default=None, help="Agent name (required when file declares multiple @agent functions)")
@click.option("--api-key", "run_api_key", default=None, help="LLM API key for local runs")
@click.pass_context
def run(ctx: click.Context, file: str, input_text: str, use_local: bool, agent_id: str | None, agent_name: str | None, run_api_key: str | None) -> None:
    """Run an agent locally or in the cloud."""
    if use_local:
        _run_local(None, input_text, run_api_key)
        return

    registrations = _discover_agents(file)
    if len(registrations) == 1:
        reg = registrations[0]
    else:
        if not agent_name:
            names = ", ".join(r.name for r in registrations)
            click.echo(
                f"Error: {file} declares {len(registrations)} agents ({names}).\n"
                "  Pass --name <agent-name> to pick one.",
                err=True,
            )
            sys.exit(1)
        matches = [r for r in registrations if r.name == agent_name]
        if not matches:
            names = ", ".join(r.name for r in registrations)
            click.echo(
                f"Error: no @agent named '{agent_name}' in {file}. Available: {names}",
                err=True,
            )
            sys.exit(1)
        reg = matches[0]

    _run_cloud(ctx, reg, file, input_text, agent_id)


def _run_local(agent: Any, input_text: str, api_key_override: str | None) -> None:
    """Local execution is BYOF — run your agent file directly.

    Papayya does not ship LLM provider adapters, so the CLI cannot run an
    agent on your behalf. Execute your agent file directly with Python
    instead (``python agent.py``) — your code owns the LLM call, papayya
    owns durable checkpointing via ``run.task(...)``.
    """
    del agent, input_text, api_key_override
    click.echo(
        "Local execution via `papayya run --local` is no longer supported.\n"
        "\n"
        "Papayya does not ship LLM provider adapters — your code calls the\n"
        "LLM directly and wraps it with `papayya.durable.run.task()` for\n"
        "durable execution. To run your agent locally, execute the file\n"
        "directly:\n"
        "\n"
        "    python agent.py\n"
        "\n"
        "To run in the cloud runtime, deploy and use `papayya run` without\n"
        "the --local flag. See docs/pages/sdk/byof for examples.",
        err=True,
    )
    sys.exit(1)


def _run_cloud(ctx: click.Context, reg: Any, file: str, input_text: str, agent_id: str | None) -> None:
    """Trigger a cloud run.

    ``reg`` is an ``AgentRegistration`` produced by ``_discover_agents``.
    """
    if not agent_id:
        click.echo(
            "Error: --agent-id is required for cloud runs.\n"
            "  Deploy first: papayya deploy --file agent.py --agent-id <id>",
            err=True,
        )
        sys.exit(1)

    resolved_key = _resolve_api_key(ctx.obj["api_key"])
    if not resolved_key:
        click.echo(
            "Error: No API key found.\n"
            "  Run `papayya signup` first, or set PAPAYYA_API_KEY.",
            err=True,
        )
        sys.exit(1)

    budget_cents = int(reg.budget_usd * 100) if reg.budget_usd is not None else 500

    config = APIConfig(api_key=resolved_key, base_url=ctx.obj["base_url"])
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

        # Poll until complete
        click.echo("Waiting for completion...")
        while True:
            time.sleep(2)
            status_resp = api.get_run(run_id)
            state = status_resp.get("status", "unknown")
            step = status_resp.get("current_step", 0)
            click.echo(f"  Step {step} — {state}")

            if state in ("completed", "failed", "cancelled", "budget_exceeded"):
                break

        # Show final result
        click.echo(f"\nFinal status: {state}")
        steps = api.get_steps(run_id)
        for s in steps:
            output = s.get("output", {})
            content = output.get("content", "") if isinstance(output, dict) else str(output)
            click.echo(f"  Step {s['step_number']} [{s['step_type']}]: {content[:200]}")

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
    resolved_key = _resolve_api_key(ctx.obj["api_key"])
    if not resolved_key:
        click.echo("Error: No API key. Run `papayya signup` first, or set PAPAYYA_API_KEY.", err=True)
        sys.exit(1)
    config = APIConfig(api_key=resolved_key, base_url=ctx.obj["base_url"])
    api = APIClient(config)

    try:
        result = api.get_run(run_id)
        click.echo(f"Run:    {result['id']}")
        click.echo(f"Status: {result['status']}")
        click.echo(f"Step:   {result.get('current_step', 0)}")
        click.echo(f"Cost:   {result.get('total_cost_cents', 0)} cents")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    finally:
        api.close()


@main.command()
@click.argument("run_id")
@click.pass_context
def logs(ctx: click.Context, run_id: str) -> None:
    """Show step-by-step logs for a run."""
    resolved_key = _resolve_api_key(ctx.obj["api_key"])
    if not resolved_key:
        click.echo("Error: No API key. Run `papayya signup` first, or set PAPAYYA_API_KEY.", err=True)
        sys.exit(1)
    config = APIConfig(api_key=resolved_key, base_url=ctx.obj["base_url"])
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
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    finally:
        api.close()


@main.group()
@click.pass_context
def secrets(ctx: click.Context) -> None:
    """Manage project secrets."""
    pass


@secrets.command("set")
@click.argument("name")
@click.argument("value")
@click.option("--project-id", required=False, envvar="PAPAYYA_PROJECT_ID", default=None, help="Project ID")
@click.pass_context
def secrets_set(ctx: click.Context, name: str, value: str, project_id: str | None) -> None:
    """Set a secret for a project."""
    resolved_key = _resolve_api_key(ctx.obj["api_key"])
    if not resolved_key:
        click.echo("Error: No API key. Run `papayya signup` first, or set PAPAYYA_API_KEY.", err=True)
        sys.exit(1)
    if not project_id:
        project_id = _load_cli_config().get("project_id")
    if not project_id:
        click.echo("Error: --project-id required (or run `papayya signup` to save one).", err=True)
        sys.exit(1)
    config = APIConfig(api_key=resolved_key, base_url=ctx.obj["base_url"])
    api = APIClient(config)

    try:
        api.set_secret(project_id, name, value)
        click.echo(f"Secret '{name}' set successfully.")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    finally:
        api.close()


@secrets.command("list")
@click.option("--project-id", required=False, envvar="PAPAYYA_PROJECT_ID", default=None, help="Project ID")
@click.pass_context
def secrets_list(ctx: click.Context, project_id: str | None) -> None:
    """List secrets for a project (names only)."""
    resolved_key = _resolve_api_key(ctx.obj["api_key"])
    if not resolved_key:
        click.echo("Error: No API key. Run `papayya signup` first, or set PAPAYYA_API_KEY.", err=True)
        sys.exit(1)
    if not project_id:
        project_id = _load_cli_config().get("project_id")
    if not project_id:
        click.echo("Error: --project-id required (or run `papayya signup` to save one).", err=True)
        sys.exit(1)
    config = APIConfig(api_key=resolved_key, base_url=ctx.obj["base_url"])
    api = APIClient(config)

    try:
        result = api.list_secrets(project_id)
        if not result:
            click.echo("No secrets found.")
            return
        for s in result:
            click.echo(f"  {s['name']}  (updated: {s.get('updated_at', '?')})")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    finally:
        api.close()


@secrets.command("delete")
@click.argument("name")
@click.option("--project-id", required=False, envvar="PAPAYYA_PROJECT_ID", default=None, help="Project ID")
@click.pass_context
def secrets_delete(ctx: click.Context, name: str, project_id: str | None) -> None:
    """Delete a secret."""
    resolved_key = _resolve_api_key(ctx.obj["api_key"])
    if not resolved_key:
        click.echo("Error: No API key. Run `papayya signup` first, or set PAPAYYA_API_KEY.", err=True)
        sys.exit(1)
    if not project_id:
        project_id = _load_cli_config().get("project_id")
    if not project_id:
        click.echo("Error: --project-id required (or run `papayya signup` to save one).", err=True)
        sys.exit(1)
    config = APIConfig(api_key=resolved_key, base_url=ctx.obj["base_url"])
    api = APIClient(config)

    try:
        api.delete_secret(project_id, name)
        click.echo(f"Secret '{name}' deleted.")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
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
    resolved_key = _resolve_api_key(ctx.obj["api_key"])
    if not resolved_key:
        click.echo("Error: No API key. Run `papayya signup` first, or set PAPAYYA_API_KEY.", err=True)
        sys.exit(1)
    project_id = _resolve_project_id(ctx.obj)
    if not project_id:
        click.echo("Error: --project-id required (or run `papayya signup` to save one).", err=True)
        sys.exit(1)
    api = APIClient(APIConfig(api_key=resolved_key, base_url=ctx.obj["base_url"]))
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


@main.command()
@click.option("--file", required=True, help="Path to agent definition file")
@click.option("--poll-interval", default=2.0, help="Poll interval in seconds")
@click.pass_context
def worker(ctx: click.Context, file: str, poll_interval: float) -> None:
    """Run a tool worker that executes tool calls for cloud runs."""
    from papayya.worker import run_worker

    agent = _load_agent_from_file(file)
    resolved_key = _resolve_api_key(ctx.obj["api_key"])
    if not resolved_key:
        click.echo("Error: No API key. Run `papayya signup` first, or set PAPAYYA_API_KEY.", err=True)
        sys.exit(1)
    config = APIConfig(api_key=resolved_key, base_url=ctx.obj["base_url"])
    api = APIClient(config)

    try:
        run_worker(agent, api, poll_interval=poll_interval)
    finally:
        api.close()


# ---------------------------------------------------------------------------
# batch — submit / inspect / cancel / retry batches
# ---------------------------------------------------------------------------

def _make_papayya_client(ctx: click.Context) -> Any:
    """Resolve auth and return a Papayya client, exiting with a friendly
    error if no API key is configured. Callers are responsible for
    ``client.close()`` in a finally block."""
    from papayya import Papayya

    resolved_key = _resolve_api_key(ctx.obj["api_key"])
    if not resolved_key:
        click.echo(
            "Error: No API key. Run `papayya signup` first, or set PAPAYYA_API_KEY.",
            err=True,
        )
        sys.exit(1)
    return Papayya(api_key=resolved_key, base_url=ctx.obj["base_url"])


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


@batch.command("status")
@click.argument("batch_id")
@click.pass_context
def batch_status(ctx: click.Context, batch_id: str) -> None:
    """Show aggregate status of a batch."""
    client = _make_papayya_client(ctx)
    try:
        b = client.batches.get(batch_id)
    except PapayyaAPIError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    finally:
        client.close()

    click.echo(f"Batch:     {b.get('id', batch_id)}")
    if b.get("name"):
        click.echo(f"Name:      {b['name']}")
    click.echo(f"Status:    {b.get('status', '?')}")
    click.echo(f"Agent:     {b.get('agent_id', '?')}")

    total = b.get("total_items", 0) or 0
    completed = b.get("completed", 0) or 0
    failed = b.get("failed", 0) or 0
    paused = b.get("paused", 0) or 0
    click.echo(f"Items:     {completed}/{total} completed, {failed} failed, {paused} paused")

    cost = b.get("aggregate_cost_cents", 0) or 0
    cap = b.get("budget_cents_cap")
    if cap:
        click.echo(f"Cost:      {cost}¢ / {cap}¢")
    else:
        click.echo(f"Cost:      {cost}¢ (no cap)")

    if b.get("created_at"):
        click.echo(f"Created:   {b['created_at']}")


@batch.command("results")
@click.argument("batch_id")
@click.option("-o", "--output", "output_path", default=None, help="Output JSONL path (default: stdout)")
@click.option("--include-failed", is_flag=True, default=False, help="Also stream failed/cancelled/budget_exceeded runs")
@click.option("--poll-interval", type=float, default=2.0, help="Polling cadence in seconds")
@click.pass_context
def batch_results(
    ctx: click.Context,
    batch_id: str,
    output_path: str | None,
    include_failed: bool,
    poll_interval: float,
) -> None:
    """Stream completed child runs of a batch as JSON lines.

    Blocks until the batch reaches a terminal status. Writes one JSON
    object per line, containing the raw run record as returned by
    ``GET /v1/batches/{id}/runs``.
    """
    client = _make_papayya_client(ctx)

    # stdout or file, opened the same way so the write loop is identical.
    if output_path is None:
        sink = sys.stdout
        close_sink = False
    else:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        sink = out.open("w", encoding="utf-8")
        close_sink = True

    count = 0
    try:
        for run in client.batches.stream_results(
            batch_id,
            poll_interval=poll_interval,
            include_failed=include_failed,
        ):
            sink.write(json.dumps(run) + "\n")
            sink.flush()
            count += 1
    except PapayyaAPIError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    finally:
        if close_sink:
            sink.close()
        client.close()

    if output_path is not None:
        click.echo(f"Wrote {count} run(s) to {output_path}")


@batch.command("cancel")
@click.argument("batch_id")
@click.pass_context
def batch_cancel(ctx: click.Context, batch_id: str) -> None:
    """Cancel a batch. Fan-out happens server-side; this returns 202."""
    client = _make_papayya_client(ctx)
    try:
        b = client.batches.cancel(batch_id)
    except PapayyaAPIError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    finally:
        client.close()

    click.echo(f"Batch {b.get('id', batch_id)} cancellation accepted (status: {b.get('status', '?')})")


@batch.command("retry")
@click.argument("batch_id")
@click.option("--failed", "retry_failed_flag", is_flag=True, required=True, help="Re-enqueue every failed child (currently the only retry mode)")
@click.pass_context
def batch_retry(ctx: click.Context, batch_id: str, retry_failed_flag: bool) -> None:
    """Re-enqueue failed children of a batch as new runs."""
    del retry_failed_flag  # --failed is required; no other mode exists yet

    client = _make_papayya_client(ctx)
    try:
        b = client.batches.retry_failed(batch_id)
    except PapayyaAPIError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    finally:
        client.close()

    click.echo(f"Batch {b.get('id', batch_id)} re-enqueued (total_items now {b.get('total_items', '?')})")

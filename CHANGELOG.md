# Changelog

All notable changes to the `papayya` Python package.

## 0.2.0 — 2026-04-29

The launch release. Worker-pool runtime, declarative config, dead letter queue, BYOF observability.

### Added

#### Declarative config + multi-env (`papayya.yaml`)
- New `papayya.yaml` describes schedules, webhooks, and per-env project mapping. A single `papayya deploy` uploads the agent bundle and reconciles triggers against the selected env. Replaces the imperative UUID-copy dance that was the day-1 friction point for trigger setup.
- `papayya envs list / use / create / link` subcommands. `create` provisions a project + API key via the stored JWT; `link` attaches an existing project.
- `--env` flag accepted in either position on every command (e.g. `papayya --env staging run …` and `papayya run --env staging …` both work).
- `papayya init` now writes a minimal `papayya.yaml` (`version: 1`); no more agent.py/.env scaffold dropped into your repo.

#### Worker-pool runtime + local dispatcher
- `python -m papayya.runtime` — long-running worker process that pulls items from a dispatcher and runs your `@agent` against each one. The foundation for the upcoming hosted runtime pivot.
- `python -m papayya.runtime.dispatcher --port 8765 --enqueue agent:item1,item2` — local dispatcher with `POST /enqueue` and `GET /stats` so you can curl items in and feel the iteration loop without any cloud setup.
- Production-hardening already in this release: lease TTL + worker heartbeats, graceful SIGTERM drain, idempotent `/complete` with retry, per-item soft timeout watchdog, exponential backoff on dispatcher unreachability, version-tagged lineage, lineage write retries with local journal.
- `PAPAYYA_LOCAL_DB_PATH` honored by `papayya dev` so you can point the dashboard at a worker's SQLite without flag-juggling.
- `papayya example` scaffolds `local_demo_agent.py` — a keyless two-step durable run you can execute immediately. Use `--print` to pipe to stdout.

#### Dead letter queue
- `papayya dlq list / replay / skip / acknowledge` for triaging failed runs. Idempotent dispositions, guarded by `status='failed'`.
- `client.runs.list_dlq(batch_id)`, `replay()`, `skip()`, `acknowledge()` SDK methods.
- Batch status is now ternary: `completed` / `partial` / `failed`. A batch with any failures is `partial` until every dead letter is resolved, then promotes to `completed`.
- Failed-run input is captured into `runs.input_snapshot` so replays don't need the original trigger payload.

#### BYOF observability
- `run.step("…", fn, kind="llm")` records that a step is an LLM call without wrapping the SDK or shipping a pricing table. Tokens, duration, model, and cost render alongside the step in `papayya dev`.
- Provider-error classification exported from the SDK: `from papayya import CreditExhausted, BudgetExceeded, is_credit_exhaustion_error, classify_provider_error`. Used internally by `run.step(kind="llm")` to pause runs on credit exhaustion vs. transient errors.

#### Live event stream (SSE)
- `client.runs.stream(run_id)` opens an SSE connection to `/v1/runs/{id}/events` and yields step-level events as they happen. Server-side ships in this release; dashboard integration is next.

#### Rate cards
- `papayya rate-card show / set / remove / import / edit` lets you provide your own per-model pricing so the dashboard can render `≈$` estimates next to token counts. No built-in pricing table — your numbers, your authority.

#### CLI polish
- `papayya --version` (was missing in 0.1.1).
- `papayya dev` prints a clear actionable error when its port is already in use.
- `papayya signup` refuses to clobber an existing config; `--force` to override.
- `papayya run <slug>` resolves agent slugs against your project; legacy file paths still work.
- All API errors render through a single SafeGroup handler — consistent formatting, no more raw stack traces on auth failures.

### Fixed

- `papayya run` no longer crashes when invoked against an `@agent`-decorated file (was returning the decorator symbol instead of the wrapped function).
- DLQ replay unpacks `dict` input snapshots as kwargs so agents with multi-arg signatures replay correctly.
- Hosted CloudStore and the dashboard now use the canonical serialization helper for non-JSON values, matching the local SQLite path.
- README durable-execution snippet was missing the import — fixed.

### Changed

- `~/.papayya/config.json` migrated from flat v1 to multi-env v2 (`envs.<name>` + `current_env` + top-level `auth.jwt`). Migration is transparent on load with a one-time notice. If you have tooling that reads this file directly, update it for the new shape.
- `papayya init` no longer writes `agent.py`, `requirements.txt`, or `.env.example`. Use `papayya example` for a runnable demo or `papayya init --help` to see what it now does.

### Dependencies

- Added `pydantic>=2,<3` and `pyyaml>=6,<7` (for the new declarative config).
- Continues to require `httpx>=0.27,<1` and `click>=8,<9`.

---

## 0.1.1

- Batches SDK resource (`client.batches.create / get / list / cancel`) and `papayya batch` CLI subcommand group.
- Local budget enforcement removed — `budget_usd` is now metadata used by the cloud runtime only. Local `PapayyaRun` is durability-only.
- URL metadata refresh: `getpapayya.com` domain, `papayya-python` repo rename.
- `.env` added to the init scaffold's `.gitignore`.

## 0.1.0

- Initial release. `@agent` decorator, durable `papayya().run().step()`, `papayya init / deploy / dev / status / signup / login`, local SQLite checkpointing, cloud control plane integration.

# Changelog

All notable changes to the `papayya` Python package.

## Unreleased (0.3.0)

Plan 34 noun consolidation — one vocabulary everywhere:

> An **agent** is loaded once by the worker pool; each **run** processes
> **items**; every item shows what it did and what it cost, and you can
> **replay** the ones that didn't work.

The shift-by-one: what the SDK used to call a *batch* is now a **run**
(one invocation — one `map()` call, one cron fire); what it used to call
a *run* is now an **item** (one record processed); a *task* is a **step**
(one node in an item's trace).

### BREAKING

These names persist with NEW semantics — they could not be aliased:

- `Papayya().runs` was the per-item resource; it is now the invocation
  surface (the old `.batches` submission methods). Per-item access moved
  to `Papayya().items` (same methods, same frozen HTTP endpoints).
- `papayya.Client` (deprecated in 0.2.x with a removal notice) is removed
  as a distinct class; the name now aliases `Papayya`. The v1 trigger
  methods (`run(agent_id, input)`, `run_sync`, `get_status`, `get_steps`)
  are gone — their endpoints retired with the v1 DROP. Use
  `Papayya().items.*` / `Papayya().runs.create(...)`.
- Local SQLite schema migrates v11 → v12: `batches`→`runs`,
  `runs`→`items` (PK `run_id`→`id`, FK `batch_id`→`run_id`),
  `tasks`→`steps` (customer `item_id`→`customer_item_id`, FK
  `run_id`→`item_id`); the dead legacy `steps` LLM-call log (no
  production writer since v2) is dropped. A one-time
  `local.db.backup-v11` sibling file is written before migrating.
- `papayya runs` CLI group: was hosted per-item verbs; `runs list` now
  lists LOCAL runs (invocations, with the outcome rollup) and
  `runs submit` is the old `batch submit`. Per-item verbs moved to
  `papayya items` (`list`, `get`, `stream`); `runs stream` is gone —
  same word could not carry both meanings.
- `papayya dev` item rows emit `run_id` with the NEW meaning (the
  invocation); the record uuid is `id` everywhere. The pre-0.3.0 dev API
  used `run_id` for the record uuid — one key could not carry both. The
  `/api/runs/<record>/steps` [] stub is removed with the page that
  fetched it. (`/api/batches*` aliases and old page paths survive one
  release; `/api/runs/<record>` falls back to per-record detail.)

### Added

- **Invocation minting**: `papayya.map()` / `papayya.iter()` mint ONE run
  row per call and link every processed item to it — a 1,000-item map()
  is one run of 1,000 items, not 1,000 runs-of-one. Direct calls keep the
  implicit run-of-one (`single-…`) wrapping.
- **Slice replay**: `papayya replay --run <run_id>` now replays a run's
  not-ok slice — the items whose `worst_outcome_status != 'ok'` — into a
  NEW run linked via `replayed_from` (run- and item-level). `--tenant`
  narrows the slice to one `partition_key`. SDK:
  `papayya.durable.replay_slice(run_id, tenant=…, handler=…/agent_module=…)`.
  Single-item replay stays available as `papayya replay --item <id>`
  (and `--run <old per-item id>` still falls back to it).
- `papayya().item(...)` returning `Item` — the rename of
  `papayya().run(...)` / `PapayyaRun`; old names kept as silent aliases.
  `Item.id` is the record's surrogate uuid (`item_id` stays reserved for
  customer identity).
- `papayya.active_item()` returns the active item handle;
  `active_run_id()` kept as a deprecated alias.
- `agent=` keyword on `map()`/`iter()`; `workload=` accepted as a silent
  alias for one release.
- OTel baggage/span attributes dual-emit `papayya.agent` alongside
  `papayya.workload` (old key drops one release after the control-pane
  reads the new one).

- **`papayya dev` speaks the vocabulary and shows the wedge** (Unit 3):
  nav is Agents → Runs → Items; the runs list renders
  `worst_outcome_status` and "N of M degraded" (+ degraded-tenant count
  and token totals) per run, so a degradation incident is visible
  without clicking anything. Run detail adds a per-tenant blast-radius
  table; items and steps carry worked/degraded/failed badges with
  outcome reasons. Two item keyspaces split into two pages: `/record`
  (one record in one run, by surrogate uuid) and `/item` (customer
  identity over time). Thrashing detection rebuilt on v12 step rows
  (same bare label + same input, occurrence-suffix aware); step search
  filters by label / error category / outcome.
- **Tiered CLI help** (Unit 4): `papayya --help` lists the rung-0 loop
  (`init`, `example`, `dev`, `deploy`, `replay`, `login`) first, then
  run/inspect commands, then platform ops.
- `papayya triggers` group (create/list/delete) — the rename of
  `webhooks`; the old group name survives one release as a hidden alias.
  `papayya.yaml` accepts `triggers:` alongside the legacy `webhooks:`
  key (same schema; merged, duplicate names rejected).
- `papayya batch submit` survives as a hidden alias of `runs submit`.
- `papayya example` scaffold rewritten as a quickstart: `papayya.map` +
  `@papayya.llm` over canned tickets across two tenants, with two items
  coming back degraded (a refusal on a 200) so `papayya dev` shows
  ran-vs-worked on the very first run. The hardcoded
  `PAPAYYA_LOCAL_DB_PATH=/tmp/...` is gone — the run lands in
  `.papayya/local.db`, where `papayya dev` reads.

### Fixed

- **Backup storm on fresh databases**: a brand-new local DB used to be
  created at schema v1 and chain-migrated to head, leaving one
  `backup-vN` file per migration step. Fresh DBs are now created at the
  current schema version directly, with no backups.
- `map()`/`iter()` over an empty iterable no longer strands a
  forever-'running' run row.

### Unchanged (wire freezes)

- The durable checkpoint HTTP contract is frozen at the old paths and
  field names (`/v1/durable/runs/{run_id}`, `run_id`/`item_id`/
  `parent_run_id`) until the control-pane renames (Plan 34 Unit 5).
- The runtime lease protocol (dispatcher/worker) is frozen likewise.
- `papayya dev` API routes and JSON field names keep the pre-v12 wire
  shape (plus new-name fields where non-colliding) for one release; the
  UI-facing rename lands with the dev-dashboard pass.

## 0.2.1 — 2026-05-14

### Documentation

- README rewritten around the workload — periodic and batch LLM jobs (KB ingestion, evals, enrichment, doc extraction, post-processing, codemod) — instead of the generic "durable background jobs for AI agents" framing. Concrete failure modes and the per-item visibility wedge now lead the page.
- Quickstart code now shows a pile-of-items loop (`for company in companies: papayya().run(..., item_id=...)`) so the per-item tagging pattern is visible from the first read.
- `papayya()` factory from `papayya.durable` documented as the canonical entry point. The `Papayya` class export is retained for back-compat but no longer appears in copy.
- New "Examples" section pointing at [github.com/papayya-zero/examples](https://github.com/papayya-zero/examples).
- Project description in `pyproject.toml` updated to match.

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

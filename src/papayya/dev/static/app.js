// Papayya Dev Dashboard — vanilla JS, zero build step.
//
// Nouns (Plan 34): agent → run (one invocation) → item (one record
// processed) → step (one trace node). Two item keyspaces, never merged:
//   /record?id=<items.id>       one record, in one run   (surrogate uuid)
//   /item?id=<customer item id> one item, over time      (lineage)
//
// Three shared helpers only (convention, guarded in PR review):
//   - fetchJSON(url)          : GET + JSON parse with status-aware error
//   - renderTable(cols, rows) : generic table renderer; cols declare label + render
//   - renderBadge(kind,label) : status / category pill
//
// Adding a fourth helper requires justification. This ceiling is
// deliberate — if it feels limiting, that's the point. Grow carefully.
//
// Per-page init is dispatched from `document.body.dataset.page`. Each
// page renders into `#app`. Filter state lives in `location.search` and
// is reflected back on navigation so back/refresh/share all work.

(function () {
    "use strict";

    // ----- utilities -----

    function fetchJSON(url) {
        return fetch(url).then(async (r) => {
            const body = await r.json().catch(() => ({}));
            if (!r.ok) {
                const msg = body && body.error ? body.error : `HTTP ${r.status}`;
                const err = new Error(msg);
                err.status = r.status;
                throw err;
            }
            return body;
        });
    }

    function el(tag, attrs, ...children) {
        const node = document.createElement(tag);
        if (attrs) {
            for (const [k, v] of Object.entries(attrs)) {
                if (v === null || v === undefined || v === false) continue;
                if (k === "class") node.className = v;
                else if (k === "html") node.innerHTML = v;
                else if (k.startsWith("on") && typeof v === "function") {
                    node.addEventListener(k.slice(2), v);
                } else {
                    node.setAttribute(k, v);
                }
            }
        }
        for (const c of children) {
            if (c === null || c === undefined || c === false) continue;
            if (typeof c === "string" || typeof c === "number") {
                node.appendChild(document.createTextNode(String(c)));
            } else {
                node.appendChild(c);
            }
        }
        return node;
    }

    function renderTable(cols, rows, opts) {
        opts = opts || {};
        const thead = el("thead", null,
            el("tr", null, ...cols.map((c) => el("th", null, c.label))));
        const tbody = el("tbody", null,
            ...rows.map((row) => {
                const tr = el("tr", opts.onRowClick ? { class: "clickable" } : null,
                    ...cols.map((c) => {
                        const val = c.render ? c.render(row) : row[c.key];
                        if (val && val.nodeType) return el("td", { class: c.cls || "" }, val);
                        return el("td", { class: c.cls || "" }, val == null ? "" : String(val));
                    }));
                if (opts.onRowClick) tr.addEventListener("click", () => opts.onRowClick(row));
                return tr;
            }));
        return el("table", null, thead, tbody);
    }

    function renderBadge(kind, label) {
        return el("span", { class: "pill pill-" + (kind || "muted") }, label);
    }

    function badgeForStatus(status) {
        const k = {
            completed: "ok", running: "info", queued: "muted",
            failed: "err", cancelled: "muted", paused: "warn",
            partial: "warn",
        }[status] || "muted";
        return renderBadge(k, status || "unknown");
    }

    // The wedge badge: ran vs WORKED. `ok` renders as "worked" so the
    // difference from a plain status "completed" is visible at a glance.
    function badgeForOutcome(outcome) {
        if (outcome === "degraded") return renderBadge("warn", "degraded");
        if (outcome === "failed") return renderBadge("err", "failed");
        return renderBadge("ok", "worked");
    }

    // An item that hard-failed shows "failed" even if no inspector fired.
    function itemOutcome(row) {
        if (row.status === "failed" || row.worst_outcome_status === "failed") return "failed";
        return row.worst_outcome_status || "ok";
    }

    // Run-level outcome cell: "N of M items degraded" is the incident
    // signal that must read off the runs list without clicking anything.
    function runOutcomeCell(r) {
        const total = r.item_count != null ? r.item_count : (r.total_items || 0);
        const parts = [];
        if (r.degraded_items) {
            parts.push(renderBadge("warn", r.degraded_items + " of " + total + " degraded"));
            if (r.degraded_tenants > 0) {
                parts.push(" ");
                parts.push(el("span", { class: "muted" },
                    r.degraded_tenants + " tenant" + (r.degraded_tenants === 1 ? "" : "s")));
            }
        }
        if (r.failed_items) {
            if (parts.length) parts.push(" ");
            parts.push(renderBadge("err", r.failed_items + " failed"));
        }
        if (!parts.length) parts.push(renderBadge("ok", "worked"));
        return el("div", null, ...parts);
    }

    function badgeForErrorCategory(cat) {
        const k = {
            provider: "warn", tool: "warn", timeout: "muted", logic: "err",
        }[cat] || "muted";
        return renderBadge(k, cat);
    }

    function fmtInt(n) {
        if (n == null) return "0";
        return Number(n).toLocaleString();
    }

    // Local cost signal: the ledger has no $ column (rate cards are
    // hosted-side), so token totals are the honest per-item cost proxy.
    function fmtTokens(n) {
        if (n == null) return "—";
        return fmtInt(n) + " tok";
    }

    function shortId(id) {
        if (!id) return "";
        return id.length > 14 ? id.slice(0, 12) + "…" : id;
    }

    function fmtDuration(ms) {
        if (ms == null || isNaN(ms)) return "—";
        const n = Number(ms);
        if (n < 1000) return Math.round(n) + "ms";
        if (n < 60000) return (n / 1000).toFixed(1) + "s";
        return (n / 60000).toFixed(1) + "m";
    }

    function timeAgo(iso) {
        if (!iso) return "";
        const t = Date.parse(iso);
        if (isNaN(t)) return "";
        const s = Math.floor((Date.now() - t) / 1000);
        if (s < 60) return s + "s ago";
        if (s < 3600) return Math.floor(s / 60) + "m ago";
        if (s < 86400) return Math.floor(s / 3600) + "h ago";
        return Math.floor(s / 86400) + "d ago";
    }

    function qs(params) {
        const sp = new URLSearchParams(params);
        return sp.toString();
    }

    function readQuery() {
        const out = {};
        new URLSearchParams(location.search).forEach((v, k) => { out[k] = v; });
        return out;
    }

    function pushQuery(params) {
        const url = location.pathname + (params ? "?" + qs(params) : "");
        history.pushState(null, "", url);
    }

    function copyToClipboard(text) {
        navigator.clipboard && navigator.clipboard.writeText(text);
    }

    function mount(node) {
        const app = document.getElementById("app");
        app.innerHTML = "";
        app.appendChild(node);
    }

    function showError(err) {
        mount(el("div", { class: "panel" },
            el("h2", null, "Error"),
            el("p", null, err.message || String(err))));
    }

    function recordHref(row) {
        const run = row.run_id ? "&run=" + encodeURIComponent(row.run_id) : "";
        return "/record?id=" + encodeURIComponent(row.id || row.item_id) + run;
    }

    // ----- empty-state helpers -----

    const QUICKSTART = "papayya example   # scaffolds agent.py\npython agent.py   # one keyless run over canned items\n";

    function emptyRoot(title, body) {
        const pre = el("pre", null, QUICKSTART);
        const copyBtn = el("button", {
            class: "copy-btn",
            onclick: () => copyToClipboard(QUICKSTART),
        }, "Copy commands");
        return el("div", { class: "empty" },
            el("h3", null, title),
            el("p", null, body),
            pre,
            copyBtn);
    }

    function stat(label, value) {
        return el("div", { class: "stat" },
            el("div", { class: "label" }, label),
            el("div", { class: "value" }, value == null ? "—" : String(value)));
    }

    // ----- page: agents -----

    function pageAgents() {
        fetchJSON("/api/agents").then((agents) => {
            if (!agents.length) {
                mount(emptyRoot(
                    "No agents yet",
                    "Run something with the SDK and every agent shows up here with its runs and outcomes."));
                return;
            }
            const cols = [
                { label: "Agent", render: (a) => el("a", { href: "/runs?agent=" + encodeURIComponent(a.agent) }, a.agent) },
                { label: "Runs", cls: "num", render: (a) => fmtInt(a.run_count) },
                { label: "Items", cls: "num", render: (a) => fmtInt(a.item_count) },
                { label: "Degraded", cls: "num", render: (a) => a.degraded_items ? renderBadge("warn", fmtInt(a.degraded_items)) : "0" },
                { label: "Failed", cls: "num", render: (a) => a.failed_items ? renderBadge("err", fmtInt(a.failed_items)) : "0" },
                { label: "Tokens", cls: "num", render: (a) => fmtTokens(a.total_tokens) },
                { label: "Last run", cls: "muted", render: (a) => timeAgo(a.last_run_at) },
            ];
            mount(el("div", null,
                el("h1", null, "Agents"),
                el("p", { class: "subtitle" }, "Every deployable unit that has written to this ledger."),
                renderTable(cols, agents, {
                    onRowClick: (a) => { location.href = "/runs?agent=" + encodeURIComponent(a.agent); },
                })));
        }).catch(showError);
    }

    // ----- page: runs (invocations) -----

    function pageRuns() {
        const agentFilter = readQuery().agent;
        const url = "/api/runs" + (agentFilter ? "?agent=" + encodeURIComponent(agentFilter) : "");
        Promise.all([fetchJSON(url), fetchJSON("/api/stats")])
            .then(([runs, stats]) => {
                if (!runs.length) {
                    mount(emptyRoot(
                        "No runs yet",
                        "One map() call, one cron fire, or one direct call is one run. Scaffold the demo to see your first:"));
                    return;
                }

                const statRow = el("div", { class: "stats-row" },
                    stat("Runs", stats.runs_total),
                    stat("In progress", stats.runs_in_progress),
                    stat("Items", stats.items_total),
                    stat("Degraded items", stats.items_degraded));

                const cols = [
                    { label: "Run", render: (r) => el("a", { href: "/run?id=" + encodeURIComponent(r.run_id) }, shortId(r.run_id)) },
                    { label: "Agent", key: "agent" },
                    { label: "Status", render: (r) => badgeForStatus(r.status) },
                    { label: "Outcome", render: (r) => runOutcomeCell(r) },
                    { label: "Progress", render: (r) => progressCell(r) },
                    { label: "Tokens", cls: "num", render: (r) => fmtTokens(r.total_tokens) },
                    { label: "Started", cls: "muted", render: (r) => timeAgo(r.created_at) },
                ];
                const tbl = renderTable(cols, runs, {
                    onRowClick: (r) => { location.href = "/run?id=" + encodeURIComponent(r.run_id); },
                });

                mount(el("div", null,
                    el("h1", null, "Runs", agentFilter ? el("span", { class: "muted" }, " · " + agentFilter) : null),
                    el("p", { class: "subtitle" }, "One run per invocation. The Outcome column is ran-vs-worked: a run can complete and still have degraded items."),
                    statRow, tbl));
            })
            .catch(showError);
    }

    function progressCell(run) {
        const done = (run.completed || 0) + (run.failed || 0);
        const total = run.total_items || 1;
        const pct = Math.min(100, Math.round(100 * done / total));
        const bar = el("div", { class: "progress" },
            el("div", { class: "bar" + (run.failed ? " failed" : ""), style: "width:" + pct + "%" }));
        return el("div", null, bar,
            el("div", { class: "progress-label" }, done + " / " + total + " (" + pct + "%)"));
    }

    // ----- page: run detail -----

    function pageRun() {
        const runId = readQuery().id;
        if (!runId) { showError(new Error("missing ?id")); return; }

        fetchJSON("/api/runs/" + encodeURIComponent(runId)).then((run) => {
            // Legacy deep link: a pre-0.3.0 "run id" was a record uuid. The
            // API falls back to the item row; send the browser to /record.
            if (run.id) {
                location.replace(recordHref(run));
                return;
            }
            return Promise.all([
                fetchJSON("/api/runs/" + encodeURIComponent(runId) + "/items"),
                fetchJSON("/api/runs/" + encodeURIComponent(runId) + "/tenants").catch(() => []),
                fetchJSON("/api/runs/" + encodeURIComponent(runId) + "/clusters").catch(() => []),
                fetchJSON("/api/runs/" + encodeURIComponent(runId) + "/outliers").catch(() => []),
                fetchJSON("/api/runs/" + encodeURIComponent(runId) + "/dlq").catch(() => []),
            ]).then(([items, tenants, clusters, outliers, dlq]) => {
                renderRunDetail(runId, run, items, tenants, clusters, outliers, dlq);
            });
        }).catch(showError);
    }

    function renderRunDetail(runId, run, items, tenants, clusters, outliers, dlq) {
        const header = el("div", null,
            el("h1", null, "Run ", el("code", null, run.run_id), " ",
                badgeForStatus(run.status), " ", runOutcomeCell(run)),
            el("p", { class: "subtitle" }, run.agent,
                run.replayed_from
                    ? el("span", null, " · replay of ",
                        el("a", { href: "/run?id=" + encodeURIComponent(run.replayed_from) }, shortId(run.replayed_from)))
                    : ""));

        const worked = Math.max(0, (run.item_count || 0) - (run.degraded_items || 0) - (run.failed_items || 0));
        const stats = el("div", { class: "stats-row" },
            stat("Items", run.item_count),
            stat("Worked", worked),
            stat("Degraded", run.degraded_items),
            stat("Failed", run.failed_items),
            stat("Tokens", run.total_tokens == null ? "—" : fmtInt(run.total_tokens)));

        const cancelBtn = ["completed", "failed", "cancelled", "partial"].includes(run.status)
            ? null
            : el("button", {
                class: "copy-btn",
                onclick: () => cancelRun(runId),
            }, "Cancel run");

        // Per-tenant blast radius — shown whenever anything didn't work,
        // or the run spans multiple tenants.
        const showTenants = tenants.length > 1
            || tenants.some((t) => t.degraded_items || t.failed_items);
        const tenantSection = showTenants && tenants.length
            ? el("div", null,
                el("h2", null, "By tenant"),
                renderTable([
                    { label: "Tenant", key: "tenant" },
                    { label: "Items", cls: "num", key: "item_count" },
                    { label: "Worked", cls: "num", render: (t) => Math.max(0, t.item_count - (t.degraded_items || 0) - (t.failed_items || 0)) },
                    { label: "Degraded", cls: "num", render: (t) => t.degraded_items ? renderBadge("warn", t.degraded_items) : "0" },
                    { label: "Failed", cls: "num", render: (t) => t.failed_items ? renderBadge("err", t.failed_items) : "0" },
                ], tenants))
            : null;

        const clusterSection = clusters.length
            ? el("div", null,
                el("h2", null, "Failure clusters ", el("span", { class: "count" }, "(" + clusters.length + ")")),
                renderTable([
                    { label: "Error", render: (c) => renderBadge("err", c.error_category) },
                    { label: "Count", cls: "num", key: "count" },
                    { label: "Sample step", key: "sample_label", cls: "muted" },
                    { label: "Sample reason", key: "sample_reason", cls: "muted" },
                ], clusters))
            : null;

        const outlierSection = outliers.length
            ? el("div", null,
                el("h2", null, "Longest-running items"),
                renderTable([
                    { label: "Item", render: (r) => el("a", { href: "/record?id=" + encodeURIComponent(r.id) + "&run=" + encodeURIComponent(runId) }, r.item_id || shortId(r.id)) },
                    { label: "Outcome", render: (r) => badgeForOutcome(itemOutcome(r)) },
                    { label: "Status", render: (r) => badgeForStatus(r.status) },
                    { label: "Duration", cls: "num", render: (r) => fmtDuration(r.duration_ms) },
                ], outliers))
            : null;

        const itemsSection = el("div", null,
            el("h2", null, "Items ", el("span", { class: "count" }, "(" + items.length + ")")),
            renderTable([
                { label: "Item", render: (i) => el("a", { href: recordHref(i) }, i.item_id || shortId(i.id)) },
                { label: "Outcome", render: (i) => badgeForOutcome(itemOutcome(i)) },
                { label: "Status", render: (i) => badgeForStatus(i.status) },
                { label: "Tenant", key: "partition_key", cls: "muted" },
                { label: "Steps", cls: "num", key: "step_count" },
                { label: "Tokens", cls: "num", render: (i) => fmtTokens(i.total_tokens) },
                { label: "Started", cls: "muted", render: (i) => timeAgo(i.created_at) },
            ], items, {
                onRowClick: (i) => { location.href = recordHref(i); },
            }));

        const dlqSection = dlq.length
            ? el("div", null,
                el("h2", null, "Dead letter queue ",
                    el("span", { class: "count" }, "(" + dlq.length + ")")),
                el("p", { class: "subtitle" },
                    "Failed items waiting for triage. Replay re-drives the item from its captured input; skip / acknowledge resolves it without re-running."),
                el("div", { class: "dlq-list" },
                    ...dlq.map((d) => renderDlqRow(runId, d))))
            : null;

        const upgradeCard = maybeRenderRunUpgradeCard(run);

        mount(el("div", null, header, upgradeCard, cancelBtn, stats, dlqSection, tenantSection, clusterSection, outlierSection, itemsSection));
    }

    function renderDlqRow(runId, d) {
        const snap = parseSnapshot(d.input_snapshot);
        const snapBlock = snap === undefined
            ? null
            : el("pre", { class: "dlq-input" },
                snap === null ? "null" : JSON.stringify(snap, null, 2));

        const errLine = d.error
            ? el("div", { class: "dlq-error" }, String(d.error))
            : null;

        const actions = el("div", { class: "dlq-actions" },
            el("button", {
                class: "dlq-btn dlq-btn-primary",
                onclick: () => dlqReplay(runId, d.id),
            }, "Replay"),
            el("button", {
                class: "dlq-btn",
                onclick: () => dlqDispose(runId, d.id, "skip"),
            }, "Skip"),
            el("button", {
                class: "dlq-btn",
                onclick: () => dlqDispose(runId, d.id, "acknowledge"),
            }, "Acknowledge"));

        return el("div", { class: "dlq-item" },
            el("div", { class: "dlq-head" },
                el("a", { href: "/record?id=" + encodeURIComponent(d.id) + "&run=" + encodeURIComponent(runId) },
                    d.item_id || shortId(d.id)),
                d.item_id ? el("span", null, " · ",
                    el("a", { href: "/item?id=" + encodeURIComponent(d.item_id) }, "history")) : null,
                d.partition_key ? el("span", { class: "muted" }, " · " + d.partition_key) : null,
                el("span", { class: "muted" }, " · " + timeAgo(d.created_at))),
            errLine,
            snapBlock,
            actions);
    }

    function dlqDispose(runId, recordId, action) {
        const url = "/api/runs/" + encodeURIComponent(runId)
            + "/dlq/" + encodeURIComponent(recordId) + "/" + action;
        fetch(url, { method: "POST" })
            .then((r) => r.json().then((body) => ({ ok: r.ok, body })))
            .then(({ ok, body }) => {
                if (!ok) {
                    alert("Action failed: " + (body.error || "unknown"));
                    return;
                }
                // Refresh the run page so the DLQ list updates.
                pageRun();
            })
            .catch((e) => alert("Action failed: " + e.message));
    }

    function dlqReplay(runId, recordId) {
        const url = "/api/runs/" + encodeURIComponent(runId)
            + "/dlq/" + encodeURIComponent(recordId) + "/replay";
        // Replay can take up to ~120s (the server-side timeout). Show an
        // inline "running..." state rather than a browser-level spinner.
        const existingBtns = document.querySelectorAll(".dlq-btn");
        existingBtns.forEach((b) => { b.disabled = true; });
        fetch(url, { method: "POST" })
            .then((r) => r.json().then((body) => ({ ok: r.ok, body })))
            .then(({ ok, body }) => {
                existingBtns.forEach((b) => { b.disabled = false; });
                if (!ok) {
                    alert("Replay failed: " + (body.error || "unknown"));
                    return;
                }
                if (body.exit_code !== 0) {
                    alert("Replay exited " + body.exit_code
                        + (body.stderr ? "\n\n" + body.stderr : ""));
                }
                pageRun();
            })
            .catch((e) => {
                existingBtns.forEach((b) => { b.disabled = false; });
                alert("Replay failed: " + e.message);
            });
    }

    function cancelRun(id) {
        fetch("/api/runs/" + encodeURIComponent(id) + "/cancel", { method: "POST" })
            .then((r) => r.json())
            .then((body) => {
                alert(body.noop ? "Run already terminal — nothing to cancel" : "Cancelled");
                pageRun();
            })
            .catch((e) => alert("Cancel failed: " + e.message));
    }

    // ----- page: record (one item, in one run — surrogate-uuid keyspace) -----

    function pageRecord() {
        const q = readQuery();
        const recordId = q.id;
        if (!recordId) { showError(new Error("missing ?id")); return; }

        // Internal links always carry &run=. A bare ?id= (old bookmark)
        // resolves the parent run via the legacy /api/runs/<record> fallback.
        const withRun = q.run
            ? Promise.resolve(q.run)
            : fetchJSON("/api/runs/" + encodeURIComponent(recordId)).then((row) => {
                if (!row.id) throw new Error("that id is a run — open /run?id=" + recordId);
                return row.run_id;
            });

        withRun.then((runId) => Promise.all([
            fetchJSON("/api/runs/" + encodeURIComponent(runId) + "/items/" + encodeURIComponent(recordId)),
            fetchJSON("/api/thrashing?item=" + encodeURIComponent(recordId)).catch(() => []),
        ])).then(([detail, thrash]) => {
            const item = detail.item;
            const steps = detail.steps || [];

            // Record page title: "Item <customer id> in run <short>" — the
            // lineage page ("Item <id> — history") is the other keyspace.
            const header = el("div", null,
                el("h1", null,
                    "Item ", el("code", null, item.item_id || shortId(item.id)), " ",
                    badgeForOutcome(itemOutcome(item)), " ",
                    badgeForStatus(item.status)),
                el("p", { class: "subtitle" },
                    item.agent,
                    " · in run ",
                    el("a", { href: "/run?id=" + encodeURIComponent(item.run_id) }, shortId(item.run_id)),
                    item.item_id ? el("span", null, " · ",
                        el("a", { href: "/item?id=" + encodeURIComponent(item.item_id) }, "history")) : "",
                    item.partition_key ? " · tenant " + item.partition_key : "",
                    item.replayed_from ? el("span", null, " · replay of ",
                        el("a", { href: "/record?id=" + encodeURIComponent(item.replayed_from) }, shortId(item.replayed_from))) : ""));

            const stats = el("div", { class: "stats-row" },
                stat("Steps", steps.length),
                stat("Degraded steps", item.degraded_count || 0),
                stat("Tokens", item.total_tokens == null ? "—" : fmtInt(item.total_tokens)),
                stat("Started", timeAgo(item.created_at)));

            const thrashBanner = thrash && thrash.length
                ? el("div", { class: "banner warn" },
                    el("div", null,
                        el("strong", null, "Thrashing detected"),
                        el("p", null,
                            "Step ", el("code", null, thrash[0].label),
                            " journaled ", String(thrash[0].repeat_count),
                            " times with identical input — a common sign of a model loop or missing stop condition.")))
                : null;

            const timeline = steps.length
                ? el("div", { class: "timeline" },
                    ...steps.map((t, i) => el("div", { class: "step" },
                        el("div", { class: "idx" }, "#" + (i + 1)),
                        el("div", null,
                            el("div", null,
                                t.label || el("em", null, "untitled"),
                                " ",
                                t.outcome_status && t.outcome_status !== "ok"
                                    ? badgeForOutcome(t.outcome_status) : null,
                                " ",
                                ...renderLlmBadges(t),
                                " ",
                                t.error_category ? badgeForErrorCategory(t.error_category) : null),
                            el("div", { class: "meta" },
                                t.outcome_reason ? el("span", null, t.outcome_reason) : null,
                                t.kind === "llm" && t.llm_model ? el("span", null, t.llm_model) : null,
                                el("span", null, fmtDuration(t.duration_ms)))),
                        el("div", null, timeAgo(t.completed_at)))))
                : el("div", { class: "empty" },
                    el("h3", null, "No steps yet"),
                    el("p", null, "Steps appear when the item's trace records work — an @papayya.llm / @papayya.step call, or run.step(...)."));

            mount(el("div", null, header, thrashBanner, stats, timeline));
        }).catch(showError);
    }

    // Render badges for a step's LLM usage metadata. Returns an array of
    // nodes (possibly empty) so callers can spread them into a parent.
    // Only emits anything when the step was recorded with kind="llm".
    function renderLlmBadges(t) {
        if (t.kind !== "llm") return [];
        const out = [];
        if (t.llm_provider_shape) {
            out.push(renderBadge("info", t.llm_provider_shape));
            out.push(" ");
        }
        if (t.llm_total_tokens != null) {
            out.push(renderBadge("muted", fmtInt(t.llm_total_tokens) + " tok"));
            out.push(" ");
        }
        if (t.llm_stop_reason) {
            const normalized = String(t.llm_stop_reason).toLowerCase();
            // Gemini returns "STOP" uppercase; OpenAI "stop" and
            // Anthropic "end_turn" are the other standards. Only
            // surface genuine non-standard stop reasons — "length"
            // and provider-specific cutoffs are what operators care about.
            if (normalized !== "stop" && normalized !== "end_turn") {
                out.push(renderBadge("warn", t.llm_stop_reason));
                out.push(" ");
            }
        }
        return out;
    }

    // ----- page: items (latest records across all runs) -----

    function pageItems() {
        const params = readQuery();
        const query = [];
        if (params.outcome) query.push("outcome=" + encodeURIComponent(params.outcome));
        if (params.agent) query.push("agent=" + encodeURIComponent(params.agent));
        fetchJSON("/api/items" + (query.length ? "?" + query.join("&") : "")).then((items) => {
            if (!items.length) {
                mount(emptyRoot(
                    "No items yet",
                    "Every record a run processes shows up here with its ran-vs-worked outcome."));
                return;
            }
            const cols = [
                { label: "Item", render: (i) => el("a", { href: recordHref(i) }, i.item_id || shortId(i.id)) },
                { label: "Outcome", render: (i) => badgeForOutcome(itemOutcome(i)) },
                { label: "Status", render: (i) => badgeForStatus(i.status) },
                { label: "Agent", key: "agent" },
                { label: "Run", render: (i) => i.run_id ? el("a", { href: "/run?id=" + encodeURIComponent(i.run_id) }, shortId(i.run_id)) : "—" },
                { label: "Tenant", key: "partition_key", cls: "muted" },
                { label: "Tokens", cls: "num", render: (i) => fmtTokens(i.total_tokens) },
                { label: "History", render: (i) => i.item_id ? el("a", { href: "/item?id=" + encodeURIComponent(i.item_id) }, "history") : "" },
                { label: "Started", cls: "muted", render: (i) => timeAgo(i.created_at) },
            ];
            const filterBar = el("div", { class: "filter-bar" },
                ...["", "degraded", "failed", "ok"].map((o) =>
                    el("a", {
                        class: "pill pill-" + (params.outcome === o || (!params.outcome && o === "") ? "info" : "muted"),
                        href: "/items" + (o ? "?outcome=" + o : ""),
                        style: "margin-right:8px",
                    }, o === "" ? "all" : o)));
            mount(el("div", null,
                el("h1", null, "Items"),
                el("p", { class: "subtitle" }, "Latest processed records across every run. Click an item for its trace, or its history for the same item over time."),
                filterBar,
                renderTable(cols, items, {
                    onRowClick: (i) => { location.href = recordHref(i); },
                })));
        }).catch(showError);
    }

    // ----- page: item history (customer-identity keyspace) -----

    function pageItem() {
        const itemId = readQuery().id;
        if (!itemId) { showError(new Error("missing ?id")); return; }

        fetchJSON("/api/items/" + encodeURIComponent(itemId)).then((data) => {
            const records = data.records || [];
            const steps = data.steps || [];

            const header = el("div", null,
                el("h1", null, "Item ", el("code", null, data.item_id), " — history"),
                el("p", { class: "subtitle" },
                    "The same item over time, across runs. ",
                    records.length + " record" + (records.length === 1 ? "" : "s"),
                    " · ",
                    steps.length + " step" + (steps.length === 1 ? "" : "s")));

            const degraded = records.filter((r) => itemOutcome(r) === "degraded").length;
            const failed = records.filter((r) => itemOutcome(r) === "failed").length;
            const stats = el("div", { class: "stats-row" },
                stat("Records", records.length),
                stat("Degraded", degraded),
                stat("Failed", failed),
                stat("First seen", records[0] ? timeAgo(records[0].created_at) : "—"));

            const recordsSection = records.length
                ? el("div", null,
                    el("h2", null, "Records"),
                    renderTable([
                        { label: "Record", render: (r) => el("a", { href: recordHref(r) }, shortId(r.id)) },
                        { label: "Outcome", render: (r) => badgeForOutcome(itemOutcome(r)) },
                        { label: "Status", render: (r) => badgeForStatus(r.status) },
                        { label: "Run", render: (r) => r.run_id ? el("a", { href: "/run?id=" + encodeURIComponent(r.run_id) }, shortId(r.run_id)) : "—" },
                        { label: "Agent", key: "agent" },
                        { label: "When", cls: "muted", render: (r) => timeAgo(r.created_at) },
                    ], records, {
                        onRowClick: (r) => { location.href = recordHref(r); },
                    }))
                : null;

            const timeline = steps.length
                ? el("div", null,
                    el("h2", null, "Steps"),
                    el("div", { class: "timeline" },
                        ...steps.map((t) => renderItemStep(t))))
                : el("div", { class: "empty" },
                    el("h3", null, "No steps yet"),
                    el("p", null, "Steps tagged with this item id appear here once a run records them."));

            mount(el("div", null, header, stats, recordsSection, timeline));
        }).catch(showError);
    }

    function renderItemStep(t) {
        const inSnap = parseSnapshot(t.input_snapshot);
        const outSnap = parseSnapshot(t.output_snapshot);
        const diff = diffShallow(inSnap, outSnap);
        const diffBadges = diff.length
            ? el("div", { class: "meta" },
                ...diff.map((d) => renderBadge(d.kind === "changed" ? "warn" : d.kind === "added" ? "ok" : "muted",
                    (d.kind === "added" ? "+" : d.kind === "removed" ? "−" : "~") + d.key)))
            : null;

        const snapshotBlock = (label, value) => value === undefined
            ? null
            : el("div", { class: "snapshot" },
                el("div", { class: "meta" }, label),
                el("pre", null, value === null ? "null" : JSON.stringify(value, null, 2)));

        return el("div", { class: "step" },
            el("div", { class: "idx" }, "→"),
            el("div", null,
                el("div", null,
                    t.label || el("em", null, "untitled"),
                    " ",
                    t.outcome_status && t.outcome_status !== "ok"
                        ? badgeForOutcome(t.outcome_status) : null,
                    " ",
                    t.record_agent ? renderBadge("info", t.record_agent) : null,
                    " ",
                    ...renderLlmBadges(t),
                    " ",
                    t.error_category ? badgeForErrorCategory(t.error_category) : null),
                el("div", { class: "meta" },
                    el("a", { href: "/record?id=" + encodeURIComponent(t.item_id) + (t.run_id ? "&run=" + encodeURIComponent(t.run_id) : "") }, "record " + shortId(t.item_id)),
                    t.kind === "llm" && t.llm_model ? el("span", null, t.llm_model) : null,
                    el("span", null, fmtDuration(t.duration_ms))),
                diffBadges,
                snapshotBlock("input", inSnap),
                snapshotBlock("output", outSnap)),
            el("div", null, timeAgo(t.completed_at)));
    }

    function parseSnapshot(raw) {
        if (raw === null || raw === undefined) return undefined;
        try { return JSON.parse(raw); } catch (_) { return raw; }
    }

    function diffShallow(a, b) {
        if (!isObj(a) || !isObj(b)) return [];
        const out = [];
        const keys = new Set([...Object.keys(a), ...Object.keys(b)]);
        keys.forEach((k) => {
            const inA = k in a, inB = k in b;
            if (inA && !inB) out.push({ kind: "removed", key: k });
            else if (!inA && inB) out.push({ kind: "added", key: k });
            else if (JSON.stringify(a[k]) !== JSON.stringify(b[k])) out.push({ kind: "changed", key: k });
        });
        return out;
    }

    function isObj(v) {
        return v && typeof v === "object" && !Array.isArray(v);
    }

    // ----- page: search -----

    function pageSearch() {
        const params = readQuery();
        const form = el("form", {
            class: "filter-bar",
            onsubmit: (e) => {
                e.preventDefault();
                const data = new FormData(e.target);
                const next = {};
                data.forEach((v, k) => { if (v) next[k] = v; });
                pushQuery(next);
                pageSearch();
            },
        },
            el("label", null, "Step label",
                el("input", { name: "label", value: params.label || "" })),
            el("label", null, "Error category",
                el("input", { name: "error_category", value: params.error_category || "" })),
            el("label", null, "Outcome",
                el("input", { name: "outcome", value: params.outcome || "", placeholder: "degraded" })),
            el("button", { type: "submit" }, "Search"));

        const hasFilters = Object.values(params).some(Boolean);
        const resultsHost = el("div", null);

        if (hasFilters) {
            fetchJSON("/api/steps/search?" + qs(params))
                .then((rows) => {
                    if (!rows.length) {
                        resultsHost.appendChild(el("div", { class: "empty" },
                            el("h3", null, "No matching steps"),
                            el("p", null, "Try relaxing the filters.")));
                        return;
                    }
                    resultsHost.appendChild(renderTable([
                        { label: "Step", key: "label" },
                        { label: "Item", render: (r) => el("a", { href: "/record?id=" + encodeURIComponent(r.item_id) }, shortId(r.item_id)) },
                        { label: "Outcome", render: (r) => badgeForOutcome(r.outcome_status || "ok") },
                        { label: "Error", render: (r) => r.error_category ? badgeForErrorCategory(r.error_category) : "—" },
                        { label: "Duration", cls: "num", render: (r) => fmtDuration(r.duration_ms) },
                        { label: "When", cls: "muted", render: (r) => timeAgo(r.completed_at) },
                    ], rows));
                })
                .catch((e) => resultsHost.appendChild(el("p", null, "Error: " + e.message)));
        } else {
            resultsHost.appendChild(el("p", { class: "subtitle" },
                "Search every step across every run. Filter by label, error category, or outcome (ok / degraded / failed)."));
        }

        mount(el("div", null, el("h1", null, "Search steps"), form, resultsHost));
    }

    // ----- page: upgrade -----

    function pageUpgrade() {
        fetchJSON("/api/tier-recommendation").then((body) => {
            const p = body.projection;
            const rec = body.recommendation;
            const primary = rec.primary;

            const statsBlock = el("div", { class: "stats-row" },
                stat("Items", fmtInt(p.total_items)),
                stat("Runs", fmtInt(p.runs_total)),
                stat("Largest run", fmtInt(p.largest_run)),
                stat("Compute minutes", (p.compute_minutes || 0).toFixed(1)),
                stat("Peak concurrency", fmtInt(body.peak_concurrency)));

            const recCard = el("div", { class: "panel" },
                el("h2", null, "Recommended tier"),
                el("div", { style: "display:flex;align-items:baseline;gap:12px;margin-bottom:8px" },
                    el("div", { style: "font-size:24px;font-weight:600;text-transform:capitalize" }, primary.name),
                    el("div", { class: "pill pill-info" }, "$" + primary.price_usd_per_month + "/mo"),
                    rec.secondary
                        ? el("div", { class: "pill pill-muted" },
                            "or " + rec.secondary.name + " ($" + rec.secondary.price_usd_per_month + "/mo)")
                        : null),
                el("p", { class: "subtitle", style: "margin:0" }, rec.reason));

            const cta = el("div", { class: "banner" },
                el("div", null,
                    el("strong", null, "Ready to run this from the cloud?"),
                    el("p", null, "Sign up and import your local history so you don't start from zero.")),
                el("a", {
                    class: "pill pill-info",
                    href: "https://app.getpapayya.com/signup?ref=dev-upgrade&tier=" + encodeURIComponent(primary.name),
                }, "Sign up for " + primary.name));

            mount(el("div", null,
                el("h1", null, "Your last 30 days"),
                el("p", { class: "subtitle" },
                    "Here's what your agents look like at scale. Papayya Cloud runs this while your laptop's closed."),
                statsBlock, recCard, cta));
        }).catch(showError);
    }

    // ----- contextual upgrade card on run detail -----

    function maybeRenderRunUpgradeCard(run) {
        // Show the card when the user has actually done pile-shaped work.
        // Conditions (any one triggers it): >20 items or >5 min elapsed.
        const items = Number(run.total_items || 0);
        const started = Date.parse(run.created_at || "");
        const ended = Date.parse(run.completed_at || new Date().toISOString());
        const elapsedMin = isNaN(started) || isNaN(ended) ? 0 : (ended - started) / 60000;

        const trigger = items > 20 || elapsedMin > 5;
        if (!trigger) return null;

        const key = "papayya.dev.upgrade-dismissed." + run.run_id;
        if (localStorage.getItem(key) === "1") return null;

        const card = el("div", { class: "banner" },
            el("div", null,
                el("strong", null, "Running this nightly?"),
                el("p", null, "Papayya Cloud runs this while your laptop's closed — with team dashboards, alerts, and persistent history."),
                el("a", {
                    href: "https://app.getpapayya.com/signup?ref=dev-run&run=" + encodeURIComponent(run.run_id),
                }, "See your recommended tier →")),
            el("button", {
                class: "dismiss",
                title: "Dismiss",
                onclick: (e) => {
                    localStorage.setItem(key, "1");
                    e.target.closest(".banner").remove();
                },
            }, "×"));
        return card;
    }

    // ----- dispatcher -----

    const ROUTES = {
        agents:  pageAgents,
        runs:    pageRuns,
        run:     pageRun,
        record:  pageRecord,
        items:   pageItems,
        item:    pageItem,
        search:  pageSearch,
        upgrade: pageUpgrade,
    };

    function highlightNav() {
        const page = document.body.dataset.page;
        const navKey = { run: "runs", record: "items", item: "items" }[page] || page;
        document.querySelectorAll(".topnav a").forEach((a) => {
            if (a.dataset.nav === navKey) a.classList.add("active");
        });
    }

    function boot() {
        highlightNav();
        const page = document.body.dataset.page;
        const handler = ROUTES[page];
        if (handler) handler();
    }

    window.pDev = { refresh: boot };
    window.addEventListener("DOMContentLoaded", boot);
    window.addEventListener("popstate", boot);
})();

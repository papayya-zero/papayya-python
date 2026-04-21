// Papayya Dev Dashboard — vanilla JS, zero build step.
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
                throw new Error(msg);
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

    // ----- empty-state helpers -----

    const QUICKSTART = `from papayya import agent\n\n@agent\ndef my_agent(run):\n    return run.task("search", lambda: "hello")()\n`;

    function emptyRoot(title, body) {
        const pre = el("pre", null, QUICKSTART);
        const copyBtn = el("button", {
            class: "copy-btn",
            onclick: () => copyToClipboard(QUICKSTART),
        }, "Copy snippet");
        return el("div", { class: "empty" },
            el("h3", null, title),
            el("p", null, body),
            pre,
            copyBtn);
    }

    // ----- page: batches -----

    function pageBatches() {
        Promise.all([fetchJSON("/api/batches"), fetchJSON("/api/stats")])
            .then(([batches, stats]) => {
                if (!batches.length) {
                    mount(emptyRoot(
                        "No batches yet",
                        "Run an agent to see its runs grouped as a batch of 1. When you process a list, they'll appear here as one batch."));
                    return;
                }

                const statRow = el("div", { class: "stats-row" },
                    stat("Total batches", stats.total_batches),
                    stat("In progress", stats.batches_in_progress),
                    stat("Completed", stats.batches_completed),
                    stat("Runs", stats.total_runs));

                const cols = [
                    { label: "Batch", render: (r) => el("a", { href: "/batch?id=" + encodeURIComponent(r.batch_id) }, r.batch_id) },
                    { label: "Agent", key: "agent" },
                    { label: "Status", render: (r) => badgeForStatus(r.status) },
                    { label: "Progress", render: (r) => progressCell(r) },
                    { label: "Started", cls: "muted", render: (r) => timeAgo(r.created_at) },
                ];
                const tbl = renderTable(cols, batches, {
                    onRowClick: (r) => { location.href = "/batch?id=" + encodeURIComponent(r.batch_id); },
                });

                mount(el("div", null,
                    el("h1", null, "Batches"),
                    el("p", { class: "subtitle" }, "Every run is part of a batch. Single runs show up as batches of 1."),
                    statRow, tbl));
            })
            .catch(showError);
    }

    function stat(label, value) {
        return el("div", { class: "stat" },
            el("div", { class: "label" }, label),
            el("div", { class: "value" }, value == null ? "—" : String(value)));
    }

    function progressCell(batch) {
        const done = (batch.completed || 0) + (batch.failed || 0);
        const total = batch.total_items || 1;
        const pct = Math.min(100, Math.round(100 * done / total));
        const bar = el("div", { class: "progress" },
            el("div", { class: "bar" + (batch.failed ? " failed" : ""), style: "width:" + pct + "%" }));
        return el("div", null, bar,
            el("div", { class: "progress-label" }, done + " / " + total + " (" + pct + "%)"));
    }

    // ----- page: batch detail -----

    function pageBatch() {
        const batchId = readQuery().id;
        if (!batchId) { showError(new Error("missing ?id")); return; }

        Promise.all([
            fetchJSON("/api/batches/" + encodeURIComponent(batchId)),
            fetchJSON("/api/batches/" + encodeURIComponent(batchId) + "/runs"),
            fetchJSON("/api/batches/" + encodeURIComponent(batchId) + "/clusters"),
            fetchJSON("/api/batches/" + encodeURIComponent(batchId) + "/outliers"),
            fetchJSON("/api/batches/" + encodeURIComponent(batchId) + "/items").catch(() => []),
            fetchJSON("/api/batches/" + encodeURIComponent(batchId) + "/dlq").catch(() => []),
        ]).then(([batch, runs, clusters, outliers, items, dlq]) => {
            const header = el("div", null,
                el("h1", null, "Batch ", el("code", null, batch.batch_id), " ", badgeForStatus(batch.status)),
                el("p", { class: "subtitle" }, batch.agent));

            const stats = el("div", { class: "stats-row" },
                stat("Items", batch.total_items),
                stat("Completed", batch.completed),
                stat("Failed", batch.failed));

            const cancelBtn = ["completed", "failed", "cancelled", "partial"].includes(batch.status)
                ? null
                : el("button", {
                    class: "copy-btn",
                    onclick: () => cancelBatch(batchId),
                }, "Cancel batch");

            const clusterSection = clusters.length
                ? el("div", null,
                    el("h2", null, "Failure clusters ", el("span", { class: "count" }, "(" + clusters.length + ")")),
                    renderTable([
                        { label: "Error", render: (c) => renderBadge("err", c.error_code) },
                        { label: "Count", cls: "num", key: "count" },
                        { label: "Sample label", key: "sample_label", cls: "muted" },
                    ], clusters))
                : null;

            const outlierSection = outliers.length
                ? el("div", null,
                    el("h2", null, "Longest-running runs"),
                    renderTable([
                        { label: "Run", render: (r) => el("a", { href: "/run?id=" + encodeURIComponent(r.run_id) }, r.run_id) },
                        { label: "Status", render: (r) => badgeForStatus(r.status) },
                        { label: "Duration", cls: "num", render: (r) => fmtDuration(r.duration_ms) },
                    ], outliers))
                : null;

            const runsSection = el("div", null,
                el("h2", null, "Runs ", el("span", { class: "count" }, "(" + runs.length + ")")),
                renderTable([
                    { label: "Run", render: (r) => el("a", { href: "/run?id=" + encodeURIComponent(r.run_id) }, r.run_id) },
                    { label: "Status", render: (r) => badgeForStatus(r.status) },
                    { label: "Started", cls: "muted", render: (r) => timeAgo(r.created_at) },
                ], runs));

            const itemsSection = items && items.length
                ? el("div", null,
                    el("h2", null, "Items ", el("span", { class: "count" }, "(" + items.length + ")")),
                    renderTable([
                        { label: "Item", render: (i) => el("a", { href: "/item?id=" + encodeURIComponent(i.item_id) }, i.item_id) },
                        { label: "Runs", cls: "num", key: "run_count" },
                        { label: "Steps", cls: "num", key: "step_count" },
                        { label: "Failed runs", cls: "num", render: (i) => i.failed_runs ? renderBadge("err", i.failed_runs) : "0" },
                        { label: "Last seen", cls: "muted", render: (i) => timeAgo(i.last_seen) },
                    ], items))
                : null;

            const dlqSection = dlq.length
                ? el("div", null,
                    el("h2", null, "Dead letter queue ",
                        el("span", { class: "count" }, "(" + dlq.length + ")")),
                    el("p", { class: "subtitle" },
                        "Failed runs waiting for triage. Replay rebuilds the run from its captured input; skip / acknowledge resolves it without re-running."),
                    el("div", { class: "dlq-list" },
                        ...dlq.map((d) => renderDlqRow(batchId, d))))
                : null;

            const upgradeCard = maybeRenderBatchUpgradeCard(batch);

            mount(el("div", null, header, upgradeCard, cancelBtn, stats, dlqSection, clusterSection, outlierSection, itemsSection, runsSection));
        }).catch(showError);
    }

    function renderDlqRow(batchId, d) {
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
                onclick: () => dlqReplay(batchId, d.run_id),
            }, "Replay"),
            el("button", {
                class: "dlq-btn",
                onclick: () => dlqDispose(batchId, d.run_id, "skip"),
            }, "Skip"),
            el("button", {
                class: "dlq-btn",
                onclick: () => dlqDispose(batchId, d.run_id, "acknowledge"),
            }, "Acknowledge"));

        return el("div", { class: "dlq-item" },
            el("div", { class: "dlq-head" },
                el("a", { href: "/run?id=" + encodeURIComponent(d.run_id) }, d.run_id),
                d.item_id ? el("span", null, " · ",
                    el("a", { href: "/item?id=" + encodeURIComponent(d.item_id) }, "item " + d.item_id)) : null,
                el("span", { class: "muted" }, " · " + timeAgo(d.created_at))),
            errLine,
            snapBlock,
            actions);
    }

    function dlqDispose(batchId, runId, action) {
        const url = "/api/batches/" + encodeURIComponent(batchId)
            + "/dlq/" + encodeURIComponent(runId) + "/" + action;
        fetch(url, { method: "POST" })
            .then((r) => r.json().then((body) => ({ ok: r.ok, body })))
            .then(({ ok, body }) => {
                if (!ok) {
                    alert("Action failed: " + (body.error || "unknown"));
                    return;
                }
                // Refresh the batch page so the DLQ list updates.
                pageBatch();
            })
            .catch((e) => alert("Action failed: " + e.message));
    }

    function dlqReplay(batchId, runId) {
        const url = "/api/batches/" + encodeURIComponent(batchId)
            + "/dlq/" + encodeURIComponent(runId) + "/replay";
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
                pageBatch();
            })
            .catch((e) => {
                existingBtns.forEach((b) => { b.disabled = false; });
                alert("Replay failed: " + e.message);
            });
    }

    function cancelBatch(id) {
        fetch("/api/batches/" + encodeURIComponent(id) + "/cancel", { method: "POST" })
            .then((r) => r.json())
            .then((body) => {
                alert(body.noop ? "Batch already terminal — nothing to cancel" : "Cancelled");
                pageBatch();
            })
            .catch((e) => alert("Cancel failed: " + e.message));
    }

    // ----- page: run detail -----

    function pageRun() {
        const runId = readQuery().id;
        if (!runId) { showError(new Error("missing ?id")); return; }

        Promise.all([
            fetchJSON("/api/runs/" + encodeURIComponent(runId)),
            fetchJSON("/api/runs/" + encodeURIComponent(runId) + "/tasks"),
            fetchJSON("/api/thrashing?run_id=" + encodeURIComponent(runId)).catch(() => []),
        ]).then(([run, tasks, thrash]) => {
            const header = el("div", null,
                el("h1", null, "Run ", el("code", null, run.run_id), " ", badgeForStatus(run.status)),
                el("p", { class: "subtitle" }, run.agent,
                    run.batch_id ? el("span", null, " · ", el("a", { href: "/batch?id=" + encodeURIComponent(run.batch_id) }, "batch " + run.batch_id)) : "",
                    run.item_id ? el("span", null, " · ", el("a", { href: "/item?id=" + encodeURIComponent(run.item_id) }, "item " + run.item_id)) : ""));

            const stats = el("div", { class: "stats-row" },
                stat("Steps", tasks.length),
                stat("Started", timeAgo(run.created_at)));

            const thrashBanner = thrash && thrash.length
                ? el("div", { class: "banner warn" },
                    el("div", null,
                        el("strong", null, "Thrashing detected"),
                        el("p", null,
                            "Tool ", el("code", null, thrash[0].tool_name),
                            " called ", String(thrash[0].repeat_count),
                            " times with identical input — a common sign of a model loop or missing stop condition.")))
                : null;

            const timeline = tasks.length
                ? el("div", { class: "timeline" },
                    ...tasks.map((t, i) => el("div", { class: "step" },
                        el("div", { class: "idx" }, "#" + (i + 1)),
                        el("div", null,
                            el("div", null,
                                t.label || el("em", null, "untitled"),
                                " ",
                                ...renderLlmBadges(t),
                                " ",
                                t.error_category ? badgeForErrorCategory(t.error_category) : null),
                            el("div", { class: "meta" },
                                t.kind === "llm" && t.llm_model ? el("span", null, t.llm_model) : null,
                                el("span", null, fmtDuration(t.duration_ms)))),
                        el("div", null, timeAgo(t.completed_at)))))
                : el("div", { class: "empty" },
                    el("h3", null, "No steps yet"),
                    el("p", null, "Steps appear when the agent wraps work in run.step(...) or run.step(..., kind=\"llm\")."));

            mount(el("div", null, header, thrashBanner, stats, timeline));
        }).catch(showError);
    }

    // Render badges for a task's LLM usage metadata. Returns an array of
    // nodes (possibly empty) so callers can spread them into a parent.
    // Only emits anything when the task was wrapped with kind="llm".
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
        if (t.llm_stop_reason && t.llm_stop_reason !== "stop" && t.llm_stop_reason !== "end_turn") {
            // Only surface non-standard stop reasons — "length" and
            // provider-specific cutoffs are the ones worth flagging.
            out.push(renderBadge("warn", t.llm_stop_reason));
            out.push(" ");
        }
        return out;
    }

    // ----- page: item timeline (Slice 6) -----

    function pageItem() {
        const itemId = readQuery().id;
        if (!itemId) { showError(new Error("missing ?id")); return; }

        fetchJSON("/api/items/" + encodeURIComponent(itemId)).then((data) => {
            const runs = data.runs || [];
            const tasks = data.tasks || [];

            const firstRun = runs[0];
            const header = el("div", null,
                el("h1", null, "Item ", el("code", null, data.item_id)),
                el("p", { class: "subtitle" },
                    runs.length + " run" + (runs.length === 1 ? "" : "s"),
                    " · ",
                    tasks.length + " step" + (tasks.length === 1 ? "" : "s"),
                    firstRun && firstRun.batch_id
                        ? el("span", null, " · ", el("a", { href: "/batch?id=" + encodeURIComponent(firstRun.batch_id) }, "batch " + firstRun.batch_id))
                        : ""));

            const stats = el("div", { class: "stats-row" },
                stat("Runs", runs.length),
                stat("Steps", tasks.length),
                stat("First seen", firstRun ? timeAgo(firstRun.created_at) : "—"));

            const timeline = tasks.length
                ? el("div", { class: "timeline" },
                    ...tasks.map((t) => renderItemStep(t)))
                : el("div", { class: "empty" },
                    el("h3", null, "No steps yet"),
                    el("p", null, "Steps with this item_id appear here once a run.step(...) call tags them."));

            mount(el("div", null, header, stats, timeline));
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
                    t.run_agent ? renderBadge("info", t.run_agent) : null,
                    " ",
                    ...renderLlmBadges(t),
                    " ",
                    t.error_category ? badgeForErrorCategory(t.error_category) : null),
                el("div", { class: "meta" },
                    el("a", { href: "/run?id=" + encodeURIComponent(t.run_id) }, "run " + t.run_id),
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
            el("label", null, "Tool name",
                el("input", { name: "tool_name", value: params.tool_name || "" })),
            el("label", null, "Error code",
                el("input", { name: "error_code", value: params.error_code || "" })),
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
                        { label: "Run", render: (r) => el("a", { href: "/run?id=" + encodeURIComponent(r.run_id) }, r.run_id.slice(0, 12)) },
                        { label: "Tool", render: (r) => r.tool_name ? renderBadge("info", r.tool_name) : "—" },
                        { label: "Error", render: (r) => r.error_code ? renderBadge("err", r.error_code) : "—" },
                        { label: "Duration", cls: "num", render: (r) => fmtDuration(r.duration_ms) },
                        { label: "When", cls: "muted", render: (r) => timeAgo(r.created_at) },
                    ], rows));
                })
                .catch((e) => resultsHost.appendChild(el("p", null, "Error: " + e.message)));
        } else {
            resultsHost.appendChild(el("p", { class: "subtitle" },
                "Search every step across every run. Filter by tool or error."));
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
                stat("Runs", fmtInt(p.total_runs)),
                stat("Batches", fmtInt(p.total_batches)),
                stat("Largest batch", fmtInt(p.largest_batch)),
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
                    "Here's what your workload looks like at scale. Papayya Cloud runs this while your laptop's closed."),
                statsBlock, recCard, cta));
        }).catch(showError);
    }

    // ----- contextual upgrade card on batch detail -----

    function maybeRenderBatchUpgradeCard(batch) {
        // Show the card when the user has actually done pile-shaped work.
        // Conditions (any one triggers it): >20 items or >5 min elapsed.
        const items = Number(batch.total_items || 0);
        const started = Date.parse(batch.created_at || "");
        const ended = Date.parse(batch.completed_at || new Date().toISOString());
        const elapsedMin = isNaN(started) || isNaN(ended) ? 0 : (ended - started) / 60000;

        const trigger = items > 20 || elapsedMin > 5;
        if (!trigger) return null;

        const key = "papayya.dev.upgrade-dismissed." + batch.batch_id;
        if (localStorage.getItem(key) === "1") return null;

        const card = el("div", { class: "banner" },
            el("div", null,
                el("strong", null, "Running this nightly?"),
                el("p", null, "Papayya Cloud runs batches while your laptop's closed — with team dashboards, alerts, and persistent history."),
                el("a", {
                    href: "https://app.getpapayya.com/signup?ref=dev-batch&batch=" + encodeURIComponent(batch.batch_id),
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
        batches: pageBatches,
        batch:   pageBatch,
        run:     pageRun,
        item:    pageItem,
        search:  pageSearch,
        upgrade: pageUpgrade,
    };

    function highlightNav() {
        const page = document.body.dataset.page;
        const navKey = ["batch", "run", "item"].includes(page) ? "batches" : page;
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

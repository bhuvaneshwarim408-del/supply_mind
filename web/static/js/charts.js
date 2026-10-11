(() => {
  "use strict";

  const css = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
  const C = { text: css("--text"), muted: css("--muted"), line: css("--line"), brand: css("--brand"), ok: css("--ok"), warn: css("--warn"), bad: css("--bad"), info: css("--info"), band: css("--band") };
  const POLICY_COLORS = ["#8FA3C4", "#4F8CFF", "#7C5CFF", "#34D399", "#FBBF24"];
  Chart.defaults.font.family = "'Manrope', system-ui, sans-serif";
  Chart.defaults.color = C.muted;
  Chart.defaults.borderColor = C.line;

  const body = document.body;
  const cur = body.dataset.currency || "";
  const $ = (id) => document.getElementById(id);
  const fmt = (n) => Math.round(n).toLocaleString("en-IN");
  const money = (n) => cur + fmt(n);
  const setText = (id, text) => { const el = $(id); if (el) el.textContent = text; };
  const shortDate = (iso) => new Date(iso).toLocaleDateString("en-IN", { day: "numeric", month: "short" });
  const longDate = (iso) => new Date(iso).toLocaleDateString("en-IN", { weekday: "short", day: "numeric", month: "short" });
  const grid = { color: C.line };

  async function getJSON(url) {
    // First-time forecasts can take about a minute to train: tell the user after a short wait.
    const timer = setTimeout(() => { const el = $("load-info"); if (el) el.hidden = false; }, 1500);
    try {
      const res = await fetch(url, { headers: { Accept: "application/json" }, credentials: "same-origin" });
      if (res.status === 401) { window.location.href = "/login"; throw new Error("Sign in required"); }
      if (!res.ok) {
        let detail = "The server returned an error (" + res.status + ")";
        try { const j = await res.json(); if (j.error) detail = j.error; } catch (_) { /* keep default */ }
        throw new Error(detail);
      }
      return await res.json();
    } finally {
      clearTimeout(timer);
      const el = $("load-info"); if (el) el.hidden = true;
    }
  }

  function hbar(canvas, rows, key, unitLabel, color, tipFmt) {
    return new Chart(canvas, {
      type: "bar",
      data: { labels: rows.map((x) => x.sku), datasets: [{ data: rows.map((x) => x[key]), backgroundColor: color, borderRadius: 4 }] },
      options: {
        indexAxis: "y", maintainAspectRatio: false,
        plugins: { legend: { display: false }, tooltip: { callbacks: { label: (c) => tipFmt(c.parsed.x) } } },
        scales: { x: { title: { display: true, text: unitLabel }, grid }, y: { grid: { display: false } } },
      },
    });
  }

  const STATUS_CLASS = { "Reorder now": "bad", "On order": "info", "Low": "warn", "Healthy": "good" };
  const STATUS_LABEL = { "Reorder now": "Urgent", "On order": "On order", "Low": "Warning", "Healthy": "Healthy" };

  function badge(status) {
    const b = document.createElement("span");
    b.className = "badge " + (STATUS_CLASS[status] || "");
    b.textContent = STATUS_LABEL[status] || status;
    return b;
  }

  function addRow(tbody, cells, opts = {}) {
    const tr = document.createElement("tr");
    cells.forEach((v, i) => {
      const td = document.createElement("td");
      if (v instanceof Node) td.appendChild(v); else td.textContent = v;
      if (opts.num && opts.num.includes(i)) td.className = "num";
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  }

  function skuLink(base, sku) {
    const a = document.createElement("a");
    a.href = base + "?sku=" + encodeURIComponent(sku);
    a.textContent = sku;
    return a;
  }

  const apiQuery = window.location.search;

  // ---------- Dashboard ----------
  async function dashboard() {
    const d = await getJSON("/api/dashboard");
    const k = d.kpis, sc = d.status_counts;
    setText("k-cost", money(k.total_cost));
    setText("k-cert", k.analysed ? "Average forecast certainty " + k.avg_certainty_pct.toFixed(1) + "%" : "No SKUs analysed yet");
    setText("k-units", fmt(k.reorder_units));
    setText("k-units-l", "units across " + k.reorder_now + " SKU" + (k.reorder_now === 1 ? "" : "s") + " at or below the reorder point");
    setText("t-urgent", sc["Reorder now"] || 0);
    setText("t-warning", (sc["Low"] || 0) + (sc["On order"] || 0));
    setText("t-healthy", sc["Healthy"] || 0);
    setText("k-analysed", k.analysed);
    setText("k-analysed-l", "of " + k.sku_count + " SKUs have a forecast");
    if (!k.analysed) { $("empty-state").hidden = false; $("dash-body").hidden = true; return; }

    const names = ["Urgent", "Warning", "Healthy"];
    const vals = [sc["Reorder now"] || 0, (sc["Low"] || 0) + (sc["On order"] || 0), sc["Healthy"] || 0];
    new Chart($("c-status"), {
      type: "doughnut",
      data: { labels: names, datasets: [{ data: vals, backgroundColor: [C.bad, C.warn, C.ok], borderColor: css("--surface"), borderWidth: 3 }] },
      options: { maintainAspectRatio: false, cutout: "64%", plugins: { legend: { position: "bottom" } } },
    });
    hbar($("c-cost"), d.cost_by_sku, "cost", "Expected cost", C.brand, (v) => money(v));
    hbar($("c-orders"), d.top_orders, "qty", "Units per order", "#7C5CFF", (v) => fmt(v) + " units");

    const tbody = $("risk-body");
    const base = tbody.dataset.optimizerUrl;
    d.at_risk.forEach((r) => addRow(tbody, [skuLink(base, r.sku), fmt(r.stock_now), fmt(r.reorder_point), r.days_of_cover.toFixed(1), badge(r.status)], { num: [1, 2, 3] }));
  }

  // ---------- Alerts ----------
  async function alerts() {
    const rows = await getJSON("/api/alerts");
    if (!rows.length) { $("al-empty").hidden = false; return; }
    $("al-panel").hidden = false;
    const tbody = $("al-body"), base = tbody.dataset.optimizerUrl;
    const draw = () => {
      const f = $("al-filter").value;
      const shown = rows.filter((r) => f === "all" || r.status === f || (f === "Low" && r.status === "On order"));
      tbody.textContent = "";
      shown.forEach((r) => addRow(tbody, [skuLink(base, r.sku), badge(r.status), fmt(r.stock_now), fmt(r.reorder_point), r.days_of_cover.toFixed(1), fmt(r.order_qty), r.certainty_pct.toFixed(0) + "%"], { num: [2, 3, 4, 5, 6] }));
      setText("al-count", "Showing " + shown.length + " of " + rows.length + " analysed SKUs.");
    };
    $("al-filter").addEventListener("change", draw);
    draw();
  }

  // ---------- Forecast ----------
  async function forecast() {
    const box = $("forecast-box");
    const d = await getJSON("/api/forecast/" + encodeURIComponent(box.dataset.sku) + apiQuery);
    const m = d.metrics, split = d.split_index;
    setText("k-total", fmt(m.horizon_total));
    setText("k-total-l", "Units forecast, next " + m.horizon_days + " days");
    setText("k-avg", fmt(m.avg_daily));
    setText("k-bounds", fmt(m.lower_avg) + " to " + fmt(m.upper_avg));
    setText("k-cert", m.certainty_pct.toFixed(1) + "%");

    new Chart($("c-forecast"), {
      type: "line",
      data: {
        labels: d.dates.map(shortDate),
        datasets: [
          { label: "Lower bound (80%)", data: d.lower, borderColor: C.muted, borderWidth: 1.5, borderDash: [2, 3], pointRadius: 0, fill: false },
          { label: "Upper bound (80%)", data: d.upper, borderColor: C.muted, borderWidth: 1.5, borderDash: [2, 3], pointRadius: 0, fill: "-1", backgroundColor: C.band },
          { label: "Forecast", data: d.forecast, borderColor: C.brand, borderWidth: 2.5, pointRadius: 0, tension: 0.25,
            segment: { borderDash: (ctx) => (ctx.p0DataIndex >= split - 1 ? [6, 4] : undefined) } },
          { label: "Actual demand", data: d.actual, borderColor: C.text, borderWidth: 2, pointRadius: 0, spanGaps: false },
        ],
      },
      options: {
        maintainAspectRatio: false, interaction: { mode: "index", intersect: false },
        plugins: { legend: { position: "bottom" }, tooltip: { filter: (i) => i.parsed.y !== null } },
        scales: { x: { grid: { display: false }, ticks: { maxTicksLimit: 10, autoSkip: true } }, y: { beginAtZero: true, title: { display: true, text: "Units per day" }, grid } },
      },
    });

    const tb = $("bounds-body");
    for (let i = split; i < d.dates.length; i++) addRow(tb, [longDate(d.dates[i]), fmt(d.lower[i]), fmt(d.forecast[i]), fmt(d.upper[i])], { num: [1, 2, 3] });
  }

  // ---------- Optimizer ----------
  function showExplanation(ex) {
    const box = $("explain-box"), list = $("explain-sources");
    box.textContent = "";
    const p = document.createElement("p");
    if (ex && ex.explanation) {
      p.textContent = ex.explanation;
      if (ex.sources && ex.sources.length) {
        ex.sources.forEach((s) => { const li = document.createElement("li"); li.textContent = s; list.appendChild(li); });
        list.hidden = false;
      }
    } else {
      p.className = "muted";
      p.textContent = ex && ex.error ? "The explanation could not be generated: " + ex.error : "No explanation is available for this view yet. Sign in and open this SKU to generate one.";
    }
    box.appendChild(p);
    if (ex && ex.check) {
      const c = document.createElement("p");
      c.className = "muted";
      c.textContent = ex.check.ok
        ? "Number check: all " + ex.check.checked + " figures in this explanation match the optimizer and forecast output."
        : "Number check: these figures do not match the optimizer or forecast output, so verify them: " + ex.check.unmatched.join(", ") + ".";
      box.appendChild(c);
    }
  }

  async function optimizer() {
    const box = $("optimizer-box");
    const d = await getJSON("/api/optimizer/" + encodeURIComponent(box.dataset.sku) + apiQuery);
    setText("k-qty", fmt(d.order_qty));
    setText("k-rop", fmt(d.reorder_point));
    setText("k-safety", fmt(d.safety_stock));
    setText("k-cost", money(d.total_cost));
    setText("t-solver", d.solver_status);
    setText("t-conf", d.certainty_pct.toFixed(1) + "%");
    setText("t-avg", fmt(d.avg_daily) + " units");
    setText("t-now", fmt(d.stock_now) + " units");
    setText("t-cover", d.days_of_cover.toFixed(1) + " days");
    setText("t-lead", d.lead_time_days + " days");
    setText("t-orders", d.orders_placed);
    setText("t-served", d.projected_service_pct.toFixed(1) + "%");
    if (d.projected) {
      setText("stock-title", "Projected stock for " + d.sku);
      setText("stock-sub", "Stock from today if you follow this plan and demand matches the forecast. An order is placed each time stock reaches the reorder point.");
    }
    showExplanation(d.explanation);
    const rec = $("ov-rec"); if (rec) rec.value = d.order_qty;

    const flat = (v) => d.dates.map(() => v);
    new Chart($("c-stock"), {
      type: "line",
      data: {
        labels: d.dates.map(shortDate),
        datasets: [
          { label: d.projected ? "Projected stock" : "Stock on hand", data: d.stock, borderColor: C.brand, borderWidth: 2.5, pointRadius: 0, tension: 0, fill: true, backgroundColor: C.band },
          { label: "Reorder point", data: flat(d.reorder_point), borderColor: C.warn, borderWidth: 2, borderDash: [6, 4], pointRadius: 0 },
          { label: "Safety stock", data: flat(d.safety_stock), borderColor: C.bad, borderWidth: 2, borderDash: [2, 3], pointRadius: 0 },
        ],
      },
      options: {
        maintainAspectRatio: false, interaction: { mode: "index", intersect: false }, plugins: { legend: { position: "bottom" } },
        scales: { x: { grid: { display: false }, ticks: { maxTicksLimit: 10, autoSkip: true } }, y: { beginAtZero: true, title: { display: true, text: "Units in stock" }, grid } },
      },
    });
  }

  // ---------- shared: dataset picker for Experiments and RL Lab ----------
  async function loadExperiments(emptyId) {
    let d;
    try { d = await getJSON("/api/experiments"); }
    catch (err) { if (/No experiment/.test(err.message)) { $(emptyId).hidden = false; return null; } throw err; }
    const sel = $("ds");
    Object.keys(d.datasets).forEach((k) => { const o = document.createElement("option"); o.value = k; o.textContent = d.datasets[k].name; sel.appendChild(o); });
    return d;
  }
  const policyNames = (r) => Object.keys(r.policies.ample).filter((n) => !n.startsWith("_"));
  const FC = "Forecast base-stock (squared-error)", CV = "CVXPY optimizer", RLN = "RL correction";

  // ---------- Experiments ----------
  let expCharts = [];
  async function experiments() {
    const d = await loadExperiments("exp-empty");
    if (!d) return;
    const keys = Object.keys(d.datasets), sel = $("ds");
    $("exp-body").hidden = false;
    const render = (k) => {
      const r = d.datasets[k];
      expCharts.forEach((c) => c.destroy()); expCharts = [];
      const names = policyNames(r), sv = (sc) => r.savings[sc];
      [["ample", "e-save-ample"], ["tight", "e-save-tight"]].forEach(([sc, id]) => {
        setText(id, sv(sc).savings_pct.toFixed(1) + "%");
        setText(id + "-l", sv(sc).savings_pct > 0 ? "Lower cost than the simple rule, " + sc + " storage (best: " + sv(sc).best_policy + ")"
          : "Lower cost than the simple rule, " + sc + " storage (the simple rule was already best)");
      });
      setText("e-tau", Math.round(r.settings.tuned_service_level * 100) + "%");
      setText("e-shadow", r.shadow_prices.tight.mean.toFixed(2));
      const q = r.forecast_quality, naive = q[0].wape_pct, ridge = q[1].wape_pct;
      const delta = (sc) => 100 * (r.policies[sc][CV].total_cost - r.policies[sc][FC].total_cost) / r.policies[sc][FC].total_cost;
      setText("e-note", "The forecast model has " + (100 * (naive - ridge) / naive).toFixed(0) + "% lower error than a plain 28-day average. " +
        "Against the plain forecast policy, the CVXPY optimizer changes cost by " + delta("ample").toFixed(1) + "% with ample storage and " + delta("tight").toFixed(1) +
        "% with tight storage (positive means more costly). The decision-focused and RL rows are included for comparison; the table shows whether they helped on this dataset. Tested on " + r.n_skus + " SKUs, " + r.split.test_days + " unseen days.");
      expCharts.push(new Chart($("c-exp-cost"), {
        type: "bar",
        data: { labels: names, datasets: [
          { label: "Ample storage", data: names.map((n) => r.policies.ample[n].total_cost), backgroundColor: C.brand, borderRadius: 4 },
          { label: "Tight storage", data: names.map((n) => r.policies.tight[n].total_cost), backgroundColor: "#7C5CFF", borderRadius: 4 }] },
        options: { maintainAspectRatio: false, plugins: { legend: { position: "bottom" }, tooltip: { callbacks: { label: (c) => c.dataset.label + ": " + fmt(c.parsed.y) } } },
          scales: { x: { grid: { display: false }, ticks: { maxRotation: 40, minRotation: 0 } }, y: { beginAtZero: true, title: { display: true, text: "Total cost" }, grid } } },
      }));
      const body = $("e-policy-body"); body.textContent = "";
      names.forEach((n) => addRow(body, [n, fmt(r.policies.ample[n].total_cost), fmt(r.policies.tight[n].total_cost), r.policies.tight[n].fill_rate_pct.toFixed(1) + "%"], { num: [1, 2, 3] }));
      const qb = $("e-quality-body"); qb.textContent = "";
      q.forEach((row) => addRow(qb, [row.model, row.wape_pct == null ? "–" : row.wape_pct.toFixed(1) + "%", row.coverage_pct == null ? "–" : row.coverage_pct.toFixed(1) + "%", row.newsvendor_cost == null ? "–" : row.newsvendor_cost.toFixed(3)], { num: [1, 2, 3] }));
      const curves = r.stock_curves.tight, len = curves[names[0]].length;
      expCharts.push(new Chart($("c-exp-stock"), {
        type: "line",
        data: { labels: Array.from({ length: len }, (_, i) => i + 1), datasets: names.map((n, i) => ({ label: n, data: curves[n], borderColor: POLICY_COLORS[i % POLICY_COLORS.length], borderWidth: 2, pointRadius: 0, tension: 0.2 })) },
        options: { maintainAspectRatio: false, interaction: { mode: "index", intersect: false }, plugins: { legend: { position: "bottom" } },
          scales: { x: { title: { display: true, text: "Test day" }, grid: { display: false }, ticks: { maxTicksLimit: 10 } }, y: { title: { display: true, text: "Units in stock, all SKUs" }, grid } } },
      }));
    };
    sel.addEventListener("change", () => render(sel.value));
    render(keys[0]);
  }

  // ---------- RL Lab ----------
  let rlCharts = [];
  async function rlLab() {
    const d = await loadExperiments("rl-empty");
    if (!d) return;
    const keys = Object.keys(d.datasets), sel = $("ds");
    $("rl-body").hidden = false;
    const render = (k) => {
      const r = d.datasets[k], rl = r.rl;
      rlCharts.forEach((c) => c.destroy()); rlCharts = [];
      const rel = (sc) => 100 * (r.policies[sc][RLN].total_cost - r.policies[sc][FC].total_cost) / r.policies[sc][FC].total_cost;
      const a = rel("ample"), t = rel("tight");
      const sign = (v) => (v > 0 ? "+" : "") + v.toFixed(1) + "%";
      setText("r-ample", sign(a)); setText("r-tight", sign(t));
      setText("r-eps", fmt(rl.episodes)); setText("r-states", rl.states + " states, " + rl.actions.length + " actions");
      setText("r-note", a > 0 && t > 0
        ? "On this dataset the RL correction cost more than the forecast policy it was correcting, in both storage cases (positive means more costly). It did not help here. The agent is small and the demand is noisy, so it cannot find a reliable edge; the page shows what it learned."
        : a < 0 && t < 0 ? "On this dataset the RL correction cost less than the forecast policy in both storage cases (negative means cheaper)."
        : "The RL correction was cheaper than the forecast policy in one storage case and costlier in the other, so the benefit is not consistent.");
      const step = rl.episodes / rl.curve.length;
      rlCharts.push(new Chart($("c-rl-curve"), {
        type: "line",
        data: { labels: rl.curve.map((_, i) => fmt((i + 1) * step)), datasets: [{ label: "Average reward", data: rl.curve, borderColor: C.brand, borderWidth: 2.5, pointRadius: 0, tension: 0.25 }] },
        options: { maintainAspectRatio: false, plugins: { legend: { display: false } },
          scales: { x: { title: { display: true, text: "Training episodes" }, grid: { display: false }, ticks: { maxTicksLimit: 8 } }, y: { title: { display: true, text: "Reward (higher is better)" }, grid } } },
      }));
      rlCharts.push(new Chart($("c-rl-actions"), {
        type: "bar",
        data: { labels: rl.actions.map((x) => "×" + x.toFixed(2)), datasets: [
          { label: "Ample storage", data: rl.action_share_pct.ample, backgroundColor: C.brand, borderRadius: 4 },
          { label: "Tight storage", data: rl.action_share_pct.tight, backgroundColor: "#7C5CFF", borderRadius: 4 }] },
        options: { maintainAspectRatio: false, plugins: { legend: { position: "bottom" } }, scales: { x: { grid: { display: false }, title: { display: true, text: "Order-up-to level, as a multiple of the forecast target" } }, y: { beginAtZero: true, title: { display: true, text: "% of days chosen" }, grid } } },
      }));
      const pick = ["Simple rule", FC, CV, RLN];
      rlCharts.push(new Chart($("c-rl-cmp"), {
        type: "bar",
        data: { labels: pick, datasets: [
          { label: "Ample storage", data: pick.map((n) => r.policies.ample[n].total_cost), backgroundColor: C.brand, borderRadius: 4 },
          { label: "Tight storage", data: pick.map((n) => r.policies.tight[n].total_cost), backgroundColor: "#7C5CFF", borderRadius: 4 }] },
        options: { maintainAspectRatio: false, plugins: { legend: { position: "bottom" }, tooltip: { callbacks: { label: (c) => c.dataset.label + ": " + fmt(c.parsed.y) } } },
          scales: { x: { grid: { display: false } }, y: { beginAtZero: true, title: { display: true, text: "Total cost" }, grid } } },
      }));
    };
    sel.addEventListener("change", () => render(sel.value));
    render(keys[0]);
  }

  // ---------- What-if ----------
  let whatifChart = null;
  function whatif() {
    const out = (id, fn) => { const el = $(id), o = $(id + "-out"); const f = () => { o.textContent = fn(el.value); }; el.addEventListener("input", f); f(); };
    out("wi-demand", (v) => (v >= 0 ? "+" : "") + v + "%");
    out("wi-delay", (v) => v + (v === "1" ? " day" : " days"));
    const run = async () => {
      const sku = $("wi-sku").value;
      const qs = new URLSearchParams({ demand_change: $("wi-demand").value, delay: $("wi-delay").value, capacity: $("wi-cap").value });
      try {
        $("load-error").hidden = true;
        const d = await getJSON("/api/whatif/" + encodeURIComponent(sku) + "?" + qs.toString());
        const P = d.policies;
        setText("w-rule", fmt(P["Simple rule"].total_cost));
        setText("w-fc", fmt(P["Forecast base-stock"].total_cost));
        setText("w-cv", fmt(P["CVXPY optimizer"].total_cost));
        const best = Object.keys(P).reduce((a, b) => (P[a].total_cost <= P[b].total_cost ? a : b));
        setText("w-fill", P[best].fill_rate_pct.toFixed(1) + "%");
        setText("wi-sub", "Lowest cost: " + best + ". The forecast planned for a " + d.settings.planned_lead_days + "-day supplier; the real supplier took " + (d.settings.planned_lead_days + d.settings.delay_days) + " days.");
        if (whatifChart) whatifChart.destroy();
        const colors = { "Simple rule": POLICY_COLORS[0], "Forecast base-stock": POLICY_COLORS[1], "CVXPY optimizer": POLICY_COLORS[3] };
        whatifChart = new Chart($("c-whatif"), {
          type: "line",
          data: { labels: d.dates.map(shortDate), datasets: Object.keys(P).map((n) => ({ label: n, data: P[n].stock, borderColor: colors[n], borderWidth: 2, pointRadius: 0, tension: 0 })) },
          options: { maintainAspectRatio: false, interaction: { mode: "index", intersect: false }, plugins: { legend: { position: "bottom" } },
            scales: { x: { grid: { display: false }, ticks: { maxTicksLimit: 10 } }, y: { beginAtZero: true, title: { display: true, text: "Units in stock" }, grid } } },
        });
      } catch (err) { const el = $("load-error"); el.textContent = err.message; el.hidden = false; }
    };
    $("wi-run").addEventListener("click", run);
    $("wi-sku").addEventListener("change", run);
    run();
  }

  // ---------- theme toggle + wiring ----------
  const toggle = $("theme-toggle");
  if (toggle) {
    const now = document.documentElement.getAttribute("data-theme");
    toggle.textContent = now === "dark" ? "Switch to light theme" : "Switch to dark theme";
    toggle.addEventListener("click", () => {
      try { localStorage.setItem("supplymind-theme", now === "dark" ? "light" : "dark"); } catch (e) { /* ignore */ }
      window.location.reload();
    });
  }
  document.querySelectorAll("[data-autosubmit]").forEach((el) => el.addEventListener("change", () => el.form.submit()));

  const pages = { dashboard, alerts, forecast, optimizer, experiments, rl: rlLab, whatif };
  const run = pages[body.dataset.page];
  if (run) {
    Promise.resolve().then(run).catch((err) => {
      const el = $("load-error");
      if (el) { el.textContent = err.message + ". Fix the value or reload the page to try again."; el.hidden = false; }
    });
  }
})();

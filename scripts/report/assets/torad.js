/* ============================================================================
   Torad Train — shared client runtime
   Fetch/format helpers, the app-shell injector (sidebar + live status dot),
   and the ported analysis logic (installLevel, gate grading, loss-curve canvas,
   mini-bars). Vanilla, no build. Pages call mountShell() then render from /api.
   ============================================================================ */
(function (T) {

  /* ── tiny DOM + format ─────────────────────────────────────────────── */
  T.$  = (s, r = document) => r.querySelector(s);
  T.$$ = (s, r = document) => [...r.querySelectorAll(s)];
  T.el = (tag, cls, html) => { const n = document.createElement(tag); if (cls) n.className = cls; if (html != null) n.innerHTML = html; return n; };
  T.esc = s => String(s == null ? "" : s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  T.fmt = (n, d = 0) => (n == null || Number.isNaN(n)) ? "—" : Number(n).toLocaleString(undefined, { minimumFractionDigits: d, maximumFractionDigits: d });
  T.pct = (x, d = 0) => (x == null || Number.isNaN(x)) ? "—" : (x * 100).toFixed(d) + "%";
  T.qs  = name => new URLSearchParams(location.search).get(name);

  T.relDate = ts => {
    if (!ts) return "—";
    const d = new Date(ts * 1000), now = Date.now() / 1000, dt = now - ts;
    if (dt < 90) return "just now";
    if (dt < 3600) return Math.round(dt / 60) + "m ago";
    if (dt < 86400) return Math.round(dt / 3600) + "h ago";
    if (dt < 86400 * 6) return Math.round(dt / 86400) + "d ago";
    return d.toLocaleDateString(undefined, { month: "short", day: "numeric" });
  };

  // ── objective (festival domain) — persisted, auto-scopes every API call ──
  T.objective = () => localStorage.getItem("torad_objective") || "edc";
  T.setObjective = id => localStorage.setItem("torad_objective", id);

  T.fetchJSON = async url => {
    if (url.startsWith("/api/") && !url.includes("objective=")) {
      url += (url.includes("?") ? "&" : "?") + "objective=" + encodeURIComponent(T.objective());
    }
    const r = await fetch(url);
    const j = await r.json().catch(() => ({ error: "bad json" }));
    if (!r.ok) throw new Error(j.error || ("HTTP " + r.status));
    return j;
  };

  T.poll = (fn, ms) => {
    let stop = false, h = null;
    const tick = async () => { if (stop) return; try { await fn(); } catch (e) { /* keep polling */ } if (!stop) h = setTimeout(tick, ms); };
    tick();
    return () => { stop = true; clearTimeout(h); };
  };

  /* ── status → pill ─────────────────────────────────────────────────── */
  const PILL = { passed: ["pass", "PASS"], failed: ["fail", "FAIL"], trained: ["trained", "NOT GATED"],
                 running: ["run", "TRAINING"], pass: ["pass", "PASS"], fail: ["fail", "FAIL"],
                 warn: ["warn", "WARN"], skipped: ["skipped", "SKIP"], stopped: ["trained", "STOPPED"] };
  T.pill = (status, label) => { const [c, t] = PILL[status] || ["trained", String(status || "?").toUpperCase()]; return `<span class="pill ${c}"><span class="pd"></span>${label || t}</span>`; };

  /* gate grading: dir 'max' → value must stay ≤ thr (fabrication); 'min' → ≥ thr (acc, cov) */
  T.grade = (v, thr, dir) => v == null ? "q-dim" : (dir === "max" ? (v <= thr ? "q-good" : "q-bad") : (v >= thr ? "q-good" : "q-bad"));
  T.GATE = { fabrication: 0.25, accuracy: 0.55, coverage: 0.40 };

  /* ── CPT-install verdict (ported from coach.html, CHANCE=1/7) ──────── */
  const CHANCE = 1 / 7;
  T.installLevel = (geo, da) => {
    let ratio = null, fails = [];
    if (geo && geo.length) { const p = geo[geo.length - 1]["p@1_cpt"]; if (p != null) ratio = p / CHANCE; }
    if (da && da.dimensions) for (const k of ["density", "balance", "coverage", "disjoint_static", "disjoint_latent"]) {
      const d = da.dimensions[k]; if (d && d.verdict === "fail") fails.push(k);
    }
    let level = (ratio != null && ratio >= 2.5 && !fails.length) ? "strong"
      : ((ratio != null && ratio < 1.8) || fails.length >= 2) ? "weak" : "partial";
    return { level, ratio, fails };
  };
  T.INSTALL_PILL = { strong: ["pass", "STRONG"], partial: ["warn", "PARTIAL"], weak: ["fail", "WEAK"] };
  T.installPill = lvl => { const [c, t] = T.INSTALL_PILL[lvl] || ["trained", "—"]; return `<span class="pill ${c}"><span class="pd"></span>${t}</span>`; };

  /* ── mini bar ──────────────────────────────────────────────────────── */
  T.miniBar = (v, max, thr, cls) => {
    const w = Math.max(0, Math.min(1, (v || 0) / (max || 1))) * 100;
    const tx = thr != null ? `<span class="thresh" style="left:${Math.min(100, thr / (max || 1) * 100)}%"></span>` : "";
    return `<div class="mbar"><i class="${cls || ""}" style="width:${w}%"></i>${tx}</div>`;
  };

  /* ── loss curve canvas ─────────────────────────────────────────────── */
  T.drawLoss = (canvas, points, opts = {}) => {
    if (!canvas) return;
    const dpr = window.devicePixelRatio || 1;
    const W = canvas.clientWidth || 600, H = canvas.clientHeight || 220;
    canvas.width = W * dpr; canvas.height = H * dpr;
    const g = canvas.getContext("2d"); g.scale(dpr, dpr);
    g.clearRect(0, 0, W, H);
    const css = getComputedStyle(document.body);
    const ox = css.getPropertyValue("--ox").trim() || "#7B2D26";
    const line = css.getPropertyValue("--line-2").trim() || "#D5CDB8";
    const ink3 = css.getPropertyValue("--ink-3").trim() || "#8A8175";
    const padL = 44, padR = 12, padT = 12, padB = 24;
    const plotW = W - padL - padR, plotH = H - padT - padB;
    if (!points || !points.length) { g.fillStyle = ink3; g.font = "12px 'IBM Plex Mono'"; g.fillText("no steps yet", padL, padT + plotH / 2); return; }
    const total = opts.total || points[points.length - 1].total || points[points.length - 1].step;
    const maxL = Math.max(...points.map(p => p.loss)) * 1.05;
    const x = s => padL + (s / total) * plotW;
    const y = l => padT + plotH - (l / maxL) * plotH;
    // grid + y labels
    g.strokeStyle = line; g.fillStyle = ink3; g.font = "10px 'IBM Plex Mono'"; g.lineWidth = 1;
    for (let i = 0; i <= 4; i++) { const v = maxL * i / 4, yy = y(v);
      g.globalAlpha = .5; g.beginPath(); g.moveTo(padL, yy); g.lineTo(W - padR, yy); g.stroke(); g.globalAlpha = 1;
      g.fillText(v.toFixed(2), 6, yy + 3); }
    g.fillText("0", x(0) - 2, H - 8); g.fillText(String(total), x(total) - 14, H - 8);
    // area + line
    g.beginPath(); points.forEach((p, i) => { const X = x(p.step), Y = y(p.loss); i ? g.lineTo(X, Y) : g.moveTo(X, Y); });
    g.lineTo(x(points[points.length - 1].step), padT + plotH); g.lineTo(x(points[0].step), padT + plotH); g.closePath();
    const grad = g.createLinearGradient(0, padT, 0, padT + plotH); grad.addColorStop(0, "rgba(123,45,38,.13)"); grad.addColorStop(1, "rgba(123,45,38,0)");
    g.fillStyle = grad; g.fill();
    g.beginPath(); points.forEach((p, i) => { const X = x(p.step), Y = y(p.loss); i ? g.lineTo(X, Y) : g.moveTo(X, Y); });
    g.strokeStyle = ox; g.lineWidth = 1.8; g.lineJoin = "round"; g.stroke();
    // current point
    const last = points[points.length - 1];
    g.fillStyle = ox; g.beginPath(); g.arc(x(last.step), y(last.loss), 3.2, 0, 7); g.fill();
    g.fillStyle = "#fff"; g.beginPath(); g.arc(x(last.step), y(last.loss), 1.3, 0, 7); g.fill();
  };

  /* ── app shell ─────────────────────────────────────────────────────── */
  const NAV = [
    { sep: "Console" },
    { href: "/", page: "home", ic: "◈", label: "Overview" },
    { href: "/coach", page: "coach", ic: "✦", label: "Coach" },
    { href: "/data", page: "data", ic: "▤", label: "Data" },
    { href: "/train", page: "train", ic: "◉", label: "Train" },
    { href: "/runs", page: "runs", ic: "≣", label: "Runs" },
    { sep: "Reference" },
    { href: "/docs", page: "docs", ic: "¶", label: "Best practices" },
  ];

  T.mountShell = ({ page, crumbs } = {}) => {
    const app = T.el("div", "app");
    const side = T.el("aside", "side");
    side.innerHTML = `
      <div class="brand">
        <div class="mark"><span class="dot"></span>Torad&nbsp;Train</div>
        <div class="tag">CPT · SFT console</div>
      </div>
      <div class="objbar"><div class="lab" style="margin-bottom:5px">Objective</div>
        <select class="objsel" id="objSel"><option>${T.esc(T.objective())}</option></select></div>
      <nav>${NAV.map(n => n.sep
        ? `<div class="sep">${n.sep}</div>`
        : `<a href="${n.href}" class="${n.page === page ? "active" : ""}"><span class="ic">${n.ic}</span>${n.label}</a>`).join("")}</nav>
      <div class="foot"><span class="live" id="liveDot"></span><span id="liveTxt">idle</span></div>`;
    const main = T.el("main", "main");
    if (crumbs) {
      const c = T.el("div", "crumbs");
      c.innerHTML = crumbs.map((x, i) => (i ? '<span class="s">/</span>' : "") + (x.href ? `<a href="${x.href}">${T.esc(x.label)}</a>` : `<span>${T.esc(x.label)}</span>`)).join("");
      main.appendChild(c);
    }
    const slot = T.el("div", "slot"); main.appendChild(slot);
    app.appendChild(side); app.appendChild(main);
    document.body.appendChild(app);
    // objective switcher — populate from the registry, switch reloads the page (all data refetches scoped)
    (async () => {
      try {
        const { objectives } = await fetch("/api/objectives").then(r => r.json());
        const sel = T.$("#objSel"); const cur = T.objective();
        sel.innerHTML = objectives.map(o => {
          const tag = o.status === "planned" ? " (planned)" : o.n_runs ? ` · ${o.n_runs} runs` : "";
          return `<option value="${o.id}" ${o.id === cur ? "selected" : ""}>${T.esc(o.name)}${tag}</option>`;
        }).join("");
        if (!objectives.some(o => o.id === cur) && objectives[0]) { T.setObjective(objectives[0].id); sel.value = objectives[0].id; }
        sel.onchange = () => { T.setObjective(sel.value); location.reload(); };
      } catch (e) { /* switcher stays as-is */ }
    })();
    // live footer dot — polls the active-run heartbeat
    T.poll(async () => {
      const a = await T.fetchJSON("/api/train/active").catch(() => ({}));
      const dot = T.$("#liveDot"), txt = T.$("#liveTxt");
      if (a.active) { dot.className = "live run"; txt.textContent = (a.step == null && a.phase) ? `${a.run} · ${a.phase}` : `${a.run} · step ${a.step ?? "—"}/${a.total_steps ?? "—"}`; }
      else if (a.run && a.status === "stopped") { dot.className = "live on"; txt.textContent = `${a.run} · stopped`; }
      else { dot.className = "live"; txt.textContent = "idle · no run"; }
    }, 3000);
    return slot;
  };

  T.crumbSep = '<span class="s">/</span>';
  T.err = (slot, msg) => { slot.innerHTML = `<div class="errbox">⚠ ${T.esc(msg)}</div>`; };

})(window.T = window.T || {});

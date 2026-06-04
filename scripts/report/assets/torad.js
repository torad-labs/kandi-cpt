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

  // ── shared {objective, run, layer} scope — the SaaS context spine. Objective persists across
  // sessions (a preference); run/layer are an ephemeral working focus (this tab), shown as clearable
  // chips in the context bar so a run picked on /runs stays scoped as you move to /coach or /compare. ──
  T.scopeRun   = () => sessionStorage.getItem("torad_run") || "";
  T.setScopeRun   = r => { r ? sessionStorage.setItem("torad_run", r) : sessionStorage.removeItem("torad_run"); };
  T.scopeLayer = () => sessionStorage.getItem("torad_layer") || "";
  T.setScopeLayer = l => { l ? sessionStorage.setItem("torad_layer", l) : sessionStorage.removeItem("torad_layer"); };

  T.fetchJSON = async url => {
    if (url.startsWith("/api/") && !url.includes("objective=")) {
      url += (url.includes("?") ? "&" : "?") + "objective=" + encodeURIComponent(T.objective());
    }
    const r = await fetch(url);
    const j = await r.json().catch(() => ({ error: "bad json" }));
    if (!r.ok) throw new Error(j.error || ("HTTP " + r.status));
    return j;
  };

  // POST an action to the catalog (same actions the MCP server exposes). Throws the action's actionable
  // error sentence (e.g. the GPU-busy refusal) so callers can show it verbatim.
  T.postAction = async (name, args = {}) => {
    const r = await fetch("/api/action/" + name, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(args),
    });
    const j = await r.json().catch(() => ({ error: "bad json" }));
    if (!r.ok || j.ok === false) { const e = new Error(j.error || ("HTTP " + r.status)); e.kind = j.kind; throw e; }
    return j.result;
  };

  // settling metric: ease a number element from its last value to `to`, writing fmt(v) each frame.
  // Date.now() is unavailable in some sandboxes but fine in the browser; guard anyway.
  T.tween = (el, to, fmt = (v => v.toFixed(0)), dur = 650) => {
    if (!el) return;
    const from = (el.__tv != null) ? el.__tv : (to * 0.0);
    el.__tv = to;
    if (from === to || !window.requestAnimationFrame) { el.textContent = fmt(to); return; }
    const t0 = performance.now();
    const step = now => {
      const p = Math.min(1, (now - t0) / dur);
      const e = 1 - Math.pow(1 - p, 3);            // easeOutCubic — settles, doesn't snap
      el.textContent = fmt(from + (to - from) * e);
      if (p < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  };
  // tween every [data-tween] number in a container to its data-tween value (a "settling instrument")
  T.settle = (root, fmt = (v => (v * 100).toFixed(0) + "%")) => {
    T.$$("[data-tween]", root).forEach(el => { const to = parseFloat(el.getAttribute("data-tween")); if (!Number.isNaN(to)) T.tween(el, to, fmt); });
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
  // single-source from the server (cpt_config.GATE_THRESHOLDS). The literal above is the synchronous
  // default + a drift alarm: if the server ever disagrees we update in place and warn, never silently diverge.
  fetch("/api/gate").then(r => r.json()).then(d => {
    if (d && d.gate) for (const k of ["fabrication", "accuracy", "coverage"]) {
      if (typeof d.gate[k] === "number" && d.gate[k] !== T.GATE[k]) {
        console.warn(`T.GATE.${k}: ${T.GATE[k]} → server ${d.gate[k]} (updated)`); T.GATE[k] = d.gate[k];
      }
    }
  }).catch(() => {});

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

  /* ── the MOAT made visible: an engine badge that travels with every artifact ───────────────
     Native FP4 (Quartet-II) compute + INT8-master 1-byte weight storage = the model trains in the
     precision it ships in on-device. Sourced from the run contract (precision, int8_master, optimizer_lowbit). */
  T.engineBadge = c => {
    if (!c) return "";
    const prec = String(c.precision || "?").toUpperCase();
    const fp4 = /fp4|nvfp4|quartet/i.test(prec);
    const int8 = c.int8_master && c.int8_master !== "off";
    const parts = [fp4 ? "Native FP4" : prec];
    if (fp4) parts.push("Quartet-II");
    if (int8) parts.push("INT8-master" + (c.int8_master === "storage" ? "" : `·${c.int8_master}`));
    if (c.optimizer_lowbit) parts.push(`${c.optimizer_lowbit}-bit opt`);
    return `<span class="engine-badge" title="Trained in the low precision it ships in on-device — no post-hoc quantization cliff between what you gated and what runs on the phone.">${parts.map(p => `<span>${T.esc(p)}</span>`).join("")}</span>`;
  };
  // the plain-language explainer that pairs with the badge (Coach voice)
  T.moatExplainer = c => {
    const fp4 = /fp4|nvfp4|quartet/i.test(String((c || {}).precision || ""));
    const int8 = c && c.int8_master && c.int8_master !== "off";
    if (!fp4 && !int8) return "";
    return `This model was trained in the ${fp4 ? "4-bit (FP4) math" : "low precision"} it ships in${int8 ? ", with weights stored as 1-byte INT8 masters" : ""} — so there's <b>no quantization cliff</b> between what the gate measured and what runs on-device. The numbers you see here are the numbers the phone gets.`;
  };

  /* ── Simple ↔ Expert mode + plain-language glossary ────────────────────────────────────────────
     A training-naive domain expert defaults to Simple Mode: every ML term shows a plain-language alias
     with a [?] tooltip carrying the real term + definition. Expert Mode restores the technical labels
     (today's product) — a real interface for power users, not a warning. Persisted; toggling reloads so
     pages re-render their baked-in terms. Static .simple-only/.expert-only show/hide live via body class. */
  T.mode = () => localStorage.getItem("torad_mode") || "simple";
  T.setMode = m => { localStorage.setItem("torad_mode", m); document.body.classList.remove("mode-simple", "mode-expert"); document.body.classList.add("mode-" + m); };
  T.isSimple = () => T.mode() !== "expert";

  // key → { s: plain (Simple), x: technical (Expert), d: definition (always in the [?] tooltip) }
  T.GLOSSARY = {
    cpt:          { s: "Knowledge Phase",  x: "CPT",          d: "Continued Pre-Training — teaches the model facts from your knowledge base. Runs first; takes longer." },
    sft:          { s: "Behavior Phase",   x: "SFT",          d: "Supervised Fine-Tuning — teaches the model when to answer vs say “I don't know”. Runs after the Knowledge Phase; quick." },
    gate:         { s: "Quality test",     x: "gate",         d: "Checks the model on three things at once: it doesn't make things up, it knows your domain, and it actually answers. All three must pass." },
    fabrication:  { s: "Makes things up",  x: "Fabrication",  d: "How often it invents answers about things that never happened. Lower is better (must stay under 25%)." },
    coverage:     { s: "Answers what it knows", x: "Coverage", d: "How much of what it actually knows it's willing to answer, instead of dodging. Higher is better (at least 40%)." },
    accuracy:     { s: "Gets it right",    x: "Accuracy",     d: "When it does answer, how often it's correct. Higher is better (at least 55%)." },
    install:      { s: "Knowledge depth", x: "Install depth", d: "Whether the facts are really in the model's weights (deep) or it just learned to talk right (shallow). A run can pass the quality test on a shallow install — the trap." },
    calibration:  { s: "Overall balance", x: "Calibration",   d: "A single 0–1 score for how well-balanced the model is across all three checks. Higher is better." },
    hydrate:      { s: "Make teaching examples", x: "Hydrate", d: "Turn structured facts into many varied sentences that teach the same fact — diversity is what installs knowledge." },
    checkpoint:   { s: "Saved snapshot",   x: "Checkpoint",   d: "A saved copy of the model at a point in training." },
    lever:        { s: "What to change",   x: "Lever",        d: "A concrete change to the next run's data or recipe that improves a weak area." },
    loss:         { s: "Training progress", x: "Loss",        d: "A number that drops as the model learns. Trending down = good; flat or rising = trouble." },
    int8_master:  { s: "On-device model", x: "INT8-master",   d: "The model stored in 1-byte weights so it runs efficiently on phones — trained in the precision it ships in, no quality cliff." },
    fp4:          { s: "Phone-ready precision", x: "FP4",      d: "The ultra-efficient 4-bit math the model trains AND ships in — the moat: no post-training quality drop." },
    over_refusal: { s: "Dodges too much", x: "Over-refusal",  d: "Says “I don't know” to things it actually knows. The opposite failure to making things up — both hurt." },
    lr:           { s: "Learning pace",    x: "Learning rate", d: "How big a step the model takes each update. Too high and it destabilizes; too low and it barely learns." },
    max_steps:    { s: "Training length",  x: "Max steps",     d: "How many update steps to train for. More steps = more learning, up to a point — then it overfits." },
    absent_neg:   { s: "“Not real” examples", x: "Absent negatives", d: "Made-up entities the model is taught to refuse. More of them cuts fabrication but can push it toward dodging." },
    peakmem:      { s: "Peak GPU memory", x: "Peak memory",    d: "The most GPU memory the run used. Lower means it fits on smaller hardware — the INT8-master win." },
    sps:          { s: "Training speed",   x: "Steps/sec",     d: "How many update steps run per second — the throughput of the GPU during training." },
    p1:           { s: "Top answer is right", x: "P@1",        d: "Precision@1 — how often the single best retrieved fact is the correct one. Higher is better." },
    mrr:          { s: "Right answer ranks high", x: "MRR",    d: "Mean Reciprocal Rank — rewards putting the correct fact near the top of the list. Higher is better." },
    auc:          { s: "Tells known from unknown", x: "AUC",   d: "How cleanly the model separates facts it knows from ones it doesn't, by confidence. 1.0 is perfect, 0.5 is a coin flip." },
    separation:   { s: "Knows what it knows", x: "Basin separation", d: "The confidence gap between known and unknown facts. A wide gap means the knowledge is really installed, not faked." },
    triples:      { s: "Structured facts", x: "Triples",       d: "Facts written as subject–relation–object (e.g. “EDC — held in — Las Vegas”) — the structured form before hydration." },
    replay:       { s: "Old-knowledge mix", x: "Replay mix",   d: "A fraction of general text mixed into training so the model keeps its broad abilities while learning your domain." },
  };
  // mode-aware word (no tooltip) — for titles / nav / short labels
  T.word = key => { const g = T.GLOSSARY[key]; return g ? (T.isSimple() ? g.s : g.x) : key; };
  // mode-aware term WITH a [?] glossary tooltip — for inline body text
  T.term = key => { const g = T.GLOSSARY[key]; if (!g) return T.esc(key);
    return `<span class="term hint" data-tip="${T.esc((T.isSimple() ? g.x + ": " : "") + g.d)}">${T.esc(T.isSimple() ? g.s : g.x)}<sup>?</sup></span>`; };
  // a bare [?] help marker carrying any explanation (not tied to the glossary)
  T.help = tip => `<span class="term hint" data-tip="${T.esc(tip)}"><sup>?</sup></span>`;

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
  // Journey-ordered nav (the SaaS spine): reads top-to-bottom as the user's path —
  // PREPARE → TRAIN → TEST → ANALYZE → SHIP → LEARN. Each item carries a one-line tip; Simple Mode
  // shows plain labels (labelS), Expert shows the technical name.
  const NAV = [
    { href: "/", page: "home", ic: "◈", label: "Overview", tip: "Your dashboard — where you are in the journey" },
    { sep: "Prepare" },
    { href: "/build", page: "build", ic: "⊕", label: "Build", labelS: "Build data", tip: "Drop documents and run the autonomous data pipeline" },
    { href: "/data", page: "data", ic: "▤", label: "Corpus", labelS: "Your data", tip: "See whether your knowledge made it into the training set" },
    { sep: "Train" },
    { href: "/train", page: "train", ic: "◉", label: "Train", tip: "Launch a run and watch the model learn" },
    { href: "/runs", page: "runs", ic: "≣", label: "Runs", tip: "Every model you've trained, with its quality-test result" },
    { href: "/sweeps", page: "sweeps", ic: "⋔", label: "Sweeps", labelS: "Tuning", tip: "Try different settings to make the model better (optional)" },
    { sep: "Test" },
    { href: "/playground", page: "playground", ic: "◐", label: "Playground", labelS: "Test the model", tip: "Ask the model questions about your domain" },
    { sep: "Analyze" },
    { href: "/coach", page: "coach", ic: "✦", label: "Coach", tip: "Plain-language read of why a run passed or failed, and what to fix" },
    { href: "/compare", page: "compare", ic: "⇄", label: "Compare", tip: "See what changed between two models" },
    { sep: "Ship" },
    { href: "/registry", page: "registry", ic: "◆", label: "Registry", labelS: "Ship", tip: "Export a passing model to run in your app" },
    { sep: "Learn" },
    { href: "/docs", page: "docs", ic: "¶", label: "Best practices", labelS: "Guide", tip: "The rules behind the quality checks" },
  ];

  T.mountShell = ({ page, crumbs } = {}) => {
    document.body.classList.add("mode-" + T.mode());   // drives .simple-only / .expert-only live
    const app = T.el("div", "app");
    const side = T.el("aside", "side");
    const simple = T.isSimple();
    side.innerHTML = `
      <div class="brand">
        <div class="mark"><span class="dot"></span>Torad&nbsp;Train</div>
        <div class="tag">on-device model trainer</div>
        <div class="modetg tg" id="modeTg" title="Simple Mode shows plain language; Expert Mode shows the ML terms">
          <button data-mode="simple" class="${simple ? "on" : ""}">Simple</button><button data-mode="expert" class="${simple ? "" : "on"}">Expert</button></div>
      </div>
      <div class="objbar"><div class="lab" style="margin-bottom:5px">Objective</div>
        <select class="objsel" id="objSel"><option>${T.esc(T.objective())}</option></select></div>
      <nav>${NAV.map(n => n.sep
        ? `<div class="sep">${n.sep}</div>`
        : `<a href="${n.href}" class="${n.page === page ? "active" : ""}" title="${T.esc(n.tip || "")}"><span class="ic">${n.ic}</span>${T.esc(simple && n.labelS ? n.labelS : n.label)}</a>`).join("")}</nav>
      <div class="foot"><span class="live" id="liveDot"></span><span id="liveTxt">idle</span></div>`;
    const main = T.el("main", "main");
    // auto-scope the run from the URL so the focus follows you across pages (Runs → Coach → Compare)
    const urlRun = T.qs("id") || T.qs("run"); if (urlRun) T.setScopeRun(urlRun);
    main.appendChild(T._ctxBar());   // persistent context bar: objective · run · layer · ⌘K
    if (crumbs) {
      const c = T.el("div", "crumbs");
      c.innerHTML = crumbs.map((x, i) => (i ? '<span class="s">/</span>' : "") + (x.href ? `<a href="${x.href}">${T.esc(x.label)}</a>` : `<span>${T.esc(x.label)}</span>`)).join("");
      main.appendChild(c);
    }
    const slot = T.el("div", "slot"); main.appendChild(slot);
    app.appendChild(side); app.appendChild(main);
    document.body.appendChild(app);
    T._wirePalette();   // ⌘K / Ctrl-K command palette (jump to any page / run / objective)
    // Simple ↔ Expert mode toggle — reload so every page re-renders its baked-in terms in the new voice
    T.$$("#modeTg button").forEach(b => b.onclick = () => { if (b.dataset.mode !== T.mode()) { T.setMode(b.dataset.mode); location.reload(); } });
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
  // recover: optional next-step HTML (a link / instruction) shown under the error so a dead end becomes a path forward.
  T.err = (slot, msg, recover) => { slot.innerHTML = `<div class="errbox">⚠ ${T.esc(msg)}${recover ? `<div class="err-recover">${recover}</div>` : ""}</div>`; };
  // an empty-state with a next step — distinct from an error: nothing went wrong, there's just nothing here yet.
  T.empty = (msg, recover) => `<div class="empty">${T.esc(msg)}${recover ? `<div class="err-recover">${recover}</div>` : ""}</div>`;

  /* ── context bar: the shared {objective · run · layer} scope, always visible ───────────
     Chips reflect the current focus; the run chip links to its detail and is clearable, so
     the scope you set on /runs rides with you to /coach and /compare. ⌘K opens the palette. */
  T._renderCtx = bar => {
    const run = T.scopeRun(), layer = T.scopeLayer();
    const chips = [`<span class="ctx-chip obj" title="Objective — persisted across sessions"><i>objective</i><b>${T.esc(T.objective())}</b></span>`];
    if (run) chips.push(`<a class="ctx-chip run" href="/run?id=${encodeURIComponent(run)}" title="Scoped run"><i>run</i><b>${T.esc(run)}</b><span class="cx" data-clear="run" title="clear">✕</span></a>`);
    if (layer) chips.push(`<span class="ctx-chip layer" title="Scoped layer"><i>layer</i><b>${T.esc(layer)}</b><span class="cx" data-clear="layer" title="clear">✕</span></span>`);
    bar.innerHTML = `<div class="ctx-chips">${chips.join("")}</div>
      <button class="ctx-k" title="Command palette (⌘K)"><kbd>⌘</kbd><kbd>K</kbd><span>Jump to…</span></button>`;
    bar.querySelectorAll(".cx").forEach(x => x.onclick = e => {
      e.preventDefault(); e.stopPropagation();
      x.dataset.clear === "run" ? T.setScopeRun("") : T.setScopeLayer("");
      T._renderCtx(bar);
    });
    bar.querySelector(".ctx-k").onclick = () => T.openPalette();
  };
  T._ctxBar = () => { const bar = T.el("div", "ctxbar"); T._renderCtx(bar); return bar; };

  /* ── ⌘K command palette: jump to any page / run / objective, or compare against the scoped run ── */
  T._wirePalette = () => {
    if (T._palWired) return; T._palWired = true;
    document.addEventListener("keydown", e => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") { e.preventDefault(); T.openPalette(); }
    });
  };

  T._paletteItems = async () => {
    const items = [];
    for (const n of NAV) if (!n.sep) items.push({ ic: n.ic, label: n.label, sub: "Page", go: () => location.href = n.href });
    const cur = T.scopeRun();
    try {
      const { runs } = await T.fetchJSON("/api/runs");
      for (const r of (runs || [])) {
        if (!r.id) continue;
        items.push({ ic: "≣", label: r.id, sub: "Open run" + (r.status ? " · " + r.status : ""), go: () => location.href = "/run?id=" + encodeURIComponent(r.id) });
        if (cur && r.id !== cur) items.push({ ic: "⇄", label: `${cur} ⇄ ${r.id}`, sub: "Compare runs", go: () => location.href = `/compare?runs=${encodeURIComponent(cur)},${encodeURIComponent(r.id)}` });
      }
    } catch (e) { /* runs unavailable — pages still work */ }
    try {
      const { objectives } = await fetch("/api/objectives").then(r => r.json());
      for (const o of (objectives || [])) items.push({ ic: "◈", label: o.name, sub: "Switch objective", go: () => { T.setObjective(o.id); location.reload(); } });
    } catch (e) { /* switcher unavailable */ }
    if (cur) items.push({ ic: "✕", label: "Clear run scope", sub: "Action", go: () => { T.setScopeRun(""); location.reload(); } });
    if (T.scopeLayer()) items.push({ ic: "✕", label: "Clear layer scope", sub: "Action", go: () => { T.setScopeLayer(""); location.reload(); } });
    return items;
  };

  T.openPalette = async () => {
    if (T.$(".pal-wrap")) return;
    const prevFocus = document.activeElement;   // restore focus here on close (a11y: don't strand the user)
    const wrap = T.el("div", "pal-wrap");
    wrap.innerHTML = `<div class="pal" role="dialog" aria-modal="true" aria-label="Command palette">
      <input class="pal-in" type="text" name="palette" placeholder="Jump to a page, run, or objective…" aria-label="Jump to a page, run, or objective" autocomplete="off" spellcheck="false">
      <div class="pal-list"></div>
      <div class="pal-foot"><kbd>↑</kbd><kbd>↓</kbd> navigate&nbsp;&nbsp;<kbd>↵</kbd> open&nbsp;&nbsp;<kbd>esc</kbd> close</div></div>`;
    document.body.appendChild(wrap);
    const inp = wrap.querySelector(".pal-in"), list = wrap.querySelector(".pal-list");
    let items = [{ ic: "·", label: "Loading…", sub: "", go: () => {} }], shown = items, sel = 0;
    const close = () => { wrap.remove(); if (prevFocus && prevFocus.focus) prevFocus.focus(); };
    const paint = () => list.querySelectorAll(".pal-it").forEach((el, i) => el.classList.toggle("on", i === sel));
    const ensureVis = () => { const el = list.querySelector(".pal-it.on"); if (el) el.scrollIntoView({ block: "nearest" }); };
    const render = () => {
      list.innerHTML = shown.length ? shown.map((it, i) => `<div class="pal-it ${i === sel ? "on" : ""}" data-i="${i}">
        <span class="pi-ic">${it.ic}</span><span class="pi-l">${T.esc(it.label)}</span><span class="pi-s">${T.esc(it.sub)}</span></div>`).join("")
        : `<div class="pal-empty">no matches</div>`;
      list.querySelectorAll(".pal-it").forEach(el => {
        el.onmouseenter = () => { sel = +el.dataset.i; paint(); };
        el.onclick = () => { shown[+el.dataset.i].go(); close(); };
      });
    };
    const filter = () => {
      const q = inp.value.trim().toLowerCase();
      shown = q ? items.filter(it => (it.label + " " + it.sub).toLowerCase().includes(q)) : items;
      sel = 0; render();
    };
    render(); inp.focus();
    inp.oninput = filter;
    wrap.onclick = e => { if (e.target === wrap) close(); };
    // wire keys BEFORE the await so Escape/arrows work during the brief item-load too
    document.addEventListener("keydown", function onkey(e) {
      if (!document.body.contains(wrap)) { document.removeEventListener("keydown", onkey); return; }
      if (e.key === "Escape") { e.preventDefault(); close(); }
      else if (e.key === "Tab") { e.preventDefault(); inp.focus(); }   // focus trap: the input is the only control
      else if (e.key === "ArrowDown") { e.preventDefault(); sel = Math.min(shown.length - 1, sel + 1); paint(); ensureVis(); }
      else if (e.key === "ArrowUp") { e.preventDefault(); sel = Math.max(0, sel - 1); paint(); ensureVis(); }
      else if (e.key === "Enter") { e.preventDefault(); if (shown[sel]) { shown[sel].go(); close(); } }
    });
    items = await T._paletteItems(); filter();
  };

})(window.T = window.T || {});

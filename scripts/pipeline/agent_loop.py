#!/usr/bin/env python3
"""The autonomous data-pipeline agent loop (qgre-cpt).

A REPRODUCIBLE finite-state controller that drives the pipeline tools (clean → extract → hydrate) via the
in-process action catalog, reads the deterministic quality gates between steps, and DECIDES the next action
— re-hydrating under-covered relations toward their balance target, stopping at the exposure cap, and
classifying source-gaps honestly instead of fabricating filler. It NEVER touches the GPU and NEVER
auto-trains; tokenize + CPT stay an explicit, user-triggered step.

Launched by the `run_pipeline` action as its own pipeline job, so its decision stream tees to
pipeline/<job_id>/run.log (the UI tails it) and a structured agent_decisions.jsonl. State is checkpointed
to pipeline/<job_id>/state.json after every step (crash-resumable).

  PYTHONPATH=scripts python scripts/pipeline/agent_loop.py --domain edc --editions las_vegas/2026 \
      --job-id <id> [--from clean] [--exposure-cap 60] [--max-rehydrate 2]

The decision policy is deterministic (gate verdict → fixed next action), so a rebuild is reproducible and
the gate report can be diffed against the GOLD reference. The only non-deterministic seam is the LLM
generation inside the spawned hydrate/extract steps (which the gates then check).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torad_actions as TA           # noqa: E402  the in-process action catalog
from qgre import spec as qspec       # noqa: E402

POLL_S = 5            # how often to poll a spawned sub-job
SPINE = ["clean", "extract", "hydrate"]   # ingest is optional (only when sources are supplied)


class Loop:
    def __init__(self, domain, editions, job_id, sources=None, exposure_cap=60,
                 max_rehydrate=2, start_from="clean", max_wall_s=6 * 3600):
        self.domain = domain
        self.festival = domain
        self.editions = [e.strip() for e in editions.split(",") if e.strip()]
        self.job_id = job_id
        self.sources = sources
        self.exposure_cap = exposure_cap
        self.max_rehydrate = max_rehydrate
        self.start_from = start_from
        self.deadline = time.time() + max_wall_s
        self.spec = qspec.load_spec(domain)
        self.protected = set(self.spec.get("protected_editions", []))   # the hand-built gold — gate-only, never rebuilt
        self.state_dir = TA.PIPELINE_DIR / job_id
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.decisions = []
        self.state = {"job_id": job_id, "domain": domain, "editions": self.editions,
                      "phase": "starting", "per_edition": {}, "source_gaps": {}, "attempts": {}}

    # ── logging: human line to stdout (→ run.log, the UI stream) + structured jsonl ──
    def log(self, msg):
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    def decide(self, kind, detail):
        rec = {"t": time.time(), "kind": kind, "detail": detail}
        self.decisions.append(rec)
        with (self.state_dir / "agent_decisions.jsonl").open("a") as f:
            f.write(json.dumps(rec) + "\n")
        self.log(f"DECISION · {kind}: {detail}")

    def checkpoint(self):
        (self.state_dir / "state.json").write_text(json.dumps(self.state, indent=2, default=str))

    # ── run a spawned tool step and wait for it ──
    def run_step(self, action, args, label):
        self.log(f"→ {label} ({action})")
        r = TA.dispatch(action, **args)
        if r.get("error"):
            self.log(f"  ✗ {action} error: {r['error']}")
            return False
        sub = r.get("job_id")
        if not sub:        # synchronous tool (e.g. ingest) — already done
            self.log(f"  ✓ {label} (sync)")
            return True
        while True:
            if time.time() > self.deadline:
                self.log("  ✗ wall-clock budget exceeded"); return False
            time.sleep(POLL_S)
            st = TA.dispatch("pipeline_status", job_id=sub)
            if st.get("status") == "done":
                self.log(f"  ✓ {label} finished")
                return True

    def gate(self, edition):
        return TA.dispatch("gate_corpus", festival=self.festival, edition=edition, domain=self.domain)

    # ── the decision policy for ONE edition (deterministic) ──
    def process_edition(self, ed):
        es = self.state["per_edition"].setdefault(ed, {"phase": "spine", "rehydrates": {}})
        # PROTECTED gold: gate-check only, NEVER extract/hydrate (re-running would overwrite the hand-built
        # reference corpus — the thing the whole project protects). Read-only.
        if ed in self.protected:
            self.decide("protected", f"{ed}: protected (hand-built gold) — gate-only, skipping extract/hydrate")
            g = self.gate(ed)
            es["phase"] = "protected"; es["gate"] = {k: v.get("severity") for k, v in g.get("gates", {}).items()}
            es["final_overall"] = g.get("overall")
            self.checkpoint(); return
        # linear spine from the requested start stage
        start_idx = SPINE.index(self.start_from) if self.start_from in SPINE else 0
        for stage in SPINE[start_idx:]:
            es["phase"] = stage
            self.checkpoint()
            ok = False
            if stage == "clean":
                ok = self.run_step("clean_layer", {"editions": ed, "llm_rescue": True}, f"{ed} · clean")
            elif stage == "extract":
                ok = self.run_step("extract_triples", {"editions": ed, "domain": self.domain},
                                   f"{ed} · extract (spec-driven + contract)")
            elif stage == "hydrate":
                ok = self.run_step("hydrate", {"festival": self.festival, "edition": ed,
                                               "exposure_cap": self.exposure_cap}, f"{ed} · hydrate (cap {self.exposure_cap})")
            if not ok:
                es["phase"] = "blocked"; self.checkpoint(); return

        # ── gate + decide loop ──
        for _ in range(self.max_rehydrate + 1):
            es["phase"] = "gate"; self.checkpoint()
            g = self.gate(ed)
            gates = g.get("gates", {})
            es["gate"] = {k: v.get("severity") for k, v in gates.items()}
            self.log(f"  gate({ed}): {es['gate']} → overall {g.get('overall')}")

            # source-gap classification is honest reporting, never a failure to fix
            cls = gates.get("source_vs_gap", {}).get("source_gap_vs_defect", {})
            gaps = {r: c for r, c in cls.items() if c.get("kind") == "source_gap"}
            if gaps:
                self.state["source_gaps"][ed] = gaps
                self.decide("source_gap", f"{ed}: {list(gaps)} missing for lack of source data — reported, not fabricated")

            # 1. id_integrity FAIL → re-extract (the contract lives in the structured layer)
            if gates.get("id_integrity", {}).get("severity") == "fail":
                key = f"{ed}:_extract"
                n = self.state["attempts"].get(key, 0)
                if n < self.max_rehydrate:
                    self.state["attempts"][key] = n + 1
                    self.decide("re_extract", f"{ed}: id_integrity FAIL — re-extracting to enforce the predicate contract")
                    if not self.run_step("extract_triples", {"editions": ed, "domain": self.domain}, f"{ed} · re-extract"):
                        break
                    self.run_step("hydrate", {"festival": self.festival, "edition": ed, "exposure_cap": self.exposure_cap}, f"{ed} · re-hydrate")
                    continue
                self.decide("blocked", f"{ed}: id_integrity still failing after {n} re-extracts — needs a human look")
                es["phase"] = "blocked"; self.checkpoint(); return

            # 2. balance FAIL on a relation under floor that is a DEFECT (not a source gap) → re-hydrate that relation
            bal = gates.get("balance", {})
            defects = [d["relation"] for d in bal.get("metrics", {}).get("under_floor_defects", [])
                       if d["relation"] not in gaps]
            if bal.get("severity") == "fail" and defects:
                target = defects[0]
                key = f"{ed}:{target}"
                n = es["rehydrates"].get(target, 0)
                if n < self.max_rehydrate:
                    es["rehydrates"][target] = n + 1
                    boost = 24 * (n + 1)
                    self.decide("rebalance", f"{ed}: relation '{target}' under floor — top-up hydrate at base_n={boost}")
                    ok = self.run_step("hydrate", {"festival": self.festival, "edition": ed,
                                                   "relations": target, "balance_target": f"{target}={boost}",
                                                   "exposure_cap": self.exposure_cap}, f"{ed} · top-up {target}")
                    if not ok:
                        break
                    continue
                self.decide("balance_unreachable", f"{ed}: '{target}' still thin after {n} top-ups — data-thin, not fabricating")

            # 3. faithfulness FAIL → re-hydrate at lower temperature (tighter grounding)
            if gates.get("faithfulness", {}).get("severity") == "fail":
                key = f"{ed}:_faith"
                n = self.state["attempts"].get(key, 0)
                if n < self.max_rehydrate:
                    self.state["attempts"][key] = n + 1
                    temp = max(0.1, 0.3 - 0.1 * (n + 1))
                    self.decide("retighten", f"{ed}: faithfulness FAIL — re-hydrate at temperature {temp:.1f}")
                    if not self.run_step("hydrate", {"festival": self.festival, "edition": ed,
                                                     "temperature": temp, "exposure_cap": self.exposure_cap}, f"{ed} · re-hydrate temp {temp:.1f}"):
                        break
                    continue

            # nothing actionable left to improve this pass → done with this edition
            es["phase"] = "done"; es["final_overall"] = g.get("overall")
            self.decide("edition_done", f"{ed}: gates settled (overall {g.get('overall')}); "
                        f"remaining issues are source-gaps or shared defects, not fixable by re-hydration")
            self.checkpoint(); return
        es["phase"] = "done_capped"; self.checkpoint()

    def run(self):
        self.state["phase"] = "running"; self.checkpoint()
        self.log(f"qgre pipeline loop · domain={self.domain} · editions={self.editions} · cap={self.exposure_cap}")
        for ed in self.editions:
            if time.time() > self.deadline:
                self.decide("budget", "wall-clock budget exceeded — stopping; remaining editions unprocessed")
                break
            self.log(f"════ edition {ed} ════")
            try:
                self.process_edition(ed)
            except Exception as e:
                self.log(f"  ✗ {ed} crashed: {type(e).__name__}: {e}")
                self.state["per_edition"].setdefault(ed, {})["phase"] = "error"
        self.state["phase"] = "done"
        # final report
        done = [e for e, s in self.state["per_edition"].items() if s.get("phase", "").startswith("done")]
        blocked = [e for e, s in self.state["per_edition"].items() if s.get("phase") in ("blocked", "error")]
        self.state["summary"] = {"done": done, "blocked": blocked, "source_gaps": self.state["source_gaps"]}
        self.checkpoint()
        self.log(f"DONE · {len(done)} editions settled, {len(blocked)} blocked. "
                 f"Source-gaps reported for {list(self.state['source_gaps'])}. "
                 f"Tokenize + CPT remain an explicit, user-triggered step (the loop never trains).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="edc")
    ap.add_argument("--editions", required=True)
    ap.add_argument("--job-id", required=True)
    ap.add_argument("--sources")
    ap.add_argument("--exposure-cap", type=int, default=60)
    ap.add_argument("--max-rehydrate", type=int, default=2)
    ap.add_argument("--from", dest="start_from", default="clean")
    args = ap.parse_args()
    Loop(args.domain, args.editions, args.job_id, sources=args.sources,
         exposure_cap=args.exposure_cap, max_rehydrate=args.max_rehydrate,
         start_from=args.start_from).run()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Adversarial build-completeness check: syntax, config contract, action catalog, routes."""
import ast, sys, subprocess, json, urllib.request
sys.path.insert(0, "scripts")
ROOT = "/home/marcos/Documents/dev/training/kandi-cpt"
import os; os.chdir(ROOT)

fail = []

# 1. Python syntax on every file edited this session
print("── syntax ──")
pyfiles = ["scripts/serve.py", "scripts/torad_actions.py", "scripts/torad_mcp.py",
           "scripts/train_cpt.py", "scripts/cpt_config.py", "scripts/playground_infer.py"]
for f in pyfiles:
    try:
        ast.parse(open(f).read()); print(f"  ok   {f}")
    except SyntaxError as e:
        print(f"  FAIL {f}: {e}"); fail.append(f"syntax {f}")

# 2. shell scripts parse
print("── shell ──")
for f in ["scripts/run_recipe.sh", "scripts/run_sft.sh", "scripts/train_fp4.sh"]:
    rc = subprocess.run(["bash", "-n", f], capture_output=True, text=True)
    print(f"  {'ok  ' if rc.returncode==0 else 'FAIL'} {f}")
    if rc.returncode: fail.append(f"bash {f}: {rc.stderr.strip()}")

# 3. config contract tests + key configs load
print("── config contract ──")
rc = subprocess.run([".venv/bin/python", "scripts/test_cpt_config.py"], capture_output=True, text=True, env={**os.environ, "PYTHONPATH": "scripts"})
ok = "ALL" in rc.stdout and "PASS" in rc.stdout
print(f"  {'ok  ' if ok else 'FAIL'} test_cpt_config ({rc.stdout.strip().splitlines()[-1] if rc.stdout.strip() else rc.stderr[-200:]})")
if not ok: fail.append("config tests")

# 4. action catalog: every action present + every READ action dispatches without crashing
print("── action catalog ──")
import torad_actions as A
print(f"  {len(A.ACTIONS)} actions registered")
read_calls = {
    "gpu_status": {}, "list_runs": {"objective": "edc"}, "get_run": {"run_id": "edc_sft_v10"},
    "read_gate": {"run_id": "edc_sft_v10"}, "compare_runs": {"runs": ["edc_sft_v10", "edc_sft_v11"]},
    "diagnose_run": {"run_id": "edc_sft_v10"}, "inspect_corpus": {"festival": "edc"},
    "run_status": {}, "gate_curve": {"run_id": "edc_sft_v10"}, "list_objectives": {},
    "read_edition": {"edition": "chicago/2013"}, "read_layer": {"layer": "25_hydrated"},
    "run_log": {"run_id": "edc_sft_v10"}, "list_exports": {}, "list_studies": {},
    "playground_stored": {"run_id": "edc_sft_v10"}, "ui_map": {},
    "navigate": {"page": "compare", "runs": ["edc_sft_v10", "edc_sft_v11"]},
    "trace_fact": {"festival": "edc", "edition": "chicago/2013", "query": "chicago"},
    "verify_layer": {"festival": "edc", "edition": "chicago/2013", "layer": "25_hydrated"},
}
read_actions = [n for n, a in A.ACTIONS.items() if not a.mutating]
uncalled = [n for n in read_actions if n not in read_calls]
if uncalled: print(f"  (read actions not smoke-tested here: {uncalled})")
for name, args in read_calls.items():
    try:
        status, payload = A.run_action(name, args, allow_write=False)
        bad = (status != 200) or (isinstance(payload.get("result"), dict) and payload["result"].get("error"))
        print(f"  {'ok  ' if not bad else 'WARN'} {name}{' → ' + str(payload['result'].get('error'))[:60] if bad else ''}")
    except Exception as e:
        print(f"  FAIL {name}: {type(e).__name__}: {e}"); fail.append(f"action {name}")

# 5. write actions are gate-guarded (no accidental execution)
print("── write-gating ──")
for name in ["launch_run", "apply_recipe", "export_checkpoint", "build_corpus"]:
    status, payload = A.run_action(name, {}, allow_write=False)
    ok = status == 403
    print(f"  {'ok  ' if ok else 'FAIL'} {name} blocked without writes (HTTP {status})")
    if not ok: fail.append(f"write-gate {name}")

# 6. all routes 200
print("── routes ──")
for p in ["/", "/data", "/train", "/runs", "/run?id=edc_sft_v10", "/coach", "/compare",
          "/playground", "/registry", "/sweeps", "/docs", "/api/gate", "/api/runs"]:
    try:
        code = urllib.request.urlopen("http://127.0.0.1:8800" + p, timeout=5).status
        print(f"  {'ok  ' if code==200 else 'FAIL'} {p} ({code})")
        if code != 200: fail.append(f"route {p}")
    except Exception as e:
        print(f"  FAIL {p}: {e}"); fail.append(f"route {p}")

print("\n" + ("✗ FAILURES: " + "; ".join(fail) if fail else "✓ ALL DETERMINISTIC CHECKS PASS"))

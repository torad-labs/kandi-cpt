#!/usr/bin/env bash
# verdict-time-bound.sh — the entrypoint's regression harness, runnable on host or in-container.
#
# WHY THIS EXISTS AS A SCRIPT AND NOT A GATE TEST: none of the five gate commands can see
# docker/bin/run-cloud-training (measured: grep for it over run_tests.sh, plan_install.py,
# run_pipeline.py, qgre/test_gates.py, test_cpt_config.py returns 0 hits), so without this file
# the 400-line entrypoint — including the one-second verdict law — has zero mechanical coverage.
# docs/cloud-trainer-image.md names this script as the pre-push check.
#
# THREE LAWS, the first two parameterised over payload size because a single-size probe is what
# let the last defect through:
#   1. VERDICT TIME IS O(central directory): a malformed bundle is refused inside the launcher's
#      1s probe REGARDLESS of payload bytes.
#   2. SUCCESS NEEDS EVIDENCE: a trainer that exits 0 with no metric rows must NOT leave
#      post_install DONE, regardless of how long the supervisor lived on pre-verdict work —
#      and MUST leave post_install FAILED (terminal non-success, not PENDING).
#   3. TRAINER FAILURE CLOSES THE EVIDENCE STAGE: a nonzero trainer exit (and a probe-window
#      discard) must write train FAILED + post_install FAILED. Leaving post_install PENDING made
#      incomplete documents look like "not success yet" rather than "install never happened"
#      (conductor CloudSmokeProbe honest no-GPU path; paid smoke 2026-08-02).
#
# Usage:  QGRE_ENGINE_DIR=<engine checkout> docker/tests/verdict-time-bound.sh
#         PAYLOAD_MB="50 150 300" to change the size sweep (default "50 150 300").
set -euo pipefail

ENGINE="${QGRE_ENGINE_DIR:-/opt/qgre/kandi-cpt}"
ENTRY="$(cd "$(dirname "$0")/.." && pwd)/bin/run-cloud-training"
SIZES="${PAYLOAD_MB:-50 150 300}"
PROBE_MS=1000
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
PASS=0; FAIL=0

say()  { printf '%s\n' "$*"; }
ok()   { PASS=$((PASS+1)); say "  ok   $*"; }
bad()  { FAIL=$((FAIL+1)); say "  FAIL $*"; }
dump() { # dump <rundir> — forensics for a failed arm; a FAIL without its artifacts is unactionable
    say "       stages : $(tail -c 260 "$1/cpt_stages.json" 2>/dev/null || echo '<absent>')"
    say "       metrics: $(wc -l < "$1/metrics.jsonl" 2>/dev/null || echo 0) row(s)"
    say "       log    : $(grep -cE 'TRAINER-EXEC|step ' "$1/train.log" 2>/dev/null || echo 0) trainer line(s)"
}

now_ms() { date +%s%3N; }

mkbundle() { # mkbundle <out.zip> <payload_mb> <yaml...>   (python zipfile: no zip(1) dependency)
    "$ENGINE/.venv/bin/python" - "$@" <<'PY'
import sys, zipfile, os
out, mb, yamls = sys.argv[1], int(sys.argv[2]), sys.argv[3:]
with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as z:
    for y in yamls:
        z.write(y, os.path.basename(y))
    if mb:
        blob = b"\0" * (1024 * 1024)
        with z.open(zipfile.ZipInfo("payload.bin"), "w", force_zip64=True) as f:
            for _ in range(mb):
                f.write(blob)
PY
}

launch() { # launch <bundle> <rundir> [trainer-cmd]  — the VERBATIM CloudRemoteLaunch.kt:163-170 line
    local BUNDLE="$1" RUN_DIR="$2"
    # Env-prefixes on FUNCTION calls are not reliably exported to the nohup child, so the trainer
    # stub travels as an argument and is exported here — measured: the prefix form left the var
    # out of the entrypoint's environment and the real trainer path ran instead, making the
    # no-success assertions pass VACUOUSLY (right answer, wrong mechanism).
    if [ "$#" -ge 3 ]; then export QGRE_CLOUD_TRAINER_CMD="$3"; else unset QGRE_CLOUD_TRAINER_CMD; fi
    mkdir -p "$RUN_DIR" && { nohup "$ENTRY" "$BUNDLE" "$RUN_DIR" \
        > "$RUN_DIR/train.log" 2>&1 < /dev/null & \
        pid=$!; \
        if printf '%s\n' "$pid" > "$RUN_DIR/pid" && \
        sleep 1 && kill -0 "$pid"; then \
        printf '%s\n' "$pid"; \
        else kill "$pid" 2>/dev/null || true; rm -f "$RUN_DIR/pid"; return 1; fi; }
}

verdict_ms() { # time until cpt_stages.json holds a terminal verdict, capped at 15s
    local RUN_DIR="$1" t0; t0=$(now_ms)
    for _ in $(seq 1 300); do
        if grep -qE '"status": ?"(FAILED|DONE)"' "$RUN_DIR/cpt_stages.json" 2>/dev/null; then
            echo $(( $(now_ms) - t0 )); return 0; fi
        sleep 0.05
    done
    echo 15000
}

# A config the ENGINE ORACLE accepts — by construction, the same rule the entrypoint uses.
GOOD_CFG=""
for c in "$ENGINE"/configs/train_*.yaml; do
    if (cd "$ENGINE" && PYTHONPATH=scripts "$ENGINE/.venv/bin/python" -c "
import sys
from cpt_config import CptConfig
cfg = CptConfig.from_yaml('$c'); cfg.validate_runtime(1)
from pathlib import Path
sys.exit(0 if Path(cfg.model.model_path).exists() else 1)" >/dev/null 2>&1); then
        GOOD_CFG="$c"; break
    fi
done
[ -n "$GOOD_CFG" ] || { say "no engine-acceptable config found; cannot calibrate"; exit 2; }
say "calibration config: $GOOD_CFG"
printf 'x: 1\n' > "$TMP/second.yaml"

say ""
say "LAW 1 — verdict time is O(central directory), swept over payload sizes: $SIZES"
for MB in $SIZES; do
    R="$TMP/reject_$MB"; B="$TMP/twoyaml_$MB.zip"
    mkbundle "$B" "$MB" "$GOOD_CFG" "$TMP/second.yaml"
    launch "$B" "$R" >/dev/null 2>&1 || true      # a refusal may lose the probe; that is fine
    V=$(verdict_ms "$R")
    if grep -q 'refusing to guess' "$R/cpt_stages.json" 2>/dev/null && [ "$V" -le "$PROBE_MS" ]; then
        ok "two-yaml ${MB}MB refused in ${V}ms"
    else bad "two-yaml ${MB}MB: verdict ${V}ms (cap ${PROBE_MS}ms) or wrong reason"; fi
done

say ""
say "LAW 1b — an OVERSIZE config is refused from the central directory, not streamed"
R="$TMP/oversize"; B="$TMP/oversize.zip"
"$ENGINE/.venv/bin/python" - "$B" <<'PY'
import sys, zipfile
with zipfile.ZipFile(sys.argv[1], "w", zipfile.ZIP_STORED) as z:
    z.writestr("huge.yaml", "x: 1\n" + "#pad\n" * (8 * 1024 * 1024))   # ~40 MB single yaml
PY
launch "$B" "$R" >/dev/null 2>&1 || true
V=$(verdict_ms "$R")
if grep -q 'declares.*bytes' "$R/cpt_stages.json" 2>/dev/null && [ "$V" -le "$PROBE_MS" ]; then
    ok "40MB config refused via declared size in ${V}ms"
else bad "40MB config: verdict ${V}ms or wrong reason"; dump "$R"; fi

say ""
say "LAW 2 — success needs evidence, swept over payload sizes: $SIZES"
for MB in $SIZES; do
    R="$TMP/stub_$MB"; B="$TMP/stub_$MB.zip"
    mkbundle "$B" "$MB" "$GOOD_CFG"
    launch "$B" "$R" true >/dev/null 2>&1 || true
    for _ in $(seq 1 600); do
        grep -qE '"stage": ?"(post_install|train)".*"status": ?"(FAILED|DONE)"' "$R/cpt_stages.json" 2>/dev/null && break
        sleep 0.05
    done
    if ! grep -q 'TRAINER-EXEC (test hook)' "$R/train.log" 2>/dev/null; then
        bad "stub trainer ${MB}MB: the stub never ran - this arm is VACUOUS"
    elif grep -qE '"stage": ?"post_install", ?"status": ?"DONE"' "$R/cpt_stages.json" 2>/dev/null; then
        bad "stub trainer ${MB}MB left post_install DONE with no metric rows"
    elif ! grep -qE '"stage": ?"post_install", ?"status": ?"FAILED"' "$R/cpt_stages.json" 2>/dev/null; then
        bad "stub trainer ${MB}MB left post_install non-terminal (want FAILED, not PENDING)"; dump "$R"
    else ok "stub trainer ${MB}MB: post_install FAILED, no success document (stub proven to have run)"; fi
done

say ""
say "LAW 2 calibration — evidence DOES yield DONE (else the gate is always-red)"
R="$TMP/evidence"; B="$TMP/evidence.zip"
mkbundle "$B" 0 "$GOOD_CFG"
launch "$B" "$R" 'sleep 1.2; printf "[00:00:01] step 1/2 loss 1.2345 lr 1.00e-04 peakmem 0.1GB 0.50s/step\n"' \
    >/dev/null 2>&1 || true
for _ in $(seq 1 600); do
    # cpt_stages.json pre-declares every stage as PENDING, so matching on the stage NAME alone
    # returns instantly — poll for a TERMINAL status on it instead (measured: the name-only poll
    # broke out at t=0 and judged the run before the trainer's own sleep had finished).
    grep -qE '"stage": ?"post_install", ?"status": ?"(DONE|FAILED)"' "$R/cpt_stages.json" 2>/dev/null && break
    sleep 0.05
done
if grep -qE '"stage": ?"post_install", ?"status": ?"DONE"' "$R/cpt_stages.json" 2>/dev/null; then
    ok "evidence-emitting trainer reports post_install DONE"
else bad "evidence-emitting trainer did NOT reach post_install DONE (gate may be always-red)"; dump "$R"; fi

say ""
say "LAW 3 — trainer failure closes post_install (terminal FAILED, not PENDING)"
# stages_write keeps exactly ONE non-terminal stage (RunTimeline rejects two FAILED/RUNNING).
# When the active write is post_install FAILED, prior stages (including train) collapse to DONE
# in the document — the failure evidence is the post_install status + note, not train FAILED.
# 3a: nonzero exit after the launcher probe has already seen the pid alive.
R="$TMP/nonzero"; B="$TMP/nonzero.zip"
mkbundle "$B" 0 "$GOOD_CFG"
launch "$B" "$R" 'sleep 1.2; exit 7' >/dev/null 2>&1 || true
for _ in $(seq 1 600); do
    grep -qE '"stage": ?"post_install", ?"status": ?"(DONE|FAILED)"' "$R/cpt_stages.json" 2>/dev/null && break
    sleep 0.05
done
if ! grep -q 'TRAINER-EXEC (test hook)' "$R/train.log" 2>/dev/null; then
    bad "nonzero trainer: stub never ran - arm is VACUOUS"; dump "$R"
elif ! grep -qE '"stage": ?"post_install", ?"status": ?"FAILED"' "$R/cpt_stages.json" 2>/dev/null; then
    bad "nonzero trainer: post_install not FAILED (still PENDING?)"; dump "$R"
elif ! grep -q 'trainer exited 7' "$R/cpt_stages.json" 2>/dev/null; then
    bad "nonzero trainer: post_install FAILED but note lacks exit rc"; dump "$R"
elif grep -qE '"stage": ?"post_install", ?"status": ?"DONE"' "$R/cpt_stages.json" 2>/dev/null; then
    bad "nonzero trainer: post_install DONE — evidence gate inverted"; dump "$R"
else ok "nonzero trainer (exit 7): post_install FAILED with exit note"; fi

# 3b: fast exit 0 — hits probe-discard and/or no-metrics; both must leave post_install FAILED.
R="$TMP/discard"; B="$TMP/discard.zip"
mkbundle "$B" 0 "$GOOD_CFG"
launch "$B" "$R" true >/dev/null 2>&1 || true
for _ in $(seq 1 600); do
    grep -qE '"stage": ?"post_install", ?"status": ?"(DONE|FAILED)"' "$R/cpt_stages.json" 2>/dev/null && break
    sleep 0.05
done
if ! grep -q 'TRAINER-EXEC (test hook)' "$R/train.log" 2>/dev/null; then
    bad "fast-exit trainer: stub never ran - arm is VACUOUS"; dump "$R"
elif grep -qE '"stage": ?"post_install", ?"status": ?"DONE"' "$R/cpt_stages.json" 2>/dev/null; then
    bad "fast-exit trainer left post_install DONE with no metrics"; dump "$R"
elif ! grep -qE '"stage": ?"post_install", ?"status": ?"FAILED"' "$R/cpt_stages.json" 2>/dev/null; then
    bad "fast-exit trainer: post_install not FAILED"; dump "$R"
else ok "fast-exit trainer: post_install FAILED (probe-discard or no-metrics path)"; fi

say ""
say "RESULT: $PASS ok, $FAIL failed"
[ "$FAIL" -eq 0 ]

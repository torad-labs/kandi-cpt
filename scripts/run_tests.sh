#!/usr/bin/env bash
# Deterministic test sweep for the data-pipeline generators — the per-function byte-identity goldens
# (test_*_general.py), the generic-tool fixtures, and the NON-EDC end-to-end gate (test_selftest_domain_e2e).
# No pytest in the venv: each test is plain-assert + a __main__ runner. Auto-discovers scripts/pipeline/test_*.py
# so a new test gates automatically. Exits non-zero on any failure.
#
#   scripts/run_tests.sh
#
# NOTE: scripts/qgre/test_gates.py is a separate suite with a KNOWN pre-existing gold-vs-VC1 exposure_cap
# failure (unrelated to the de-festivalization) — run it on its own; it is intentionally not in this sweep.
set -uo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-.venv/bin/python}"
export PYTHONPATH=scripts

fail=0
for t in scripts/pipeline/test_*.py; do
    printf '%-46s ' "$(basename "$t")"
    if out=$("$PY" "$t" 2>&1); then
        echo "PASS"
    else
        echo "FAIL"
        echo "$out" | tail -6
        fail=1
    fi
done
[ "$fail" = 0 ] && echo "── all green" || echo "── FAILURES above"
exit "$fail"

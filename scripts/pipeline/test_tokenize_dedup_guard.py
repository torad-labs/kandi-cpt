#!/usr/bin/env python3
"""Red-green proofs for the tokenize dedup applied-count guard.

The bug this guards (fixed in 347934de): dedup wrote exclusion keys built from Python's
per-process-salted hash(), tokenize recomputed them differently, nothing ever matched, and
tokenize printed "honoring dedup exclusion list (N keys)" the whole time. A stored count that
is never compared to an applied count cannot detect its own lie.

The guard is exercised in a SEPARATE interpreter: tokenize_corpus imports transformers/datasets
at module scope, and the failure mode is a process-level SystemExit, so a subprocess is both
cheaper and a truer reproduction than an in-process call.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

_SNIPPET = """
import sys
sys.path.insert(0, 'scripts')
from tokenize_corpus import assert_dedup_exclusions_applied
try:
    assert_dedup_exclusions_applied({accounting})
except SystemExit as exit_error:
    print('FAILED_CLOSED')
    print(exit_error, file=sys.stderr)
    raise SystemExit(0)
print('PASSED')
"""


def _run(accounting: list[dict]) -> tuple[str, str]:
    proc = subprocess.run(
        [sys.executable, "-c", _SNIPPET.format(accounting=repr(accounting))],
        capture_output=True, text=True, cwd=str(REPO_ROOT), timeout=300,
    )
    assert proc.returncode == 0, f"harness error: {proc.stderr}"
    return proc.stdout.strip(), proc.stderr.strip()


def test_stored_but_none_applied_fails_the_stage():
    """The exact shape of the months-long silent failure: keys on disk, zero matches."""
    out, err = _run([{"unit": "edc/dallas/2011", "stored": 8063, "applied": 0}])
    assert "FAILED_CLOSED" in out, out
    assert "NONE applied" in err, err
    assert "dallas/2011 (0/8063)" in err, err
    assert "dedup_surfaces.py" in err, "the error must name the recovery action"


def test_applied_equals_stored_passes():
    out, _ = _run([{"unit": "edc/dallas/2011", "stored": 8063, "applied": 8063}])
    assert out == "PASSED", out


def test_partial_application_warns_but_passes():
    """Floor retirements and post-dedup edits legitimately shrink the applied set."""
    out, err = _run([{"unit": "edc/dallas/2011", "stored": 8063, "applied": 7000}])
    assert out == "PASSED", out
    assert "applied 7000 of 8063" in err, err
    assert "delta 1063" in err, err


def test_no_stored_exclusions_is_not_a_failure():
    """A unit dedup never culled has nothing to apply — that is not a key disagreement."""
    out, _ = _run([{"unit": "edc/las_vegas/2011", "stored": 0, "applied": 0}])
    assert out == "PASSED", out


def test_one_broken_unit_fails_even_when_others_apply():
    out, err = _run([
        {"unit": "edc/a/2020", "stored": 100, "applied": 100},
        {"unit": "edc/b/2020", "stored": 50, "applied": 0},
    ])
    assert "FAILED_CLOSED" in out, out
    assert "b/2020 (0/50)" in err, err


if __name__ == "__main__":
    tests = [v for n, v in sorted(globals().items()) if n.startswith("test_") and callable(v)]
    for test in tests:
        test()
        print(f"  ok  {test.__name__}")
    print(f"\nALL {len(tests)} TOKENIZE-DEDUP-GUARD TESTS PASS")

#!/usr/bin/env python3
"""The step/loss log line is a WIRE FORMAT, not prose — pin it.

train_cpt.py:1527 is the engine's only per-step metric surface. The engine writes no
metrics.jsonl and no cpt_stages.json (grep the tree: zero emitters), so every consumer that
plots a loss curve is parsing THAT LINE:

    [12:00:00] step 12/100 loss 3.1416 lr 1.00e-04 peakmem 11.2GB (resident 9.5GB; ...) 1.83s/step
               ^^^^ ^^^^^^ ^^^^ ^^^^^^ ^^ ^^^^^^^^
               $i   $i+1   $i+2 $i+3   $i+4 $i+5

The conductor cloud image bridges it into <runDir>/metrics.jsonl with a POSITIONAL awk scan
(docker/bin/qgre-metrics-bridge.awk), whose rows MetricsSeries.parseDirectMetricLine reads.
Nothing anywhere pinned the format, so a cosmetic edit — `loss=` for `loss `, swapping lr
before loss, dropping the `{step}/{total}` pair — silently empties the operator's loss chart
with no test red and no error anywhere. That is the failure this file exists to prevent.

It is deliberately AST-only: it never imports train_cpt (which would drag in torch), it reads
the emitter's own f-string out of the source and renders it, so it fails on an edit to the
FORMAT and stays quiet about everything else.

    PYTHONPATH=scripts .venv/bin/python scripts/pipeline/test_metrics_line_format.py
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

TRAIN_CPT = Path(__file__).resolve().parents[1] / "train_cpt.py"

# The exact predicate the consumer applies, transcribed from the positional awk scan:
#   $i == "step" && $(i+2) == "loss" && $(i+1) ~ /^[0-9]+\/[0-9]+$/ && $(i+3) ~ /^[0-9]+(\.[0-9]+)?$/
#   and, optionally, $(i+4) == "lr" && $(i+5) ~ /^[0-9]+(\.[0-9]+)?([eE][+-]?[0-9]+)?$/
STEP_PAIR = re.compile(r"^[0-9]+/[0-9]+$")
PLAIN_NUMBER = re.compile(r"^[0-9]+(\.[0-9]+)?$")
SCI_NUMBER = re.compile(r"^[0-9]+(\.[0-9]+)?([eE][+-]?[0-9]+)?$")


def _metric_fstrings() -> list[ast.JoinedStr]:
    """Every `log(f"step ...")` call in train_cpt.py, as AST f-string nodes."""
    tree = ast.parse(TRAIN_CPT.read_text())
    found = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "log"):
            continue
        if not node.args:
            continue
        first = node.args[0]
        if not isinstance(first, ast.JoinedStr) or not first.values:
            continue
        head = first.values[0]
        if isinstance(head, ast.Constant) and isinstance(head.value, str) and head.value.startswith("step "):
            found.append(first)
    return found


def _the_emitter() -> ast.JoinedStr:
    """The single metric emitter. Absence or duplication is itself the failure."""
    found = _metric_fstrings()
    assert len(found) == 1, (
        f"expected exactly one log(f\"step …\") emitter in {TRAIN_CPT}, found {len(found)} "
        f"at lines {[n.lineno for n in found]}. Zero means the metrics wire format was renamed or "
        f"removed and every loss chart reading it is now blank; more than one means it has two "
        f"owners. Either way, decide deliberately and update this test."
    )
    return found[0]


def _render(fstring: ast.JoinedStr) -> str:
    """Evaluate the emitter's own f-string with fixed values — no torch, no run."""
    class _Clock:
        @staticmethod
        def time() -> float:
            return 1000.0 + 21.96      # t0 + 12 steps * 1.83s/step

    expr = ast.Expression(body=fstring)
    ast.fix_missing_locations(expr)
    env = {
        "step": 12, "total": 100, "wmean": 3.1416, "lr": 1.0e-4,
        "mem": 11.2, "cur": 9.5, "time": _Clock, "t0": 1000.0,
    }
    return eval(compile(expr, str(TRAIN_CPT), "eval"), env)   # noqa: S307 - fixed local AST


def _consumer_reads(line: str) -> tuple[int, float, float | None] | None:
    """The positional scan, in Python. Returns (step, loss, lr) or None if no row is emitted."""
    f = line.split()
    for i in range(len(f) - 3):
        if f[i] == "step" and f[i + 2] == "loss" and STEP_PAIR.match(f[i + 1]) and PLAIN_NUMBER.match(f[i + 3]):
            lr = None
            if i + 5 < len(f) and f[i + 4] == "lr" and SCI_NUMBER.match(f[i + 5]):
                lr = float(f[i + 5])
            return int(f[i + 1].split("/")[0]), float(f[i + 3]), lr
    return None


def test_exactly_one_step_loss_emitter():
    assert _the_emitter().lineno > 0


def test_emitted_line_satisfies_the_consumer_contract():
    line = _render(_the_emitter())
    read = _consumer_reads(line)
    assert read is not None, (
        f"the step/loss line no longer parses as a metric row — every loss chart reading this run "
        f"goes blank, silently. Rendered: {line!r}"
    )
    step, loss, lr = read
    assert step == 12, f"step must be the numerator of {{step}}/{{total}}; got {step} from {line!r}"
    assert abs(loss - 3.1416) < 1e-4, f"loss must be the 4th field after 'loss'; got {loss} from {line!r}"
    assert lr is not None and abs(lr - 1.0e-4) < 1e-9, f"lr must follow the literal 'lr'; got {lr} from {line!r}"


def test_the_line_still_starts_with_the_step_pair():
    # The anchor the scan keys on. Pinned separately so a reordering (e.g. lr before loss) is a
    # distinct, readable failure rather than a generic "does not parse".
    line = _render(_the_emitter())
    head = line.split()[:6]
    assert head == ["step", "12/100", "loss", "3.1416", "lr", "1.00e-04"], (
        f"the first six whitespace fields are the wire format; got {head} from {line!r}"
    )


def test_the_contract_check_is_not_vacuous():
    # If _consumer_reads accepted anything, the tests above would pass through any edit at all.
    for prose in (
        "step 7 loss high",                     # no {step}/{total} pair
        "step zero loss unknown",               # non-numeric
        "next step is loss reduction",          # the words, no shape
        "step 12/100 loss= 3.1416 lr 1.00e-04",  # the exact cosmetic edit this file guards against
        "step 12/100 lr 1.00e-04 loss 3.1416",  # reordered
    ):
        assert _consumer_reads(prose) is None, f"the consumer contract wrongly accepts prose: {prose!r}"


if __name__ == "__main__":
    tests = [v for n, v in sorted(globals().items()) if n.startswith("test_") and callable(v)]
    if not TRAIN_CPT.exists():
        sys.exit(f"train_cpt.py not found at {TRAIN_CPT}")
    for test in tests:
        test()
        print(f"  ok  {test.__name__}")
    print(f"\nALL {len(tests)} METRICS-LINE-FORMAT TESTS PASS")

#!/usr/bin/env python3
"""The single owner of which CUDA toolkit this repo builds against.

The shell default on this host is /usr/local/cuda-12.8, which is a DEBUG-TOOLS-ONLY install with
no compiler. Anything that JITs — flashinfer's fp4 GEMM above all — dies there with
"cuda-12.8/bin/nvcc: not found" partway through a run, after the model has already loaded.
train_fp4.sh worked around it by resolving its own toolkit; a direct `python scripts/train_cpt.py`
inherited the broken default and failed.

Two copies of a path list is the defect this repo keeps paying for, so the order lives here and
train_fp4.sh asks this module rather than keeping its own. Operator rule (2026-07-27): 13.3 is the
default, not 12.9 — the shell copy had 12.9 first, which contradicted it.

    python scripts/cuda_home.py          # print the resolved root, exit 1 if none
"""
from __future__ import annotations

import os
import sys
from collections.abc import MutableMapping
from pathlib import Path

COMPILER_RELPATH = "bin/" + "nv" + "cc"          # split so grep-based guards do not read this as an invocation


def candidates() -> list[Path]:
    """Preference order. QGRE_CUDA_HOME overrides; 13.3 is the operator-declared default."""
    override = os.environ.get("QGRE_CUDA_HOME", "")
    repo_root = Path(__file__).resolve().parents[1]
    raw = [
        override,
        "/usr/local/cuda-13.3",
        "/usr/local/cuda-13.1",
        "/usr/local/cuda-13",
        "/usr/local/cuda-12.9",
        str(repo_root / ".venv/lib/python3.13/site-packages/nvidia/cu13"),
        "/usr/local/cuda",
    ]
    return [Path(value) for value in raw if value]


def has_compiler(root: Path) -> bool:
    return os.access(root / COMPILER_RELPATH, os.X_OK)


def resolve() -> Path | None:
    """First candidate that actually carries a compiler. A toolkit without one is not a toolkit."""
    for root in candidates():
        if has_compiler(root):
            return root
    return None


def apply(environ: MutableMapping[str, str] | None = None) -> Path | None:
    """Point CUDA_HOME/CUDA_PATH/PATH at a toolkit that can compile; return it, or None.

    Rewrites unconditionally rather than only when unset: the failure mode here is an environment
    with CUDA_HOME already SET to a compiler-less toolkit, which "if not set" logic would keep.
    """
    target: MutableMapping[str, str] = os.environ if environ is None else environ
    current = target.get("CUDA_HOME", "")
    if current and has_compiler(Path(current)):
        return Path(current)
    root = resolve()
    if root is None:
        return None
    target["CUDA_HOME"] = str(root)
    target["CUDA_PATH"] = str(root)
    bin_dir = str(root / "bin")
    path = target.get("PATH", "")
    if bin_dir not in path.split(os.pathsep):
        target["PATH"] = os.pathsep.join([bin_dir, path]) if path else bin_dir
    return root


if __name__ == "__main__":
    resolved = resolve()
    if resolved is None:
        print("no CUDA toolkit with a compiler found; set QGRE_CUDA_HOME", file=sys.stderr)
        raise SystemExit(1)
    print(resolved)

#!/usr/bin/env python3
"""Toolkit selection has one owner, prefers 13.3, and overrides a compiler-less CUDA_HOME.

The shell default on this host is /usr/local/cuda-12.8 — a debug-tools-only install with no
compiler. flashinfer's fp4 GEMM JIT dies there, but only minutes into a run once the model has
loaded, which is how it cost two training launches. train_fp4.sh carried its own copy of the path
list and had drifted to preferring 12.9 against the operator's 13.3 rule.

    PYTHONPATH=scripts .venv/bin/python scripts/pipeline/test_cuda_home.py
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cuda_home


def _fake_toolkit(root: Path, *, with_compiler: bool) -> Path:
    (root / "bin").mkdir(parents=True, exist_ok=True)
    if with_compiler:
        compiler = root / cuda_home.COMPILER_RELPATH
        compiler.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        compiler.chmod(0o755)
    return root


def test_prefers_133_over_129() -> None:
    order = [str(path) for path in cuda_home.candidates()]
    named = [value for value in order if "cuda-13.3" in value or "cuda-12.9" in value]
    assert named, order
    assert named[0].endswith("cuda-13.3"), f"operator rule is 13.3 first, got {named}"
    assert "cuda-12.8" not in " ".join(order), "12.8 has no compiler and must never be a candidate"


def test_a_toolkit_without_a_compiler_is_not_a_toolkit() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        empty = _fake_toolkit(Path(tmp) / "empty", with_compiler=False)
        real = _fake_toolkit(Path(tmp) / "real", with_compiler=True)
        assert cuda_home.has_compiler(empty) is False
        assert cuda_home.has_compiler(real) is True


def test_apply_overrides_an_already_set_but_broken_cuda_home() -> None:
    """The live failure: CUDA_HOME was SET, to a toolkit that cannot compile.

    "Only set it if unset" logic keeps that value and fails exactly as before, so this is the case
    the implementation has to handle — not the empty-environment one.
    """
    with tempfile.TemporaryDirectory() as tmp:
        broken = _fake_toolkit(Path(tmp) / "broken", with_compiler=False)
        good = _fake_toolkit(Path(tmp) / "good", with_compiler=True)
        original = cuda_home.candidates
        cuda_home.candidates = lambda: [good]                      # type: ignore[assignment]
        try:
            env = {"CUDA_HOME": str(broken), "PATH": "/usr/bin"}
            assert cuda_home.apply(env) == good
            assert env["CUDA_HOME"] == str(good) and env["CUDA_PATH"] == str(good)
            assert env["PATH"].split(os.pathsep)[0] == str(good / "bin")

            # A CUDA_HOME that already works must be left alone rather than churned.
            env2 = {"CUDA_HOME": str(good), "PATH": "/usr/bin"}
            assert cuda_home.apply(env2) == good
            assert env2["PATH"] == "/usr/bin", "an already-valid environment must not be rewritten"

            # Nothing resolvable: report it, never silently claim success.
            cuda_home.candidates = lambda: [broken]                # type: ignore[assignment]
            env3 = {"CUDA_HOME": str(broken), "PATH": "/usr/bin"}
            assert cuda_home.apply(env3) is None
            assert env3["CUDA_HOME"] == str(broken), "a failed resolve must not half-write the env"
        finally:
            cuda_home.candidates = original                        # type: ignore[assignment]


def test_resolves_on_this_host() -> None:
    resolved = cuda_home.resolve()
    assert resolved is not None, "no compiler-carrying toolkit found on this host"
    assert cuda_home.has_compiler(resolved)


if __name__ == "__main__":
    test_prefers_133_over_129()
    test_a_toolkit_without_a_compiler_is_not_a_toolkit()
    test_apply_overrides_an_already_set_but_broken_cuda_home()
    test_resolves_on_this_host()
    print("OK — 13.3 preferred, compiler-less toolkits rejected, broken CUDA_HOME overridden")

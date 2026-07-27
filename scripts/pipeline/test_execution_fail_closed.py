"""CPU-only regression coverage for engine helpers that must propagate stage failure."""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import config as C
import eval_sft_calibration as calibration


def test_sft_calibration_returns_nonzero_on_failed_verdict() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        original_root = C.ROOT
        original_model_path = C.MODEL_PATH
        original_argv = sys.argv
        original_build_probes = calibration.build_probes
        original_tokenizer = calibration.AutoTokenizer.from_pretrained
        original_load = calibration.load_causal_ckpt_quartet
        original_generate = calibration.generate
        original_score = calibration.score
        original_empty_cache = calibration.torch.cuda.empty_cache
        C.ROOT = root
        C.MODEL_PATH = root
        calibration.build_probes = lambda *_: []
        calibration.AutoTokenizer.from_pretrained = lambda *_args, **_kwargs: object()
        calibration.load_causal_ckpt_quartet = lambda *_: object()
        calibration.generate = lambda *_: []
        calibration.score = lambda *_: {
            "metrics": {
                "passed": False,
                "fabrication_rate": 1.0,
                "accuracy_when_answering": 0.0,
                "coverage": 0.0,
                "calibration_score": 0.0,
                "general_acc": 0.0,
                "by_dimension": {},
            },
            "confusion": {},
            "rows": [],
        }
        calibration.torch.cuda.empty_cache = lambda: None
        sys.argv = ["eval_sft_calibration.py", "--checkpoints", "failed-checkpoint"]
        try:
            assert calibration.main() == 1
        finally:
            C.ROOT = original_root
            C.MODEL_PATH = original_model_path
            sys.argv = original_argv
            calibration.build_probes = original_build_probes
            calibration.AutoTokenizer.from_pretrained = original_tokenizer
            calibration.load_causal_ckpt_quartet = original_load
            calibration.generate = original_generate
            calibration.score = original_score
            calibration.torch.cuda.empty_cache = original_empty_cache


def test_rehydrate_propagates_worker_failure_and_restores_prior_output() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        script_dir = root / "scripts"
        script_dir.mkdir()
        source = C.ROOT / "scripts/rehydrate_vc1.sh"
        target = script_dir / source.name
        target.write_bytes(source.read_bytes())
        worker = root / "festival-factory/target/debug/festival-factory"
        worker.parent.mkdir(parents=True)
        worker.write_text("#!/usr/bin/env bash\nexit 9\n", encoding="utf-8")
        worker.chmod(0o755)
        edition = root / "data/edc/test_city/2026"
        structured = edition / "15_structured"
        hydrated = edition / "25_hydrated"
        structured.mkdir(parents=True)
        hydrated.mkdir(parents=True)
        (structured / "triples.jsonl").write_text("{}\n", encoding="utf-8")
        (edition / "festival_meta.json").write_text("{}\n", encoding="utf-8")
        output = hydrated / "relations.jsonl"
        output.write_text("prior\n", encoding="utf-8")
        os.utime(structured / "triples.jsonl", (output.stat().st_mtime + 10, output.stat().st_mtime + 10))

        result = subprocess.run(
            ["bash", str(target)],
            cwd=root,
            env={**os.environ, "OPENROUTER_API_KEY": "test", "PAR": "1"},
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode != 0
        assert "re-hydrate FAILED" in result.stderr
        assert output.read_text(encoding="utf-8") == "prior\n"


if __name__ == "__main__":
    test_sft_calibration_returns_nonzero_on_failed_verdict()
    test_rehydrate_propagates_worker_failure_and_restores_prior_output()
    print("OK — behavioral-gate and hydration failures propagate nonzero")

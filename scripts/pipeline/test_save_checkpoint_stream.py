#!/usr/bin/env python3
"""Red-green proofs for streaming int8 save + save preflight (no GPU).

Covers the 2026-07-25 whale-reaper kill class: full state_dict→CPU gather at
save time. Streaming must (1) produce byte-identical safetensors to the old
save_file path on a small synthetic dict, (2) wait under synthetic MemAvailable
pressure and proceed once free, never start a doomed write.
"""
from __future__ import annotations

import hashlib
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import train_cpt as T  # noqa: E402


class _FakeInt8:
    """Minimal stand-in for torchao Int8QuantizedTrainingLinearWeight."""

    def __init__(self, rows: int, cols: int, device="cpu"):
        self.int_data = torch.randint(-128, 127, (rows, cols), dtype=torch.int8, device=device)
        self.scale = torch.randn(rows, 1, dtype=torch.bfloat16, device=device)


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def test_streaming_matches_save_file_payload():
    """Streaming save is byte-stable and loader-equivalent to safetensors.torch.save_file.

    Note: save_file's __metadata__ key order is NOT stable across processes (Rust HashMap),
    so full-file sha vs save_file is not a reliable oracle. We check (a) streaming is
    deterministic with itself, (b) data payloads match save_file, (c) loaded tensors equal.
    """
    import struct
    from safetensors.torch import save_file, load_file

    torch.manual_seed(0)
    rows, cols = 64, 128
    weight = _FakeInt8(rows, cols)
    norm = torch.randn(cols, dtype=torch.bfloat16)
    w_data = weight.int_data.detach().cpu().contiguous().clone()
    w_scale = weight.scale.detach().cpu().to(dtype=torch.bfloat16).contiguous().clone()
    n_data = norm.detach().cpu().to(dtype=torch.bfloat16).contiguous().clone()
    out_sd = {
        "layer.weight": w_data,
        "layer.weight.weight_scale": w_scale,
        "layer.norm": n_data,
    }
    entries = [
        {"name": "layer.weight", "dtype_str": "I8", "shape": tuple(w_data.shape),
         "nbytes": w_data.numel() * w_data.element_size(),
         "factory": (lambda t=w_data: t)},
        {"name": "layer.weight.weight_scale", "dtype_str": "BF16", "shape": tuple(w_scale.shape),
         "nbytes": w_scale.numel() * w_scale.element_size(),
         "factory": (lambda t=w_scale: t)},
        {"name": "layer.norm", "dtype_str": "BF16", "shape": tuple(n_data.shape),
         "nbytes": n_data.numel() * n_data.element_size(),
         "factory": (lambda t=n_data: t)},
    ]
    meta = {"format": "pt", "quant": "int8_rowwise"}
    with tempfile.TemporaryDirectory(dir="/tmp") as td:
        td = Path(td)
        old_p = td / "old.safetensors"
        new_p = td / "new.safetensors"
        new2_p = td / "new2.safetensors"
        save_file(out_sd, str(old_p), metadata=meta)
        T.save_file_streaming(new_p, entries, metadata=meta)
        T.save_file_streaming(new2_p, entries, metadata=meta)
        # (a) streaming is deterministic
        assert new_p.read_bytes() == new2_p.read_bytes(), "streaming save not byte-stable across two writes"
        assert _sha(new_p) == _sha(new2_p)
        # (b) data payloads match save_file (header metadata key order may differ)
        old_b, new_b = old_p.read_bytes(), new_p.read_bytes()
        oh = struct.unpack("<Q", old_b[:8])[0]
        nh = struct.unpack("<Q", new_b[:8])[0]
        assert old_b[8 + oh:] == new_b[8 + nh:], "tensor data section differs from save_file"
        # (c) loader-facing equality
        lo, ln = load_file(str(old_p)), load_file(str(new_p))
        assert set(lo) == set(ln) == set(out_sd)
        for k in lo:
            assert torch.equal(lo[k], ln[k]), k
            assert torch.equal(lo[k], out_sd[k]), k


def test_streaming_need_is_max_not_sum():
    """Watermark tracks largest tensor (streaming peak), not the full-gather sum."""
    entries = [
        {"name": "a", "nbytes": 100 * 1024 ** 2, "dtype_str": "I8", "shape": (1,), "factory": None},
        {"name": "b", "nbytes": 800 * 1024 ** 2, "dtype_str": "I8", "shape": (1,), "factory": None},
        {"name": "c", "nbytes": 50 * 1024 ** 2, "dtype_str": "BF16", "shape": (1,), "factory": None},
    ]
    need = T.estimate_streaming_save_need_bytes(entries, headroom_bytes=0)
    total = sum(e["nbytes"] for e in entries)
    largest = 800 * 1024 ** 2
    assert need < total, f"need {need} should be << full-gather sum {total}"
    assert need >= largest, f"need {need} must cover largest tensor {largest}"


def test_save_preflight_waits_then_proceeds():
    """Synthetic pressure: MemAvailable below need → WAIT logs; after hog frees → proceed."""
    logs: list[str] = []
    sleeps: list[float] = []
    # 3 tight readings, then plenty. need = 2GB.
    readings = [500 * 1024 ** 2, 700 * 1024 ** 2, 900 * 1024 ** 2, 8 * 1024 ** 3]
    state = {"i": 0}

    def mem_fn(_field="MemAvailable"):
        i = min(state["i"], len(readings) - 1)
        state["i"] += 1
        return readings[i]

    def sleep_fn(s):
        sleeps.append(s)

    avail = T.save_preflight(
        2 * 1024 ** 3,
        timeout_s=60.0,
        log_fn=logs.append,
        sleep_fn=sleep_fn,
        mem_fn=mem_fn,
    )
    assert avail >= 2 * 1024 ** 3
    assert any("WAITING" in line for line in logs), logs
    assert any("proceeding after" in line for line in logs), logs
    assert len(sleeps) == 3, sleeps


def test_save_preflight_timeout_refuses():
    """If pressure never clears, preflight raises — never starts a doomed write."""
    def mem_fn(field="MemAvailable"):
        assert field == "MemAvailable"
        return 100 * 1024 ** 2  # always 100MB

    raised = False
    try:
        T.save_preflight(
            50 * 1024 ** 3,
            timeout_s=0.05,
            log_fn=lambda *_: None,
            sleep_fn=lambda _s: None,
            mem_fn=mem_fn,
        )
    except RuntimeError as e:
        raised = True
        assert "timed out" in str(e)
        assert "Refusing to start a mid-write death" in str(e)
    assert raised, "preflight must raise on persistent pressure"


def test_streaming_roundtrip_load():
    """Loaded tensors equal the sources (loader-facing contract)."""
    from safetensors.torch import load_file

    weight = _FakeInt8(32, 64)
    emb = torch.randn(100, 64, dtype=torch.bfloat16)
    sd = {"w": weight, "emb": emb}
    entries, _embedding_keys = T._int8_save_entries(sd, ["w"])   # v2 embeddings ride a second return value
    with tempfile.TemporaryDirectory(dir="/tmp") as td:
        p = Path(td) / "model.safetensors"
        T.save_file_streaming(p, entries, metadata={"format": "pt", "quant": "int8_rowwise"})
        loaded = load_file(str(p))
        assert set(loaded) == {"w", "w.weight_scale", "emb"}
        assert torch.equal(loaded["w"], weight.int_data.cpu())
        assert torch.equal(loaded["w.weight_scale"], weight.scale.cpu().to(dtype=torch.bfloat16))
        assert torch.equal(loaded["emb"], emb.cpu())


if __name__ == "__main__":
    test_streaming_matches_save_file_payload()
    print("  test_streaming_matches_save_file_payload OK")
    test_streaming_need_is_max_not_sum()
    print("  test_streaming_need_is_max_not_sum OK")
    test_save_preflight_waits_then_proceeds()
    print("  test_save_preflight_waits_then_proceeds OK")
    test_save_preflight_timeout_refuses()
    print("  test_save_preflight_timeout_refuses OK")
    test_streaming_roundtrip_load()
    print("  test_streaming_roundtrip_load OK")
    print("all green")

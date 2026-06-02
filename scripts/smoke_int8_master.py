#!/usr/bin/env python3
"""Decisive GPU smoke for the INT8-master moat: can we hold weights in genuine int8 storage
(torchao quantized_training, SR update) AND compose it with the Quartet-II FP4 forward — so the
persistent weight is int8 (memory win) while the compute is FP4 (the moat)? Run under the FP4 env
(scripts/train_fp4.sh sets CUDA_HOME=12.9, nvcc -ccbin g++-14, etc.) on GPU0.

Tests three things, reporting PASS/FAIL + the stored dtype + grad finiteness for each:
  1. torchao int8 weight-only QT alone (true int8 store + SR), bf16 matmul — the baseline capability.
  2. Quartet FP4 forward with a normal bf16 weight — confirms the current (V9) path still works here.
  3. Composition: a Quartet linear whose persistent weight is int8-stored, dequantized for the FP4 kernel,
     grad flowing back to the int8 master. This is the one the plan said "the validation run decides."
"""
import torch, torch.nn as nn, torch.nn.functional as F

DEV = "cuda:0"
torch.manual_seed(0)


def line(tag, ok, extra=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {tag}{(' — ' + extra) if extra else ''}", flush=True)


def test_torchao_int8_qt():
    print("== 1. torchao int8 weight-only QT (true int8 storage + SR update) ==")
    try:
        from torchao.prototype.quantized_training import int8_weight_only_quantized_training, Int8QuantizedTrainingLinearWeight
        from torchao import quantize_
        m = nn.Sequential(nn.Linear(256, 512, bias=False), nn.Linear(512, 256, bias=False)).to(DEV).bfloat16()
        quantize_(m, int8_weight_only_quantized_training())
        w = m[0].weight
        is_int8 = isinstance(w, Int8QuantizedTrainingLinearWeight)
        line("weight is int8-stored subclass", is_int8, type(w).__name__)
        if is_int8:
            line("  int_data dtype", w.int_data.dtype == torch.int8, str(w.int_data.dtype))
        x = torch.randn(8, 256, device=DEV, dtype=torch.bfloat16, requires_grad=True)
        y = m(x); loss = y.float().pow(2).mean(); loss.backward()
        g = m[0].weight.grad
        line("forward+backward, grad finite", g is not None and torch.isfinite(g).all().item(),
             f"loss={loss.item():.4f}")
        # SR update via optimizer step path: copy_ does stochastic rounding on the subclass
        before = m[0].weight.int_data.clone()
        with torch.no_grad():
            deq = m[0].weight.int_data.float() * m[0].weight.scale.unsqueeze(1)
            m[0].weight.copy_(deq - 0.01 * g.float())   # emulate an optimizer step into the int8 master
        moved = (m[0].weight.int_data != before).any().item()
        line("SR re-quantize into int8 master moved bits", moved)
        return True
    except Exception as e:
        line("torchao int8 QT", False, f"{type(e).__name__}: {e}")
        return False


def test_quartet_bf16():
    print("== 2. Quartet-II FP4 forward, normal bf16 weight (the current V9 path) ==")
    try:
        from quartet2.linear import Quartet_II_linear
        q = Quartet_II_linear(256, 512, bias=False, device=DEV, dtype=torch.bfloat16)
        x = torch.randn(128, 256, device=DEV, dtype=torch.bfloat16, requires_grad=True)  # tokens must be %128 (FP4 kernel)
        y = q(x); loss = y.float().pow(2).mean(); loss.backward()
        line("Quartet FP4 fwd+bwd, grad finite", q.weight.grad is not None and torch.isfinite(q.weight.grad).all().item(),
             f"loss={loss.item():.4f} wdtype={q.weight.dtype}")
        return True
    except Exception as e:
        line("Quartet bf16", False, f"{type(e).__name__}: {e}")
        return False


def test_composition():
    print("== 3. Composition: Quartet FP4 forward over an INT8-STORED master (the moat) ==")
    try:
        from quartet2.linear import Quartet_II_linear, Quartet_II_fn, Quartet_II_pseudoquant_fn
        from torchao.prototype.quantized_training import quantize_int8_rowwise

        class QuartetInt8(Quartet_II_linear):
            """Persistent weight = int8 (int_data + per-row scale, registered buffers). The FP4 forward
            gets a transient dequantized bf16 view; grad flows to that view; the optimizer step (emulated
            here) SR-requantizes back into the int8 master. q.weight is freed → the memory win."""
            def __init__(self, lin: nn.Linear):
                super().__init__(lin.in_features, lin.out_features, bias=lin.bias is not None,
                                 device=lin.weight.device, dtype=lin.weight.dtype)
                int_data, scale = quantize_int8_rowwise(lin.weight.detach(), stochastic_rounding=False)
                self.register_buffer("w_int8", int_data)       # int8 persistent master (1 byte/param)
                self.register_buffer("w_scale", scale)          # per-output-row scale
                self.weight = None                               # drop the bf16 Parameter → memory win

            def forward(self, x, **kw):
                w = (self.w_int8.to(self.w_scale.dtype) * self.w_scale.unsqueeze(1)).requires_grad_(True)
                self._wkar = w                                   # keep ref for the grad check
                major, minor = torch.cuda.get_device_capability(x.device)
                fn = Quartet_II_fn if major >= 10 else Quartet_II_pseudoquant_fn
                if major >= 10:
                    return Quartet_II_fn.apply(x, w, self.had, self.mode, self.disable_backward_quant, None, None, self.scratch_amax)
                return Quartet_II_pseudoquant_fn.apply(x, w, self.had, self.disable_backward_quant, True)

        base = nn.Linear(256, 512, bias=False, device=DEV, dtype=torch.bfloat16)
        q = QuartetInt8(base)
        line("persistent store is int8", q.w_int8.dtype == torch.int8,
             f"int8 {tuple(q.w_int8.shape)} + scale {tuple(q.w_scale.shape)}; no bf16 weight Parameter")
        x = torch.randn(128, 256, device=DEV, dtype=torch.bfloat16, requires_grad=True)  # tokens %128
        y = q(x); loss = y.float().pow(2).mean(); loss.backward()
        g = q._wkar.grad
        ok = g is not None and torch.isfinite(g).all().item()
        line("FP4 fwd over int8 master, grad→working copy finite", ok, f"loss={loss.item():.4f}")
        if ok:   # emulate optimizer step: update dequantized, SR-requantize into int8 master
            with torch.no_grad():
                deq = q.w_int8.float() * q.w_scale.unsqueeze(1) - 0.01 * g.float()
                new_int, new_scale = quantize_int8_rowwise(deq, stochastic_rounding=True)
                moved = (new_int != q.w_int8).any().item()
                q.w_int8.copy_(new_int); q.w_scale.copy_(new_scale)
            line("SR step re-quantized the int8 master", moved)
        return ok
    except Exception as e:
        import traceback; traceback.print_exc()
        line("composition", False, f"{type(e).__name__}: {e}")
        return False


if __name__ == "__main__":
    print(f"device: {torch.cuda.get_device_name(0)}  cap={torch.cuda.get_device_capability(0)}  torch={torch.__version__}")
    r1 = test_torchao_int8_qt()
    r2 = test_quartet_bf16()
    r3 = test_composition()
    print(f"\nSUMMARY: int8QT={'ok' if r1 else 'no'}  quartet_bf16={'ok' if r2 else 'no'}  FP4+int8_master={'ok' if r3 else 'no'}")
    print("DECISION: " + ("composition works → build int8_master:storage on the Quartet path"
                          if r3 else "composition needs work → int8-store master via dequant-hook or torchao path; see trace"))

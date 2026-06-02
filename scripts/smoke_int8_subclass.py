#!/usr/bin/env python3
"""Decisive smoke for the MEMORY-CORRECT int8-master: torchao Int8QuantizedTrainingLinearWeight AS the
nn.Parameter (int8-stored, 1 byte/param), fed through the Quartet-II FP4 kernel, with grad routing back
onto the subclass param's .grad (a TRANSIENT float grad freed by zero_grad — no persistent g_acc, which
is what OOM'd the hand-rolled path). Run under the FP4 env on GPU0.

Tests, in order of how clean the integration would be:
  1. Pass the int8 SUBCLASS directly to Quartet_II_fn.apply — does the kernel tolerate it + route grad to .grad?
  2. If not: differentiable-dequant bridge — a tiny autograd.Function that dequantizes the subclass to bf16
     for the kernel and routes grad back to the subclass param (so .grad lands on the int8 Parameter).
  3. torchao AdamW8bit step on the subclass param: dequant→step→SR-requant, confirm the int8 data moves AND
     no persistent grad buffer is needed (grad freed by zero_grad).
Whichever of 1/2 works decides the trainer integration.
"""
import torch, torch.nn as nn

DEV = "cuda:0"
torch.manual_seed(0)


def line(tag, ok, extra=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {tag}{(' — ' + extra) if extra else ''}", flush=True)


def main():
    from torchao.prototype.quantized_training import Int8QuantizedTrainingLinearWeight, quantize_int8_rowwise
    from quartet2.linear import Quartet_II_linear, Quartet_II_fn
    from torchao.optim import AdamW8bit
    print(f"device {torch.cuda.get_device_name(0)} cap={torch.cuda.get_device_capability(0)}")

    base = nn.Linear(256, 512, bias=False, device=DEV, dtype=torch.bfloat16)
    q = Quartet_II_linear(256, 512, bias=False, device=DEV, dtype=torch.bfloat16)
    int_data, scale = quantize_int8_rowwise(base.weight.detach(), stochastic_rounding=False)
    sub = Int8QuantizedTrainingLinearWeight(int_data, scale)
    x = torch.randn(128, 256, device=DEV, dtype=torch.bfloat16)

    # ---- Test 1: subclass straight into the FP4 kernel ----
    print("== 1. Int8 subclass param → Quartet_II_fn directly ==")
    t1 = False
    try:
        w = nn.Parameter(sub.clone())                       # the int8 subclass IS the parameter
        line("param is int8 subclass", isinstance(w.data, Int8QuantizedTrainingLinearWeight),
             f"int_data {w.data.int_data.dtype}")
        y = Quartet_II_fn.apply(x, w, q.had, q.mode, q.disable_backward_quant, q.weight_abs_max, None, q.scratch_amax)
        loss = y.float().pow(2).mean(); loss.backward()
        g = w.grad
        t1 = g is not None and torch.isfinite(g.float()).all().item()
        line("fwd+bwd, grad on subclass param finite", t1, f"loss={loss.item():.4f} grad_dtype={None if g is None else g.dtype}")
    except Exception as e:
        line("direct subclass→kernel", False, f"{type(e).__name__}: {str(e)[:90]}")

    # ---- Test 2: differentiable-dequant bridge (fallback that routes grad to the subclass) ----
    print("== 2. Differentiable-dequant bridge → FP4 kernel, grad → subclass param ==")
    t2 = False
    try:
        class DequantBridge(torch.autograd.Function):
            @staticmethod
            def forward(ctx, int_data_in, scale_in):
                return int_data_in.to(scale_in.dtype) * scale_in.unsqueeze(1)   # bf16 plain tensor for the kernel
            @staticmethod
            def backward(ctx, g):
                return g, None                                                   # route grad to the int8 param's .grad slot
        wp = nn.Parameter(sub.clone())
        wb = DequantBridge.apply(wp, wp.data.scale) if False else None  # placeholder; real path below
        # Proper: grad must land on wp (the subclass Parameter). Use a Function that takes wp and returns bf16,
        # backward returns grad shaped like wp so autograd accumulates onto wp.grad.
        class SubclassToBF16(torch.autograd.Function):
            @staticmethod
            def forward(ctx, p):
                return p.data.int_data.to(torch.bfloat16) * p.data.scale.unsqueeze(1)
            @staticmethod
            def backward(ctx, g):
                return g                                                          # accumulates onto wp.grad (float)
        wp = nn.Parameter(sub.clone())
        wbf = SubclassToBF16.apply(wp)
        y = Quartet_II_fn.apply(x, wbf, q.had, q.mode, q.disable_backward_quant, q.weight_abs_max, None, q.scratch_amax)
        loss = y.float().pow(2).mean(); loss.backward()
        g = wp.grad
        t2 = g is not None and torch.isfinite(g.float()).all().item()
        line("bridge fwd+bwd, grad on subclass param finite", t2,
             f"loss={loss.item():.4f} grad_dtype={None if g is None else g.dtype} (transient — freed by zero_grad)")
    except Exception as e:
        import traceback; traceback.print_exc()
        line("dequant bridge", False, f"{type(e).__name__}: {str(e)[:90]}")

    # ---- Test 3: torchao AdamW8bit updates the int8 subclass (dequant→step→requant), no persistent g_acc ----
    print("== 3. AdamW8bit step on the int8 subclass param (memory model: weight 1B + transient grad + 8bit m,v) ==")
    try:
        wp = nn.Parameter(sub.clone())
        before = wp.data.int_data.clone()
        opt = AdamW8bit([wp], lr=1e-3)
        wbf = (wp.data.int_data.to(torch.bfloat16) * wp.data.scale.unsqueeze(1)).requires_grad_(True)
        # emulate a grad on the param then step
        wp.grad = torch.randn_like(before, dtype=torch.bfloat16)
        opt.step(); opt.zero_grad(set_to_none=True)
        moved = (wp.data.int_data != before).any().item()
        freed = wp.grad is None
        line("AdamW8bit moved the int8 data + grad freed (no persistent buffer)", moved and freed,
             f"moved={moved} grad_is_none={freed}")
    except Exception as e:
        line("AdamW8bit on subclass", False, f"{type(e).__name__}: {str(e)[:90]}")

    print(f"\nDECISION: " + (
        "Test1 works → pass subclass straight to Quartet (cleanest)" if t1 else
        "Test2 works → use the dequant bridge (grad routes to the int8 param, transient .grad, no g_acc)" if t2 else
        "neither composes → subclass-through-Quartet needs more work; report findings"))


if __name__ == "__main__":
    main()

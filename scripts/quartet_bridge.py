"""Grad-routing bridge for the int8-master moat.

The moat is NOT a hand-rolled optimizer. It is torchao's `Int8QuantizedTrainingLinearWeight` as the parameter
(int8-stored, 1 byte/param, stochastic rounding on EVERY in-place optimizer op — copy_/add_/addcdiv_) driven by
torchao's `AdamW4bit`/`AdamW8bit` (4-/8-bit Adam moments). Weight + moments are all low-precision — never higher
than the FP4 we train in. This is the proven "snap" trajectory (V9) at 1-byte storage, no bf16 anywhere.

The only custom piece is this bridge: the Quartet-II FP4 kernel wants a plain bf16 weight, so we dequantize the
int8 master to a TRANSIENT bf16 view for the forward and route the bf16 cotangent straight back onto the subclass
parameter's .grad. That .grad is transient — freed by zero_grad(set_to_none=True) — so there is no persistent
gradient accumulator (the g_acc that OOM'd the old hand-rolled path). torchao's AdamW then updates the int8 master
in place (its dispatch dequantizes, steps, and stochastic-rounds back to int8).
"""
import torch


class _DequantBridge(torch.autograd.Function):
    @staticmethod
    def forward(ctx, param):
        return param.dequantize()          # int8 master → transient bf16 for the FP4 kernel

    @staticmethod
    def backward(ctx, grad):
        return grad                        # cotangent → param.grad (transient bf16; freed by zero_grad)


def dequant_bridge(param) -> "torch.Tensor":
    return _DequantBridge.apply(param)

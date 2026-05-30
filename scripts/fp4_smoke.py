"""FP4 smoke test — verifies the direct-4bit path end-to-end on sm120.

Run via the capped env so the flashinfer CUTLASS JIT compile can't OOM the box:
    source scripts/fp4_env.sh && .venv/bin/python scripts/fp4_smoke.py

Exercises Quartet-II's FP4 linear: forward = CUTLASS sm120 FP4 GEMM (flashinfer
mm_fp4, backend=cutlass), backward = all-GEMM FP4 with RHT128 Hadamard rotation.
Success = forward + backward complete with finite grads, no multi-libcudart crash.
"""
import torch                      # import FIRST: loads cu13 libcudart into the process
import quartet2                   # noqa: F401  (registers the quartet2::fp4_mm op)
from quartet2.linear import Quartet_II_linear

torch.manual_seed(0)
dev = "cuda:0"
M, K, N = 128, 512, 256           # rows%128==0; in/out features 128-multiples

lin = Quartet_II_linear(K, N, bias=False).to(dev).to(torch.bfloat16)
x = torch.randn(M, K, device=dev, dtype=torch.bfloat16, requires_grad=True)

y = lin(x)                        # FP4 forward GEMM (CUTLASS sm120)
print(f"[fwd] y.shape={tuple(y.shape)} dtype={y.dtype} finite={torch.isfinite(y).all().item()}")

loss = y.pow(2).mean()            # bf16 reduction (no FP32 materialization)
loss.backward()                   # all-GEMM FP4 backward (RHT128 + EDEN)
print(f"[bwd] x.grad finite={torch.isfinite(x.grad).all().item()} "
      f"w.grad finite={torch.isfinite(lin.weight.grad).all().item()}")
print(f"[loss] {loss.item():.6f}")
print("DIRECT-FP4 FWD+BWD: OK")

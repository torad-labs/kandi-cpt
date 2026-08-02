# vendor/ — the pieces the package indexes cannot give back

## quartet2-0.0.2-cp312-abi3-linux_x86_64.whl

The fp4 training kernels (`train_cpt.py` imports `quartet2.linear` on every `precision: fp4`
run). Upstream is IST-DASLab/Quartet-II (the ICML 2026 NVFP4 paper); it is on NO package index,
and until this wheel existed the only copy of the built extension lived in this machine's venv —
the single reason the trainer image could not be rebuilt anywhere else.

**This wheel is a REPACK of the proven install, not a rebuild.** The `.so` inside is
byte-identical to `.venv/.../quartet2/_quartet2.abi3.so` (sha256
7b0fd396eeb72be021d9ccac28b0c02d50529293529a57a8812ad166c9deca07, verified at packing). That is
deliberate: it is the exact binary every proven fp4 run trained through.

Rebuild-from-source status, measured 2026-08-01: upstream HEAD (5f2a47e) FAILS to link —
duplicate `ptx_type_name<...>` symbols (an ODR bug introduced after our artifact was built).
A fresh checkout cannot currently reproduce this wheel; that is exactly why the proven binary
is vendored rather than a build recipe.

Caveats, honestly:
- **CUDA arch: sm_120 only** (this box's Blackwell). The fast path (`Quartet_II_fn`) is gated
  `capability >= 10` anyway; every older GPU takes `Quartet_II_pseudoquant_fn`, which is pure
  torch + Triton (JIT per-GPU at runtime) and never touches these cubins. A non-Blackwell cloud
  box therefore works, on the fallback path.
- **Python tag cp312-abi3**: stable-ABI, so it installs into 3.12+ (the venv runs 3.13).
- **Upstream ships NO LICENSE file.** Default all-rights-reserved: this wheel may live in this
  PRIVATE repo and in PRIVATE images, and must never ride into anything public.

## venv-freeze.txt

`pip freeze --all` of the proven venv (153 distributions, torch 2.12.0+cu130 stack). With the
quartet2 line pointing at the wheel above, the venv is reconstructible in principle from public
indexes + this directory — remove "only this machine can build the trainer image" from the risk
register. The docker image still BAKES the proven venv (docs/cloud-trainer-image.md); a
pip-reconstructed venv is NOT proven to train bit-identically until someone runs that experiment.

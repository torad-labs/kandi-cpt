# GPU/CUDA kernel engineering playbook — transferable techniques

Distilled from the TQ4 megakernel + spec-decode verify campaign (2026-06). These are **reusable
techniques**, not project specifics — they transfer to any future kernel / CUDA / low-precision work
(the megatrainer included). Four categories, because they're genuinely four different disciplines:
**(1) correctness proofs, (2) kernel coding idioms, (3) measurement, (4) posture.** Read this before
touching a hot kernel; it's the accumulated "how we don't get burned."

---

## 1 — Correctness-proof techniques (make the change un-wrong before you trust it)

- **Exhaustive small-domain bit-exactness.** When a transform has a *small finite* input domain, don't
  sample — **enumerate the whole domain** on-GPU and assert bit-identity vs the reference. The PRMT-LUT
  dequant was proven over **all 65,536 nibble-pairs × adversarial codebooks**, not a random subset. If
  the domain fits in a sweep, a sweep is a *proof*, not a test.
- **Byte-identity gating for default-off levers.** Any new code path that ships **default-off** must be
  proven to leave the proven path **byte-identical** — sha256 the emitted PTX/cubin for the unchanged
  recipes. Then the lever *cannot* regress the baseline by construction, and review is about the opt-in
  path only. (build.rs folds the env flag into the PTX cache key so default and the flag can't drift.)
- **Cross-body parity as the invariant.** An optimized "fast" path is gated against a slow **reference**
  path, row-for-row, bit-exact (`verify-step-parity-gate.sh`: M-row verify argmax == M=1 `step()` argmax,
  128/128). Generalize: *every* fast path earns its keep only by bit-matching a trusted reference path.
  Lock it as a CI gate so it can't silently drift.
- **Adversarial parameters, not just random inputs.** Argmax-flip / rounding-edge bugs (the "token-17"
  class) hide behind *near-ties*, not random data. Sweep **adversarial** parameters (worst-case
  codebooks, near-tie logits) — those are where bf16-vs-f32 rounding and accumulator-order bugs surface.
- **Prove the order, not the width, when precision-matching.** Two paths that must agree losslessly must
  match the **operation order** (the sequence of rounds), not merely the storage type. Lossless breakage
  is almost always a reordered scale-combine, not a "needs more bits" — fix the order, don't raise precision.

## 2 — Kernel coding idioms (the SASS-level moves that pay)

- **`byte_perm` / PRMT register-LUT.** A small LUT (≤16 entries) packed into a few registers + a
  `__byte_perm` chain replaces per-element SMEM/global gathers → **zero memory ops** for the lookup. The
  TQ4 codebook (16 bf16 centroids) → 4 packed regs + byte-permute. Transfers to any small-codebook dequant
  (and to bf16-pair LUTs at ~2× chain cost).
- **Accumulate WIDE, store NARROW.** The accumulator width is forced by the **running-sum magnitude**
  (correctness), independent of the I/O precision. int8×int8 → **int32** (exact integer, mandatory — the
  sum overflows int8/int16) → ×scale → **bf16** stored. Float MMA → **f32** internal → narrow. The wide
  accumulator is a transient exact container, *not* a precision raise; the *stored* precision is the floor.
- **Scale-first vs scale-last (the scheme fork).** *When* you apply the quant scales decides the
  accumulator magnitude → the required width → the hardware path:
  - **scale-LAST (integer path):** accumulate raw integer products → int32 → scale at the end. Uses int8
    tensor cores (IMMA/dp4a). Pays: int8 act-quant + int32 register width. Worth it only when **compute-bound**.
  - **scale-FIRST (dequant-to-float):** dequant weight → bf16, multiply real activation, accumulate bf16.
    Normal-magnitude accumulator, no int8 machinery. Better for **memory/latency-bound** decode. (This is
    what llama.cpp / the Mali engine do.)
  - **Pick by what the kernel is bound on** (§3), not by reflex. They produce the same answer.
- **Offline weight repack (Marlin/Machete).** Pre-shuffle quantized weights *offline* into the exact
  layout the warp consumes, so runtime loads are **static-index 128-bit** and the dequant is
  register-cheap. Only the storage layout changes; the quantization is untouched. (TRT-LLM
  `preprocess_weights_for_mixed_gemm`; MNN's `[l/lp,h,lp]` Image layout — see the steal-lists.)
- **Multi-row-output matvec.** Compute N output rows per workgroup so the **activation loads + addressing
  amortize across the rows** (the K/M decode lever). Empirically falls off past ~16 rows on both CUDA and
  Mali — that's M-saturation, a real ceiling, not a tuning miss.
- **Everything experimental behind an env flag, default-off.** New kernel paths ship gated
  (`QGRE_PRMT_DEQUANT`, `QGRE_SCHEME_B`, `QGRE_MROWS_SKIP`) so the proven path is the default, experiments
  are opt-in, and A/B is a one-env-var flip with both paths in the same binary.

## 3 — Measurement & diagnosis (profile, don't infer — global §13)

- **Profile before optimizing.** `ncu` the kernel; never reason from wall-clock + a mental model. The one
  set of numbers that settles the bound: **DRAM%** (bandwidth), **SM%/FMA%** (compute), **LSU%** (load/store
  issue), **No-Eligible / warps-active** (latency starvation). A 30-second `ncu` has repeatedly redirected
  days of planned work aimed at the wrong bottleneck.
- **Per-op attribution via skip-deltas.** Env-gate each op class to **skip** it, measure the wall-clock
  delta, attribute the cost (`QGRE_MROWS_SKIP`). Turns "the kernel is slow" into "gate_up is 26%, down is
  22%, residual is 30%" — so you optimize the op that matters.
- **Live wall-clock ≠ profiler replay.** `ncu` *app-replay* per-kernel duration is **not** live throughput
  (it reruns under instrumentation, in isolation). Always confirm a claimed speedup with a **live, no-ncu**
  bench over many iters. (We caught a reported "74 ms" that was an app-replay artifact; live verify was
  unchanged at ~100 ms.)
- **The "does the constraint walk?" test.** After an optimization, **re-profile**: did the binding
  constraint *move* (regs off the cap, No-Eligible down, the bound shift type) — or did you just shave a
  few % without moving the wall? A free lossless win that **doesn't walk the constraint** is a *keep*, not
  a *breakthrough* — bank it and say so honestly. Cheapening an op that isn't the bound won't move tok/s.
- **The resource-cascade model.** Register pressure → occupancy cliffs (`⌊RF/T⌋`; Mali 0-32/33-64); the
  binding constraint *walks* as you remove each one; **every lever is inert until its constraint is the
  binding one.** You cannot skip a step. (See `MEGAKERNEL-KNOB-CASCADE.md`; the Mali twin in
  `MALI-ARM-STEAL-LIST.md`.)
- **Stale batch-size / regime assumptions.** "Decode is memory-bound, tensor cores useless" is true at M=1
  and false the moment you batch (M-row verify). Re-`ncu` the assumption at the new regime before inheriting
  an old conclusion.

## 4 — Engineering posture (the judgment that separates good from impressive)

- **Don't oversell.** Report the keeper-that-didn't-walk-the-constraint honestly. "KEEP — free + lossless,
  but the bound is still there and here's why" is the senior signal, not "shipped a speedup."
- **Commit at every green step.** A context reset / `git reset` already ate hours of uncommitted kernel
  work once. Every gate-green increment is a commit; a reset must cost minutes, not hours.
- **The gates are the truth — independent of how good the reasoning was.** No reasoning, however deep,
  substitutes for the bit-exact gate. Re-run the gates on every commit before trusting a number; the one
  time a report was wrong (a stale "diverges" line) was caught by a re-run, not by the reasoning.
- **Localize with evidence first (§5).** Read the actual kernel + the actual profile before proposing.
  Bisect to pin the failing op with `file:line` evidence; don't pattern-match a fix from the symptom.
- **Keep the precision floor; never silently raise it (§15).** The accumulator-is-wider point (§2) is the
  *only* sanctioned widening, and it's an exact transient container, not a stored-precision raise. Any
  actual upcast of a stored tensor/accumulator is a deliberate, evidence-backed, human decision.

---

*Companion docs:* `MEGAKERNEL-KNOB-CASCADE.md` (the resource model), `TRTLLM-STEAL-LIST.md` +
`MALI-ARM-STEAL-LIST.md` (reference implementations), `VERIFY-HIGHK-SPILL-PLAN.md` (the live campaign).

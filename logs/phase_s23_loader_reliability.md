# phase_s23 — eval-side loader reliability incident + fixes (2026-07-26, 01:00–04:00)

## 2026-07-26 PM addendum — the bf16 purge (operator law: no bf16 residency, ever)

The reliability fixes below were superseded the same day by a harder directive: **bf16 must
not exist as a resident/stored precision anywhere in the loop** (brain law #916/#925/#833:
"there is no place where bf16 is required — it's a choice"). What changed:

- **Convert-once**: `scripts/convert_base_int8.py` (streaming, ~5GB peak) produced
  `models/gemma-4-E2B-it-int8/` — int8 masters bit-exact vs trainer init
  (`quantize_int8_rowwise` determinism; verified on random linears). Per-run bf16 ingestion
  is dead.
- **Evals run the TRAINING FORWARD**: `pipeline.model_load.load_causal_ckpt_quartet` —
  meta skeleton → Quartet_II_linear at exactly the quant_keys linears → int8 masters from
  the shard (no requantization, no bf16 decode) → fp4 GEMM forward. Scorer pads to the fp4
  kernel's 128-token multiple; GA uses a manual greedy loop (generate() breaks the shape
  constraint). `load_causal_ckpt_bf16` family = S4 legacy, scheduled for removal.
- **Trainer init native**: `build_model` init_from detects quant_keys → quartet-native load
  (trainable masters), skipping the dequant round-trip; downstream swaps no-op by isinstance.
- **from_pretrained bf16 fallbacks → fail-loud** in every loader (no silent residency).
- **The gate**: `scripts/qgre/check_no_bf16.sh` (wired into `run_tests.sh`) — code-level
  bf16 usage outside declared seams S1–S5 FAILS. Green as of 2026-07-26 PM.
- **Eval env law**: CUDA_HOME must be a cu13 toolkit (session baseline is toolkit-less
  12.8; the flashinfer JIT cache was built with 13.3) — set in the rebaseline/mopup scripts.
- **First certified quartet-instrument numbers**: r005 FC 0.6301/0.2945, s2 FC 0.7329
  [0.6793,0.7804] — PASS on both; stable ~−3.5pt offset vs the bf16 instrument (fp4 forward
  vs upcast decode). bf16 series preserved at logs/runs/bf16_instrument/ (s2's detailed
  bf16 report was overwritten by the re-run; its number survives in the session transcript).

Pending operator recipe decisions (declared in the gate, not defaulted): TQ4-embed
(S3 selective embeddings), fp8 scales (save format), S5 legacy instruments
(geometry_probe/eval_voice/eval_cpt/roberta embedders) re-baseline-or-retire, S4 removal
when bf16-era checkpoints convert.

---



Companion to the save-side fix (streaming save + preflight + oom-shield, landed 2026-07-25).
That fix held: **5/5 checkpoints survived** (s2_reasoning, s3_r005/r010/r020/r030). This file
is the EVAL-side story: seven eval processes died to the whale-reaper in one night, and the
root causes were all in the model-LOAD path, not ambient pressure.

## Kill chain (all SIGTERM, whale-reaper journal)

| time | process | rss | note |
|---|---|---|---|
| 23:07 | s2 GA (score_sentinel_live) | — | candidate load, pre-fix |
| 00:04 | r005 GA | 20.1GB | candidate load, avail=6% |
| 01:00 | r010 GA | ~16.4GB | candidate load, avail=9% |
| 03:12 | r005 GA ×2 | 19.8/18.7GB | BASE load at FC→GA handoff |
| 03:25 | r010 FC | 18.8GB | base load |
| 03:03 | s2 FC ×2 | (CUDA OOM) | different cause — see §2 |

## §1 The loader was the whale — three stacked causes

1. **fp32 construction.** `Gemma4ForCausalLM(cfg)` materializes params in torch's default
   dtype: 4.63B × 4B = **18.5GB before the final `.to(bf16)`**. Every death was the reaper
   catching that climb. Fix: construct on `meta`, `.to(bf16)` (free on meta), `to_empty(cpu)`
   → 9.3GB bf16, no fp32 phase.
2. **full-dict reads.** `load_file` + dequant built the raw dict (8.6GB) and the bf16 dict
   (~10GB) simultaneously. Fix: per-tensor streaming (`iter_causal_bf16`, 64-tensor chunks
   into `load_state_dict(strict=False)` — per-tensor copy semantics, order-independent).
3. **mmap page cache in VmRSS.** `safe_open` mmaps the shard; faulted pages count in VmRSS
   (= the reaper's metric): RssFile 9.24GB on top of RssAnon 9.55GB.
   `posix_fadvise(DONTNEED)` **cannot evict still-mapped pages** — proven no-op on this box
   (RssFile 2054MB before AND after). Fix: `_SafetensorsRead`, a ~40-line pread-based
   safetensors reader (pread caps at 0x7ffff000/call — loop for >2GB tensors). Page cache
   stays kernel-owned; VmRSS shows anonymous only.
   Result: base load peak **18→10.8GB total** (RssAnon 10.47 + RssFile 0.37).

`to_empty` leaves persistent=False buffers uninitialized — re-materialized exactly as
`__init__` computes them (rotary inv_freq ×4, embed_scale ×2), fp32 math then copy_-round
to bf16. Deliberately unguarded: a missed buffer = NaN outputs = silently wrong evals;
an AttributeError is the honest failure.

## §2 The release-contract bug (s2 FC CUDA OOM, 03:03)

`release_eval_model(model)` did `del model` — deleting the HELPER's local binding while the
CALLER's loop variable kept the base model GPU-resident: 12.56GiB allocated at the
candidate's `.to(device)` → OOM on a 15.5GB card. The sentinel GA steps survived only
because their frames exit right after release. Fix: caller `del model` FIRST, then
`release_eval_model()` (no arg) does gc + empty_cache + malloc_trim. Contract is in the
docstring; the four call sites updated (eval_probes ×1, score_sentinel_live ×3).

NOTE: the same OOM was independently diagnosed elsewhere as from_pretrained materializing
the multimodal tower. The OOM message says otherwise ("this process has 12.91 GiB in use"
at the candidate move = base never freed). The text-only reroute in eval_probes.load_model
is kept anyway (leaner GPU resident), now with the streaming treatment that makes it safe.

## §3 Receipts writer shape gap

`phase_s23_write_receipts.py._probe_summary` guessed flat `report.tuned_acc` /
`forced_choice`; `eval_probes.build_report` nests under `install_strength.forced_choice`.
All extraction paths missed → replay-curve fit would read "incomplete" over landed evals.
Fixed to read `install_strength.*` first.

## Exactness proofs (all passed)

- Streaming dequant vs old load_file path: 50 tensors (20 int8), torch.equal, 0 mismatches.
- 6 rematerialized buffers vs fp32-recompute→bf16-round (old construct path): exact.
- `_SafetensorsRead` vs safetensors' own reader: 25 keys int8 ckpt + 3 biggest base-shard
  tensors (incl. 4.7GB per-layer embedding): bit-identical.
- Live red-green: r020 GA install 40/54=0.7407 / GA 9-10/10, r030 41/54=0.7593 / 10/10,
  s2 43/54=0.7963 / 9/10 — base reproduces 16/54 install + 10/10 GA every time.

## Cross-reference

- Save side: streaming save + `save_preflight` + oom-shield in train_cpt.py / train_fp4.sh.
- The fill/mop-up machinery: logs/runs/phase_s23_{fc_ga_fill,mopup}.sh (+ .log).
- psi_full=92 thrash episode ~03:34 (swap 94% used) killed the first fill script mid-retry;
  "no unprotected whale >8G — earlyoom owns this case" in the journal.

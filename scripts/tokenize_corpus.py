#!/usr/bin/env python3
"""SUPERSEDED by `festival-factory tokenize` (Rust; tokenizers + arrow-ipc; structural parity, 2026-05-31).
The orchestrator now drives the Rust tokenizer; the trainer's make_loader reads either format. Kept as fallback.

Tokenize the multi-edition corpus into a HF Datasets Arrow file for the trainer.

Collects, across all editions (or --editions): curated 10_clean docs (filtered by the curation
keep-list, holdout excluded), the 20_synthetic expansion, plus data/_shared/knowledge — then
optionally mixes ~10% data/_shared/replay (generic text, prevents forgetting; skipped if empty).
Adds <bos>/<eos> per doc, seeded-shuffles doc order, packs into seq_len blocks.

    PYTHONPATH=scripts python scripts/tokenize_corpus.py [--editions …] [--replay-frac 0.10]
        [--seq-len 512] [--no-synthetic] [--no-lever-metadata]

Output: data/_build/30_tokenized/corpus_v1.arrow + corpus_v1_stats.json.
  Columns: input_ids (int) and, unless --no-lever-metadata, the lever metadata aligned to the packed
  blocks — entity_mask (0/1 per token, Lever 1), src (source-type id per token, Lever 2), and fact_ids
  (the distinct (artist|stage|day) facts each block touches, Lever 4). The trainer reads input_ids by
  default and only consumes the metadata when the corresponding lever is enabled, so emitting it is
  backward-compatible.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter

from transformers import AutoTokenizer
from datasets import Dataset

import config as C
from pipeline.editions import iter_editions

# source-type ids (Lever 2). raw_knowledge covers facts/set-times/rosters/hydrated/encyclopedic prose.
SRC = {"raw_knowledge": 0, "qa": 1, "conversation": 2, "replay": 4}


def _src_of(tag: str, path: str) -> int:
    p = path.lower()
    if "qa_pairs__" in p:
        return SRC["qa"]
    if "friend_chat__" in p or "/_cultural/" in p:
        return SRC["conversation"]
    return SRC["raw_knowledge"]


def chunk_ids(ids: list, seq_len: int) -> list:
    n_full = len(ids) // seq_len
    return [ids[i * seq_len:(i + 1) * seq_len] for i in range(n_full)]


def collect(domain, editions_only, use_synth: bool):
    """Return list of dicts {tag, path, text}. tag ∈ clean|synthetic|hydrated|knowledge (for stats);
    path is the full posix path (or jsonl path) used to classify source-type."""
    keep_path = C.BUILD / "curation" / "keep_list.json"
    keep = set(json.loads(keep_path.read_text())) if keep_path.exists() else None
    docs = []
    for ed in iter_editions(domain, only=editions_only, with_raw=False):   # consumes 15_structured/25_hydrated, not 00_raw
        clean_root = ed.dir / "10_clean"
        for md in sorted(clean_root.rglob("*.md")) if clean_root.is_dir() else []:
            if md.name == "README.md":
                continue
            if keep is not None and str(md.relative_to(C.ROOT)) not in keep:
                continue
            t = md.read_text(encoding="utf-8", errors="replace").strip()
            if t:
                docs.append({"tag": "clean", "path": md.as_posix(), "text": t})
        if use_synth:
            synth_root = ed.dir / "20_synthetic"
            for md in sorted(synth_root.rglob("*.md")) if synth_root.is_dir() else []:
                # _facts/ + _rosters/ are the OLD templated relational corpus — SUPERSEDED by the Rust
                # engine's 25_hydrated/ (grounded, faithfulness-gated). Keep only chunk-strategy prose,
                # plus _cultural/ + _settimes/ (the new agent-generated installs).
                p = md.as_posix()
                if "/_facts/" in p or "/_rosters/" in p:
                    continue
                t = md.read_text(encoding="utf-8", errors="replace").strip()
                if t:
                    docs.append({"tag": "synthetic", "path": p, "text": t})
            hyd_root = ed.dir / "25_hydrated"
            for jf in sorted(hyd_root.rglob("*.jsonl")) if hyd_root.is_dir() else []:
                if jf.with_name(jf.name + ".partial").exists():   # a synth aborted mid-write → this file is STALE
                    print(f"  WARNING: {jf} has a sibling .partial — the last synth for this dir ABORTED; "
                          f"SKIPPING this stale file (re-run the synth stage to refresh it before tokenizing — "
                          f"tokenize is the one-way door into training).", file=sys.stderr)
                    continue
                for line in jf.read_text(encoding="utf-8", errors="replace").splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        txt = json.loads(line).get("text", "").strip()
                    except Exception:
                        continue
                    if txt:
                        docs.append({"tag": "hydrated", "path": jf.as_posix(), "text": txt})
    for md in sorted((C.SHARED / "knowledge").rglob("*.md")) if (C.SHARED / "knowledge").is_dir() else []:
        t = md.read_text(encoding="utf-8", errors="replace").strip()
        if t:
            docs.append({"tag": "knowledge", "path": md.as_posix(), "text": t})
    return docs


def _block_fact_ids(doc_ranges, n_blocks, seq_len):
    """Per-block union of fact_ids, via two-pointer over docs sorted by token start (both ordered)."""
    out = [[] for _ in range(n_blocks)]
    lo = 0
    for b in range(n_blocks):
        bs, be = b * seq_len, (b + 1) * seq_len
        while lo < len(doc_ranges) and doc_ranges[lo][1] <= bs:   # range fully before block
            lo += 1
        seen, j = set(), lo
        while j < len(doc_ranges) and doc_ranges[j][0] < be:      # range starts before block end
            if doc_ranges[j][1] > bs:                             # and overlaps
                seen.update(doc_ranges[j][2])
            j += 1
        out[b] = sorted(seen)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--editions", help="restrict to 'city/year,…' (default: all with corpus)")
    ap.add_argument("--domain", default="edc", help="domain slug — picks data/<domain> + its registry (was hardcoded 'edc')")
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--replay-frac", type=float, default=0.10, help="~general-text fraction (5–10% sweet spot)")
    ap.add_argument("--no-synthetic", action="store_true")
    ap.add_argument("--no-lever-metadata", action="store_true",
                    help="skip entity_mask/src/fact_ids emission (faster; levers 1/2/4 then unavailable)")
    ap.add_argument("--out", default="corpus_v1", help="output arrow/stats basename under TOKENIZED_DIR "
                    "(default corpus_v1; use a distinct name for lever A/B corpora to avoid clobbering)")
    args = ap.parse_args()
    emit_meta = not args.no_lever_metadata

    tok = AutoTokenizer.from_pretrained(str(C.MODEL_PATH), use_fast=True)
    bos, eos = tok.bos_token_id, tok.eos_token_id
    if bos is None or eos is None:
        sys.exit("tokenizer missing bos/eos")
    if emit_meta:
        from pipeline import span_match as SM

    docs = collect(args.domain, args.editions, not args.no_synthetic)
    random.seed(0)
    random.shuffle(docs)
    if not docs:
        sys.exit("no corpus docs found (run extract/synth first)")

    all_ids: list[int] = []
    all_mask: list[int] = []
    all_src: list[int] = []
    doc_ranges: list[tuple] = []                 # (tok_start, tok_end, fact_ids) for per-block fact join
    by_tag: Counter = Counter()
    by_source_tokens: Counter = Counter()
    masked_tokens = 0
    for d in docs:
        text = d["text"]
        if emit_meta:
            enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
            ids = enc["input_ids"]
            mask = SM.entity_mask(enc["offset_mapping"], text)
            src = _src_of(d["tag"], d["path"])
            fids = SM.fact_ids_in(text)
            start = len(all_ids)
            all_ids += [bos, *ids, eos]
            all_mask += [0, *mask, 0]
            all_src += [src] * (len(ids) + 2)
            doc_ranges.append((start, len(all_ids), tuple(fids)))
            masked_tokens += sum(mask)
        else:
            ids = tok.encode(text, add_special_tokens=False)
            all_ids += [bos, *ids, eos]
        by_tag[d["tag"]] += 1
        by_source_tokens[d["tag"]] += len(ids)
    corpus_tokens = len(all_ids)

    # replay mix (generic text, ~replay_frac of corpus) — skip if _shared/replay empty
    replay_dir = C.REPLAY
    replay_files = list(replay_dir.rglob("*")) if replay_dir.is_dir() else []
    replay_files = [f for f in replay_files if f.is_file() and f.suffix in (".md", ".txt")]
    replay_tokens = 0
    if args.replay_frac > 0 and replay_files:
        target = int(corpus_tokens * args.replay_frac)
        pool: list[int] = []
        for f in replay_files:
            pool += tok.encode(f.read_text(encoding="utf-8", errors="replace"), add_special_tokens=False)
            if len(pool) >= target:
                break
        rep = pool[:target]
        all_ids += [bos, *rep, eos]
        if emit_meta:
            all_mask += [0] * (len(rep) + 2)
            all_src += [SRC["replay"]] * (len(rep) + 2)
        replay_tokens = len(rep)
    elif args.replay_frac > 0:
        print(f"[warn] _shared/replay is empty — skipping the {args.replay_frac:.0%} replay mix "
              f"(populate it with a generic-text slice for the full run).", file=sys.stderr)

    total = len(all_ids)
    chunks = chunk_ids(all_ids, args.seq_len)
    cols = {"input_ids": chunks}
    if emit_meta:
        cols["entity_mask"] = chunk_ids(all_mask, args.seq_len)
        cols["src"] = chunk_ids(all_src, args.seq_len)
        cols["fact_ids"] = _block_fact_ids(doc_ranges, len(chunks), args.seq_len)

    OUT = C.TOKENIZED_DIR
    OUT.mkdir(parents=True, exist_ok=True)
    Dataset.from_dict(cols).save_to_disk(str(OUT / f"{args.out}.arrow"))
    stats = {
        "total_tokens": total, "num_chunks": len(chunks), "seq_len": args.seq_len,
        "docs_by_tag": dict(by_tag), "tokens_by_tag": dict(by_source_tokens),
        "replay_tokens": replay_tokens, "editions": args.editions or "all",
        "lever_metadata": emit_meta,
    }
    if emit_meta:
        stats["entity_masked_tokens"] = masked_tokens
        stats["entity_mask_frac"] = round(masked_tokens / max(1, corpus_tokens), 4)
        stats["src_token_dist"] = {k: sum(1 for s in all_src if s == v) for k, v in SRC.items()}
    (OUT / f"{args.out}_stats.json").write_text(json.dumps(stats, indent=2))
    print(f"docs: {dict(by_tag)}  |  corpus_tokens={corpus_tokens:,}  replay={replay_tokens:,}  "
          f"total={total:,}  chunks={len(chunks)}")
    if emit_meta:
        print(f"lever-metadata: entity_mask {masked_tokens:,}/{corpus_tokens:,} "
              f"({stats['entity_mask_frac']:.1%}) | src {stats['src_token_dist']}")
    print(f"saved {OUT/(args.out + '.arrow')}")


if __name__ == "__main__":
    main()

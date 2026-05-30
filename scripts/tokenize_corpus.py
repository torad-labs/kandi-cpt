#!/usr/bin/env python3
"""Tokenize one edition's cleaned corpus into a HF Datasets Arrow file.

Points at a SINGLE edition: data/edc/las_vegas/2026/10_clean/. Walks its
source subfolders (official/, youtube/, reference/, …), reads every .md file,
adds <bos>/<eos> markers, and chunks into seq_len blocks for causal-LM training.

NOTE — multi-edition (for the rewrite): the corpus is now partitioned per
edition at data/<festival>/<city>/<year>/. Training across festivals means
iterating those editions (+ data/_shared/) rather than this one hard-coded
path. Also still UNMIXED here: the edition's 20_synthetic, data/_shared/knowledge,
and data/_shared/replay (~10–30%). Wiring that mix is the fact-density rebuild's
job, deliberately not done in this path-repoint.

Run:
    python scripts/tokenize_corpus.py

Output:
    data/_build/30_tokenized/corpus_v1.arrow — one column `input_ids` (int32),
    each row a seq_len-token block.
"""
import json
import sys

from transformers import AutoTokenizer
from datasets import Dataset

from config import MODEL_PATH, CLEAN as CORPUS_DIR, TOKENIZED_DIR as OUT_DIR
SEQ_LEN = 512   # batch=1/seq=512/vocab=262144/bf16 logits = 0.25 GB; CE backward ~0.5 GB

def collect_documents() -> list[tuple[str, str]]:
    """Return (relative_path, text) for every .md file under the edition's 10_clean/."""
    docs = []
    for md in sorted(CORPUS_DIR.rglob("*.md")):
        if md.name == "README.md":      # stage signpost, not corpus content
            continue
        rel = md.relative_to(CORPUS_DIR)
        text = md.read_text(encoding="utf-8").strip()
        if text:
            docs.append((str(rel), text))
    return docs

def chunk_ids(ids: list[int], seq_len: int) -> list[list[int]]:
    """Pack token ids into fixed seq_len chunks, dropping the final remainder."""
    n_full = len(ids) // seq_len
    return [ids[i * seq_len : (i + 1) * seq_len] for i in range(n_full)]

def main():
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH), use_fast=True)
    bos = tokenizer.bos_token_id
    eos = tokenizer.eos_token_id
    if bos is None or eos is None:
        sys.exit("Tokenizer missing bos_token_id or eos_token_id")

    docs = collect_documents()
    print(f"Loaded {len(docs)} documents from {CORPUS_DIR}")

    all_ids: list[int] = []
    per_doc_tokens: dict[str, int] = {}
    for rel, text in docs:
        ids = tokenizer.encode(text, add_special_tokens=False)
        per_doc_tokens[rel] = len(ids)
        all_ids.append(bos)
        all_ids.extend(ids)
        all_ids.append(eos)

    total = len(all_ids)
    chunks = chunk_ids(all_ids, SEQ_LEN)
    print(f"Total tokens: {total:,}  |  chunks of {SEQ_LEN}: {len(chunks)}  |  dropped tail: {total - len(chunks)*SEQ_LEN}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ds = Dataset.from_dict({"input_ids": chunks})
    out_path = OUT_DIR / "corpus_v1.arrow"
    ds.save_to_disk(str(out_path))

    stats_path = OUT_DIR / "corpus_v1_stats.json"
    stats_path.write_text(json.dumps({
        "total_tokens": total,
        "num_chunks": len(chunks),
        "seq_len": SEQ_LEN,
        "per_document_tokens": per_doc_tokens,
    }, indent=2))
    print(f"Saved {out_path}  +  {stats_path}")

if __name__ == "__main__":
    main()

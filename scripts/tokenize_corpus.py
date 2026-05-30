#!/usr/bin/env python3
"""Tokenize the corpus/ directory into a HF Datasets Arrow file.

Walks corpus/{tier1_edc,tier2_releases,tier2_safety,tier2_reddit,tier3_history,general_anchor}/
and reads every .md file. Each file becomes one document with <bos> and <eos>
markers. All documents are concatenated and chunked into seq_len-sized blocks
for causal-LM training.

Run:
    python scripts/tokenize_corpus.py

Output:
    tokenized/corpus_v1.arrow — dataset with one column `input_ids` (int32),
    each row is a seq_len-token block.
"""
from pathlib import Path
import json
import sys

from transformers import AutoTokenizer
from datasets import Dataset

ROOT = Path(__file__).resolve().parents[1]
CORPUS_DIR = ROOT / "corpus"
OUT_DIR = ROOT / "tokenized"
MODEL_PATH = Path("/home/marcos/Documents/dev/models/gemma-4-E2B-it")
SEQ_LEN = 512   # batch=1/seq=512/vocab=262144/bf16 logits = 0.25 GB; CE backward ~0.5 GB

def collect_documents() -> list[tuple[str, str]]:
    """Return (relative_path, text) for every .md file under corpus/."""
    docs = []
    for md in sorted(CORPUS_DIR.rglob("*.md")):
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

# _build — cross-edition build artifacts

Not per-festival. Holds trainer-ready output mixed across editions.

- `30_tokenized/` — packed parquet/arrow shards from `scripts/tokenize_corpus.py`.
  **Gitignored + regenerable** (`mkdir`'d on run); does not survive a fresh clone.

The clean + synthetic + replay mix that feeds this is the fact-density rebuild's job —
today `tokenize_corpus.py` reads one edition's `10_clean` only.

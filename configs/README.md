# configs — pipeline configuration

Versioned knobs for the pipeline, separated from code so runs are reproducible and diffable:
generation configs + prompt versions (for `synth_corpus.py`), data-mix ratios
(clean : synthetic : replay), and tokenizer settings (seq_len, special tokens).

- **Git:** committed. Reference configs by version in run logs so any corpus build is reproducible.

"""Canonical paths + base-model location for Kandi CPT scripts.

Single source of truth. Folder restructures and the base-model path live HERE,
not duplicated across every script — the base-model path was hardcoded in 6
scripts (AST-grep `Path($$$)` flagged it), and the data paths were repointed by
hand on every layout change. Import from here instead.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = Path("/home/marcos/Documents/dev/infra/models/gemma-4-E2B-it")  # base Gemma 4 E2B (bf16)

DATA = ROOT / "data"
SHARED = DATA / "_shared"
REPLAY = SHARED / "replay"
BUILD = DATA / "_build"
TOKENIZED_DIR = BUILD / "30_tokenized"
TOKENIZED_ARROW = TOKENIZED_DIR / "corpus_v1.arrow"
CHECKPOINTS = ROOT / "checkpoints"
CPT_RUNS = CHECKPOINTS / "cpt"     # full-param CPT run outputs (gitignored)
LOGS = ROOT / "logs"
CONFIGS = ROOT / "configs"


def edition(festival: str, city: str, year) -> Path:
    """Per-edition root: data/<festival>/<city>/<year>/."""
    return DATA / festival / city / str(year)


def raw_dir(festival: str, city: str, year, source: str) -> Path:
    """Scrape landing for one source channel: <edition>/00_raw/<source>/."""
    return edition(festival, city, year) / "00_raw" / source


def load_editions(festival: str = "edc") -> dict:
    """The worldwide edition registry the scrapers iterate (configs/editions_<festival>.json)."""
    import json
    return json.loads((CONFIGS / f"editions_{festival}.json").read_text())


# Default edition for the single-edition scripts. Multi-edition iteration is the
# ingestion rewrite's job — see the tokenize_corpus.py / synth_corpus.py docstrings.
EDITION = edition("edc", "las_vegas", 2026)
CLEAN = EDITION / "10_clean"
SYNTHETIC = EDITION / "20_synthetic"

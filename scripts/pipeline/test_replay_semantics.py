"""Replay is a RATIO TO DOMAIN, not a share of the final mixture — and the stats must say which.

`--replay-to-domain-ratio r` emits `domain_tokens * r` replay tokens, so replay ends up at
r/(1+r) of the mixture: 0.10 is 9.09%, not 10%. The old flag was named `--replay-frac` and helped
as "general-text fraction", which reads as the final share and overstates replay by ~10% relative.
Renaming rather than redefining keeps every existing checkpoint's meaning intact.

A replay pool smaller than the target silently under-delivers, so the realized share is counted
from the token stream, never from the request.

    PYTHONPATH=scripts .venv/bin/python scripts/pipeline/test_replay_semantics.py
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TOKENIZE = ROOT / "scripts" / "tokenize_corpus.py"


def _source() -> str:
    return TOKENIZE.read_text(encoding="utf-8")


def test_ratio_to_final_share_arithmetic():
    """The identity the whole finding rests on: ratio r lands at r/(1+r) of the mixture."""
    for ratio, expected in ((0.05, 0.047619), (0.10, 0.090909), (0.20, 0.166667), (0.30, 0.230769)):
        assert abs(ratio / (1.0 + ratio) - expected) < 1e-5, ratio
    # The gap is not cosmetic: a "30%" knob delivers under 24% of the mixture.
    assert 0.30 / 1.30 < 0.24


def test_stats_stamp_all_three_replay_numbers():
    """A receipt quoting one replay number is not interpretable — require the ratio, the share it
    implies, and what actually landed."""
    src = _source()
    for key in ("replay_to_domain_ratio", "requested_final_replay_fraction",
                "realized_final_replay_fraction", "replay_target_tokens"):
        assert f'"{key}"' in src, f"stats must stamp {key}"
    # Realized must come from the token stream, not from the requested target — otherwise an
    # exhausted replay pool reports the share it wanted rather than the share it got.
    assert "replay_stream_tokens" in src
    assert 'sum(1 for t in all_block_tag if t == "replay")' in src


def test_realized_is_not_computed_from_the_request():
    """Guard the specific regression: realized_final_replay_fraction must not be derived from the
    ratio (that would make it a restatement of the request and blind to pool exhaustion)."""
    src = _source()
    line = next(ln for ln in src.splitlines() if "realized_final_replay_fraction" in ln and "round(" in ln)
    assert "replay_stream_tokens" in line, line
    assert "replay_to_domain_ratio" not in line, line


def test_pool_shortfall_is_reported_not_silent():
    src = _source()
    assert "replay pool exhausted" in src, "an under-delivered replay mix must warn"
    assert "replay_tokens < replay_target" in src


def test_deprecated_alias_still_accepted_and_mutually_exclusive():
    """Renaming must not break callers; passing both must be a loud error, not a silent pick."""
    src = _source()
    assert "--replay-to-domain-ratio" in src
    assert "--replay-frac" in src, "the old flag must keep working"
    assert "not both" in src, "passing both flags must fail loudly"


def test_replay_documents_are_separated_like_domain_documents():
    """Domain docs are added as [bos, *ids, eos] each; replay must be too.

    Concatenating replay files bare and wrapping the pool once fuses N files into a single
    pseudo-document, so unrelated general text conditions on itself at every file boundary. Latent
    with one replay file on disk today; it fires on the second, and it corrupts the one stream whose
    purpose is preserving base-distribution behaviour.
    """
    src = _source()
    assert "pool += [bos, *ids, eos]" in src, "each replay document needs its own BOS/EOS"
    assert "all_ids += [bos, *rep, eos]" not in src, (
        "wrapping the whole replay pool once is the fusion defect")
    # A truncated tail document must still terminate.
    assert "rep.append(eos)" in src
    # Tag/mask spans must follow the new length (wrappers are now inside rep, not added after).
    assert 'all_block_tag += ["replay"] * len(rep)' in src
    assert "(len(rep) + 2)" not in src, "stale +2 span arithmetic left over from the single wrapper"


def test_help_no_longer_calls_the_ratio_a_fraction_of_the_corpus():
    """The original defect was the WORDS: 'general-text fraction' reads as the final share."""
    src = _source()
    assert "~general-text fraction (5–10% sweet spot)" not in src, (
        "the misleading help string is the defect; it must not survive")
    assert "per domain token" in src or "replay tokens per domain token" in src


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
    print(f"ALL {len(tests)} REPLAY-SEMANTICS TESTS PASS")

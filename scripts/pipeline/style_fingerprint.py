#!/usr/bin/env python3
"""Style fingerprint — the style-domain analog of the ontology (docs/cpt-overhaul/05).

Builds a reviewable **style card** from a user-dropped corpus:
  - mechanical stats (lexical, sentence length, casing/emoji/exclamation, discourse markers)
  - optional LLM prose summary of the register (provider; omit on device-only)

The card is human-approved like an ontology. Generation and judging both anchor on it.
Below style minimums (~2k messages / ~20k tokens of real user text) this stage REFUSES.

    PYTHONPATH=scripts python scripts/pipeline/style_fingerprint.py \
        --domain <d> [--editions …] [--min-tokens 20000] [--min-docs 50] [--device-only]

Outputs:
  data/_build/style/<domain>/fingerprint.json
  data/_build/style/<domain>/style_card.md
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

import config as C
from pipeline.editions import iter_editions


_WORD_RE = re.compile(r"[A-Za-z0-9']+")
_SENT_RE = re.compile(r"[^.!?]+[.!?]+|[^.!?]+$")
# Non-overlapping ranges only (1F600–1F64F sits inside 1F300–1F9FF — listing it twice
# tripped CodeQL py/overly-large-range).
_EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001F9FF"  # misc symbols & pictographs through supplemental symbols
    "\U00002700-\U000027BF"  # dingbats
    "]+",
    flags=re.UNICODE,
)
_DISCOURSE = (
    "anyway", "honestly", "basically", "literally", "like", "you know",
    "i mean", "tbh", "imo", "fwiw", "ngl", "lol", "lmao", "haha",
)


def load_style_docs(domain: str, editions: str | None) -> list[dict]:
    """Load cleaned prose for style fingerprinting (10_clean + optional 00_raw text)."""
    docs: list[dict] = []
    try:
        units = list(iter_editions(domain, only=editions, with_raw=False))
    except Exception:
        units = []
    for ed in units:
        clean = ed.dir / "10_clean"
        if clean.is_dir():
            for path in sorted(clean.rglob("*")):
                if path.suffix.lower() not in {".md", ".txt"} or path.name == "README.md":
                    continue
                text = path.read_text(encoding="utf-8", errors="replace").strip()
                if text:
                    docs.append({"path": str(path), "text": text, "unit": ed.slug})
    # Flat drop folder support: data/<domain>/style_inbox/
    inbox = C.DATA / domain / "style_inbox"
    if inbox.is_dir():
        for path in sorted(inbox.rglob("*")):
            if path.suffix.lower() not in {".md", ".txt", ".jsonl"}:
                continue
            if path.suffix == ".jsonl":
                for i, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines()):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        text = str(obj.get("text") or obj.get("content") or "").strip()
                    except Exception:
                        text = line
                    if text:
                        docs.append({"path": f"{path}:{i}", "text": text, "unit": "inbox"})
            else:
                text = path.read_text(encoding="utf-8", errors="replace").strip()
                if text:
                    docs.append({"path": str(path), "text": text, "unit": "inbox"})
    return docs


def mechanical_stats(docs: list[dict]) -> dict:
    texts = [d["text"] for d in docs]
    all_text = "\n".join(texts)
    words = _WORD_RE.findall(all_text)
    lower_words = [w.lower() for w in words]
    sents: list[str] = []
    for t in texts:
        sents.extend(s.strip() for s in _SENT_RE.findall(t) if s.strip())
    sent_lens = [len(_WORD_RE.findall(s)) for s in sents if s]
    chars = len(all_text)
    letters = sum(1 for c in all_text if c.isalpha())
    upper = sum(1 for c in all_text if c.isupper())
    lower = sum(1 for c in all_text if c.islower())
    excl = all_text.count("!")
    qmarks = all_text.count("?")
    emojis = _EMOJI_RE.findall(all_text)
    markers = Counter()
    blob = " ".join(lower_words)
    for m in _DISCOURSE:
        markers[m] = blob.count(m)
    top_lex = Counter(lower_words).most_common(40)
    openers = Counter()
    for text in texts:
        for line in text.splitlines():
            match = re.match(r"\s*([a-z][a-z_-]{2,24})\s*:", line.lower())
            if match:
                openers[match.group(1)] += 1
    return {
        "n_docs": len(docs),
        "n_chars": chars,
        "n_tokens_est": len(words),
        "n_sentences": len(sents),
        "mean_sent_len": round(statistics.mean(sent_lens), 2) if sent_lens else 0.0,
        "median_sent_len": statistics.median(sent_lens) if sent_lens else 0.0,
        "upper_frac": round(upper / letters, 4) if letters else 0.0,
        "lower_frac": round(lower / letters, 4) if letters else 0.0,
        "exclamation_per_1k_chars": round(1000 * excl / chars, 3) if chars else 0.0,
        "question_per_1k_chars": round(1000 * qmarks / chars, 3) if chars else 0.0,
        "emoji_count": len(emojis),
        "emoji_per_1k_chars": round(1000 * len(emojis) / chars, 3) if chars else 0.0,
        "discourse_markers": dict(markers.most_common(20)),
        "top_lexical": top_lex,
        "signature_openers": [name for name, count in openers.most_common(8) if count >= 5],
        "habits": _habit_candidates(
            upper_frac=(upper / letters) if letters else 0.0,
            excl_rate=(excl / chars) if chars else 0.0,
            emoji_n=len(emojis),
            mean_sent=statistics.mean(sent_lens) if sent_lens else 0.0,
        ),
    }


def _habit_candidates(*, upper_frac: float, excl_rate: float, emoji_n: int, mean_sent: float) -> list[str]:
    """Card-derived checkable habits for the fidelity floor (no judge)."""
    habits = []
    if upper_frac < 0.02:
        habits.append("almost_never_uses_uppercase")
    if upper_frac > 0.15:
        habits.append("heavy_uppercase")
    if excl_rate < 0.0005:
        habits.append("rarely_uses_exclamation")
    if excl_rate > 0.005:
        habits.append("frequent_exclamation")
    if emoji_n == 0:
        habits.append("no_emoji")
    if emoji_n > 20:
        habits.append("emoji_heavy")
    if mean_sent and mean_sent < 8:
        habits.append("short_sentences")
    if mean_sent and mean_sent > 25:
        habits.append("long_sentences")
    return habits


def render_style_card(domain: str, stats: dict, prose: str | None = None) -> str:
    habits = stats.get("habits") or []
    markers = stats.get("discourse_markers") or {}
    top = stats.get("top_lexical") or []
    lines = [
        f"# Style card — `{domain}`",
        "",
        "_Human-approved register spec. Generation and judging both anchor here._",
        "",
        "## Corpus size",
        f"- docs: {stats.get('n_docs')}",
        f"- est. tokens: {stats.get('n_tokens_est')}",
        f"- sentences: {stats.get('n_sentences')}",
        "",
        "## Mechanical fingerprint",
        f"- mean sentence length (words): {stats.get('mean_sent_len')}",
        f"- uppercase letter frac: {stats.get('upper_frac')}",
        f"- exclamations / 1k chars: {stats.get('exclamation_per_1k_chars')}",
        f"- questions / 1k chars: {stats.get('question_per_1k_chars')}",
        f"- emoji / 1k chars: {stats.get('emoji_per_1k_chars')}",
        "",
        "## Checkable habits (fidelity floor)",
    ]
    if habits:
        lines.extend(f"- `{h}`" for h in habits)
    else:
        lines.append("- _(none auto-detected — author manually)_")
    lines += ["", "## Discourse markers (counts)"]
    if markers:
        lines.extend(f"- {k}: {v}" for k, v in list(markers.items())[:12])
    else:
        lines.append("- _(none)_")
    lines += ["", "## Top lexical items"]
    lines.extend(f"- {w}: {c}" for w, c in top[:20])
    lines += ["", "## Signature openers"]
    openers = stats.get("signature_openers") or []
    lines.extend(f"- `{value}:`" for value in openers) if openers else lines.append("- _(none)_")
    lines += ["", "## Register prose"]
    if prose:
        lines.append(prose.strip())
    else:
        lines.append(
            "_(LLM prose summary omitted — device-only or unavailable. "
            "Operator: describe what this voice sounds like and what it never does.)_"
        )
    lines += [
        "",
        "## Approval",
        "- [ ] Operator approved this card as the generation/judge anchor",
        "",
    ]
    return "\n".join(lines) + "\n"


def check_minimums(stats: dict, *, min_tokens: int, min_docs: int) -> tuple[bool, str]:
    n_tok = int(stats.get("n_tokens_est") or 0)
    n_docs = int(stats.get("n_docs") or 0)
    if n_docs < min_docs or n_tok < min_tokens:
        return False, (
            f"STYLE_MINIMUM_NOT_MET: docs={n_docs} (need ≥{min_docs}), "
            f"tokens≈{n_tok} (need ≥{min_tokens}). "
            "Not enough of your writing to learn your voice — add exports/notes."
        )
    return True, "ok"


def build_fingerprint(
    domain: str,
    editions: str | None = None,
    *,
    min_tokens: int = 20_000,
    min_docs: int = 50,
    device_only: bool = True,
) -> dict:
    docs = load_style_docs(domain, editions)
    stats = mechanical_stats(docs) if docs else {
        "n_docs": 0, "n_tokens_est": 0, "n_sentences": 0, "habits": [],
        "discourse_markers": {}, "top_lexical": [],
    }
    ok, reason = check_minimums(stats, min_tokens=min_tokens, min_docs=min_docs)
    prose = None
    if ok and not device_only:
        prose = None  # provider path reserved; mechanical card is enough for v1
    card = render_style_card(domain, stats, prose=prose)
    source_digest = hashlib.sha256()
    for doc in sorted(docs, key=lambda row: str(row.get("path") or "")):
        source_digest.update(str(doc.get("path") or "").encode("utf-8"))
        source_digest.update(b"\0")
        source_digest.update(str(doc.get("text") or "").encode("utf-8"))
        source_digest.update(b"\0")
    return {
        "domain": domain,
        "editions": editions,
        "status": "PASS" if ok else "REFUSE",
        "refuse_reason": None if ok else reason,
        "stats": stats,
        "signature_openers": stats.get("signature_openers") or [],
        "source_sha256": source_digest.hexdigest(),
        "style_card_md": card,
        "device_only": device_only,
        "min_tokens": min_tokens,
        "min_docs": min_docs,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--domain", required=True)
    ap.add_argument("--editions", default=None)
    ap.add_argument("--min-tokens", type=int, default=20_000)
    ap.add_argument("--min-docs", type=int, default=50)
    ap.add_argument("--device-only", action="store_true", default=True)
    ap.add_argument("--allow-provider", action="store_true", help="permit LLM prose summary")
    args = ap.parse_args()
    result = build_fingerprint(
        args.domain,
        args.editions,
        min_tokens=args.min_tokens,
        min_docs=args.min_docs,
        device_only=not args.allow_provider,
    )
    out = C.BUILD / "style" / args.domain
    out.mkdir(parents=True, exist_ok=True)
    fp_path = out / "fingerprint.json"
    card_path = out / "style_card.md"
    payload = {k: v for k, v in result.items() if k != "style_card_md"}
    tmp = fp_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, fp_path)
    card_path.write_text(result["style_card_md"], encoding="utf-8")
    print(f"status={result['status']} docs={result['stats'].get('n_docs')} "
          f"tokens≈{result['stats'].get('n_tokens_est')}")
    print(f"  wrote {fp_path}")
    print(f"  wrote {card_path}")
    if result["status"] == "REFUSE":
        print(result["refuse_reason"], file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()

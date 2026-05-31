"""00_raw → 10_clean extraction driver.

    PYTHONPATH=scripts python -m extract [--editions las_vegas/2024,…] [--sources news,reddit,…]
        [--limit N] [--llm-rescue] [--force] [--check-key]

Walks each edition's 00_raw/<source>/*.meta.json, dispatches to the per-source cleaner, and writes
10_clean/<subfolder>/<stem>.md (canonical header + body) + a <stem>.clean.json provenance sidecar.
Idempotent (skips when the sidecar already records the raw sha, unless --force) and never
overwrites a hand-curated doc (a .md with no pipeline sidecar). Prints a coverage table.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter

from pipeline.editions import iter_editions
from pipeline import provenance as P
from extract.cleaners import get_cleaners, SUBFOLDER


def _check_key() -> bool:
    try:
        import synth_corpus  # noqa: F401  (module-level get_api_key resolves the key or sys.exits)
        return True
    except SystemExit:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--editions", help="restrict to 'city/year,…' (default: all 97 with raw)")
    ap.add_argument("--sources", help="restrict to 'news,reddit,youtube,official,mfw'")
    ap.add_argument("--limit", type=int, help="max artifacts per (edition, source)")
    ap.add_argument("--llm-rescue", action="store_true", help="LLM-rescue HTML that fails the quality gate")
    ap.add_argument("--force", action="store_true", help="re-extract even if the sidecar is current")
    ap.add_argument("--check-key", action="store_true", help="preflight the OpenRouter key and exit")
    args = ap.parse_args()

    if args.check_key:
        print("OpenRouter key:", "OK" if _check_key() else "MISSING")
        return
    if args.llm_rescue and not _check_key():
        sys.exit("--llm-rescue needs the OpenRouter key (not found in env or libsecret).")

    cleaners = get_cleaners()
    only_sources = set(args.sources.split(",")) if args.sources else None
    stats: Counter = Counter()
    per_edition: dict[str, Counter] = {}

    for ed in iter_editions("edc", only=args.editions):
        ec = per_edition.setdefault(ed.slug, Counter())
        raw_root = ed.dir / "00_raw"
        for source, clean_fn in cleaners.items():
            if only_sources and source not in only_sources:
                continue
            src_dir = raw_root / source
            if not src_dir.is_dir():
                continue
            metas = sorted(src_dir.glob("*.meta.json"))
            if args.limit:
                metas = metas[: args.limit]
            sub = SUBFOLDER[source]
            out_dir = ed.dir / "10_clean" / sub
            for meta_path in metas:
                artifact = meta_path.with_name(meta_path.name[: -len(".meta.json")])
                if not artifact.exists():
                    continue
                meta = P.read_meta(meta_path)
                raw_sha = meta.get("sha256", "")
                stem = artifact.stem
                out_md = out_dir / f"{stem}.md"
                sidecar = out_dir / f"{stem}.clean.json"

                # protect hand-curated docs (a .md with no pipeline sidecar)
                if out_md.exists() and not sidecar.exists() and not args.force:
                    stats["protected"] += 1; ec["protected"] += 1
                    continue
                if P.is_unchanged(sidecar, raw_sha) and not args.force:
                    stats["unchanged"] += 1; ec["unchanged"] += 1
                    continue

                try:
                    doc = clean_fn(artifact, meta, llm_rescue=args.llm_rescue)
                except Exception as e:
                    stats["error"] += 1; ec["error"] += 1
                    print(f"  ! {ed.slug}/{source}/{stem}: {type(e).__name__}: {e}", file=sys.stderr)
                    continue
                if doc is None:
                    if sidecar.exists():        # stale pipeline output that no longer qualifies → remove
                        out_md.unlink(missing_ok=True)
                        sidecar.unlink(missing_ok=True)
                        stats["removed_stale"] += 1; ec["removed_stale"] += 1
                    stats["skipped"] += 1; ec["skipped"] += 1
                    continue

                out_dir.mkdir(parents=True, exist_ok=True)
                text = P.clean_header(doc.title, doc.source, meta.get("domain_class", ""),
                                      meta.get("fetched_at")) + doc.body.strip() + "\n"
                out_md.write_text(text, encoding="utf-8")
                sidecar.write_text(json.dumps({
                    "raw_sha256": raw_sha,
                    "clean_sha256": hashlib.sha256(text.encode()).hexdigest(),
                    "extractor": doc.extractor, "llm_rescued": doc.llm_rescued,
                    "quality_gate": doc.quality_gate,
                    "domain_class": meta.get("domain_class", ""), "source_channel": source,
                    "source_url": doc.source, "title": doc.title,
                }, indent=2), encoding="utf-8")
                stats["cleaned"] += 1; ec["cleaned"] += 1
                if doc.llm_rescued:
                    stats["rescued"] += 1; ec["rescued"] += 1

    print("\n=== extraction coverage ===")
    for slug in sorted(per_edition):
        c = per_edition[slug]
        if c:
            print(f"  {slug:22s} " + "  ".join(f"{k}={v}" for k, v in sorted(c.items())))
    print(f"--- totals: " + "  ".join(f"{k}={v}" for k, v in sorted(stats.items())))


if __name__ == "__main__":
    main()

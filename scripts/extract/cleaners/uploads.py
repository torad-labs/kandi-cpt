"""Passthrough cleaner for user-uploaded documents (source_channel='uploads').

Uploaded docs are already source material (markdown / text / html / json), not scraped HTML — so this is a
light pass: markdown & text flow through verbatim; html goes through trafilatura like the news cleaner;
json passes through (the structured extractor reads its text). The first non-empty line is the title.
This is what lets the UI document-drop feed the same clean → structured → hydrate pipeline as scraping.
"""
from __future__ import annotations

from pathlib import Path

from . import CleanDoc

MIN_BODY = 40   # below this, the doc is too thin to teach anything — skip


def clean(artifact_path, meta, *, llm_rescue=False):
    p = Path(artifact_path)
    raw = p.read_text(encoding="utf-8", errors="replace")
    ext = p.suffix.lower().lstrip(".")
    if ext in ("html", "htm"):
        try:
            import trafilatura
            body = trafilatura.extract(raw) or raw
        except Exception:
            body = raw
    else:  # md, txt, json, anything text-like — already clean source material
        body = raw
    body = body.strip()
    if len(body) < MIN_BODY:
        return None
    title = next((ln.lstrip("# ").strip() for ln in body.splitlines() if ln.strip()), p.stem)
    return CleanDoc(title=title[:200], body=body, source=meta.get("url") or "upload",
                    extractor="uploads-passthrough")

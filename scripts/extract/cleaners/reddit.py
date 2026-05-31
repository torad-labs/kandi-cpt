"""Reddit — old.reddit search-listing JSON → one .md per listing. Pure parse, no LLM.

Saved artifacts are search listings (~10 t3 posts with title+selftext), not comment trees, so
the post title+selftext is the signal. Skip link-only posts (empty selftext)."""
import json

from . import CleanDoc

MIN_SELFTEXT = 20


def clean(artifact_path, meta, *, llm_rescue=False):
    try:
        data = json.loads(artifact_path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return None
    children = (data.get("data") or {}).get("children", [])
    blocks, kept = [], 0
    for ch in children:
        d = ch.get("data") or {}
        title = (d.get("title") or "").strip()
        selftext = (d.get("selftext") or "").strip()
        if not title or len(selftext) < MIN_SELFTEXT:
            continue
        score = d.get("score")
        tag = f"  (score {score})" if isinstance(score, int) else ""
        blocks.append(f"### {title}{tag}\n\n{selftext}")
        kept += 1
    if not blocks:
        return None

    sub = (meta.get("extra") or {}).get("subreddit", "")
    title = f"Reddit r/{sub} discussion — {meta.get('edition',{}).get('city','')} {meta.get('edition',{}).get('year','')}".strip()
    return CleanDoc(title=title, body="\n\n".join(blocks),
                    source=meta.get("url") or "", extractor="json-reddit",
                    quality_gate={"posts": kept})

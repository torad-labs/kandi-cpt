"""YouTube — yt-dlp info JSON → title + description (+ subtitles if present). Pure parse, no LLM.

Subtitles are usually absent in the saved info JSON, so the description is the real signal;
skip bare-title videos (a title alone is not corpus)."""
import json

from . import CleanDoc

MIN_DESC = 50


def _subtitle_text(info: dict) -> str:
    """Best-effort: yt-dlp sometimes inlines auto-caption events; usually empty."""
    subs = info.get("subtitles") or info.get("automatic_captions") or {}
    for lang in ("en", "en-US", "en-orig"):
        for track in subs.get(lang, []):
            data = track.get("data")
            if isinstance(data, str) and data.strip():
                return data.strip()
    return ""


def clean(artifact_path, meta, *, llm_rescue=False):
    try:
        info = json.loads(artifact_path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return None
    title = (info.get("title") or "").strip()
    desc = (info.get("description") or "").strip()
    subs = _subtitle_text(info)
    if not title or (len(desc) < MIN_DESC and not subs):
        return None

    uploader = info.get("uploader") or info.get("channel") or ""
    date = info.get("upload_date") or ""
    parts = [desc] if desc else []
    if subs:
        parts.append("## Transcript\n\n" + subs)
    meta_line = " ".join(x for x in [f"Uploaded {date}" if date else "",
                                     f"by {uploader}" if uploader else ""] if x).strip()
    if meta_line:
        parts.append(f"*{meta_line}.*")
    body = "\n\n".join(parts)
    return CleanDoc(title=title, body=body,
                    source=info.get("webpage_url") or meta.get("url") or "",
                    extractor="json-youtube", quality_gate={"desc_chars": len(desc), "has_subs": bool(subs)})

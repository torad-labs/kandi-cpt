"""YouTube via yt-dlp — metadata + descriptions for EDC aftermovies / set recordings /
recaps. No video download (skip_download). One JSON record per video, deduped by id."""
import json
from pathlib import Path

import config as C
from scrape.fetcher import RawDoc, save_doc

DOMAIN_CLASS = "social-vetted"
KEEP = ("id", "title", "description", "upload_date", "uploader", "channel", "channel_id",
        "duration", "view_count", "like_count", "tags", "categories", "webpage_url")


def scrape(festival, ed, year, *, http, browser, limit=12) -> list[Path]:
    import yt_dlp

    query = f"EDC {ed['city'].replace('_', ' ')} {year}"
    dest = C.raw_dir(festival, ed["city"], year, "youtube")
    tag = {"festival": festival, "city": ed["city"], "year": year}
    opts = {"quiet": True, "no_warnings": True, "skip_download": True, "ignoreerrors": True}
    with yt_dlp.YoutubeDL(opts) as ydl:
        try:
            res = ydl.extract_info(f"ytsearch{limit}:{query}", download=False)
        except Exception:
            return []
    saved = []
    for e in (res or {}).get("entries") or []:
        if not e or not e.get("id"):
            continue
        info = {k: e.get(k) for k in KEEP}
        payload = json.dumps(info, ensure_ascii=False, indent=2).encode("utf-8")
        raw = RawDoc(url=e.get("webpage_url") or f"https://youtu.be/{e['id']}",
                     final_url=e.get("webpage_url") or "", status=200,
                     content=payload, content_type="application/json", fetcher="http")
        p = save_doc(raw, dest, e["id"], source_channel="youtube", domain_class=DOMAIN_CLASS,
                     edition=tag, extra={"query": query, "title": info.get("title")})
        if p:
            saved.append(p)
    return saved

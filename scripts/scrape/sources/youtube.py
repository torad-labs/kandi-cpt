"""YouTube via yt-dlp — title + description + TRANSCRIPT for EDC aftermovies / set recordings / recaps.

No video download. yt-dlp's extract_info returns caption *URLs* (manual `subtitles` and auto `automatic_captions`),
not inline text — so we fetch the best English track (json3 preferred, vtt fallback), parse it to plain text, and
inline it as `subtitles.en[0].data`, the shape the cleaner already reads. One JSON record per video, deduped by id."""
import json
import re
import urllib.request
from pathlib import Path

import config as C
from scrape.fetcher import RawDoc, save_doc

DOMAIN_CLASS = "social-vetted"
KEEP = ("id", "title", "description", "upload_date", "uploader", "channel", "channel_id",
        "duration", "view_count", "like_count", "tags", "categories", "webpage_url")
_EN = ("en", "en-US", "en-GB", "en-orig", "en-auto")
_UA = {"User-Agent": "Mozilla/5.0"}


def _parse_json3(raw: str) -> str:
    try:
        d = json.loads(raw)
    except Exception:
        return ""
    parts = []
    for ev in d.get("events") or []:
        seg = "".join(s.get("utf8", "") for s in (ev.get("segs") or []))
        if seg.strip():
            parts.append(seg.strip())
    return re.sub(r"\s+", " ", " ".join(parts)).strip()


def _parse_vtt(raw: str) -> str:
    out = []
    for ln in raw.splitlines():
        ln = ln.strip()
        if not ln or ln == "WEBVTT" or "-->" in ln or ln.isdigit():
            continue
        if ln.startswith(("Kind:", "Language:", "NOTE")):
            continue
        ln = re.sub(r"<[^>]+>", "", ln)              # strip <c>/<00:00:00.000> caption tags
        if ln and (not out or out[-1] != ln):        # auto-captions roll/repeat → drop consecutive dups
            out.append(ln)
    return re.sub(r"\s+", " ", " ".join(out)).strip()


def _fetch_transcript(e: dict) -> str:
    for table in (e.get("subtitles") or {}, e.get("automatic_captions") or {}):
        for lang in _EN:
            tracks = table.get(lang) or []
            for fmt, parse in (("json3", _parse_json3), ("vtt", _parse_vtt), ("srv1", _parse_vtt)):
                for t in tracks:
                    if t.get("ext") != fmt or not t.get("url"):
                        continue
                    try:
                        raw = urllib.request.urlopen(
                            urllib.request.Request(t["url"], headers=_UA), timeout=25).read().decode("utf-8", "replace")
                    except Exception:
                        continue
                    txt = parse(raw)
                    if len(txt) >= 50:
                        return txt
    return ""


def scrape(festival, ed, year, *, http, browser, limit=12) -> list[Path]:  # noqa: F821
    import yt_dlp

    query = f"EDC {ed['city'].replace('_', ' ')} {year}"
    dest = C.raw_dir(festival, ed["city"], year, "youtube")
    tag = {"festival": festival, "city": ed["city"], "year": year}
    # full (non-flat) extract so each entry carries caption-track URLs; skip the actual media download.
    opts = {"quiet": True, "no_warnings": True, "skip_download": True, "ignoreerrors": True,
            "subtitleslangs": list(_EN), "writesubtitles": False, "writeautomaticsub": False}
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
        transcript = _fetch_transcript(e)
        if transcript:                                # inline as the shape cleaners/youtube._subtitle_text reads
            info["subtitles"] = {"en": [{"ext": "txt", "data": transcript}]}
        payload = json.dumps(info, ensure_ascii=False, indent=2).encode("utf-8")
        raw = RawDoc(url=e.get("webpage_url") or f"https://youtu.be/{e['id']}",
                     final_url=e.get("webpage_url") or "", status=200,
                     content=payload, content_type="application/json", fetcher="http")
        p = save_doc(raw, dest, e["id"], source_channel="youtube", domain_class=DOMAIN_CLASS,
                     edition=tag, extra={"query": query, "title": info.get("title"),
                                         "transcript_chars": len(transcript)})
        if p:
            saved.append(p)
    return saved

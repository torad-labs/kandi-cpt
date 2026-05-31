"""Scrape EDC Las Vegas 2026 SET TIMES (per-artist start/end by stage by day).

Faithfulness rule: only times actually stated in a real source are written.
Nothing is invented or estimated. A row is emitted only if BOTH a start and an
end clock time parse cleanly out of the source DOM.

Source decision (see report / recon in the PR thread):
  - The OFFICIAL insomniac.com / lasvegas.electricdaisycarnival.com set-times page
    is no longer web-accessible for 2026: the site has rolled to 2027, and even the
    surviving /lineup/set-times/ page renders its grid entirely via blocked
    client-side JS (zero schedule__artist nodes in the static HTML, no backing
    lineup JSON feed). BrowserFetcher bypasses Cloudflare fine, but there is no
    2026 set-time content to parse. Confirmed by direct probe.
  - festtimetable.com publishes a clean, server-rendered, DAY-SEPARATED, per-artist
    grid (one `bystage-list__item` per set: artist title + start/end `datetime`
    nodes, grouped under per-stage `action-subarea` headers whose titles are the
    exact camelCase stage names). It transcribes the official Insomniac schedule.
    Plain HTTP works (no Cloudflare). This is the source we parse.

Because festtimetable is a third-party republisher (not insomniac.com itself), every
row is tagged confidence="community" per the task's {official|community} vocabulary.

Run:  PYTHONPATH=scripts .venv/bin/python scripts/scrape_set_times.py
Writes: data/edc/las_vegas/2026/set_times.jsonl  (one JSON object per line)
        data/edc/las_vegas/2026/00_raw/set_times/  (raw HTML + provenance sidecars)
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from selectolax.parser import HTMLParser

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
from scrape.fetcher import HttpFetcher, save_doc

SOURCE_URL = "https://festtimetable.com/en-US/events/edc-lasvegas-2026"
DOMAIN_CLASS = "community-aggregator"      # third-party schedule republisher
CONFIDENCE = "community"
EDITION = {"festival": "edc", "city": "las_vegas", "year": 2026}

# festtimetable day path -> (canonical day name, calendar start date of that night)
DAYS = {
    "day-1": ("Friday", "May 15"),
    "day-2": ("Saturday", "May 16"),
    "day-3": ("Sunday", "May 17"),
}

# camelCase stage title (as rendered on the page) -> our 9 main-stage slugs.
# Secondary/art-car stages have no slug → slug stays "" but the row is still written
# with its raw stage string (downstream can map or ignore them).
STAGE_SLUGS = {
    "kineticfield": "kinetic_field",
    "cosmicmeadow": "cosmic_meadow",
    "circuitgrounds": "circuit_grounds",
    "neongarden": "neon_garden",
    "basspod": "basspod",
    "wasteland": "wasteland",
    "quantumvalley": "quantum_valley",
    "stereobloom": "stereo_bloom",
    "bionicjungle": "bionic_jungle",
}

# Rows that are production segments, not artist performances — excluded from set times.
NON_ARTIST_RE = re.compile(
    r"^(opening ceremony|closing ceremony|intermezzo|main fireworks|fireworks|"
    r"kinetic launch|ceremony)\b", re.I)

# Start nodes prefix a date whose day-number runs straight into the hour with no
# separator ("May 18" + "4:14 AM" -> "May 184:14 AM"; "May 16" + "1:47 AM" ->
# "May 161:47 AM"). Strip a leading "<Month> <day>" date token FIRST, then parse the
# clock — otherwise a greedy 2-digit hour eats the trailing date digit (61:47).
DATE_PREFIX_RE = re.compile(
    r"^\s*(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s*\d{1,2}", re.I)
CLOCK_RE = re.compile(r"(\d{1,2}):(\d{2})\s*([AP]M)", re.I)


def norm_clock(s: str) -> str | None:
    """Pull the 'H:MM AM/PM' clock time out of a string; normalize spacing/case.
    Drops any leading '<Month> <day>' date prefix so its digits can't bleed in."""
    s = DATE_PREFIX_RE.sub("", (s or "").strip())
    m = CLOCK_RE.search(s)
    if not m:
        return None
    h, mm, ap = int(m.group(1)), m.group(2), m.group(3).upper()
    if not (1 <= h <= 12):
        return None
    return f"{h}:{mm} {ap}"


def parse_day(html: str, day_name: str, day_date: str) -> list[dict]:
    """Parse one festtimetable day page into set-time rows.
    `day_date` is the calendar date the night opens on (e.g. 'May 15')."""
    tree = HTMLParser(html)
    rows: list[dict] = []
    for sub in tree.css(".action-subarea--bystage"):
        title_node = sub.css_first(".action-subarea__title")
        if not title_node:
            continue
        raw_stage = title_node.text().strip()
        slug = STAGE_SLUGS.get(re.sub(r"[^a-z0-9]", "", raw_stage.lower()), "")
        for item in sub.css(".bystage-list__item"):
            tnode = item.css_first(".bystage-column__title")
            if not tnode:
                continue
            # Artist name lives in .show-title__name (the .bystage-column__title text
            # would glue the trailing tag, e.g. "ARMIN VAN BUURENSUNRISE SET").
            name_node = tnode.css_first(".show-title__name")
            artist = re.sub(r"\s+", " ", (name_node.text() if name_node else tnode.text()).strip())
            if not artist or NON_ARTIST_RE.search(artist):
                continue
            snode = item.css_first(".datetime--start")
            enode = item.css_first(".datetime--end")
            start = norm_clock(snode.text()) if snode else None
            end = norm_clock(enode.text()) if enode else None
            if not (start and end):       # faithfulness: need both, else omit
                continue
            # Post-midnight sets carry an explicit date prefix ("May 16") on the start
            # node — its real calendar date. Pre-midnight sets have none → the night's
            # own opening date. (Faithful: the date is read, never assumed from the clock.)
            dm = DATE_PREFIX_RE.search(snode.text() if snode else "")
            start_date = re.sub(r"\s+", " ", dm.group(0).strip()) if dm else day_date
            tags = [t.text().strip() for t in tnode.css(".tag-custom") if t.text().strip()]
            rows.append({
                "artist": artist,
                "stage": slug,
                "stage_raw": raw_stage,
                "day": day_name,
                "start": start,
                "end": end,
                "start_date": start_date,
                "source": SOURCE_URL,
                "confidence": CONFIDENCE,
                **({"tags": tags} if tags else {}),
            })
    return rows


def main() -> None:
    http = HttpFetcher(ignore_robots=True)
    raw_dir = C.raw_dir("edc", "las_vegas", 2026, "set_times")
    all_rows: list[dict] = []
    try:
        for path, (day_name, day_date) in DAYS.items():
            url = f"{SOURCE_URL}/{path}"
            doc = http.get(url)
            if not doc or doc.status >= 400:
                print(f"  ! {url}: fetch failed (status={getattr(doc, 'status', 'none')})", flush=True)
                continue
            save_doc(doc, raw_dir, f"festtimetable-{path}", source_channel="community",
                     domain_class=DOMAIN_CLASS, edition=EDITION, extra={"day": day_name})
            rows = parse_day(doc.content.decode("utf-8", "replace"), day_name, day_date)
            print(f"  {day_name:9s} ({path}): {len(rows)} set times", flush=True)
            all_rows.extend(rows)
    finally:
        http.close()

    out = C.edition("edc", "las_vegas", 2026) / "set_times.jsonl"
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in all_rows))
    print(f"\nwrote {len(all_rows)} set times -> {out}", flush=True)

    # coverage summary
    by_day: dict[str, int] = {}
    by_slug: dict[str, int] = {}
    main_stage = 0
    for r in all_rows:
        by_day[r["day"]] = by_day.get(r["day"], 0) + 1
        key = r["stage"] or f"(unmapped:{r['stage_raw']})"
        by_slug[key] = by_slug.get(key, 0) + 1
        if r["stage"]:
            main_stage += 1
    print(f"main-stage rows: {main_stage} / {len(all_rows)}")
    print("by day:", by_day)
    print("by stage:")
    for k in sorted(by_slug, key=lambda x: -by_slug[x]):
        print(f"  {by_slug[k]:4d}  {k}")


if __name__ == "__main__":
    main()

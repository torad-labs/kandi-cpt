"""Scrape EDC SET TIMES (per-artist start/end by stage by day) for every festtimetable-covered edition.

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

Run:  PYTHONPATH=scripts .venv/bin/python scripts/scrape_set_times.py [--editions all|city/year,...]
Writes (per edition): data/edc/<city>/<year>/set_times.jsonl  (one JSON object per line)
        data/edc/<city>/<year>/00_raw/set_times/  (raw HTML + provenance sidecars)
The feasible editions (those festtimetable actually publishes) are listed in FEASIBLE; las_vegas/2026
is the hand-built GOLD and is deliberately omitted so a run never regenerates it.
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

FT_BASE = "https://festtimetable.com/en-US/events"
DOMAIN_CLASS = "community-aggregator"      # third-party schedule republisher
CONFIDENCE = "community"

# Editions festtimetable.com actually publishes (probed 2026-06). "city/year" -> its event slug.
# las_vegas/2026 is the hand-built GOLD set_times.jsonl — intentionally excluded (protected, never regenerated).
FEASIBLE = {
    "las_vegas/2022": "edc-lasvegas-2022", "las_vegas/2023": "edc-lasvegas-2023",
    "las_vegas/2024": "edc-lasvegas-2024", "las_vegas/2025": "edc-lasvegas-2025",
    "orlando/2022": "edc-orlando-2022", "orlando/2023": "edc-orlando-2023",
    "orlando/2024": "edc-orlando-2024", "orlando/2025": "edc-orlando-2025",
    "mexico_city/2023": "edc-mexico-2023", "mexico_city/2024": "edc-mexico-2024",
    "mexico_city/2025": "edc-mexico-2025", "mexico_city/2026": "edc-mexico-2026",
    "incheon/2025": "edc-korea-2025",
    "phuket/2025": "edc-thailand-2025", "phuket/2026": "edc-thailand-2026",
    "suzhou/2023": "edc-china-2023", "suzhou/2025": "edc-china-2025",
}

# Weekday abbrev (from the page <title>) -> full day name.
WEEKDAY = {"mon": "Monday", "tue": "Tuesday", "wed": "Wednesday", "thu": "Thursday",
           "fri": "Friday", "sat": "Saturday", "sun": "Sunday"}
# Page <title>: "EDC Orlando 2024｜Day 1 Timetable－Fri, Nov 8" -> ("Friday", "Nov 8").
TITLE_DAY_RE = re.compile(
    r"(Mon|Tue|Wed|Thu|Fri|Sat|Sun)[a-z]*,?\s*"
    r"((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s*\d{1,2})", re.I)


def day_header(html: str, fallback_n: int) -> tuple[str, str]:
    """Derive (day_name, opening_date) from the page <title> — faithful (read, not assumed from the clock).
    Falls back to 'Day N' + empty date if the title lacks the weekday/date."""
    t = HTMLParser(html).css_first("title")
    m = TITLE_DAY_RE.search(t.text()) if t else None
    if not m:
        return (f"Day {fallback_n}", "")
    return (WEEKDAY.get(m.group(1)[:3].lower(), m.group(1)), re.sub(r"\s+", " ", m.group(2)).strip())

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


def parse_day(html: str, day_name: str, day_date: str, source_url: str) -> list[dict]:
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
                "source": source_url,
                "confidence": CONFIDENCE,
                **({"tags": tags} if tags else {}),
            })
    return rows


def scrape_edition(http: HttpFetcher, city: str, year: int, ft_slug: str) -> int:
    """Scrape one festtimetable edition -> data/edc/<city>/<year>/set_times.jsonl (+ raw provenance).
    Auto-discovers day pages (day-1.. until 404). Backs up any existing file before writing
    (set-times are expensive to re-acquire — never silently clobbered). Returns rows written."""
    edition = {"festival": "edc", "city": city, "year": year}
    raw_dir = C.raw_dir("edc", city, year, "set_times")
    rows: list[dict] = []
    for n in range(1, 8):                                  # day-1.. ; stop at the first missing day
        url = f"{FT_BASE}/{ft_slug}/day-{n}"
        doc = http.get(url)
        if not doc or doc.status >= 400:
            break
        html = doc.content.decode("utf-8", "replace")
        day_name, day_date = day_header(html, n)
        save_doc(doc, raw_dir, f"festtimetable-day-{n}", source_channel="community",
                 domain_class=DOMAIN_CLASS, edition=edition, extra={"day": day_name})
        drows = parse_day(html, day_name, day_date, url)
        print(f"    day-{n} {day_name:9s} ({day_date or '—'}): {len(drows)} set times", flush=True)
        rows.extend(drows)

    out = C.edition("edc", city, year) / "set_times.jsonl"
    if out.exists():                                       # guard: back up before replacing expensive output
        bk = C.BUILD / "backups" / "set_times" / f"{city}_{year}_{int(out.stat().st_mtime)}.jsonl"
        bk.parent.mkdir(parents=True, exist_ok=True)
        bk.write_bytes(out.read_bytes())
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    return len(rows)


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Scrape festtimetable set-times for feasible EDC editions.")
    ap.add_argument("--editions", default="all",
                    help="'all' (every feasible edition) or a comma list of 'city/year' keys from FEASIBLE.")
    args = ap.parse_args()
    keys = sorted(FEASIBLE) if args.editions == "all" else [e.strip() for e in args.editions.split(",") if e.strip()]

    http = HttpFetcher(ignore_robots=True)
    total: dict[str, int] = {}
    try:
        for key in keys:
            slug = FEASIBLE.get(key)
            if not slug:
                print(f"  ! {key}: no festtimetable coverage (not in FEASIBLE) — skipping", flush=True)
                continue
            city, year = key.split("/")
            print(f"  {key} ({slug}):", flush=True)
            total[key] = scrape_edition(http, city, int(year), slug)
    finally:
        http.close()

    print(f"\n=== set-times acquired: {sum(total.values())} rows across {len(total)} editions ===")
    for k in sorted(total):
        print(f"  {total[k]:5d}  {k}")


if __name__ == "__main__":
    main()

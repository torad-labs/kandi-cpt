#!/usr/bin/env python3
"""Wikipedia 'List of events' table → per-edition festival_meta.json (venue, dates, attendance).

The main Electric Daisy Carnival article carries one authoritative table of every edition: Year | Event |
Dates | Venue | Attendance. It does NOT have lineups (those come from the official site / MFW / last.fm),
but it is the cleanest source for the metadata gaps — the 7 venue-less san_juan editions, plus attendance
(missing for all 98) and date confirmation. We parse the wikitable (handling rowspans), map each row's
event name + year to a registry edition, and fill ONLY missing meta fields (never overwrite).

    PYTHONPATH=scripts python scripts/extract_wikipedia_meta.py [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import re

from selectolax.parser import HTMLParser

from pipeline.editions import iter_editions

URL = "https://en.wikipedia.org/wiki/Electric_Daisy_Carnival"
# event-name (lowercased) → city slug. None = disambiguate by year/venue below.
EVENT_CITY = {
    "edc los angeles": "los_angeles", "edc colorado": "denver", "edc dallas": "dallas",
    "edc puerto rico": "san_juan", "edc las vegas": "las_vegas", "edc new york": "new_york_nj",
    "edc chicago": "chicago", "edc london": "london", "edc uk": "milton_keynes",
    "edc mexico": "mexico_city", "edc brasil": "sao_paulo", "edc india": "new_delhi",
    "edc japan": "chiba", "edc guangdong": "zhuhai", "edc thailand": "phuket",
    "edc portugal": "portimao", "edc colombia": "medellin", "edc texas": "austin", "edsea": "edsea",
    "edc china": None, "edc korea": None, "edc": None,            # disambiguate by year/venue
}


def _row_city(event: str, year: int, venue: str) -> str | None:
    c = EVENT_CITY.get(event)
    if c is not None:
        return c
    if event == "edc china":
        return "shanghai" if year == 2018 else "suzhou"
    if event == "edc korea":
        return "seoul" if year <= 2019 else "incheon"
    if event == "edc":                                           # early bare "EDC" (LA-area)
        return "san_bernardino" if "san bernardino" in venue.lower() else "los_angeles"
    return None


def _cells_with_rowspan(rows):
    """Yield each row's logical cells, carrying rowspanned cells forward (Wikipedia merges the Year col)."""
    carry: dict[int, tuple[str, int]] = {}                       # col_idx -> (text, rows_remaining)
    for tr in rows:
        tds = tr.css("td, th")
        out, ci, td_i = [], 0, 0
        while td_i < len(tds) or ci in carry:
            if ci in carry:
                text, rem = carry[ci]
                out.append(text)
                carry[ci] = (text, rem - 1) if rem - 1 > 0 else None
                if carry[ci] is None:
                    del carry[ci]
                ci += 1
                continue
            td = tds[td_i]
            td_i += 1
            txt = re.sub(r"\s+", " ", td.text(strip=True)).strip()
            span = int(td.attributes.get("rowspan") or 1)
            if span > 1:
                carry[ci] = (txt, span - 1)
            out.append(txt)
            ci += 1
        yield out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    import httpx                                               # Wikipedia requires a descriptive UA + contact
    ua = "kandi-cpt-edc-research/1.0 (https://github.com/; festival knowledge dataset)"
    r = httpx.get(URL, headers={"User-Agent": ua}, follow_redirects=True, timeout=30)
    if r.status_code != 200:
        raise SystemExit(f"wikipedia fetch failed: {r.status_code}")
    tree = HTMLParser(r.text)

    # the "List of events" table = the wikitable whose header starts Year | Event | Dates | Venue
    table = None
    for t in tree.css("table.wikitable"):
        head = " ".join(th.text(strip=True).lower() for th in t.css("tr")[0].css("th, td"))
        if "year" in head and "venue" in head and ("event" in head or "dates" in head):
            table = t
            break
    if table is None:
        raise SystemExit("could not locate the List-of-events wikitable")

    # parse rows → {(city, year): {venue, dates, attendance}}
    parsed: dict[tuple, dict] = {}
    rows = table.css("tr")
    for cells in _cells_with_rowspan(rows[1:]):
        if len(cells) < 4:
            continue
        ym = re.match(r"(\d{4})", cells[0])
        if not ym:
            continue
        year = int(ym.group(1))
        event = cells[1].lower().strip()
        # columns can be [Year, Event, Dates, Venue, Attendance] OR [Year, Event, Venue, Attendance]
        dates = cells[2] if len(cells) >= 5 else ""
        venue = cells[3] if len(cells) >= 5 else cells[2]
        att = cells[4] if len(cells) >= 5 else (cells[3] if len(cells) >= 4 else "")
        city = _row_city(event, year, venue)
        if not city:
            continue
        rec = parsed.setdefault((city, year), {})
        if venue and "cancel" not in venue.lower():
            rec["venue"] = venue
        if dates:
            rec["dates"] = dates
        am = re.search(r"([\d,]{3,})", att)
        if am:
            rec["attendance"] = int(am.group(1).replace(",", ""))

    # apply to editions (fill only missing fields)
    known = {(ed.city, ed.year): ed for ed in iter_editions("edc")}
    n_v = n_a = n_ed = 0
    for (city, year), rec in sorted(parsed.items()):
        ed = known.get((city, year))
        if not ed:
            continue
        mp = ed.dir / "festival_meta.json"
        m = json.loads(mp.read_text()) if mp.exists() else {}
        changed = []
        cur_v = (m.get("venue") or "").strip().lower()
        if rec.get("venue") and (not cur_v or cur_v.startswith("various")):   # fill empty or placeholder
            m["venue"] = rec["venue"]; changed.append(f"venue={rec['venue']}"); n_v += 1
        if rec.get("attendance") and not m.get("attendance"):
            m["attendance"] = rec["attendance"]; changed.append(f"att={rec['attendance']}"); n_a += 1
        if rec.get("dates") and not (m.get("dates") or m.get("date") or m.get("start_date")):
            m["dates"] = rec["dates"]; changed.append("dates")
        if changed:
            n_ed += 1
            print(f"  {ed.slug:24s} {'; '.join(changed)}", flush=True)
            if not args.dry_run:
                mp.write_text(json.dumps(m, indent=2))

    print(f"\n{'(dry-run) ' if args.dry_run else ''}wikipedia meta: {n_ed} editions updated "
          f"(+{n_v} venues, +{n_a} attendance) from {len(parsed)} table rows")


if __name__ == "__main__":
    main()

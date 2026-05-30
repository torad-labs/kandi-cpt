"""Fetchers + provenance for the EDC scrapers.

Two ways to pull a URL, one save path:
  HttpFetcher    — httpx + browser UA, robots-honoring, rate-limited. For open sites
                   (Music Festival Wizard, most news, Wikipedia).
  BrowserFetcher — Patchright (patched undetected Playwright) driving real headed Chrome.
                   For Cloudflare-challenged / OAuth-walled sites (insomniac.com, reddit.com)
                   where every plain HTTP client gets 403. Headed + persistent context so the
                   cf_clearance cookie is solved once and reused. Needs a display (DISPLAY=:0
                   or xvfb) — the CLI handles that.

Every saved artifact gets a `<name>.meta.json` provenance sidecar whose `domain_class`
uses the schema/metadata.schema.json §7.2 trust vocabulary, so the clean stage keeps trust.
Saves are content-hash deduped → re-running a scrape is idempotent (no duplicate fetches/writes).
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import random
import re
import time
import urllib.robotparser as robotparser
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

import httpx

UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/124.0.0.0 Safari/537.36")
CONTACT = "kandi-cpt research crawler (marcos@seekapps.live)"
CHALLENGE_MARKERS = ("just a moment", "checking your browser", "attention required")


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def slugify(text: str, maxlen: int = 90) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return (s[:maxlen].rstrip("-")) or "doc"


def is_challenge(title: str, html: str) -> bool:
    t = (title or "").lower()
    return any(m in t for m in CHALLENGE_MARKERS) or "Enable JavaScript and cookies to continue" in html


@dataclass
class RawDoc:
    url: str
    final_url: str
    status: int
    content: bytes
    content_type: str
    fetcher: str  # "http" | "browser"


@dataclass
class Provenance:
    url: str
    final_url: str
    fetched_at: str
    http_status: int
    sha256: str
    fetcher: str
    source_channel: str   # official | mfw | youtube | reddit | news
    domain_class: str     # schema/metadata.schema.json §7.2 trust enum
    edition: dict         # {festival, city, year}
    extra: dict = field(default_factory=dict)


class RateLimiter:
    """Per-domain minimum spacing + jitter — polite, and avoids tripping rate defenses."""

    def __init__(self, min_interval: float = 2.0, jitter: float = 1.5):
        self.min_interval, self.jitter = min_interval, jitter
        self._last: dict[str, float] = {}

    def wait(self, url: str) -> None:
        dom = urlparse(url).netloc
        elapsed = time.monotonic() - self._last.get(dom, 0.0)
        delay = self.min_interval + random.uniform(0, self.jitter)
        if elapsed < delay:
            time.sleep(delay - elapsed)
        self._last[dom] = time.monotonic()


class _Robots:
    """Per-domain robots.txt gate. Fetches via the same client; if robots is unreadable
    (e.g. itself Cloudflare-blocked), default to allow and rely on rate-limiting."""

    def __init__(self, fetch_text, ignore: bool = False):
        self._fetch_text, self.ignore = fetch_text, ignore
        self._cache: dict[str, robotparser.RobotFileParser | None] = {}

    def allows(self, url: str) -> bool:
        if self.ignore:
            return True
        p = urlparse(url)
        base = f"{p.scheme}://{p.netloc}"
        if base not in self._cache:
            rfp = robotparser.RobotFileParser()
            txt = self._fetch_text(base + "/robots.txt")
            if txt is None:
                self._cache[base] = None        # unreadable -> allow
            else:
                rfp.parse(txt.splitlines())
                self._cache[base] = rfp
        rfp = self._cache[base]
        return True if rfp is None else rfp.can_fetch(UA, url)


class HttpFetcher:
    def __init__(self, rate: RateLimiter | None = None, ignore_robots: bool = False, timeout: float = 25.0):
        self.client = httpx.Client(
            follow_redirects=True, timeout=timeout,
            headers={"User-Agent": UA, "From": CONTACT, "Accept-Language": "en-US,en;q=0.9",
                     "Accept": "text/html,application/json;q=0.9,*/*;q=0.8"},
        )
        self.rate = rate or RateLimiter()
        self.robots = _Robots(self._robots_text, ignore=ignore_robots)
        self.fetcher_name = "http"

    def _robots_text(self, url: str) -> str | None:
        try:
            r = self.client.get(url)
            return r.text if r.status_code == 200 else None
        except Exception:
            return None

    def get(self, url: str) -> RawDoc | None:
        if not self.robots.allows(url):
            return None
        self.rate.wait(url)
        try:
            r = self.client.get(url)
        except Exception:
            return None
        return RawDoc(url=url, final_url=str(r.url), status=r.status_code,
                     content=r.content, content_type=r.headers.get("content-type", ""),
                     fetcher="http")

    def close(self):
        self.client.close()


class BrowserFetcher:
    """Headed real Chrome via Patchright. Lazily launched, one persistent context reused
    across all gets (cf_clearance + speed). Honors robots like HttpFetcher."""

    def __init__(self, user_data_dir: str = "/tmp/kandi_scrape_chrome", headless: bool = False,
                 ignore_robots: bool = False, settle_ms: int = 1500, challenge_wait_s: int = 25):
        self.user_data_dir, self.headless = user_data_dir, headless
        self.settle_ms, self.challenge_wait_s = settle_ms, challenge_wait_s
        self.fetcher_name = "browser"
        self._p = self._ctx = None
        self.rate = RateLimiter(min_interval=3.0, jitter=2.0)
        self.robots = _Robots(self._robots_text, ignore=ignore_robots)

    def _ensure(self):
        if self._ctx is None:
            from patchright.sync_api import sync_playwright
            self._p = sync_playwright().start()
            self._ctx = self._p.chromium.launch_persistent_context(
                user_data_dir=self.user_data_dir, channel="chrome",
                headless=self.headless, no_viewport=True)

    def _fetch(self, url: str, body_text: bool = False) -> RawDoc | None:
        self._ensure()
        page = self._ctx.new_page()
        try:
            resp = page.goto(url, wait_until="domcontentloaded", timeout=45000)
            status = resp.status if resp else 0
            for _ in range(self.challenge_wait_s):     # let any CF JS challenge auto-solve
                page.wait_for_timeout(1000)
                if not any(m in page.title().lower() for m in CHALLENGE_MARKERS):
                    break
            page.wait_for_timeout(self.settle_ms)
            html, final = page.content(), page.url
            # body_text: for JSON endpoints (e.g. reddit .json) Chrome wraps the payload in
            # HTML — innerText gives back the raw JSON string.
            text = page.evaluate("() => document.body && document.body.innerText") if body_text else None
        except Exception:
            page.close()
            return None
        page.close()
        if is_challenge("", html):
            return None
        if body_text:
            return RawDoc(url=url, final_url=final, status=status,
                         content=(text or "").encode("utf-8"), content_type="application/json",
                         fetcher="browser")
        return RawDoc(url=url, final_url=final, status=status,
                     content=html.encode("utf-8"), content_type="text/html", fetcher="browser")

    def _robots_text(self, url: str) -> str | None:
        doc = self._fetch(url)
        return doc.content.decode("utf-8", "replace") if doc and doc.status == 200 else None

    def get(self, url: str, body_text: bool = False, check_robots: bool = True) -> RawDoc | None:
        if check_robots and not self.robots.allows(url):
            return None
        self.rate.wait(url)
        return self._fetch(url, body_text=body_text)

    def close(self):
        if self._ctx is not None:
            try:
                self._ctx.close()
                self._p.stop()
            except Exception:
                pass
            self._ctx = self._p = None


_EXT = {"text/html": "html", "application/json": "json", "text/plain": "txt", "application/xml": "xml"}


def save_doc(raw: RawDoc, dest_dir: Path, name: str, *, source_channel: str, domain_class: str,
             edition: dict, extra: dict | None = None) -> Path | None:
    """Write artifact + .meta.json sidecar. Skips if an identical sha256 already lives here
    (idempotent re-scrape). Returns the artifact path, or None on dedup/empty."""
    if raw is None or not raw.content:
        return None
    dest_dir.mkdir(parents=True, exist_ok=True)
    digest = sha256_bytes(raw.content)
    ext = next((e for ct, e in _EXT.items() if ct in raw.content_type), "html")
    art = dest_dir / f"{slugify(name)}.{ext}"
    meta = art.with_suffix(art.suffix + ".meta.json")
    if meta.exists():
        try:
            if json.loads(meta.read_text()).get("sha256") == digest:
                return None  # unchanged — dedup
        except Exception:
            pass
    art.write_bytes(raw.content)
    prov = Provenance(url=raw.url, final_url=raw.final_url, fetched_at=now_iso(),
                      http_status=raw.status, sha256=digest, fetcher=raw.fetcher,
                      source_channel=source_channel, domain_class=domain_class,
                      edition=edition, extra=extra or {})
    meta.write_text(json.dumps(prov.__dict__, indent=2))
    return art

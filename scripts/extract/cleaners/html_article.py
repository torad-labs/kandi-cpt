"""HTML → clean markdown via trafilatura, with a quality gate and source-faithful LLM rescue.

trafilatura strips boilerplate and emits article markdown. If it returns too little (missed the
article) or too much link-text (a listing leaked), we route to an LLM EXTRACTION pass (not
generation): temperature ~0, "output the article verbatim, invent nothing". Shared by the
news/mfw/official cleaners.
"""
from __future__ import annotations

import re

import trafilatura
from selectolax.parser import HTMLParser

MIN_WORDS = 250          # below this, trafilatura likely missed the article body
MIN_KEEP_RATIO = 0.10    # extracted/visible < this → grabbed only nav/footer
MAX_ANCHOR_RATIO = 0.40  # link-text dominates → a listing page, not an article

EXTRACT_SYS = (
    "You are an HTML-to-text extractor. Output ONLY the main article body as clean markdown. "
    "Preserve the factual sentences verbatim where possible. Do NOT summarize, do NOT add facts, "
    "do NOT invent details — if something is not in the HTML it must not appear in your output. "
    "Strip navigation, ads, cookie banners, related-posts, comment forms, and share widgets. "
    "Output the article text only, no preamble."
)


def _visible_and_anchor(html: str) -> tuple[int, int]:
    """(visible_text_len, anchor_text_len) via selectolax, scripts/styles removed."""
    try:
        tree = HTMLParser(html)
    except Exception:
        return 0, 0
    for tag in tree.css("script, style, noscript"):
        tag.decompose()
    body = tree.body or tree
    visible = body.text(separator=" ", strip=True) if body else ""
    anchor = " ".join(a.text(strip=True) for a in tree.css("a"))
    return len(visible), len(anchor)


def extract_html(html: str, url: str) -> dict:
    """Run trafilatura + compute the gate. Returns dict(text, words, gate_pass, gate_reason, metrics)."""
    text = trafilatura.extract(html, url=url, output_format="markdown",
                               include_tables=True, include_comments=False,
                               favor_recall=True) or ""
    words = len(text.split())
    vis_len, anchor_len = _visible_and_anchor(html)
    keep_ratio = (len(text) / vis_len) if vis_len else 0.0
    anchor_ratio = (anchor_len / vis_len) if vis_len else 0.0

    reason = None
    if not text.strip():
        reason = "trafilatura-empty"
    elif words < MIN_WORDS:
        reason = f"too-short({words}w)"
    elif keep_ratio < MIN_KEEP_RATIO:
        reason = f"low-keep-ratio({keep_ratio:.2f})"
    elif anchor_ratio > MAX_ANCHOR_RATIO:
        reason = f"high-anchor({anchor_ratio:.2f})"
    return {"text": text, "words": words, "gate_pass": reason is None, "gate_reason": reason,
            "metrics": {"words": words, "keep_ratio": round(keep_ratio, 3),
                        "anchor_ratio": round(anchor_ratio, 3)}}


def llm_rescue(html: str, url: str) -> str | None:
    """Source-faithful LLM extraction of the article body. Returns markdown, or None on failure
    (incl. the model padding past the raw visible length — a hallucination guard)."""
    import httpx
    import synth_corpus as S   # lazy: triggers the OpenRouter key lookup only when rescue fires

    # strip script/style and cap input to keep the call cheap
    try:
        tree = HTMLParser(html)
        for t in tree.css("script, style, noscript"):
            t.decompose()
        stripped = (tree.body or tree).html or html
    except Exception:
        stripped = html
    stripped = re.sub(r"\s+", " ", stripped)[:24000]
    vis_len, _ = _visible_and_anchor(html)

    body = {"model": S.MODEL,
            "messages": [{"role": "system", "content": EXTRACT_SYS},
                         {"role": "user", "content": f"URL: {url}\n\nHTML:\n{stripped}"}],
            "temperature": 0.1, "max_tokens": 4000}
    try:
        r = httpx.Client(timeout=120).post(S.OPENROUTER_URL, json=body, headers=S.HEADERS)
        if r.status_code != 200:
            return None
        out = r.json().get("choices", [{}])[0].get("message", {}).get("content", "").strip()
    except Exception:
        return None
    low = out.lower()
    refusal = (not out
               or any(p in low for p in ("does not contain", "no main article", "no article body",
                                         "consists of navigation", "could not find", "i cannot",
                                         "i'm unable", "no readable article"))
               or low.startswith(("the provided html", "the html provided", "this html", "the page"))
               or (vis_len and len(out) > vis_len * 1.2))     # padded beyond the source
    return None if refusal else out


_NAV_JUNK = ("home", "menu", "sign in", "log in", "log out", "subscribe", "privacy", "cookie",
             "terms", "contact", "advertise", "newsletter", "follow us", "share", "read more",
             "next", "previous", "search", "account", "register")


def listing_index(html: str) -> str:
    """Faithful index of a link-heavy page: its substantive link titles + headings as a markdown
    list. NOT fabrication — every line is text present on the page (real entities/dates/titles)."""
    try:
        tree = HTMLParser(html)
    except Exception:
        return ""
    for t in tree.css("script, style, noscript, nav, footer, header"):
        t.decompose()
    items: list[str] = []
    seen: set[str] = set()
    for node in tree.css("h1, h2, h3, a"):
        txt = re.sub(r"\s+", " ", node.text(strip=True) or "").strip()
        key = txt.lower()
        if len(txt) < 25 or key in seen or any(b in key for b in _NAV_JUNK):
            continue
        seen.add(key)
        items.append(txt)
    return ("## Index\n\n" + "\n".join(f"- {x}" for x in items[:200])) if len(items) >= 3 else ""


def _title(html: str, meta: dict, fallback: str) -> str:
    t = (meta.get("extra") or {}).get("title")
    if not t:
        try:
            node = HTMLParser(html).css_first("title")
            t = node.text(strip=True) if node else ""
        except Exception:
            t = ""
    t = re.sub(r"\s+", " ", t or "").strip()
    return (t[:200] or fallback)


def clean_html(artifact_path, meta: dict, *, llm_rescue_on: bool, fallback_title: str):
    """Shared HTML→CleanDoc path for news/mfw/official. trafilatura first; LLM rescue on gate fail."""
    from . import CleanDoc
    html = artifact_path.read_text(encoding="utf-8", errors="replace")
    url = meta.get("final_url") or meta.get("url") or ""
    res = extract_html(html, url)
    title = _title(html, meta, fallback_title)

    if res["gate_pass"]:
        return CleanDoc(title=title, body=res["text"], source=url, extractor="trafilatura",
                        quality_gate=res["metrics"])
    reason = res["gate_reason"] or ""
    # A link-heavy page (high-anchor / low-keep) is a LISTING: keep its faithful index of real
    # link-titles/headings (entities, dates, what-exists) rather than discard the signal.
    if reason.startswith(("high-anchor", "low-keep")):
        idx = listing_index(html)
        if idx:
            return CleanDoc(title=title, body=idx, source=url, extractor="listing-index",
                            quality_gate={**res["metrics"], "kept_as": "listing"})
        return None
    # too-short / empty = a genuine article trafilatura under-extracted → LLM rescue (faithful).
    if not llm_rescue_on:
        return None
    rescued = llm_rescue(html, url)
    if not rescued:
        return None                          # rescue failed/refused → no fabrication, skip
    return CleanDoc(title=title, body=rescued, source=url, extractor="llm-rescue",
                    llm_rescued=True, quality_gate={**res["metrics"], "rescued_from": reason})

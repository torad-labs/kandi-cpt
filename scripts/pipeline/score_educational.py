#!/usr/bin/env python3
"""Educational-density scorer — between extract and curate (docs/cpt-overhaul/01).

Teacher pass (v1): grade each 10_clean doc 1–5 on educational density for the domain
(facts mode) or register-representativeness (style mode). Cached by doc sha — idempotent.
Provider-bound → rides the existing consent line; a device-only corpus omits the stage and
curate falls back with EDUCATIONAL_SCORE_UNAVAILABLE.

Judge default is ``qwen/qwen3-32b`` (same working default as ``eval_generative.py``).
The free Nemotron tier proved unusable live for generative judging (2026-07-18); do not
reinstate it as the default. Override via ``--model`` or ``QGRE_EVAL_JUDGE_MODEL``.

    PYTHONPATH=scripts python scripts/pipeline/score_educational.py \
        --domain <d> [--editions …] [--mode facts|style] [--dry-run] \
        [--scores-json path]   # fixture path skips the teacher (unit tests / offline)
        [--model qwen/qwen3-32b]

Outputs (non-destructive):
  data/_build/educational/<domain>/<doc_sha>.json   per-doc score sidecar
  data/_build/educational/scores_<domain>.json      aggregate map path→score
  data/_build/educational/report_<domain>.json      distribution + mode + coverage
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

import config as C
from pipeline.editions import iter_editions

DEFAULT_JUDGE_MODEL = "qwen/qwen3-32b"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MAX_ATTEMPTS = 4
REQUEST_TIMEOUT_SEC = 120

MODES = ("facts", "style")

TEACHER_SYSTEM = (
    "You are a strict educational-density grader for a knowledge-install corpus. "
    "Score the DOCUMENT from 1 to 5. Reply with ONE short sentence of reasoning, then "
    "exactly one final line:\n"
    "SCORE: <n>\n"
    "where <n> is an integer 1..5."
)

_SCORE_LINE = re.compile(r"SCORE:\s*([1-5])\b", re.IGNORECASE)


def body_of(md_text: str) -> str:
    return md_text.split("\n---\n", 1)[-1].strip() if "\n---\n" in md_text else md_text.strip()


def doc_sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_entity_types(domain: str) -> list[str]:
    """Manifest entity_types anchor the facts-mode rubric. Missing → empty list (still grades)."""
    path = C.ROOT / "configs" / "domains" / f"{domain}.yaml"
    if not path.is_file():
        return []
    try:
        import yaml
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return []
    types = raw.get("entity_types") or []
    if isinstance(types, str):
        return [types]
    return [str(t) for t in types]


def collect_docs(domain: str, editions_only: str | None) -> list[dict]:
    docs = []
    for ed in iter_editions(domain, only=editions_only, with_raw=False):
        clean_root = ed.dir / "10_clean"
        if not clean_root.is_dir():
            continue
        for md in sorted(clean_root.rglob("*.md")):
            if md.name == "README.md":
                continue
            text = body_of(md.read_text(encoding="utf-8", errors="replace"))
            if not text.strip():
                continue
            rel = str(md.relative_to(C.ROOT))
            docs.append({
                "path": rel,
                "md": md,
                "edition": ed.slug,
                "text": text,
                "sha": doc_sha(text),
            })
    return docs


# /no_think: the judge default is qwen3, which otherwise spends the whole max_tokens budget
# reasoning and never reaches the required `SCORE: n` line — measured live on edc/zhuhai/2018
# (2026-07-24), where a 200-token budget produced pure deliberation and no verdict.
# audit_recall.py carries the same directive for the same model for the same reason.
NO_THINK = "\n\n/no_think"


def rubric_prompt(mode: str, entity_types: list[str], text: str) -> str:
    sample = text[:4000]
    if mode == "style":
        return (
            "MODE: style (register-representativeness)\n"
            "Score how characteristic this text is of the corpus register / persona voice. "
            "High = distinctive discourse habit, tone, or style signal even if fact-thin "
            "(chat logs full of short replies can score high). "
            "Low = generic boilerplate or off-register noise.\n\n"
            f"DOCUMENT:\n{sample}"
            f"{NO_THINK}"
        )
    types = ", ".join(entity_types) if entity_types else "(domain entities)"
    return (
        "MODE: facts (educational / knowledge density)\n"
        f"Score how much this document asserts about the domain's entities and relations "
        f"(entity types: {types}). "
        "5 = dense, specific, teachable facts; 1 = empty, boilerplate, or off-topic.\n\n"
        f"DOCUMENT:\n{sample}"
        f"{NO_THINK}"
    )


def parse_score(raw: str) -> int | None:
    score = None
    for match in _SCORE_LINE.finditer(raw):
        score = int(match.group(1))
    return score


def secret_tool_lookup(service: str) -> str | None:
    import subprocess
    try:
        proc = subprocess.run(
            ["secret-tool", "lookup", "service", service],
            capture_output=True, text=True, timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip() or None


def resolve_openrouter_key() -> str | None:
    value = (os.environ.get("OPENROUTER_API_KEY") or "").strip()
    if value:
        return value
    return secret_tool_lookup("openrouter")


def message_text(payload: dict) -> str:
    """Completion text, falling back to reasoning when a provider leaves content empty.

    Observed live on qwen3 via OpenRouter: the answer arrives only in the reasoning channel
    and ``content`` is "". ``audit_recall.py`` carries the same fallback for its judge.
    """
    message = payload["choices"][0]["message"] or {}
    raw = message.get("content") or ""
    if not raw.strip():
        raw = str(message.get("reasoning") or message.get("reasoning_content") or "")
    return raw


def chat(model: str, key: str, system: str, user: str) -> str:
    body = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.0,
        "max_tokens": 200,
    }).encode()
    request = urllib.request.Request(
        OPENROUTER_BASE_URL + "/chat/completions",
        data=body,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
        method="POST",
    )
    last_error = None
    for attempt in range(MAX_ATTEMPTS):
        try:
            with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SEC) as response:
                payload = json.loads(response.read().decode())
            text = message_text(payload)
            if text.strip():
                return text
            # An empty completion is a provider hiccup, not a grade — retry it like a 5xx
            # rather than handing "" to parse_score and failing the stage closed.
            last_error = "empty completion"
            time.sleep(2.0 * (2 ** attempt))
        except urllib.error.HTTPError as error:
            detail = error.read().decode(errors="replace")[:200]
            last_error = f"HTTP {error.code}: {detail}"
            if error.code not in (429, 500, 502, 503, 504):
                break
            time.sleep(2.0 * (2 ** attempt))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError, IndexError) as error:
            last_error = repr(error)
            time.sleep(2.0 * (2 ** attempt))
    raise RuntimeError(f"teacher call failed: {last_error}")


def score_one(
    mode: str,
    entity_types: list[str],
    text: str,
    *,
    teacher_fn=None,
) -> int:
    """Return 1..5. teacher_fn(system, user) -> raw string for tests."""
    user = rubric_prompt(mode, entity_types, text)
    if teacher_fn is not None:
        raw = teacher_fn(TEACHER_SYSTEM, user)
    else:
        raise RuntimeError("no teacher_fn and no live path wired at call site")
    score = parse_score(raw)
    if score is None:
        raise ValueError(f"unparseable teacher score: {raw!r}")
    return score


def heuristic_score(text: str, mode: str) -> int:
    """Offline fallback for dry-run / no-provider: rough 1–5 from length + token diversity.

    Not a substitute for the teacher — recorded as method=heuristic so curate knows.
    """
    words = text.split()
    n = len(words)
    if n < 40:
        base = 1
    elif n < 120:
        base = 2
    elif n < 300:
        base = 3
    elif n < 800:
        base = 4
    else:
        base = 5
    if mode == "style":
        # short distinctive bursts still carry style signal
        uniq = len({w.lower() for w in words}) / max(1, n)
        if uniq > 0.55 and n >= 20:
            base = min(5, base + 1)
    return base


def out_dirs(domain: str) -> tuple[Path, Path, Path]:
    root = C.BUILD / "educational"
    per_doc = root / domain
    scores_path = root / f"scores_{domain}.json"
    report_path = root / f"report_{domain}.json"
    return per_doc, scores_path, report_path


def load_cached(per_doc: Path, sha: str) -> dict | None:
    path = per_doc / f"{sha}.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def write_cached(per_doc: Path, record: dict) -> None:
    per_doc.mkdir(parents=True, exist_ok=True)
    path = per_doc / f"{record['sha']}.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def run_score(
    domain: str,
    editions: str | None,
    mode: str,
    *,
    dry_run: bool = False,
    scores_json: Path | None = None,
    model: str = DEFAULT_JUDGE_MODEL,
    allow_heuristic: bool = False,
) -> dict:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    docs = collect_docs(domain, editions)
    if not docs:
        raise ValueError(f"no 10_clean docs for domain={domain}")
    entity_types = load_entity_types(domain)
    per_doc, scores_path, report_path = out_dirs(domain)

    # Optional fixture map path→score (or sha→score) for offline tests.
    fixture: dict[str, int] = {}
    if scores_json is not None:
        raw = json.loads(Path(scores_json).read_text(encoding="utf-8"))
        fixture = {str(k): int(v) for k, v in raw.items()}

    key = resolve_openrouter_key()
    use_teacher = key is not None and not fixture and not allow_heuristic

    results: list[dict] = []
    path_to_score: dict[str, float] = {}
    cache_hits = 0
    for doc in docs:
        cached = load_cached(per_doc, doc["sha"])
        if cached and cached.get("mode") == mode and isinstance(cached.get("score"), (int, float)):
            record = cached
            cache_hits += 1
        elif doc["path"] in fixture or doc["sha"] in fixture:
            score = fixture.get(doc["path"], fixture.get(doc["sha"]))
            record = {
                "path": doc["path"], "sha": doc["sha"], "edition": doc["edition"],
                "mode": mode, "score": int(score), "method": "fixture",
            }
            if not dry_run:
                write_cached(per_doc, record)
        elif use_teacher:
            def _fn(system, user, _key=key, _model=model):
                return chat(_model, _key, system, user)
            score = score_one(mode, entity_types, doc["text"], teacher_fn=_fn)
            record = {
                "path": doc["path"], "sha": doc["sha"], "edition": doc["edition"],
                "mode": mode, "score": score, "method": "teacher", "model": model,
            }
            if not dry_run:
                write_cached(per_doc, record)
        elif allow_heuristic or dry_run:
            score = heuristic_score(doc["text"], mode)
            record = {
                "path": doc["path"], "sha": doc["sha"], "edition": doc["edition"],
                "mode": mode, "score": score, "method": "heuristic",
            }
            if not dry_run:
                write_cached(per_doc, record)
        else:
            raise RuntimeError(
                "no OpenRouter key for educational teacher and no --scores-json fixture; "
                "set OPENROUTER_API_KEY, pass --scores-json, or --allow-heuristic"
            )
        results.append(record)
        path_to_score[doc["path"]] = float(record["score"])

    scores_list = [r["score"] for r in results]
    dist = Counter(scores_list)
    report = {
        "domain": domain,
        "mode": mode,
        "n_docs": len(results),
        "cache_hits": cache_hits,
        "entity_types": entity_types,
        "score_distribution": {str(k): dist[k] for k in range(1, 6)},
        "mean_score": round(sum(scores_list) / len(scores_list), 4) if scores_list else 0.0,
        "method_counts": dict(Counter(r["method"] for r in results)),
        "scores_path": str(scores_path.relative_to(C.ROOT)) if scores_path.is_relative_to(C.ROOT) else str(scores_path),
    }
    if not dry_run:
        scores_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = scores_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(path_to_score, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, scores_path)
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "domain": domain, "mode": mode, "n": len(results),
        "mean": report["mean_score"], "distribution": report["score_distribution"],
        "cache_hits": cache_hits,
    }))
    return {"scores": path_to_score, "report": report, "results": results}


def load_educational_scores(domain: str) -> dict[str, float] | None:
    """Curate consumer: path→score map, or None if the stage was not run."""
    _, scores_path, _ = out_dirs(domain)
    if not scores_path.is_file():
        return None
    try:
        raw = json.loads(scores_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(raw, dict) or not raw:
        return None
    return {str(k): float(v) for k, v in raw.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description="educational-density scorer (extract→curate)")
    ap.add_argument("--domain", required=True)
    ap.add_argument("--editions")
    ap.add_argument("--mode", choices=list(MODES), default="facts")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--scores-json", help="fixture path→score map (skips teacher)")
    ap.add_argument("--model", default=DEFAULT_JUDGE_MODEL)
    ap.add_argument("--allow-heuristic", action="store_true",
                    help="offline heuristic scores when no provider key (recorded as method=heuristic)")
    args = ap.parse_args()
    try:
        run_score(
            args.domain, args.editions, args.mode,
            dry_run=args.dry_run,
            scores_json=Path(args.scores_json) if args.scores_json else None,
            model=args.model,
            allow_heuristic=args.allow_heuristic,
        )
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as error:
        sys.exit(f"score_educational failed closed: {error}")


if __name__ == "__main__":
    main()

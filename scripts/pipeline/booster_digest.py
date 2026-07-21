#!/usr/bin/env python3
"""Booster digest — usage harvesting + re-batched human gate (docs/trust-loop/03-booster-digest.md).

Continuous collection into a candidate pool; nothing reaches weights without surviving the same
admission gates as scraped sources. Digest cadence from evidence mass, never the calendar.

    PYTHONPATH=scripts python scripts/pipeline/booster_digest.py harvest \
        --domain pokemon-lore --signal correction --text "no, Giovanni leads Team Rocket" \
        --claimed-subject Giovanni --claimed-predicate leads --claimed-object "Team Rocket"

    PYTHONPATH=scripts python scripts/pipeline/booster_digest.py build-digest --domain pokemon-lore
    PYTHONPATH=scripts python scripts/pipeline/booster_digest.py decline --domain pokemon-lore --digest-id ...
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import config as C
from pipeline.editions import iter_editions

SIGNAL_TYPES = frozenset({
    "correction",       # user said "no, it's X"
    "reask_fumble",     # deflection / re-ask
    "hot_retrieval",    # hot-tier fact keeps firing
    "delta_corpus",     # new/changed source files
})
ADMISSION_OWN = "own-data"
# Evidence-mass thresholds (derived cadence — not calendar)
DEFAULT_MIN_CANDIDATES = 3
DEFAULT_MIN_CORRECTIONS = 1
PER_SIGNAL_CAP = 20


class DigestError(ValueError):
    """Digest pipeline refused."""


def _pool_dir(domain: str) -> Path:
    return C.BUILD / "booster" / domain


def _pool_path(domain: str) -> Path:
    return _pool_dir(domain) / "candidates.jsonl"


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _cid(*parts: str) -> str:
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


def load_graph_triples(domain: str, editions: str | None = None) -> list[dict]:
    triples = []
    for ed in iter_editions(domain, only=editions, with_raw=False):
        for name in ("triples.resolved.jsonl", "triples.jsonl"):
            path = ed.dir / "15_structured" / name
            if path.is_file():
                for line in path.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        triples.append(json.loads(line))
                break
    return triples


def load_entity_names(domain: str, editions: str | None = None) -> dict[str, str]:
    """id → name_canonical and lower-name → id surfaces."""
    id_to_name: dict[str, str] = {}
    name_to_id: dict[str, str] = {}
    for ed in iter_editions(domain, only=editions, with_raw=False):
        for name in ("entities.resolved.jsonl", "entities.jsonl"):
            path = ed.dir / "15_structured" / name
            if not path.is_file():
                continue
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                e = json.loads(line)
                eid = str(e.get("id") or "")
                cname = str(e.get("name_canonical") or eid)
                if eid:
                    id_to_name[eid] = cname
                    name_to_id[cname.strip().lower()] = eid
                for a in e.get("aliases") or []:
                    name_to_id[str(a).strip().lower()] = eid
            break
    return {"id_to_name": id_to_name, "name_to_id": name_to_id}


def truth_validate_claim(
    *,
    subject: str,
    predicate: str,
    obj: str,
    triples: list[dict],
    names: dict[str, Any],
) -> dict[str, Any]:
    """Validate a claimed fact against the graph.

    Returns status:
      supported     — graph already holds an equivalent triple
      novel         — entities known, predicate known, fact absent (candidate for install)
      dispute       — graph holds a DIFFERENT object for same subject+predicate
      rejected      — subject/object not in graph OR predicate empty (trolling / garbage)
    """
    id_to_name = names.get("id_to_name") or {}
    name_to_id = names.get("name_to_id") or {}

    def resolve(surface: str) -> str | None:
        s = (surface or "").strip()
        if not s:
            return None
        if s in id_to_name:
            return s
        return name_to_id.get(s.lower())

    def display(x: str) -> str:
        return id_to_name.get(x, x)

    s_id = resolve(subject)
    o_id = resolve(obj)
    pred = (predicate or "").strip()
    if not pred or s_id is None or o_id is None:
        return {
            "status": "rejected",
            "reason": "subject/object not in graph or predicate missing — truth-rejected",
            "subject_resolved": s_id,
            "object_resolved": o_id,
        }

    # Collect graph objects for (subject, predicate)
    graph_objs = set()
    for t in triples:
        ts = str(t.get("subject_id") or "")
        tp = str(t.get("predicate") or "")
        if tp != pred:
            continue
        # match subject by id or display
        if ts != s_id and display(ts).lower() != subject.strip().lower():
            continue
        to = str(t.get("object") or "")
        graph_objs.add(to)
        graph_objs.add(display(to).lower())

    claim_forms = {o_id, obj.strip().lower(), display(o_id).lower()}
    if graph_objs & claim_forms:
        return {
            "status": "supported",
            "reason": "graph already holds this fact",
            "subject_resolved": s_id,
            "object_resolved": o_id,
        }
    # Different object present → dispute
    if graph_objs:
        return {
            "status": "dispute",
            "reason": "graph holds a different object for this subject+predicate",
            "subject_resolved": s_id,
            "object_resolved": o_id,
            "graph_objects_sample": sorted(graph_objs)[:5],
        }
    return {
        "status": "novel",
        "reason": "entities known; fact absent from graph — eligible candidate",
        "subject_resolved": s_id,
        "object_resolved": o_id,
    }


def pii_quick_scan(text: str) -> dict[str, Any]:
    """Minimal PII gate (email/phone). Full pii_scan stage remains authoritative for corpora."""
    email = bool(re.search(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", text or ""))
    phone = bool(re.search(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b", text or ""))
    hit = email or phone
    return {
        "pass": not hit,
        "email": email,
        "phone": phone,
        "reason": "PII-like pattern detected" if hit else "ok",
    }


def harvest_candidate(
    domain: str,
    *,
    signal: str,
    text: str = "",
    claimed_subject: str | None = None,
    claimed_predicate: str | None = None,
    claimed_object: str | None = None,
    source_path: str | None = None,
    editions: str | None = None,
    meta: dict | None = None,
) -> dict[str, Any]:
    """Admit one usage signal into the candidate pool (or quarantine)."""
    if signal not in SIGNAL_TYPES:
        raise DigestError(f"unknown signal {signal!r}; want {sorted(SIGNAL_TYPES)}")

    pii = pii_quick_scan(text or "")
    gates = {
        "admission_basis": ADMISSION_OWN,
        "pii": pii,
    }
    status = "admitted"
    quarantine_reason = None

    if not pii["pass"]:
        status = "quarantined"
        quarantine_reason = f"pii: {pii['reason']}"

    truth = None
    if signal in {"correction", "reask_fumble", "hot_retrieval"} and claimed_predicate:
        triples = load_graph_triples(domain, editions)
        names = load_entity_names(domain, editions)
        truth = truth_validate_claim(
            subject=str(claimed_subject or ""),
            predicate=str(claimed_predicate or ""),
            obj=str(claimed_object or ""),
            triples=triples,
            names=names,
        )
        gates["truth"] = truth
        if truth["status"] == "rejected":
            status = "quarantined"
            quarantine_reason = f"truth: {truth['reason']}"
        elif truth["status"] == "dispute":
            status = "dispute"  # not installed; surfaces in digest for adjudication
        elif truth["status"] == "supported":
            status = "quarantined"
            quarantine_reason = "truth: already in graph (no new install signal)"
        # novel → admitted
    elif signal == "delta_corpus":
        gates["truth"] = {"status": "n/a", "reason": "corpus delta — structured stages re-derive facts"}
        if not source_path:
            status = "quarantined"
            quarantine_reason = "delta_corpus requires source_path"

    cand = {
        "id": _cid(domain, signal, text, str(claimed_subject), str(claimed_predicate), str(claimed_object), _now()),
        "domain": domain,
        "signal": signal,
        "text": text,
        "claimed": {
            "subject": claimed_subject,
            "predicate": claimed_predicate,
            "object": claimed_object,
        },
        "source_path": source_path,
        "status": status,
        "quarantine_reason": quarantine_reason,
        "gates": gates,
        "meta": meta or {},
        "harvested_at": _now(),
        "digest_id": None,  # set when included in a built digest
    }

    path = _pool_path(domain)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(cand) + "\n")
    return cand


def _clean_sidecar_for(raw: Path) -> Path:
    stem = raw.stem.replace(" ", "_")
    return raw.parents[1] / "10_clean" / "wiki" / f"{stem}.clean.json"


def harvest_delta_corpus(domain: str, *, editions: str | None = None) -> dict[str, Any]:
    """Discover new/changed flat-corpus sources from their canonical clean SHA sidecars.

    Unchanged sources are counted but never re-enter the candidate pool. Each admitted delta carries
    both hashes and the sidecar path, making the later booster shard traceable to the source change.
    """
    counts = {"new": 0, "changed": 0, "unchanged": 0, "harvested": 0}
    candidate_ids: list[str] = []
    existing = {
        (str(row.get("source_path") or ""), str((row.get("meta") or {}).get("raw_sha256") or "")): row
        for row in load_candidates(domain, pending_only=False)
        if row.get("signal") == "delta_corpus"
    }
    for edition in iter_editions(domain, only=editions):
        raw_dir = edition.dir / "00_raw"
        for raw in sorted(raw_dir.iterdir()) if raw_dir.is_dir() else []:
            if not raw.is_file() or raw.suffix.lower() not in {".txt", ".html"}:
                continue
            current_sha = hashlib.sha256(raw.read_bytes()).hexdigest()
            sidecar = _clean_sidecar_for(raw)
            previous_sha = None
            if sidecar.is_file():
                try:
                    previous_sha = json.loads(sidecar.read_text(encoding="utf-8")).get("raw_sha256")
                except (OSError, json.JSONDecodeError):
                    previous_sha = None
            if previous_sha == current_sha:
                counts["unchanged"] += 1
                continue
            delta_status = "changed" if sidecar.is_file() else "new"
            counts[delta_status] += 1
            try:
                source_path = str(raw.relative_to(C.ROOT))
                sidecar_path = str(sidecar.relative_to(C.ROOT))
            except ValueError:
                source_path = str(raw)
                sidecar_path = str(sidecar)
            identity = (source_path, current_sha)
            candidate = existing.get(identity)
            if candidate is None:
                candidate = harvest_candidate(
                    domain,
                    signal="delta_corpus",
                    text=f"{delta_status.capitalize()} source ready for the specialist: {raw.name}",
                    source_path=source_path,
                    editions=editions,
                    meta={
                        "unit": edition.slug,
                        "delta_status": delta_status,
                        "raw_sha256": current_sha,
                        "previous_raw_sha256": previous_sha,
                        "clean_sidecar": sidecar_path,
                    },
                )
                existing[identity] = candidate
            candidate_ids.append(candidate["id"])
            counts["harvested"] += 1
    return {
        "domain": domain,
        "editions": editions,
        "counts": counts,
        "candidate_ids": candidate_ids,
        "plain": (
            f"Found {counts['new']} new and {counts['changed']} changed source files; "
            f"{counts['unchanged']} unchanged files stayed out of the booster."
        ),
    }


def load_candidates(domain: str, *, pending_only: bool = True) -> list[dict]:
    path = _pool_path(domain)
    if not path.is_file():
        return []
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if pending_only:
        rows = [r for r in rows if not r.get("digest_id") and r.get("status") != "dropped"]
        unique = []
        seen = set()
        for row in rows:
            meta = row.get("meta") or {}
            key = (
                "delta",
                str(row.get("source_path") or ""),
                str(meta.get("raw_sha256") or ""),
            ) if row.get("signal") == "delta_corpus" else ("candidate", str(row.get("id") or ""))
            if key in seen:
                continue
            seen.add(key)
            unique.append(row)
        rows = unique
    return rows


def evidence_mass(candidates: list[dict]) -> dict[str, Any]:
    by = {}
    for c in candidates:
        by[c["signal"]] = by.get(c["signal"], 0) + 1
    n = len(candidates)
    n_corr = by.get("correction", 0)
    ready = n >= DEFAULT_MIN_CANDIDATES or n_corr >= DEFAULT_MIN_CORRECTIONS
    return {
        "n": n,
        "by_signal": by,
        "ready_for_digest": ready,
        "thresholds": {
            "min_candidates": DEFAULT_MIN_CANDIDATES,
            "min_corrections": DEFAULT_MIN_CORRECTIONS,
        },
    }


def render_digest_plain(digest: dict[str, Any]) -> str:
    candidates = list(digest.get("candidates") or [])
    admitted = [candidate for candidate in candidates if candidate.get("status") == "admitted"]
    disputes = [candidate for candidate in candidates if candidate.get("status") == "dispute"]
    quarantined = [candidate for candidate in candidates if candidate.get("status") == "quarantined"]
    new_files = sum(
        1 for candidate in admitted
        if candidate.get("signal") == "delta_corpus"
        and (candidate.get("meta") or {}).get("delta_status") == "new"
    )
    changed_files = sum(
        1 for candidate in admitted
        if candidate.get("signal") == "delta_corpus"
        and (candidate.get("meta") or {}).get("delta_status") == "changed"
    )
    usage_facts = sum(
        1 for candidate in admitted
        if candidate.get("signal") in ("correction", "reask_fumble", "hot_retrieval")
    )
    booster_n = int(digest.get("booster_n") or 0)
    approval_line = (
        f"Booster #{booster_n} approved."
        if digest.get("status") == "approved"
        else f"Booster #{booster_n} is ready. Approve?"
    )
    return (
        f"This period for your {digest.get('domain')} specialist:\n"
        f"{len(admitted)} new things worth learning "
        f"({new_files} new files, {changed_files} changed files, {usage_facts} facts from practice)\n"
        f"{len(disputes)} disputes needing your call\n"
        f"{len(quarantined)} held out because a gate failed — shown, not installed\n"
        + approval_line
    )


def refresh_digest_presentation(domain: str, digest_id: str) -> dict[str, Any]:
    """Re-render trust-loop copy from immutable candidate evidence without changing approval."""
    path = _pool_dir(domain) / "digests" / f"{digest_id}.json"
    if not path.is_file():
        raise DigestError(f"digest not found: {path}")
    digest = json.loads(path.read_text(encoding="utf-8"))
    digest["plain"] = render_digest_plain(digest)
    digest["presentation_refreshed_at"] = _now()
    path.write_text(json.dumps(digest, indent=2) + "\n", encoding="utf-8")
    return {"digest_id": digest_id, "status": digest.get("status"), "plain": digest["plain"]}


def build_digest(
    domain: str,
    *,
    booster_n: int | None = None,
    unit: str | None = None,
) -> dict[str, Any] | None:
    """Build one digest if evidence mass is met. Empty mass → None (not sent)."""
    cands = load_candidates(domain, pending_only=True)
    mass = evidence_mass(cands)
    if not mass["ready_for_digest"]:
        return None

    # Cap per signal; overflow stays pending for next digest
    selected = []
    overflow = []
    counts: dict[str, int] = {}
    for c in cands:
        sig = c["signal"]
        counts[sig] = counts.get(sig, 0) + 1
        if counts[sig] <= PER_SIGNAL_CAP:
            selected.append(c)
        else:
            overflow.append(c)

    # Booster number
    state_path = _pool_dir(domain) / "state.json"
    state = json.loads(state_path.read_text()) if state_path.is_file() else {"last_booster_n": 0, "halted": []}
    if booster_n is None:
        booster_n = int(state.get("last_booster_n") or 0) + 1

    digest_id = _cid(domain, str(booster_n), _now())
    admitted = [c for c in selected if c["status"] == "admitted"]
    disputes = [c for c in selected if c["status"] == "dispute"]
    quarantined = [c for c in selected if c["status"] == "quarantined"]

    # Weak questions line from certificate spine if present
    weak = []
    qpath = C.ROOT / "data" / domain / "questions.json"
    if qpath.is_file():
        qs = json.loads(qpath.read_text()).get("questions") or []
        weak = [q.get("text") for q in qs[:2]]

    digest = {
        "schema": "kandi.booster_digest.v1",
        "digest_id": digest_id,
        "domain": domain,
        "unit": unit,
        "booster_n": booster_n,
        "built_at": _now(),
        "status": "pending_approval",
        "evidence_mass": mass,
        "counts": {
            "selected": len(selected),
            "admitted": len(admitted),
            "disputes": len(disputes),
            "quarantined": len(quarantined),
            "overflow_persisted": len(overflow),
        },
        "candidates": selected,
        "plain": "",
        "weak_questions_hint": weak,
        "gates_note": (
            "Every candidate carries signal type, admission_basis=own-data, PII + truth gate results. "
            "Quarantined never reach the shard. Disputes need adjudication."
        ),
    }
    digest["plain"] = render_digest_plain(digest)

    # Mark selected with digest_id in pool (rewrite file)
    _rewrite_pool_assign_digest(domain, {c["id"] for c in selected}, digest_id)

    out = _pool_dir(domain) / "digests" / f"{digest_id}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(digest, indent=2) + "\n", encoding="utf-8")
    digest["path"] = str(out)

    state["last_digest_id"] = digest_id
    state["last_booster_n_pending"] = booster_n
    state_path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    return digest


def _rewrite_pool_assign_digest(domain: str, ids: set[str], digest_id: str) -> None:
    path = _pool_path(domain)
    if not path.is_file():
        return
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    out = []
    for r in rows:
        if r.get("id") in ids and not r.get("digest_id"):
            r = dict(r)
            r["digest_id"] = digest_id
        out.append(r)
    path.write_text("\n".join(json.dumps(r) for r in out) + ("\n" if out else ""), encoding="utf-8")


def approve_digest(
    domain: str,
    digest_id: str,
    *,
    run_booster_hook: bool = True,
) -> dict[str, Any]:
    """Operator approval — the one-tap gate. Triggers install_event scaffolding for booster N."""
    path = _pool_dir(domain) / "digests" / f"{digest_id}.json"
    if not path.is_file():
        raise DigestError(f"digest not found: {path}")
    digest = json.loads(path.read_text(encoding="utf-8"))
    if digest.get("status") not in {"pending_approval", "declined"}:
        raise DigestError(f"digest status {digest.get('status')!r} not approvable")

    digest["status"] = "approved"
    digest["approved_at"] = _now()
    # Only admitted candidates become install payload; disputes stay out
    payload = [c for c in digest.get("candidates") or [] if c.get("status") == "admitted"]
    digest["install_payload"] = payload
    path.write_text(json.dumps(digest, indent=2) + "\n", encoding="utf-8")

    state_path = _pool_dir(domain) / "state.json"
    state = json.loads(state_path.read_text()) if state_path.is_file() else {}
    state["last_booster_n"] = digest["booster_n"]
    state["last_approved_digest_id"] = digest_id
    state_path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")

    result = {
        "digest_id": digest_id,
        "status": "approved",
        "booster_n": digest["booster_n"],
        "n_install_payload": len(payload),
        "next": (
            "Run delta structured→…→tokenize on install_payload + new docs, then train anneal-shaped "
            "booster; install_event.py runs sentinel+certificate on the resulting checkpoint."
        ),
    }
    if run_booster_hook:
        # Record an approval receipt the install_event can cite
        receipt = _pool_dir(domain) / "digests" / f"{digest_id}.approved.json"
        receipt.write_text(json.dumps({**result, "approved_at": digest["approved_at"]}, indent=2) + "\n")
        result["approval_receipt"] = str(receipt)
    return result


def decline_digest(domain: str, digest_id: str) -> dict[str, Any]:
    """Decline: candidates PERSIST (digest_id cleared so they re-enter the next mass check)."""
    path = _pool_dir(domain) / "digests" / f"{digest_id}.json"
    if not path.is_file():
        raise DigestError(f"digest not found: {path}")
    digest = json.loads(path.read_text(encoding="utf-8"))
    digest["status"] = "declined"
    digest["declined_at"] = _now()
    path.write_text(json.dumps(digest, indent=2) + "\n", encoding="utf-8")

    # Persist candidates: clear digest_id so they remain pending
    ids = {c["id"] for c in digest.get("candidates") or []}
    pool = _pool_path(domain)
    if pool.is_file():
        rows = [json.loads(l) for l in pool.read_text(encoding="utf-8").splitlines() if l.strip()]
        out = []
        for r in rows:
            if r.get("id") in ids:
                r = dict(r)
                r["digest_id"] = None
                r["declined_digest_ids"] = list(r.get("declined_digest_ids") or []) + [digest_id]
            out.append(r)
        pool.write_text("\n".join(json.dumps(r) for r in out) + ("\n" if out else ""), encoding="utf-8")

    return {
        "digest_id": digest_id,
        "status": "declined",
        "plain": "Digest declined — candidates persisted for the next period; nothing vanished.",
        "n_persisted": len(ids),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="booster digest")
    sub = ap.add_subparsers(dest="cmd", required=True)

    h = sub.add_parser("harvest")
    h.add_argument("--domain", required=True)
    h.add_argument("--signal", required=True, choices=sorted(SIGNAL_TYPES))
    h.add_argument("--text", default="")
    h.add_argument("--claimed-subject")
    h.add_argument("--claimed-predicate")
    h.add_argument("--claimed-object")
    h.add_argument("--source-path")
    h.add_argument("--editions")

    hd = sub.add_parser("harvest-delta")
    hd.add_argument("--domain", required=True)
    hd.add_argument("--editions")

    b = sub.add_parser("build-digest")
    b.add_argument("--domain", required=True)
    b.add_argument("--unit")
    b.add_argument("--booster-n", type=int)

    a = sub.add_parser("approve")
    a.add_argument("--domain", required=True)
    a.add_argument("--digest-id", required=True)

    d = sub.add_parser("decline")
    d.add_argument("--domain", required=True)
    d.add_argument("--digest-id", required=True)

    rp = sub.add_parser("refresh-presentation")
    rp.add_argument("--domain", required=True)
    rp.add_argument("--digest-id", required=True)

    s = sub.add_parser("status")
    s.add_argument("--domain", required=True)

    args = ap.parse_args()
    try:
        if args.cmd == "harvest":
            cand = harvest_candidate(
                args.domain,
                signal=args.signal,
                text=args.text,
                claimed_subject=args.claimed_subject,
                claimed_predicate=args.claimed_predicate,
                claimed_object=args.claimed_object,
                source_path=args.source_path,
                editions=args.editions,
            )
            print(json.dumps({"id": cand["id"], "status": cand["status"],
                              "quarantine_reason": cand.get("quarantine_reason"),
                              "truth": (cand.get("gates") or {}).get("truth")}))
        elif args.cmd == "harvest-delta":
            print(json.dumps(harvest_delta_corpus(args.domain, editions=args.editions)))
        elif args.cmd == "build-digest":
            dig = build_digest(args.domain, booster_n=args.booster_n, unit=args.unit)
            if dig is None:
                print(json.dumps({"status": "not_ready", "note": "evidence mass below threshold — no digest sent"}))
            else:
                print(json.dumps({"digest_id": dig["digest_id"], "booster_n": dig["booster_n"],
                                  "path": dig.get("path"), "counts": dig["counts"]}))
        elif args.cmd == "approve":
            print(json.dumps(approve_digest(args.domain, args.digest_id)))
        elif args.cmd == "decline":
            print(json.dumps(decline_digest(args.domain, args.digest_id)))
        elif args.cmd == "refresh-presentation":
            print(json.dumps(refresh_digest_presentation(args.domain, args.digest_id)))
        elif args.cmd == "status":
            cands = load_candidates(args.domain, pending_only=True)
            print(json.dumps({"pending": len(cands), "mass": evidence_mass(cands)}, indent=2))
    except (DigestError, OSError, json.JSONDecodeError) as error:
        sys.exit(f"booster_digest failed closed: {error}")


if __name__ == "__main__":
    main()

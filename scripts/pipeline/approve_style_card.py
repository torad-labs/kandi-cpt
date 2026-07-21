#!/usr/bin/env python3
"""Record the explicit human-review approval boundary for a style card."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import config as C


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def approve(domain: str, approved_by: str) -> dict:
    root = C.BUILD / "style" / domain
    fingerprint = root / "fingerprint.json"
    card = root / "style_card.md"
    if not fingerprint.is_file() or not card.is_file():
        raise ValueError("style approval needs the fingerprint and review card")
    payload = json.loads(fingerprint.read_text(encoding="utf-8"))
    if payload.get("status") != "PASS":
        raise ValueError("a refused style fingerprint cannot be approved")
    if not approved_by.strip():
        raise ValueError("style approval must name the approving operator")
    receipt = {
        "schema": "kandi.style_approval.v1",
        "domain": domain,
        "status": "APPROVED",
        "approved_by": approved_by.strip(),
        "approved_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "fingerprint_sha256": sha256(fingerprint),
        "style_card_sha256": sha256(card),
        "plain": "The operator approved this fictional sample style card as the generation and judge anchor.",
    }
    path = root / "approval.json"
    path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", required=True)
    parser.add_argument("--approved-by", required=True)
    args = parser.parse_args()
    print(json.dumps(approve(args.domain, args.approved_by), indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Config-only CLI for the canonical launcher's automatic post-training trust chain."""
from __future__ import annotations

import argparse
import json
import sys

from pipeline.auto_chain import AutoChainError, run_auto_chain


def main() -> None:
    parser = argparse.ArgumentParser(description="automatic post-training sentinel + certificate chain")
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", choices=["cpt", "sft"], default=None)
    args = parser.parse_args()
    try:
        chain = run_auto_chain(args.config, mode=args.mode)
    except (AutoChainError, OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        sys.exit(f"automatic install chain failed closed: {error}")
    print(json.dumps(chain, indent=2))
    sys.exit(2 if chain.get("status") == "HALTED" else 0)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""torad-train MCP server — the AGENT front-door over the ACTION CATALOG (torad_actions.py).

Every catalog entry is auto-registered as an MCP tool named `torad_<action>`. This is the same catalog
the Lab-Light UI drives over HTTP, so there is ZERO drift between what an agent can do and what the UI can do.

Safety model (mirrors the SageMaker-MCP read-by-default / --allow-write pattern):
  • READ tools are always exposed (list_runs, get_run, gpu_status, compare_runs, diagnose_run, …).
  • WRITE/COMPUTE tools (launch_run, stop_run, gate_run, build_corpus, …) are exposed ONLY when
    TORAD_AGENT_WRITE is truthy. An agent without that env literally cannot mutate the workstation.
  • TORAD_MCP_TOOLS=read,train,data,eval,export narrows the exposed categories (default: all) so a
    "data inspection" session never pays the context cost of the training tools.
  • GPU-touching actions still pass the in-process single-GPU mutex (torad_actions.acquire_gpu) — even an
    agent told to "launch two trainers" gets a structured refusal, not a second job.

Run (STDIO, the Claude Code / Cursor transport):
  TORAD_AGENT_WRITE=1 .venv/bin/python scripts/torad_mcp.py
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import torad_actions as TA  # noqa: E402

from mcp.server import Server  # noqa: E402
from mcp.server.stdio import stdio_server  # noqa: E402
from mcp.types import TextContent, Tool  # noqa: E402

WRITE = os.environ.get("TORAD_AGENT_WRITE", "").lower() in ("1", "true", "yes", "on")
_cats_env = os.environ.get("TORAD_MCP_TOOLS", "all").replace(" ", "")
CATEGORIES = None if (_cats_env in ("", "all")) else set(_cats_env.split(","))

server = Server("torad-train")


def _json_type(spec: str) -> dict:
    t = (spec or "").split(":", 1)[0].strip().lower()
    if t.startswith("list"):
        return {"type": "array", "items": {"type": "string"}}
    if t in ("int", "integer"):
        return {"type": "integer"}
    if t in ("bool", "boolean"):
        return {"type": "boolean"}
    return {"type": "string"}


def _input_schema(a: dict) -> dict:
    props = {k: {**_json_type(v), "description": v} for k, v in (a.get("inputs") or {}).items()}
    return {"type": "object", "properties": props, "additionalProperties": False}


def _exposed() -> list[dict]:
    return TA.catalog(include_writes=WRITE, categories=CATEGORIES)


def _describe(a: dict) -> str:
    tags = []
    if a["mutating"]:
        tags.append("WRITE")
    if a["touches_gpu"]:
        tags.append("GPU — passes the single-GPU mutex; refuses if GPU0 is busy")
    tag = f" [{'; '.join(tags)}]" if tags else ""
    return f"{a['doc']}{tag}".strip()


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [Tool(name=f"torad_{a['name']}", description=_describe(a), inputSchema=_input_schema(a))
            for a in _exposed()]


@server.call_tool()
async def call_tool(name: str, arguments: dict | None) -> list[TextContent]:
    action = name[len("torad_"):] if name.startswith("torad_") else name
    _status, payload = TA.run_action(action, arguments or {}, allow_write=WRITE)
    return [TextContent(type="text", text=json.dumps(payload, default=str, indent=2))]


async def _amain() -> None:
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


def main() -> None:
    asyncio.run(_amain())


if __name__ == "__main__":
    main()

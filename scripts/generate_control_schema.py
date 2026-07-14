from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pydantic import TypeAdapter

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent.control.protocol.models import METHOD_PARAMS


OUTPUT = ROOT / "schema" / "app-server-v1.json"


def build_schema() -> dict[str, object]:
    """从服务端 typed params 生成确定性的协议 schema。"""
    methods = {
        method: TypeAdapter(model).json_schema()
        for method, model in sorted(METHOD_PARAMS.items())
    }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "Akashic App Server Protocol v1",
        "protocolVersion": "1.0",
        "transport": "JSON-RPC 2.0 NDJSON",
        "methods": methods,
        "notifications": [
            "thread/started",
            "thread/deleted",
            "turn/queued",
            "turn/started",
            "turn/completed",
            "item/started",
            "item/assistantMessage/delta",
            "item/completed",
            "operation/completed",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    _ = parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    encoded = json.dumps(build_schema(), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.check:
        return 0 if OUTPUT.is_file() and OUTPUT.read_text(encoding="utf-8") == encoded else 1
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(encoded, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

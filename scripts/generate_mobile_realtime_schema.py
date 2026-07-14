from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from infra.mobile_realtime.protocol import (
    COMMAND_TYPES,
    CONTROL_TYPES,
    EVENT_TYPES,
    FRAME_ADAPTER,
    MAX_JSON_FRAME_BYTES,
    PRE_AUTH_CONTROL_TYPES,
    PROTOCOL_VERSION,
)


OUTPUT = ROOT / "schema" / "mobile-realtime-v1.json"


def build_schema() -> dict[str, object]:
    """从服务端帧模型生成确定性的移动协议 schema。"""
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "Akashic Mobile Realtime Protocol v1",
        "protocolVersion": PROTOCOL_VERSION,
        "transport": "WebSocket JSON text frames",
        "maxJsonFrameBytes": MAX_JSON_FRAME_BYTES,
        "commandTypes": sorted(COMMAND_TYPES),
        "eventTypes": sorted(EVENT_TYPES),
        "controlTypes": sorted(CONTROL_TYPES),
        "preAuthControlTypes": sorted(PRE_AUTH_CONTROL_TYPES),
        "frame": FRAME_ADAPTER.json_schema(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    _ = parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    encoded = (
        json.dumps(build_schema(), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n"
    )
    if args.check:
        matches = OUTPUT.is_file() and OUTPUT.read_text(encoding="utf-8") == encoded
        return 0 if matches else 1
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    _ = OUTPUT.write_text(encoded, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

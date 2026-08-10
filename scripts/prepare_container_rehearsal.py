#!/usr/bin/env python3
"""从运行中的 Workspace 创建只供容器预演使用的一致副本。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.container_rehearsal.prepare import prepare_rehearsal


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument("--source-workspace", type=Path, required=True)
    _ = parser.add_argument("--source-config", type=Path, required=True)
    _ = parser.add_argument(
        "--plugin-home", type=Path, default=Path("~/.akashic-plugin")
    )
    _ = parser.add_argument("--target", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    manifest = prepare_rehearsal(
        source_workspace=args.source_workspace,
        source_config=args.source_config,
        plugin_home=args.plugin_home,
        target=args.target,
    )
    print(json.dumps({"manifest": str(manifest)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""插件数据目录布局：normalize 改成 plugin-data/<name>，revert 还原成旧版的 <name>-<marketplace>。实例须已停止。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.plugins.data_layout import relabel_plugin_data  # noqa: E402
from bootstrap.workspace_lock import WorkspaceInstanceLock  # noqa: E402


# 1. 持有 workspace 锁，保证没有运行中的实例；2. 转换；3. 打印已转换与被跳过的项。
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("normalize", "revert"))
    parser.add_argument("--workspace", type=Path, required=True)
    arguments = parser.parse_args()
    workspace = arguments.workspace.expanduser().resolve()
    lock = WorkspaceInstanceLock(workspace)
    lock.acquire()
    try:
        moves, skipped = relabel_plugin_data(workspace, to_normalized=arguments.command == "normalize")
    finally:
        lock.release()
    for move in moves:
        print(f"已转换: {move.source} -> {move.target} ({move.plugin_id})")
    for move in skipped:
        print(f"跳过（目标目录已有内容）: {move.source} -> {move.target} ({move.plugin_id})")
    return 1 if skipped else 0


if __name__ == "__main__":
    raise SystemExit(main())

"""插件数据目录在旧形态 <name>-<marketplace> 与归一化形态 <name> 之间的显式转换；实例须已停止。"""
from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from agent.plugins.files import sync_directory
from agent.plugins.manifest import LAYOUT_RECEIPT, plugin_data_labels, read_layout_state
from agent.plugins.reload_journal import check_pending_publication
from agent.plugins.selection import PluginSelection
from core.common.frozen_json import json_value
from infra.persistence.json_store import atomic_write_text


@dataclass(frozen=True)
class Relabel:
    plugin_id: str
    source: str
    target: str


def _write_receipt(workspace: Path, state: str, moves: list[Relabel]) -> None:
    document = {"version": 1, "state": state,
                "moves": [{"plugin_id": move.plugin_id, "source": move.source, "target": move.target} for move in moves]}
    atomic_write_text(workspace / LAYOUT_RECEIPT, json.dumps(document, ensure_ascii=False, indent=1),
                      domain="plugin_data_layout")


# 目录可以改名：目标不存在或是空目录；源缺失而目标已在时是上次中断后的续跑。
def _movable(workspace: Path, move: Relabel) -> bool:
    source, target = workspace / move.source, workspace / move.target
    if target.is_symlink() or source.is_symlink():
        return False
    if not target.exists():
        return True
    if not source.exists():
        return True
    return target.is_dir() and not any(target.iterdir())


# 1. 规划：只处理选择记录引用的目录，未被引用的遗留目录一律不动。
def _plan(selection: PluginSelection, *, to_normalized: bool) -> list[Relabel]:
    ref = selection.read()
    plan: list[Relabel] = []
    for component in () if ref is None else selection.components(ref):
        record = selection.read_input(component)
        plugin_id = cast(str, record["plugin_id"])
        legacy, normalized = plugin_data_labels(plugin_id)
        source, target = (legacy, normalized) if to_normalized else (normalized, legacy)
        if record["data_dir"] == source:
            plan.append(Relabel(plugin_id, source, target))
    return plan


# 2. 改写：记录与 distribution_adoption 里引用旧目录的条目一起换成新形态，一次原子提交。
def _commit(selection: PluginSelection, moves: list[Relabel]) -> None:
    ref = selection.read()
    assert ref is not None
    by_id = {move.plugin_id: move for move in moves}
    refs: list[str] = []
    for component in selection.components(ref):
        record = dict(selection.read_input(component))
        move = by_id.get(cast(str, record["plugin_id"]))
        if move is not None:
            record["data_dir"] = move.target
            component = selection.prepare(record)
        refs.append(component)
    adoption = selection.adoption()
    if adoption is not None:
        entries = [
            {**entry, "data_dir": by_id[entry["plugin_id"]].target} if entry["plugin_id"] in by_id else dict(entry)
            for entry in cast(list[Mapping[str, str]], json_value(adoption["entries"]))
        ]
        adoption = {**cast(dict[str, object], json_value(adoption)), "entries": entries}
    _ = selection.commit(tuple(refs), expected_ref=ref, distribution_adoption=adoption, replace_adoption=True)


# 成功转换的与因目标已有内容而跳过的分别返回；任何一项被跳过都不影响其余项。
def relabel_plugin_data(workspace: Path, *, to_normalized: bool) -> tuple[list[Relabel], list[Relabel]]:
    workspace = workspace.resolve(strict=False)
    selection = PluginSelection(workspace)
    if selection.path.exists() and selection.read() is not None:
        check_pending_publication(workspace)
    plan = _plan(selection, to_normalized=to_normalized)
    moves = [move for move in plan if _movable(workspace, move)]
    skipped = [move for move in plan if move not in moves]
    if moves:
        # 先落中间状态再改名：中断后启动会拒绝，重跑本命令即可继续。
        _write_receipt(workspace, "normalizing" if to_normalized else "reverting", moves)
        for move in moves:
            source, target = workspace / move.source, workspace / move.target
            if source.exists():
                if target.exists():
                    target.rmdir()
                os.rename(source, target)
        sync_directory(workspace / "plugin-data")
        _commit(selection, moves)
    # 3. 终态：归一化后留回执；还原完且没有被跳过的项时删除回执，实例回到旧形态。
    if to_normalized:
        if moves or read_layout_state(workspace) != "normalized":
            _write_receipt(workspace, "normalized", moves)
    elif skipped:
        _write_receipt(workspace, "normalized", [])
    else:
        (workspace / LAYOUT_RECEIPT).unlink(missing_ok=True)
        sync_directory(workspace / "runtime")
    return moves, skipped

"""停机时的插件状态恢复点：选择文件、用户 patch、本次清单涉及插件的 cache 指针，落盘后进程崩溃仍可还原。

只依赖标准库（宿主 bootstrap）。写入方是目标镜像里的发布流程：它会推进指针、写 patch、新建 artifact，
这些都在恢复点覆盖范围内；运行期数据目录与 Python 环境缓存按内容寻址，不在其中。
"""
from __future__ import annotations

import shutil
import sqlite3
import subprocess
from pathlib import Path
from collections.abc import Mapping
from typing import Any, cast

from agent.plugins.prerequisites import RECORD_FILES
from scripts.akashic_release.manifest import atomic_write, read_json, write_json
from scripts.akashic_release.model import ReleasePaths

_SCHEMA = 1
_POINTERS = ".pointers.json"
_ARTIFACTS = ".artifacts"


def _text(path: Path) -> str | None:
    return path.read_text(encoding="utf-8") if path.is_file() else None


def _journal(workspace: Path) -> Path:
    return workspace / "runtime/plugin-reloads.sqlite3"


# 安装恢复点里当前最大的 rowid；发布期间新增的行在还原时标为已回退。
def _journal_mark(workspace: Path) -> int:
    path = _journal(workspace)
    if not path.is_file():
        return 0
    with sqlite3.connect(path) as connection:
        try:
            row = connection.execute("SELECT COALESCE(MAX(rowid),0) FROM plugin_updates").fetchone()
        except sqlite3.OperationalError:
            return 0
    return int(row[0])


# artifact 的 origin；发布可能复用已存在的同提交 artifact 并改写 origin，所以也要记下来。
def _origin(artifact: Path) -> str | None:
    if not (artifact / ".git").exists():
        return None
    result = subprocess.run(["git", "-C", str(artifact), "config", "--get", "remote.origin.url"],
                            capture_output=True, text=True, check=False)
    return (result.stdout.strip() or None) if result.returncode == 0 else None


def _restore_origin(artifact: Path, origin: str | None) -> None:
    if not (artifact / ".git").exists() or _origin(artifact) == origin:
        return
    args = ["remote", "remove", "origin"] if origin is None else ["remote", "set-url", "origin", "--", origin]
    _ = subprocess.run(["git", "-C", str(artifact), *args], capture_output=True, text=True, check=True)


def _plugin_record(home: Path, plugin_id: str) -> dict[str, object]:
    name, _, marketplace = plugin_id.rpartition("@")
    base = home / "cache" / marketplace / name
    artifacts = base / _ARTIFACTS
    candidates = (home / "marketplaces", home / "marketplaces" / marketplace, home / "cache",
                  home / "cache" / marketplace, base, artifacts)
    return {
        "plugin_id": plugin_id, "pointers": _text(base / _POINTERS),
        "artifacts": sorted(item.name for item in artifacts.iterdir()) if artifacts.is_dir() else [],
        "origins": {item.name: _origin(item) for item in artifacts.iterdir()} if artifacts.is_dir() else {},
        "absent_dirs": [str(path) for path in candidates if not path.exists()],
    }


# 当前插件状态的恢复点文档：选择、patch、涉及插件的指针与 artifact、安装恢复点水位线。
def capture_state(workspace: Path, home: Path, plugin_ids: list[str]) -> dict[str, object]:
    return {
        "schema_version": _SCHEMA, "plugins_home": str(home), "workspace": str(workspace),
        "selection": _text(workspace / "runtime/plugin-stable.json"),
        "patch": _text(workspace / "bundle.patch.toml"),
        # 前置条件记录（原因与声明）属于安装事务的一部分：失败或恢复时还原为事务前的内容。
        "prerequisite_files": {name: _text(workspace / name) for name in RECORD_FILES},
        "journal_mark": _journal_mark(workspace),
        "plugins": [_plugin_record(home, plugin_id) for plugin_id in plugin_ids],
    }


# 停机后、发布前保存恢复点并落盘。
def capture(paths: ReleasePaths, plugin_ids: list[str], destination: Path) -> None:
    write_json(destination, capture_state(paths.state / "workspace", paths.state / "plugin-home", plugin_ids))


def _restore_file(path: Path, text: str | None) -> None:
    if text is None:
        path.unlink(missing_ok=True)
        return
    atomic_write(path, text, mode=path.stat().st_mode & 0o777 if path.exists() else 0o644)


def _restore_plugin(record: dict[str, object], home: Path) -> None:
    name, _, marketplace = str(record["plugin_id"]).rpartition("@")
    base = home / "cache" / marketplace / name
    pointers = cast(str | None, record["pointers"])
    _restore_file(base / _POINTERS, pointers)
    artifacts = base / _ARTIFACTS
    kept = set(cast(list[str], record["artifacts"]))
    origins = cast(dict[str, str | None], record["origins"])
    if artifacts.is_dir():
        for item in artifacts.iterdir():
            if item.name in kept:
                _restore_origin(item, origins.get(item.name))
                continue
            if item.is_dir() and not item.is_symlink():
                shutil.rmtree(item)
            else:
                item.unlink()
    for directory in reversed(cast(list[str], record["absent_dirs"])):
        try:
            Path(directory).rmdir()
        except OSError:
            pass


# 还原 patch、指针、本次新建的 artifact 与空目录；选择文件由调用方先按自己的 CAS 规则还原。
# 安装恢复点先结算：发布期间新增的 armed（进程死在提交前）与 committed 记录一律置为已回退，
# 这样旧版启动不会再按这些记录回滚已被删除的目录；整个还原可重复执行。
def restore_plugin_state(snapshot: Path) -> None:
    restore_state(read_json(snapshot))


def restore_state(document: Mapping[str, Any]) -> None:
    if document.get("schema_version") != _SCHEMA:
        raise RuntimeError("插件状态恢复点版本不支持")
    workspace = Path(document["workspace"])
    # 1. 日志先结算，再动文件。
    journal = _journal(workspace)
    if journal.is_file():
        with sqlite3.connect(journal) as connection:
            try:
                connection.execute(
                    "UPDATE plugin_updates SET phase='rolled_back',error='release rollback' "
                    "WHERE rowid>? AND phase IN ('armed','committed')", (document["journal_mark"],))
            except sqlite3.OperationalError:
                pass
    # 2. patch、指针、新建的 artifact 与目录。
    _restore_file(workspace / "bundle.patch.toml", document["patch"])
    for name, text in cast(dict[str, str | None], document.get("prerequisite_files", {})).items():
        _restore_file(workspace / name, text)
    for record in document["plugins"]:
        _restore_plugin(record, Path(document["plugins_home"]))


def snapshot_selection(snapshot: Path) -> str:
    text = read_json(snapshot)["selection"]
    if not isinstance(text, str):
        raise RuntimeError("恢复点没有选择文件原文")
    return text


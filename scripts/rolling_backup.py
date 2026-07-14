#!/usr/bin/env python3
"""为任意文件和 SQLite 数据库创建一致性滚动快照。"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import sqlite3
import time
import tomllib
from collections.abc import Iterable
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any


DEFAULT_RETENTION = 14


@dataclass(frozen=True)
class BackupSource:
    """描述一个快照内名称、源路径和复制方式。"""

    name: str
    path: Path
    kind: str


def _parse_source(raw: str, *, kind: str) -> BackupSource:
    name, separator, path = raw.partition("=")
    if not separator or not name.strip() or not path.strip():
        raise ValueError(f"源参数必须是 NAME=PATH: {raw!r}")
    return BackupSource(
        name=_validate_snapshot_name(name.strip()),
        path=Path(path).expanduser().resolve(),
        kind=kind,
    )


def _validate_snapshot_name(name: str) -> str:
    path = PurePosixPath(name)
    if path.is_absolute() or name in {"", "."} or ".." in path.parts:
        raise ValueError(f"快照内名称必须是相对安全路径: {name!r}")
    return str(path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="把任意文件和 SQLite 数据库备份成滚动快照"
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="TOML 配置文件；使用配置时不再需要逐项传入源路径",
    )
    parser.add_argument(
        "--file",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="复制普通文件，可重复传入",
    )
    parser.add_argument(
        "--sqlite",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="使用 SQLite 在线备份 API 复制数据库，可重复传入",
    )
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--retention", type=int)
    return parser.parse_args()


def _config_string(config: dict[str, Any], key: str) -> str:
    value = config.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"备份配置 {key!r} 必须是非空字符串")
    return value.strip()


def _sources_from_config(raw_sources: object) -> list[BackupSource]:
    if not isinstance(raw_sources, list):
        raise ValueError("备份配置 sources 必须是数组")
    sources: list[BackupSource] = []
    for raw in raw_sources:
        if not isinstance(raw, dict):
            raise ValueError("备份配置中的 source 必须是对象")
        kind = _config_string(raw, "kind")
        if kind not in {"file", "sqlite"}:
            raise ValueError(f"备份配置 source.kind 无效: {kind!r}")
        sources.append(
            BackupSource(
                name=_validate_snapshot_name(_config_string(raw, "name")),
                path=Path(_config_string(raw, "path")).expanduser().resolve(),
                kind=kind,
            )
        )
    return sources


def _load_config(path: Path) -> tuple[Path, int, list[BackupSource]]:
    """读取通用 TOML 配置并转换成备份计划。"""

    # 1. 读取显式配置，缺失字段直接失败。
    with path.expanduser().open("rb") as handle:
        raw = tomllib.load(handle)
    if not isinstance(raw, dict):
        raise ValueError("备份配置根节点必须是对象")
    destination = Path(_config_string(raw, "destination")).expanduser()
    retention_value = raw.get("retention", DEFAULT_RETENTION)
    if not isinstance(retention_value, int):
        raise ValueError("备份配置 retention 必须是整数")

    # 2. 转换并校验源声明，源的语义完全由配置决定。
    sources = _sources_from_config(raw.get("sources"))
    return destination, retention_value, sources


def _validate_sources(sources: Iterable[BackupSource]) -> list[BackupSource]:
    validated = list(sources)
    names = [source.name for source in validated]
    if len(names) != len(set(names)):
        raise ValueError("快照内名称不能重复")
    if not validated:
        raise ValueError("至少需要一个 --file 或 --sqlite 源")
    for source in validated:
        _ = _validate_snapshot_name(source.name)
        if source.kind not in {"file", "sqlite"}:
            raise ValueError(f"未知备份类型: {source.kind}")
        if not source.path.is_file():
            raise FileNotFoundError(
                f"备份源不存在或不是文件: {source.name} -> {source.path}"
            )
    return validated


def _copy_sqlite(source: Path, destination: Path) -> None:
    """使用 SQLite 在线备份 API 生成一致性数据库快照。"""

    source_uri = f"file:{source}?mode=ro"
    with closing(sqlite3.connect(source_uri, uri=True)) as source_db:
        with closing(sqlite3.connect(destination)) as destination_db:
            # 1. 从运行中的数据库复制一致性快照，不直接复制主文件/WAL。
            source_db.backup(destination_db, pages=256, sleep=0.1)

            # 2. 备份完成后验证目标库，损坏时让调用方失败。
            result = destination_db.execute("PRAGMA integrity_check").fetchone()
            if result is None or result[0] != "ok":
                raise sqlite3.DatabaseError(
                    f"备份数据库完整性检查失败: {destination} ({result})"
                )
            destination_db.commit()


def _copy_source(source: BackupSource, snapshot: Path) -> None:
    destination = snapshot / source.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.kind == "sqlite":
        _copy_sqlite(source.path, destination)
        return
    shutil.copy2(source.path, destination)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_manifest(snapshot: Path, sources: Iterable[BackupSource]) -> None:
    manifest = {
        "created_at": datetime.now().astimezone().isoformat(),
        "files": {
            source.name: {
                "kind": source.kind,
                "source": str(source.path),
                "size": (snapshot / source.name).stat().st_size,
                "sha256": _sha256(snapshot / source.name),
            }
            for source in sources
        },
    }
    (snapshot / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _prune(destination: Path, retention: int) -> None:
    snapshots = sorted(
        (path for path in destination.glob("snapshot-*") if path.is_dir()),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )
    for path in snapshots[retention:]:
        shutil.rmtree(path)


def create_snapshot(
    *,
    sources: Iterable[BackupSource],
    destination: Path,
    retention: int = DEFAULT_RETENTION,
) -> Path:
    """创建所有源的一致性快照，并清理超出保留数的旧快照。"""

    # 1. 校验配置和所有源，避免生成半套快照。
    if retention < 1:
        raise ValueError("retention 必须大于等于 1")
    validated_sources = _validate_sources(sources)
    destination = destination.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)

    # 2. 用锁阻止并发备份写入同一个目标目录。
    lock_path = destination / ".backup.lock"
    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"已有备份任务正在运行: {lock_path}") from exc

        timestamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
        temporary = destination / f".snapshot-{timestamp}-{os.getpid()}.tmp"
        snapshot = destination / f"snapshot-{timestamp}"
        if snapshot.exists():
            timestamp = f"{timestamp}-{time.time_ns() % 1_000_000:06d}"
            snapshot = destination / f"snapshot-{timestamp}"
        committed = False
        try:
            # 3. 先写临时目录，完成所有源和 manifest 后再原子发布。
            temporary.mkdir()
            for source in validated_sources:
                _copy_source(source, temporary)
            _write_manifest(temporary, validated_sources)
            os.replace(temporary, snapshot)
            committed = True

            # 4. 只在新快照完整发布后滚动删除旧快照。
            _prune(destination, retention)
            return snapshot
        finally:
            if not committed and temporary.exists():
                shutil.rmtree(temporary)


def main() -> None:
    args = _parse_args()
    if args.config is not None:
        if (
            args.file
            or args.sqlite
            or args.destination is not None
            or args.retention is not None
        ):
            raise ValueError(
                "--config 不能和 --file/--sqlite/--destination/--retention 同时使用"
            )
        destination, retention, sources = _load_config(args.config)
    else:
        if args.destination is None:
            raise ValueError("未提供 --config 或 --destination")
        sources = [
            *(_parse_source(raw, kind="file") for raw in args.file),
            *(_parse_source(raw, kind="sqlite") for raw in args.sqlite),
        ]
        destination = args.destination
        retention = (
            DEFAULT_RETENTION if args.retention is None else args.retention
        )
    snapshot = create_snapshot(
        sources=sources,
        destination=destination,
        retention=retention,
    )
    print(f"备份完成: {snapshot}")


if __name__ == "__main__":
    main()

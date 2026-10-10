"""停写后把旧任务数组升级为同文件操作与触发回执，不从读取入口迁移。"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import cast

from infra.persistence.json_store import atomic_save_json, atomic_write_text
from .store import JobStore, ScheduleState


def _hash(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _backup(path: Path, raw: bytes) -> None:
    """恢复材料先 fsync；已存在但不同的文件不是本次可覆盖的备份。"""
    try:
        with path.open("xb") as stream:
            os.fchmod(stream.fileno(), 0o600)
            _ = stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError:
        if path.read_bytes() != raw:
            raise ValueError(f"调度迁移备份内容冲突：{path}") from None
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def migrate(path: Path, backup_root: Path) -> None:
    """先固定原字节和发布意图；中断后只沿这份已校验候选继续提交。"""
    store = JobStore(path)
    manifest_path = backup_root / "manifest.json"
    # 1. 已有 intent 优先恢复；不能把替换成功误认成无需保留原备份的新文件。
    if manifest_path.exists():
        raw_manifest = store.parse(manifest_path.read_text())
        if not isinstance(raw_manifest, dict):
            raise ValueError("调度迁移 manifest 无效")
        manifest = cast(dict[str, object], raw_manifest)
        if (set(manifest) != {"version", "path", "before", "after", "status"}
                or type(manifest["version"]) is not int or manifest["version"] != 1
                or manifest["path"] != str(path.resolve())
                or manifest["status"] not in {"prepared", "complete"}):
            raise ValueError("调度迁移 manifest 无效")
        before = (backup_root / "schedules.before.json").read_bytes()
        after = (backup_root / "schedules.after.json").read_bytes()
        if _hash(before) != manifest["before"] or _hash(after) != manifest["after"]:
            raise ValueError("调度迁移备份摘要不一致")
        old = store.decode_jobs(store.parse(before))
        new = store.decode(store.parse(after))
        if list(new.jobs.values()) != old or new.operations or new.fires:
            raise ValueError("调度迁移候选没有完整保留原任务")
        if manifest["status"] == "complete":
            # 成功后正常新增/取消是合法事实；重复执行不能覆盖它们。
            _ = store.read()
            return
    else:
        if not path.exists():
            return
        before = path.read_bytes()
        value = store.parse(before)
        if isinstance(value, dict):
            _ = store.decode(cast(dict[str, object], value))
            return
        jobs = store.decode_jobs(value)
        candidate = store.encode(ScheduleState({job.id: job for job in jobs}, {}, {}))
        after = json.dumps(candidate, ensure_ascii=False, indent=2).encode()
        backup_root.mkdir(parents=True, mode=0o700, exist_ok=True)
        _backup(backup_root / "schedules.before.json", before)
        _backup(backup_root / "schedules.after.json", after)
        manifest = {"version": 1, "path": str(path.resolve()), "before": _hash(before),
                    "after": _hash(after), "status": "prepared"}
        atomic_save_json(manifest_path, manifest, domain="scheduler-migration")

    # 2. 只接受原文件或同一候选；其他字节说明停写边界失效，不能继续覆盖。
    current = path.read_bytes()
    if _hash(current) == manifest["before"]:
        atomic_write_text(path, after.decode(), domain="scheduler-migration")
    elif _hash(current) != manifest["after"]:
        raise ValueError("调度迁移期间原文件已经变化")
    if path.read_bytes() != after:
        raise ValueError("调度迁移输出与已固定候选不一致")
    _ = store.read()
    manifest["status"] = "complete"
    atomic_save_json(manifest_path, manifest, domain="scheduler-migration")

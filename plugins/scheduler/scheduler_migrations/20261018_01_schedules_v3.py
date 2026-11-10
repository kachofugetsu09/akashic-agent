"""schedules.json v2 → v3：任务改为在创建它的会话里执行（ADR 0125）。

v2 的 instant/soft 换成 notify/run，message/prompt 合并成 text，目标会话按原渠道地址推出，
历史范围记为 none。旧流程未完成的 run 触发在内部会话里开始过，新流程不续跑，明确以失败结算；
notify 触发不涉及内部会话，保持 pending 由新流程补完。先留可读备份，再原子替换。
"""
from __future__ import annotations

import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from yoyo import step

from agent.migrations.context import current_migration_context

__depends__: set[str] = set()
__transactional__ = False

_UNFINISHED = "升级前未完成的旧触发，未续跑"


def _job(job: dict[str, Any]) -> dict[str, Any]:
    value = {key: item for key, item in job.items() if key not in {"tier", "message", "prompt"}}
    instant = job["tier"] == "instant"
    return {**value, "action": "notify" if instant else "run",
            "text": job["message"] if instant else job["prompt"],
            "session_id": f"{job['channel']}:{job['chat_id']}", "context": "none"}


def _fire(fire: dict[str, Any]) -> dict[str, Any]:
    job = _job(fire["job"])
    if fire["status"] == "pending" and job["action"] == "run":
        return {**fire, "job": job, "status": "failed", "error": _UNFINISHED}
    return {**fire, "job": job}


# 只升级 v2；缺文件或已是 v3 时保持原样。
def _upgrade(_connection: object) -> None:
    path = current_migration_context().workspace / "schedules.json"
    if not path.is_file():
        return
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("version") != 2:
        return
    upgraded = {**raw, "version": 3,
                "jobs": [_job(job) for job in raw["jobs"]],
                "fires": {key: _fire(fire) for key, fire in raw["fires"].items()}}
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    path.with_name(f"{path.name}.pre-v3-{stamp}").write_bytes(path.read_bytes())
    descriptor, name = tempfile.mkstemp(prefix=".schedules-v3-", dir=path.parent)
    os.close(descriptor)
    temporary = Path(name)
    temporary.write_text(json.dumps(upgraded, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


steps = [step(_upgrade)]

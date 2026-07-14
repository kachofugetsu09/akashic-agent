from __future__ import annotations

import json
import os
from pathlib import Path


class RuntimeReadiness:
    """发布并清理当前 supervised boot 的完整启动状态。"""

    def __init__(self, workspace: Path, boot_id: str) -> None:
        if not boot_id:
            raise ValueError("boot_id 不能为空")
        self.path = workspace / ".runtime-ready.json"
        self.boot_id = boot_id
        self.pid = os.getpid()
        self.ready = False

    def mark_ready(self) -> None:
        """在完整 AppRuntime.start 成功后原子发布 readiness。"""

        payload = {
            "bootId": self.boot_id,
            "pid": self.pid,
            "state": "ready",
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(
            f".{self.path.name}.{self.pid}.{self.boot_id}.tmp"
        )
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
        os.replace(temporary, self.path)
        self.ready = True

    def clear(self) -> None:
        """仅删除仍属于当前 boot 的 readiness 文件。"""

        self.ready = False
        if not self.path.exists():
            return
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if payload.get("bootId") == self.boot_id and payload.get("pid") == self.pid:
            self.path.unlink()

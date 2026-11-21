from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import cast

from agent.restart import SupervisorCommitChannel


# Read the immutable receipt written by the exiting Core. Host checks container exit separately.
def read_closed_boot(workspace: Path, boot_id: str) -> tuple[str, tuple[str, ...]]:
    if not re.fullmatch(r"[a-zA-Z0-9_-]{1,128}", boot_id):
        raise ValueError("旧 boot ID 无效")
    path = workspace / "runtime" / "closed" / f"{boot_id}.json"
    data: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise RuntimeError("旧 Core 关闭回执必须是对象")
    receipt = cast(dict[str, object], data)
    root = receipt.get("rootIdentity")
    if receipt.get("bootId") != boot_id or not isinstance(root, str) or not root:
        raise RuntimeError("旧 Core 关闭回执的 boot 或 Root 不符")
    if receipt["state"] == "closed" and "transactions" not in receipt:
        return root, ()
    raw = receipt.get("transactions")
    if receipt["state"] != "generation_cleanup_pending" or not isinstance(raw, list) or not raw:
        raise RuntimeError("旧 Core 关闭回执缺少待清理事务")
    transactions: list[str] = []
    for tx in cast(list[object], raw):
        if not isinstance(tx, str) or not re.fullmatch(r"[0-9a-f]{32}", tx) or tx in transactions:
            raise RuntimeError("旧 Core 关闭回执包含无效或重复的事务 ID")
        transactions.append(tx)
    return root, tuple(transactions)


class RuntimeReadiness:
    """发布并清理当前 supervised boot 的完整启动状态。"""

    def __init__(
        self,
        workspace: Path,
        boot_id: str,
        lifecycle: SupervisorCommitChannel | None = None,
    ) -> None:
        if not boot_id:
            raise ValueError("boot_id 不能为空")
        self.path = workspace / ".runtime-ready.json"
        self.boot_id = boot_id
        self.pid = os.getpid()
        self.ready = False
        self.lifecycle = lifecycle

    def mark_stage(self, name: str) -> None:
        """把当前启动阶段发布到私有生命周期管道。"""

        if self.lifecycle is not None:
            self.lifecycle.stage(name)

    def mark_ready(self) -> None:
        """在完整 AppRuntime.start 成功后原子发布 readiness。"""

        if self.lifecycle is not None:
            self.lifecycle.ready(self.pid)
        payload = {
            "bootId": self.boot_id,
            "pid": self.pid,
            "state": "ready",
        }
        runtime_commit = os.environ.get("AKASHIC_RUNTIME_COMMIT", "")
        runtime_checkout = os.environ.get("AKASHIC_RUNTIME_CHECKOUT", "")
        if runtime_commit:
            payload["sourceCommit"] = runtime_commit
        if runtime_checkout:
            payload["hostCheckout"] = runtime_checkout
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

    def mark_closed(self, root_identity: str | None, *, pending: tuple[str, ...] = ()) -> None:
        """保存此 boot 的退出状态及待宿主结算的事务，不覆盖已有回执。"""
        path = self.path.parent / "runtime" / "closed" / f"{self.boot_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, object] = {"bootId": self.boot_id, "pid": self.pid, "state": "closed", "rootIdentity": root_identity}
        if pending:
            payload.update(state="generation_cleanup_pending", transactions=list(pending))
        descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=".closed-")
        try:
            with os.fdopen(descriptor, "w") as stream:
                json.dump(payload, stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.link(temporary, path)
            directory = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            os.unlink(temporary)

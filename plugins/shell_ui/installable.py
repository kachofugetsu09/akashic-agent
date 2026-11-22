"""设置页“可安装”分组：读取 extra 扩展包列表并在后台子进程里安装；包管理逻辑全在 `main.py pkg`。"""
from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import signal
import sys
import time
from typing import Literal, cast

Row = dict[str, object]
TaskState = Literal["running", "succeeded", "failed"]

LIST_TIMEOUT = 60.0
INSTALL_TIMEOUT = 900.0
TAIL_LINES = 40
_PKG_ENV = ("AKASHIC_CORE_ROOT", "AKASHIC_WORKSPACE", "AKASHIC_CONFIG")


class InstallBusy(RuntimeError):
    pass


@dataclass(slots=True)
class InstallTask:
    name: str
    started_at: float
    state: TaskState = "running"
    exit_code: int | None = None
    tail: deque[str] = field(default_factory=lambda: deque(maxlen=TAIL_LINES))
    watcher: asyncio.Task[None] | None = None

    def snapshot(self) -> Row:
        return {"name": self.name, "state": self.state, "exit_code": self.exit_code,
                "tail": list(self.tail), "elapsed": round(time.monotonic() - self.started_at, 1)}


# 子命令只用运行中的 Core 自己的解释器、main.py、workspace 和配置，不猜开发目录。
def pkg_command(*arguments: str) -> list[str]:
    missing = [key for key in _PKG_ENV if not os.environ.get(key)]
    if missing:
        raise RuntimeError("宿主没有提供 " + "、".join(missing))
    main = Path(os.environ["AKASHIC_CORE_ROOT"]) / "main.py"
    return [sys.executable, str(main), "pkg", *arguments,
            "--workspace", os.environ["AKASHIC_WORKSPACE"], "--config", os.environ["AKASHIC_CONFIG"]]


# 只保留含未装包的组与未装包。
def uninstalled_groups(index: Row) -> list[Row]:
    groups: list[Row] = []
    for group in cast(list[Row], index["groups"]):
        packages = [item for item in cast(list[Row], group["packages"]) if not item["installed"]]
        if packages:
            groups.append({"id": group["id"], "title": group["title"], "description": group["description"],
                           "required": group["required"], "packages": packages})
    return groups


class PackageInstaller:
    def __init__(self) -> None:
        self._task: InstallTask | None = None
        self._known: set[str] = set()

    # 读取扩展包列表；失败时抛出带原因的 RuntimeError，由路由转成错误文字。
    async def available(self) -> Row:
        process = await asyncio.create_subprocess_exec(
            *pkg_command("groups", "--json"), stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        try:
            out, err = await asyncio.wait_for(process.communicate(), LIST_TIMEOUT)
        except TimeoutError as error:
            process.kill()
            _ = await process.wait()
            raise RuntimeError(f"超过 {int(LIST_TIMEOUT)} 秒仍没有读到") from error
        if process.returncode != 0:
            raise RuntimeError((err.decode(errors="replace").strip().splitlines() or ["命令失败"])[-1])
        groups = uninstalled_groups(cast(Row, json.loads(out.decode())))
        self._known = {cast(str, item["name"]) for group in groups for item in cast(list[Row], group["packages"])}
        return {"groups": groups}

    # 校验名字确实是未装的扩展包，再起后台安装；同一时间只允许一个任务。
    async def start(self, name: str) -> Row:
        if self._task is not None and self._task.state == "running":
            raise InstallBusy(f"正在安装 {self._task.name}，请等它结束")
        if name not in self._known:
            _ = await self.available()
        if name not in self._known:
            raise LookupError(f"{name} 不在可安装列表里，或已经安装")
        process = await asyncio.create_subprocess_exec(
            *pkg_command("install", name), stdin=asyncio.subprocess.DEVNULL, start_new_session=True,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
        task = InstallTask(name, time.monotonic())
        self._task = task
        task.watcher = asyncio.create_task(self._watch(task, process))
        return task.snapshot()

    def status(self) -> Row:
        return {"task": None if self._task is None else self._task.snapshot()}

    # 逐行收集输出；超时则杀掉整个进程组并标记失败。
    async def _watch(self, task: InstallTask, process: asyncio.subprocess.Process) -> None:
        async def pump() -> None:
            assert process.stdout is not None
            async for line in process.stdout:
                task.tail.append(line.decode(errors="replace").rstrip())
            task.exit_code = await process.wait()

        try:
            await asyncio.wait_for(pump(), INSTALL_TIMEOUT)
            task.state = "succeeded" if task.exit_code == 0 else "failed"
        except TimeoutError:
            os.killpg(process.pid, signal.SIGKILL)
            _ = await process.wait()
            task.tail.append(f"超过 {int(INSTALL_TIMEOUT // 60)} 分钟仍未结束，已终止")
            task.state = "failed"
        except Exception as error:  # 收集过程自身出错也要落到终态
            task.tail.append(str(error))
            task.state = "failed"

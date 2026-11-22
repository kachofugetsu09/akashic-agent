"""设置页“可安装”分组：读取 extra 扩展包列表并在后台子进程里安装；包管理逻辑全在 `main.py pkg`。"""
from __future__ import annotations

import asyncio
from collections import deque
from contextlib import suppress
import json
import os
from pathlib import Path
import re
import signal
import sys
import time
from typing import cast

from core.common.file_io import run_file_io

Row = dict[str, object]

LIST_TIMEOUT = 60.0
INSTALL_TIMEOUT = 900.0
# 给 pkg 在线等待终态的时间，略低于外层寿命，免得 120 秒默认值把仍在进行的安装报成失败。
PKG_WAIT = 840
TAIL_LINES = 40
KEEP_TASKS = 5
_PKG_ENV = ("AKASHIC_CORE_ROOT", "AKASHIC_WORKSPACE", "AKASHIC_CORE_CONFIG")
_OUTCOME = re.compile(r"^\S+\s+(installed|upgraded|current|restart-required|in-progress|failed|inactive|needs-switch)\b", re.M)
# 子进程把退出码写进 .exit 文件，shell_ui 换代后新一代仍能读到。
_WRAPPER = '"$@"; code=$?; echo "$code" > "$0.tmp" && mv "$0.tmp" "$0"'


class InstallBusy(RuntimeError):
    pass


# 子命令只用运行中的 Core 自己的解释器、main.py、workspace 和配置，不猜开发目录。
def pkg_command(*arguments: str) -> list[str]:
    missing = [key for key in _PKG_ENV if not os.environ.get(key)]
    if missing:
        raise RuntimeError("宿主没有提供 " + "、".join(missing))
    main = Path(os.environ["AKASHIC_CORE_ROOT"]) / "main.py"
    return [sys.executable, str(main), "pkg", *arguments,
            "--workspace", os.environ["AKASHIC_WORKSPACE"], "--config", os.environ["AKASHIC_CORE_CONFIG"]]


# 只保留含未装包的组与未装包。
def uninstalled_groups(index: Row) -> list[Row]:
    groups: list[Row] = []
    for group in cast(list[Row], index["groups"]):
        packages = [item for item in cast(list[Row], group["packages"]) if not item["installed"]]
        if packages:
            groups.append({"id": group["id"], "title": group["title"], "description": group["description"],
                           "required": group["required"], "packages": packages})
    return groups


# 进程是否还在：以 kill(pid, 0) 为准；有 /proc（Linux）时再排除僵尸，没有（macOS）不当作已死。
def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] != "Z"
    except (OSError, IndexError):
        return True


def _kill_group(pid: int) -> None:
    with suppress(ProcessLookupError, PermissionError):
        os.killpg(pid, signal.SIGKILL)


class PackageInstaller:
    def __init__(self) -> None:
        self._claimed: str | None = None
        self._known: set[str] = set()
        self._watchers: set[asyncio.Task[None]] = set()

    def _dir(self) -> Path:
        return Path(os.environ["AKASHIC_WORKSPACE"]) / "runtime" / "settings-install"

    # 读取扩展包列表；失败时抛出带原因的 RuntimeError，由路由转成错误文字。
    async def available(self) -> Row:
        process = await asyncio.create_subprocess_exec(
            *pkg_command("groups", "--json"), stdin=asyncio.subprocess.DEVNULL, start_new_session=True,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        try:
            out, err = await asyncio.wait_for(process.communicate(), LIST_TIMEOUT)
        except TimeoutError as error:
            raise RuntimeError(f"超过 {int(LIST_TIMEOUT)} 秒仍没有读到") from error
        finally:
            if process.returncode is None:
                _kill_group(process.pid)
                _ = await process.wait()
        if process.returncode != 0:
            raise RuntimeError((err.decode(errors="replace").strip().splitlines() or ["命令失败"])[-1])
        groups = uninstalled_groups(cast(Row, json.loads(out.decode())))
        self._known = {cast(str, item["name"]) for group in groups for item in cast(list[Row], group["packages"])}
        return {"groups": groups}

    # 校验名字确实是未装的扩展包，再起后台安装；同一时间只允许一个任务，占位在第一个 await 之前。
    async def start(self, name: str) -> Row:
        if self._claimed is not None:
            raise InstallBusy(f"正在安装 {self._claimed}，请等它结束")
        self._claimed = name
        try:
            running = await run_file_io(self._running_task)
            if running is not None:
                raise InstallBusy(f"正在安装 {running['name']}，请等它结束")
            if name not in self._known:
                _ = await self.available()
            if name not in self._known:
                raise LookupError(f"{name} 不在可安装列表里，或已经安装")
            return await self._spawn(name)
        finally:
            self._claimed = None

    async def _spawn(self, name: str) -> Row:
        task_id = f"{time.time_ns():020d}"
        base = self._dir() / task_id
        await run_file_io(lambda: base.parent.mkdir(parents=True, exist_ok=True))
        wrapper = ["sh", "-c", _WRAPPER, f"{base}.exit", *pkg_command("install", name, "--timeout", str(PKG_WAIT))]
        with open(f"{base}.log", "wb") as log:
            process = await asyncio.create_subprocess_exec(
                *wrapper, stdin=asyncio.subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True)
        record: Row = {"id": task_id, "name": name, "pid": process.pid, "started_at": time.time()}
        await run_file_io(lambda: Path(f"{base}.json").write_text(json.dumps(record)))
        watcher = asyncio.create_task(self._watch(task_id, process))
        self._watchers.add(watcher)
        watcher.add_done_callback(self._watchers.discard)
        await run_file_io(self._prune)
        return await run_file_io(lambda: self._snapshot(record))

    # 只有持有 Process 的这一代在超时时杀进程组（进程未被回收，PID 不会被复用）；本代被换掉（取消）时
    # 不动正在装的进程，状态仍可由新一代从磁盘读到。
    async def _watch(self, task_id: str, process: asyncio.subprocess.Process) -> None:
        try:
            _ = await asyncio.wait_for(process.wait(), INSTALL_TIMEOUT)
        except TimeoutError:
            await run_file_io(lambda: self._expire(task_id, process.pid))
            _ = await process.wait()

    def _expire(self, task_id: str, pid: int) -> None:
        exit_file = self._dir() / f"{task_id}.exit"
        if not exit_file.exists():
            _kill_group(pid)
            _ = exit_file.write_text("timeout")

    def _prune(self) -> None:
        for stale in sorted(self._dir().glob("*.json"))[:-KEEP_TASKS]:
            for path in (stale, stale.with_suffix(".exit"), stale.with_suffix(".log")):
                path.unlink(missing_ok=True)

    def _latest(self) -> Row | None:
        files = sorted(self._dir().glob("*.json")) if self._dir().is_dir() else []
        return cast(Row, json.loads(files[-1].read_text())) if files else None

    def _running_task(self) -> Row | None:
        record = self._latest()
        return record if record is not None and self._snapshot(record)["state"] == "running" else None

    # 磁盘记录加进程存活判断：有退出码就是终态；进程不在又没退出码，是换代等原因丢了结果。
    # 超过寿命仍无退出码只报 unknown、不发信号：磁盘记录里的 PID 可能已被无关进程复用；
    # pkg 自己带 --timeout，正常会在寿命内结束。
    def _snapshot(self, record: Row) -> Row:
        base = self._dir() / cast(str, record["id"])
        code = base.with_suffix(".exit").read_text().strip() if base.with_suffix(".exit").exists() else None
        pid = cast(int, record["pid"])
        elapsed = time.time() - cast(float, record["started_at"])
        live = code is None and elapsed <= INSTALL_TIMEOUT and _alive(pid)
        state = "running" if live else "unknown" if code is None else (
            "succeeded" if code == "0" else "failed")
        tail = deque(base.with_suffix(".log").read_text(errors="replace").splitlines(), maxlen=TAIL_LINES)
        if code == "timeout":
            tail.append(f"超过 {int(INSTALL_TIMEOUT // 60)} 分钟仍未结束，已终止")
        found = _OUTCOME.findall("\n".join(tail))
        return {"name": record["name"], "state": state, "exit_code": int(code) if code and code.lstrip("-").isdigit() else None,
                "outcome": found[-1] if found else None, "tail": list(tail), "elapsed": round(elapsed, 1)}

    async def status(self) -> Row:
        def read() -> Row:
            record = self._latest() if self._dir().is_dir() else None
            return {"task": None if record is None else self._snapshot(record)}
        return await run_file_io(read)

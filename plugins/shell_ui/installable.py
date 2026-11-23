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
# 无人值守首装留给网页挑选的标记；与启动器写入的 agent.plugins.package_index.FIRST_INSTALL_PENDING 是同一路径。
FIRST_INSTALL_PENDING = "runtime/first-install-pending.json"
KEEP_TASKS = 5
_PKG_ENV = ("AKASHIC_CORE_ROOT", "AKASHIC_WORKSPACE", "AKASHIC_CORE_CONFIG")
_OUTCOME = re.compile(r"^(\S+)\s+(installed|upgraded|current|restart-required|in-progress|failed|inactive|needs-switch)\b[ \t]*(.*)$", re.M)
# pkg install 逐包写到 stderr 的进度行，见 package_ops._install_planned。
_PROGRESS = re.compile(r"^\((\d+)/(\d+)\) 正在安装 (\S+)$", re.M)
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
        return {"groups": groups, "first_install": await run_file_io(self._selecting)}

    def _pending(self) -> Path:
        return Path(os.environ["AKASHIC_WORKSPACE"]) / FIRST_INSTALL_PENDING

    # 标记的两段：还没挑选（没有 task）、挑好了正在装（记着任务 id）。标记一直留到首装任务结束，
    # Core 以它判断首装窗口（从没装过的插件可以把迁移并入新建起点）。读不懂的标记按还没挑选处理。
    def _marker(self) -> Row | None:
        try:
            value = json.loads(self._pending().read_text())
        except FileNotFoundError:
            return None
        except (OSError, ValueError):
            return {}
        return cast(Row, value) if isinstance(value, dict) else {}

    def _selecting(self) -> bool:
        marker = self._settle()
        return marker is not None and "task" not in marker

    # 首装任务装成功才删掉标记、关闭首装窗口；失败或进程丢了保留标记，网页重试仍在窗口里。返回仍然有效的标记。
    def _settle(self) -> Row | None:
        marker = self._marker()
        task = None if marker is None else marker.get("task")
        record = self._dir() / f"{task}.json"
        if task is None or not record.exists():
            return marker
        if self._snapshot(cast(Row, json.loads(record.read_text())))["state"] != "succeeded":
            return marker
        self._pending().unlink(missing_ok=True)
        return None

    # 校验名字确实是未装的扩展包，再起一个后台任务按依赖顺序装完；同一时间只允许一个任务，占位在第一个 await 之前。
    # 首装窗口还开着（还没挑选，或上次首装没装全）时，这次提交就是首装的一部分：标记改记任务 id，装成功时删掉。
    # 窗口只看服务端标记，不采信调用方；任务记录的 first_install 供引导跟踪，窗口内或调用方声明是重试时为真。
    async def start(self, names: list[str], first_install: bool = False) -> Row:
        if not names or len(set(names)) != len(names):
            raise LookupError("要安装的包不能为空或重复")
        if self._claimed is not None:
            raise InstallBusy(f"正在安装 {self._claimed}，请等它结束")
        self._claimed = names[0] if len(names) == 1 else f"{names[0]} 等 {len(names)} 个包"
        try:
            running = await run_file_io(self._running_task)
            if running is not None:
                raise InstallBusy(f"正在安装 {_title(running)}，请等它结束")
            if not set(names) <= self._known:
                _ = await self.available()
            if unknown := [name for name in names if name not in self._known]:
                raise LookupError(f"{'、'.join(unknown)} 不在可安装列表里，或已经安装")
            pending = await run_file_io(self._settle) is not None
            started = await self._spawn(names, pending or first_install)
            if pending:
                marker = json.dumps({"schema_version": 1, "task": started["id"]})
                await run_file_io(lambda: self._pending().write_text(marker + "\n"))
            return started
        finally:
            self._claimed = None

    # 首装不装扩展包（只用核心）：删掉待挑选标记，引导不再出现这一步；以后仍可在插件页安装。
    async def decline(self) -> None:
        await run_file_io(lambda: self._pending().unlink(missing_ok=True))

    async def _spawn(self, names: list[str], first: bool) -> Row:
        task_id = f"{time.time_ns():020d}"
        base = self._dir() / task_id
        await run_file_io(lambda: base.parent.mkdir(parents=True, exist_ok=True))
        wrapper = ["sh", "-c", _WRAPPER, f"{base}.exit", *pkg_command("install", *names, "--timeout", str(PKG_WAIT))]
        with open(f"{base}.log", "wb") as log:
            process = await asyncio.create_subprocess_exec(
                *wrapper, stdin=asyncio.subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True)
        # 寿命按包数放宽：pkg 逐包在线等待终态，一次装全套时总时长远超单包。
        record: Row = {"id": task_id, "names": names, "pid": process.pid, "started_at": time.time(),
                       "lifetime": INSTALL_TIMEOUT * len(names), "first_install": first}
        await run_file_io(lambda: Path(f"{base}.json").write_text(json.dumps(record)))
        watcher = asyncio.create_task(self._watch(task_id, process, cast(float, record["lifetime"])))
        self._watchers.add(watcher)
        watcher.add_done_callback(self._watchers.discard)
        await run_file_io(self._prune)
        return await run_file_io(lambda: self._snapshot(record))

    # 只有持有 Process 的这一代在超时时杀进程组（进程未被回收，PID 不会被复用）；本代被换掉（取消）时
    # 不动正在装的进程，状态仍可由新一代从磁盘读到。
    async def _watch(self, task_id: str, process: asyncio.subprocess.Process, lifetime: float) -> None:
        try:
            _ = await asyncio.wait_for(process.wait(), lifetime)
        except TimeoutError:
            await run_file_io(lambda: self._expire(task_id, process.pid))
            _ = await process.wait()
        _ = await run_file_io(self._settle)

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
    # outcomes 是 pkg 末尾结果表的逐包结果，progress 是最近一行逐包进度；两者都从完整日志里取。
    def _snapshot(self, record: Row) -> Row:
        base = self._dir() / cast(str, record["id"])
        code = base.with_suffix(".exit").read_text().strip() if base.with_suffix(".exit").exists() else None
        pid = cast(int, record["pid"])
        elapsed = time.time() - cast(float, record["started_at"])
        lifetime = cast(float, record.get("lifetime", INSTALL_TIMEOUT))
        live = code is None and elapsed <= lifetime and _alive(pid)
        state = "running" if live else "unknown" if code is None else (
            "succeeded" if code == "0" else "failed")
        log = base.with_suffix(".log").read_text(errors="replace")
        tail = deque(log.splitlines(), maxlen=TAIL_LINES)
        if code == "timeout":
            tail.append(f"超过 {int(lifetime // 60)} 分钟仍未结束，已终止")
        steps = _PROGRESS.findall(log)
        return {"id": record["id"], "name": _title(record), "names": _names(record), "state": state, "first_install": bool(record.get("first_install")),
                "exit_code": int(code) if code and code.lstrip("-").isdigit() else None,
                "outcomes": [{"name": name, "result": result, "detail": detail.strip()} for name, result, detail in _OUTCOME.findall(log)],
                "progress": {"done": int(steps[-1][0]) - 1, "total": int(steps[-1][1]), "current": steps[-1][2]} if steps else None,
                "tail": list(tail), "elapsed": round(elapsed, 1)}

    async def status(self) -> Row:
        def read() -> Row:
            _ = self._settle()
            record = self._latest() if self._dir().is_dir() else None
            return {"task": None if record is None else self._snapshot(record)}
        return await run_file_io(read)

    # 新手引导"选择功能"一步的状态：还没挑选、或首装挑的包还在装，都算未完成；进度给一句人话。
    async def onboarding_status(self) -> Row:
        def read() -> Row:
            pending = self._selecting()
            record = self._latest() if self._dir().is_dir() else None
            # 只跟踪首装挑选及其重试的任务：普通安装不能让已有实例被当成首装。
            task = self._snapshot(record) if record is not None and record.get("first_install") else None
            running = task is not None and task["state"] == "running"
            progress = cast(Row | None, task["progress"]) if task is not None else None
            note = f"正在安装（{cast(int, progress['done']) + 1}/{progress['total']}）{progress['current']}" if running and progress else ""
            return {"enabled": True, "ready": not pending and not running, "blocked": False, "running": running,
                    "reason": "挑选要安装的功能" if pending else note}
        return await run_file_io(read)


# 任务记录里的包名：#2219 的单包记录只有 name。
def _names(record: Row) -> list[str]:
    return cast(list[str], record.get("names") or [record["name"]])


def _title(record: Row) -> str:
    names = _names(record)
    return names[0] if len(names) == 1 else f"{names[0]} 等 {len(names)} 个包"

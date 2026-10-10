"""宿主授予当前 Context 的执行原子能力，不包含资源目录或启动阶段。"""
from __future__ import annotations

import asyncio
import os
import sys
import subprocess
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Literal, Protocol

from agent.plugin_composition.context import Context
from agent.plugin_composition.model import ServiceKey



class ChildProcess(Protocol):
    """opaque 子进程句柄：stdio 归调用方，进程组终止语义归宿主实现。"""

    process: asyncio.subprocess.Process

    @property
    def group_id(self) -> int | None: ...
    async def terminate(self, *, timeout_s: float) -> None: ...
    async def kill(self, *, timeout_s: float) -> None: ...


class PreparedProcess:
    """宿主签发的冻结执行制品：command/cwd/env 已经过授权校验。

    归属校验靠签发者私有 token，而非可被随意拼造的公开字段；
    制品签发后不可修改，不存在公开的派生/修改入口。
    """

    __slots__ = ("_token", "_command", "_cwd", "_env")

    def __init__(
        self,
        token: object,
        *,
        command: tuple[str, ...],
        cwd: str,
        env: Mapping[str, str],
    ) -> None:
        self._token = token
        self._command = tuple(command)
        self._cwd = cwd
        self._env = MappingProxyType(dict(env))

    @property
    def command(self) -> tuple[str, ...]:
        return self._command

    @property
    def cwd(self) -> str:
        return self._cwd

    @property
    def env(self) -> Mapping[str, str]:
        return self._env

    def _issued_by(self, token: object) -> bool:
        return self._token is token


class ProcessSpawner(Protocol):
    """受控子进程来源；ExecutionGrant 结构满足，不另立 ServiceKey。"""

    def prepare_process(
        self,
        command: tuple[str, ...],
        cwd: str,
        env: Mapping[str, str],
        candidate_env: Mapping[str, str] = {},
    ) -> PreparedProcess:
        """在边界内执行授权校验并签发冻结的执行制品。"""
        ...
    async def spawn(
        self,
        prepared: PreparedProcess,
        *,
        stdin: object = None,
        stdout: object = None,
        stderr: object = None,
        limit: int | None = None,
    ) -> tuple[ChildProcess, bool]: ...


class ExecutionGrant(ProcessSpawner, Protocol):
    @property
    def mode(self) -> Literal["candidate", "formal"]: ...
    def command(self, command: tuple[str, ...], cwd: str) -> tuple[str, ...]: ...
    def cwd(self, relative: str) -> Path: ...
    def environment(self, values: Mapping[str, str], candidate_values: Mapping[str, str]) -> dict[str, str]: ...
    async def spawn(
        self,
        prepared: PreparedProcess,
        *,
        stdin: object = None,
        stdout: object = None,
        stderr: object = None,
        limit: int | None = None,
    ) -> tuple[ChildProcess, bool]:
        """受控 spawn：只消费本授权签发的 PreparedProcess，返回取消标记。"""
        ...


class ExecutionAccess(Protocol):
    def bind(self, ctx: Context) -> ExecutionGrant: ...




EXECUTION = ServiceKey[ExecutionAccess]("host.execution.v1")


def process_group_exists(group_id: int) -> bool:
    """只在进程组仍有活成员时返回 True。"""

    if sys.platform.startswith("linux"):
        return _linux_group_has_live_members(group_id)
    if sys.platform == "darwin":
        return _darwin_group_has_live_members(group_id)
    try:
        os.killpg(group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _darwin_group_has_live_members(group_id: int) -> bool:
    """Darwin 的 killpg(0) 也会命中 zombie，需检查 stat。"""

    try:
        result = subprocess.run(
            ["ps", "-axo", "pgid=,stat="],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    for line in result.stdout.splitlines():
        fields = line.split()
        if (
            len(fields) >= 2
            and fields[0].isdigit()
            and int(fields[0]) == group_id
            and not fields[1].startswith("Z")
        ):
            return True
    return False


def _linux_group_has_live_members(group_id: int) -> bool:
    """将只剩 zombie 的组视为已释放，避免容器 PID 1 延迟回收误报。"""
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / "stat").read_text()
        except OSError:
            continue
        command_end = stat.rfind(")")
        if command_end < 0:
            continue
        fields = stat[command_end + 2 :].split()
        if len(fields) < 3 or int(fields[2]) != group_id:
            continue
        if fields[0] != "Z":
            return True
    return False

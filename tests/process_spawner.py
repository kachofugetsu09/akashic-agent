"""测试侧裸启动替身：不经 Context 授权的同一回收实现，不属于生产边界。"""
from __future__ import annotations

import asyncio
from collections.abc import Mapping

from agent.host_bridge.plugin_execution import _spawn_child
from agent.plugin_composition.execution import ChildProcess, PreparedProcess
from plugins.mcp.client import _infer_cwd


class LocalProcessSpawner:
    """与 ExecutionGrant 结构相同的测试替身；prepare 不做授权校验。"""

    def __init__(self) -> None:
        self._issue_token = object()

    def prepare_process(
        self,
        command: tuple[str, ...],
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        candidate_env: Mapping[str, str] = {},
    ) -> PreparedProcess:
        return PreparedProcess(
            self._issue_token,
            command=tuple(command),
            cwd=cwd or _infer_cwd(list(command)) or ".",
            env=dict(env or {}),
        )

    async def spawn(
        self,
        prepared: PreparedProcess,
        *,
        stdin: object = None,
        stdout: object = None,
        stderr: object = None,
        limit: int | None = None,
    ) -> tuple[ChildProcess, bool]:
        if type(prepared) is not PreparedProcess or not prepared._issued_by(self._issue_token):
            raise PermissionError("spawn 只接受本授权签发的 PreparedProcess")
        return await _spawn_child(
            prepared.command, cwd=prepared.cwd, env=prepared.env,
            stdin=stdin, stdout=stdout, stderr=stderr, limit=limit,
        )

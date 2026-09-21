"""测试侧裸启动替身：不经 Context 授权的同一回收实现，不属于生产边界。"""
from __future__ import annotations

import asyncio
from collections.abc import Collection, Mapping

from agent.host_bridge.plugin_execution import HostedChildProcess, _spawn_child
from agent.plugin_composition.execution import ChildProcess, PreparedProcess
from plugins.mcp.client import _infer_cwd


class LocalProcessSpawner:
    """与 ExecutionGrant 结构相同的测试替身；prepare 不做授权校验。"""

    def __init__(self) -> None:
        self._issue_token = object()
        self._children: dict[int, HostedChildProcess] = {}

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
        env_scrub_keys: Collection[str] = frozenset(),
        stdin: object = None,
        stdout: object = None,
        stderr: object = None,
        limit: int | None = None,
    ) -> tuple[ChildProcess, bool]:
        if type(prepared) is not PreparedProcess or not prepared._issued_by(self._issue_token):
            raise PermissionError("spawn 只接受本授权签发的 PreparedProcess")
        child, cancelled = await _spawn_child(
            prepared.command, cwd=prepared.cwd, env=prepared.env,
            env_scrub_keys=env_scrub_keys,
            stdin=stdin, stdout=stdout, stderr=stderr, limit=limit,
        )
        pid = child.process.pid
        if isinstance(pid, int):
            self._children[pid] = child
        return child, cancelled

    def adopt(self, process: asyncio.subprocess.Process) -> ChildProcess:
        pid = getattr(process, "pid", None)
        child = self._children.get(pid) if isinstance(pid, int) else None
        if child is None or child.process is not process:
            raise PermissionError("adopt 只接受本授权已登记的子进程")
        return child

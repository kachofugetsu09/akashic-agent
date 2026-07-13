"""PeerAgentPoller：后台 asyncio 任务，轮询所有 pending A2A 任务。"""
from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

import httpx

from bus.events import InboundMessage
from bus.queue import MessageBus
from core.net.http import HttpRequester, RequestBudget

if TYPE_CHECKING:
    from agent.peer_agent.process_manager import PeerProcessManager

logger = logging.getLogger(__name__)

_POLL_INTERVAL_S = 10
_TASK_TIMEOUT_S = 3600      # 60 分钟硬超时（DeepResearch 复杂任务可能需要 30min+）
_IN_PROGRESS_STATES = frozenset({"queued", "running", "submitted", "working"})
_SUCCESS_STATES = frozenset({"completed"})
_FAILURE_STATES = frozenset({"failed", "canceled"})


class PeerTaskProtocolError(ValueError):
    """远端 A2A 响应不符合本地消费协议。"""


class PeerTaskDuplicateError(ValueError):
    """本地 pending catalog 已经登记同一 agent 的 task id。"""


class _PeerTaskRetryError(RuntimeError):
    """本轮外部查询暂时失败，保留 pending 供下一轮重试。"""


@dataclass
class _PendingTask:
    task_id: str
    agent_name: str
    agent_url: str
    channel: str
    chat_id: str
    goal: str
    submitted_at: float = field(default_factory=time.monotonic)
    terminal_state: str | None = None
    artifacts: dict[str, str] = field(default_factory=dict)
    failure_reason: str = ""
    notification_sent: bool = False
    termination_done: bool = False


PendingKey = tuple[str, str]


def _object(value: Any, path: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise PeerTaskProtocolError(
            f"tasks/get {path} 必须是对象，实际为 {type(value).__name__}"
        )
    return cast(dict[str, object], value)


def _required_text(value: dict[str, object], key: str, path: str) -> str:
    raw = value.get(key)
    if not isinstance(raw, str) or not raw.strip():
        raise PeerTaskProtocolError(f"tasks/get {path}.{key} 必须是非空字符串")
    return raw


def _parse_part_text(value: Any, path: str) -> str | None:
    """校验一个 A2A part，并返回其中的文本内容。"""

    # 1. 校验 part 根节点及两种实际序列化形式
    part = _object(value, path)
    if not part:
        raise PeerTaskProtocolError(f"tasks/get {path} 不能是空对象")
    if "text" in part:
        return _required_text(part, "text", path)
    if "root" in part:
        root = _object(part["root"], f"{path}.root")
        if not root:
            raise PeerTaskProtocolError(f"tasks/get {path}.root 不能是空对象")
        if "text" in root:
            return _required_text(root, "text", f"{path}.root")
        if "kind" in root:
            _required_text(root, "kind", f"{path}.root")
        return None
    if "kind" in part:
        kind = _required_text(part, "kind", path)
        if kind == "text":
            raise PeerTaskProtocolError(f"tasks/get {path}.text 缺失")
        return None
    raise PeerTaskProtocolError(f"tasks/get {path} 缺少 text、root 或 kind")


def _parse_message(value: Any, path: str) -> str:
    """校验状态消息，并拼接文本 part 供失败诊断。"""

    # 1. parts 是 A2A message 的真实消费字段，缺失或类型错误都必须暴露
    message = _object(value, path)
    raw_parts = message.get("parts")
    if not isinstance(raw_parts, list):
        raise PeerTaskProtocolError(f"tasks/get {path}.parts 必须是数组")

    # 2. 只提取文本 part，非文本 part 仍先经过结构校验
    texts = [
        text
        for index, part in enumerate(raw_parts)
        if (text := _parse_part_text(part, f"{path}.parts[{index}]")) is not None
    ]
    return " | ".join(texts)


def _parse_artifacts(value: dict[str, object]) -> dict[str, str]:
    """校验 artifacts，并提取每个 artifact 的首个文本 part。"""

    # 1. artifacts 可省略，但出现时必须是数组且每项具备名称和 parts
    raw_artifacts = value.get("artifacts", [])
    if not isinstance(raw_artifacts, list):
        raise PeerTaskProtocolError("tasks/get result.artifacts 必须是数组")
    artifacts: dict[str, str] = {}
    seen_names: set[str] = set()
    for index, raw_artifact in enumerate(raw_artifacts):
        path = f"result.artifacts[{index}]"
        artifact = _object(raw_artifact, path)
        name = _required_text(artifact, "name", path)
        raw_parts = artifact.get("parts")
        if not isinstance(raw_parts, list):
            raise PeerTaskProtocolError(f"tasks/get {path}.parts 必须是数组")
        if name in seen_names:
            raise PeerTaskProtocolError(f"tasks/get {path}.name 重复：{name!r}")
        seen_names.add(name)
        text_found = False
        for part_index, part in enumerate(raw_parts):
            text = _parse_part_text(part, f"{path}.parts[{part_index}]")
            if text is not None:
                artifacts[name] = text
                text_found = True
                break
        if not text_found:
            raise PeerTaskProtocolError(f"tasks/get {path}.parts 必须包含文本 part")
    return artifacts


class PeerAgentPoller:
    """后台轮询所有 pending A2A 任务，完成后注入 MessageBus 触发新一轮 AgentLoop。"""

    def __init__(
        self,
        bus: MessageBus,
        process_manager: PeerProcessManager,
        requester: HttpRequester,
    ) -> None:
        self._bus = bus
        self._pm = process_manager
        self._requester = requester
        self._pending: dict[PendingKey, _PendingTask] = {}
        self._agent_locks: dict[str, asyncio.Lock] = {}
        self._task: asyncio.Task[None] | None = None

    @asynccontextmanager
    async def submission_lease(self, agent_name: str) -> AsyncIterator[None]:
        """串行化同一 agent 的提交、登记和最后任务终结。"""

        lock = self._agent_locks.setdefault(agent_name, asyncio.Lock())
        async with lock:
            yield

    def has_pending(self, agent_name: str) -> bool:
        """返回 agent 是否仍有 pending 或正在重试终态副作用的任务。"""

        return any(meta.agent_name == agent_name for meta in self._pending.values())

    def register(
        self,
        *,
        task_id: str,
        agent_name: str,
        agent_url: str,
        channel: str,
        chat_id: str,
        goal: str,
    ) -> None:
        """登记唯一的 A2A task，保留终态投递和进程回收的 ownership。"""

        # 1. task id 只在单个 agent 内唯一，跨 agent 必须隔离命名空间
        key = (agent_name, task_id)
        if key in self._pending:
            raise PeerTaskDuplicateError(
                f"重复注册 A2A task：{agent_name!r}/{task_id!r}"
            )
        self._pending[key] = _PendingTask(
            task_id=task_id,
            agent_name=agent_name,
            agent_url=agent_url,
            channel=channel,
            chat_id=chat_id,
            goal=goal,
        )
        logger.info("[Poller] 注册任务 task_id=%s agent=%s", task_id, agent_name)

    def start(self) -> None:
        """启动唯一后台轮询任务。"""

        if self._task is not None:
            if not self._task.done():
                return
            task = self._task
            self._task = None
            task.result()
        self._task = asyncio.create_task(self._loop(), name="peer_agent_poller")
        logger.info("[Poller] 后台轮询已启动")

    async def stop(self) -> None:
        """停止后台轮询并等待取消完成。"""

        task = self._task
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            finally:
                if self._task is task:
                    self._task = None
        logger.info("[Poller] 已停止")

    # ── 内部 ──────────────────────────────────────────────────

    async def _loop(self) -> None:
        while True:
            await asyncio.sleep(_POLL_INTERVAL_S)
            for _, meta in list(self._pending.items()):
                try:
                    await self._check(meta.task_id, meta)
                except _PeerTaskRetryError as exc:
                    logger.warning("[Poller] 检查任务 %s/%s 出错: %s", meta.agent_name, meta.task_id, exc)

    async def _check(self, task_id: str, meta: _PendingTask) -> None:
        """读取任务终态并可靠投递通知、回收 agent，成功后删除 pending。"""

        key = (meta.agent_name, task_id)
        if self._pending.get(key) is not meta:
            return

        # 1. 终态已经确定时只重试尚未成功的副作用，避免重复查询和通知
        if meta.terminal_state is None:
            if time.monotonic() - meta.submitted_at > _TASK_TIMEOUT_S:
                logger.warning("[Poller] 任务 %s 超时（60分钟）", task_id)
                meta.terminal_state = "failed"
                meta.failure_reason = "调研超时（超过60分钟）"
            else:
                try:
                    state, artifacts, status_text = await self._get_task_status(
                        meta.agent_url, task_id
                    )
                except PeerTaskProtocolError as exc:
                    logger.exception(
                        "[Poller] 任务 %s/%s 响应协议错误: %s",
                        meta.agent_name,
                        task_id,
                        exc,
                    )
                    meta.terminal_state = "failed"
                    meta.failure_reason = f"后台任务响应协议错误：{exc}"
                    await self._finalize(task_id, meta)
                    return
                if state in _SUCCESS_STATES:
                    logger.info(
                        "[Poller] 任务 %s 完成，artifacts: %s",
                        task_id,
                        list(artifacts.keys()),
                    )
                    meta.terminal_state = state
                    meta.artifacts = artifacts
                elif state in _FAILURE_STATES:
                    logger.warning(
                        "[Poller] 任务 %s 失败 原因: %s",
                        task_id,
                        status_text or "(无消息)",
                    )
                    meta.terminal_state = state
                    meta.failure_reason = (
                        f"调研任务执行{state}：{status_text}"
                        if status_text
                        else f"调研任务执行{state}"
                    )

        # 2. 先保证通知成功，再回收进程；任一步失败都保留 pending 供下轮重试
        if meta.terminal_state is not None:
            await self._finalize(task_id, meta)

    async def _finalize(self, task_id: str, meta: _PendingTask) -> None:
        # 1. 通知不占用提交 lease；成功后 catalog 仍保留 task ownership
        if not meta.notification_sent:
            if meta.terminal_state == "completed":
                await self._inject_completion(meta, meta.artifacts)
            else:
                await self._inject_failure(meta, meta.failure_reason)
            meta.notification_sent = True

        async with self.submission_lease(meta.agent_name):
            # 2. 只有当前任务是该 agent 的最后 pending 才释放受管进程
            if not meta.termination_done:
                has_other_pending = any(
                    other is not meta and other.agent_name == meta.agent_name
                    for other in self._pending.values()
                )
                if not has_other_pending:
                    await self._pm.terminate(meta.agent_name)
                meta.termination_done = True

            # 3. 所有副作用成功后才从 catalog 删除任务
            key = (meta.agent_name, meta.task_id)
            if self._pending.get(key) is meta:
                del self._pending[key]

    async def _get_task_status(
        self, agent_url: str, task_id: str
    ) -> tuple[str, dict[str, str], str]:
        """向 A2A 服务器查询任务状态并严格解析 result、message 和 artifacts。"""

        payload = {
            "jsonrpc": "2.0",
            "id": "poll-1",
            "method": "tasks/get",
            "params": {"id": task_id},
        }

        # 1. 发送查询并确认 HTTP 成功
        try:
            response = await self._requester.post(
                agent_url,
                json=payload,
                budget=RequestBudget(total_timeout_s=8.0),
            )
            response.raise_for_status()
        except httpx.UnsupportedProtocol:
            raise
        except (httpx.TimeoutException, httpx.TransportError, httpx.HTTPStatusError) as exc:
            raise _PeerTaskRetryError(
                f"tasks/get 网络或 HTTP 查询失败：{exc}"
            ) from exc
        try:
            raw_data = response.json()
        except ValueError as exc:
            raise PeerTaskProtocolError("tasks/get 返回了非法 JSON") from exc
        data = _object(raw_data, "root")

        # 2. JSON-RPC error 与 result 互斥，错误对象也必须符合真实字段类型
        if "error" in data:
            if "result" in data:
                raise PeerTaskProtocolError(
                    "tasks/get root 不能同时包含 error 和 result"
                )
            error = _object(data["error"], "root.error")
            code = error.get("code")
            if isinstance(code, bool) or not isinstance(code, int):
                raise PeerTaskProtocolError("tasks/get root.error.code 必须是整数")
            message = _required_text(error, "message", "root.error")
            raise PeerTaskProtocolError(f"tasks/get 错误 code={code}: {message}")

        # 3. 校验 result/status/state/message/artifacts，再提取消费字段
        result = _object(data.get("result"), "root.result")
        status = _object(result.get("status"), "result.status")
        state = _required_text(status, "state", "result.status")
        status_text = ""
        if "message" in status:
            status_text = _parse_message(status["message"], "result.status.message")
        if state not in _IN_PROGRESS_STATES | _SUCCESS_STATES | _FAILURE_STATES:
            raise PeerTaskProtocolError(f"tasks/get 返回未知 state：{state!r}")
        artifacts = _parse_artifacts(result)
        if status_text:
            logger.debug("[Poller] 任务 %s 状态=%s 消息: %s", task_id, state, status_text)
        return state, artifacts, status_text

    async def _inject_completion(self, meta: _PendingTask, artifacts: dict[str, str]) -> None:
        """向 MessageBus 注入系统消息，触发 AgentLoop 新一轮处理结果并回复用户。"""

        artifact_lines = "\n".join(
            f"  - {name}: {path}" for name, path in artifacts.items()
        ) or "  （无产出文件）"
        text = (
            f"[系统通知] 后台任务已完成。\n"
            f"执行的任务：{meta.goal}\n"
            f"执行者：{meta.agent_name}\n"
            f"产出文件：\n{artifact_lines}\n\n"
            f"请根据产出内容向用户汇报结果。"
        )
        await self._bus.publish_inbound(
            InboundMessage(
                channel=meta.channel,
                sender="system",
                chat_id=meta.chat_id,
                content=text,
                metadata={"system_injected": True, "task_id": meta.task_id},
            )
        )

    async def _inject_failure(self, meta: _PendingTask, reason: str) -> None:
        text = (
            f"[系统通知] 后台任务未能完成：{reason}。\n"
            f"执行的任务：{meta.goal}\n"
            f"执行者：{meta.agent_name}\n"
            f"请告知用户，并建议他们稍后重试。"
        )
        await self._bus.publish_inbound(
            InboundMessage(
                channel=meta.channel,
                sender="system",
                chat_id=meta.chat_id,
                content=text,
                metadata={"system_injected": True, "task_id": meta.task_id},
            )
        )

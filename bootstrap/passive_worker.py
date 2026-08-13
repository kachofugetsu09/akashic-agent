from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, cast

from agent.control.errors import (
    ControlAdmissionError,
    RuntimeClosedError,
    ThreadBusyError,
)
from agent.control.models import (
    TurnItemKind,
    TurnRecord,
    TurnRequest,
    TurnResult,
    TurnStatus,
)
from agent.control.runtime import ConversationRuntime, TurnHandle
from agent.looping.core import AgentLoop
from bus.events import InboundMessage, OutboundMessage, TurnTerminalStatus
from bus.queue import MessageBus
from core.common.diagnostic_log import turn_milestone

logger = logging.getLogger(__name__)

_TERMINAL_DELIVERY_RETRY_DELAYS = (0.05, 0.1)
_TERMINAL_LANE_RETRY_DELAY = 1.0


class _TerminalHandoffRetainedError(RuntimeError):
    """终态尚未 durable delivery，handoff 仍由当前 lane 持有。"""


class PassiveMessageWorker:
    """把渠道入站消息转换为 ConversationRuntime turn。"""

    def __init__(
        self, bus: MessageBus, runtime: ConversationRuntime, legacy_loop: AgentLoop
    ) -> None:
        self._bus = bus
        self._runtime = runtime
        self._legacy_loop = legacy_loop
        self._running = False
        self._lane_queues: dict[str, asyncio.Queue[InboundMessage | object]] = {}
        self._lane_tasks: dict[str, asyncio.Task[None]] = {}
        self._result_tasks: set[asyncio.Task[None]] = set()

    async def run(self) -> None:
        self._running = True
        try:
            # 1. 仅重放有界 durable handoff 页，lane 准入由 MessageBus 统一持有。
            await self._bus.recover_durable_inbounds()
            while self._running:
                try:
                    item = await asyncio.wait_for(
                        self._bus.consume_inbound(), timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue
                self._enqueue(item)
        finally:
            self._running = False
            for task in tuple(self._lane_tasks.values()):
                task.cancel()
            if self._lane_tasks:
                await asyncio.gather(
                    *tuple(self._lane_tasks.values()),
                    return_exceptions=True,
                )
            self._lane_tasks.clear()
            self._lane_queues.clear()
            for task in tuple(self._result_tasks):
                task.cancel()
            if self._result_tasks:
                await asyncio.gather(*tuple(self._result_tasks), return_exceptions=True)
            self._result_tasks.clear()

    def _enqueue(self, item: object) -> None:
        key = cast(Any, item).session_key
        queue = self._lane_queues.setdefault(key, asyncio.Queue())
        queue.put_nowait(item)
        task = self._lane_tasks.get(key)
        if task is None or task.done():
            self._lane_tasks[key] = asyncio.create_task(
                self._run_lane(key, queue),
                name=f"passive-lane:{key}",
            )

    async def _run_lane(
        self,
        key: str,
        queue: asyncio.Queue[InboundMessage | object],
    ) -> None:
        """串行执行单 thread 队列，并隔离单条消息失败。"""

        while True:
            item = await queue.get()
            while True:
                try:
                    if isinstance(item, InboundMessage):
                        result_task = await self._admit_message(item)
                        if result_task is not None:
                            await result_task
                    else:
                        await self._legacy_loop._run_inbound_turn(cast(Any, item))
                    break
                except asyncio.CancelledError:
                    raise
                except _TerminalHandoffRetainedError as error:
                    # 终态已持久化，只重投同一权威 terminal；同 session 后续消息
                    # 保持排队，绝不重跑 Provider，也不把 accepted owner 变成孤儿。
                    logger.error(
                        "passive terminal retained; retrying thread=%s delay=%.1fs error=%s",
                        key,
                        _TERMINAL_LANE_RETRY_DELAY,
                        error,
                    )
                    await asyncio.sleep(_TERMINAL_LANE_RETRY_DELAY)
                except Exception:
                    logger.exception("passive lane message failed thread=%s", key)
                    break
            if queue.empty():
                task = asyncio.current_task()
                if self._lane_tasks.get(key) is task:
                    self._lane_tasks.pop(key)
                    self._lane_queues.pop(key)
                return

    async def _run_message(self, item: InboundMessage) -> None:
        """兼容直接调用：等待由本条消息新建的 turn 完成。"""

        result_task = await self._admit_message(item)
        if result_task is not None:
            await result_task

    async def _admit_message(
        self,
        item: InboundMessage,
    ) -> asyncio.Task[None] | None:
        """快速准入渠道消息，并把唯一终态发送职责交给新 turn owner。"""

        # 阶段1：mobile durable handoff 以权威 turn 为 owner，恢复绝不直接删 row。
        if item.channel == "mobile" and item.handoff_id is not None:
            matched = self._matched_mobile_turn(item)
            if matched is not None:
                if matched.status.is_terminal:
                    await self._redeliver_terminal(item, matched)
                    return None
                raise RuntimeError(
                    f"mobile turn 非终态未恢复: {matched.id}/{matched.status.value}"
                )
        # 阶段2：mobile 消息需要 session admission；channel 已给出则本轮复用。
        if item.channel == "mobile" and item.session_admission_id is None:
            _, item.session_admission_id = (
                self._legacy_loop.session_manager.admit_existing(item.session_key)
            )
        transferred = False
        try:
            # 阶段3：渠道信息只作为 executor 所需的受控 metadata，不改变 thread identity。
            request = TurnRequest(
                item.session_key,
                item.content,
                {
                    "channel": item.channel,
                    "chatId": item.chat_id,
                    "sender": item.sender,
                    "media": list(item.media),
                    "inputTimestamp": item.timestamp.isoformat(),
                    "inboundMetadata": dict(item.metadata),
                },
            )
            while True:
                try:
                    handle = await self._runtime.start_turn(request)
                    break
                except ThreadBusyError:
                    await self._runtime.wait_thread_available(item.session_key)
                    continue
                except ControlAdmissionError:
                    if self._runtime.admission_request_never_fits(request):
                        handle = await self._runtime.reject_never_fit_turn(request)
                        break
                    self._release_admission_once(item)
                    await self._runtime.wait_capacity_available(request)
                    if item.channel == "mobile" and item.session_admission_id is None:
                        _, item.session_admission_id = (
                            self._legacy_loop.session_manager.admit_existing(
                                item.session_key
                            )
                        )
                    continue
                except RuntimeClosedError:
                    # Restart 取消会在同一进程恢复 admission：当前 coroutine 继续
                    # 持有 accepted owner，等待栅栏后原地重试。完整 shutdown 会
                    # 取消 worker；durable row 留给下一进程恢复。
                    self._release_admission_once(item)
                    await self._runtime.wait_until_accepting_turns()
                    if item.channel == "mobile" and item.session_admission_id is None:
                        _, item.session_admission_id = (
                            self._legacy_loop.session_manager.admit_existing(
                                item.session_key
                            )
                        )
                    continue
            task = asyncio.create_task(
                self._finish_message(item, handle),
                name=f"passive-result:{handle.id}",
            )
            self._result_tasks.add(task)
            task.add_done_callback(self._result_tasks.discard)
            transferred = True
            return task
        finally:
            # 阶段4：turn owner 建立前只释放 admission，durable handoff 保留供恢复。
            if not transferred:
                self._release_admission_once(item)

    async def _finish_message(self, item: InboundMessage, handle: TurnHandle) -> None:
        """等待新 turn 的唯一终态；mobile 只在 durable delivered 后完成 inbound。

        finally 兜底本轮 session admission：取消、receipt False 与异常路径都
        只释放一次；成功路径 _complete_message 已清空 identity，finally 不重复。
        """

        try:
            result = await handle.result()
            outbound = self._terminal_outbound(item, result)
            if item.channel == "mobile" and item.handoff_id is not None:
                # 阶段1：terminal 实际送达收据是 handoff 删除的唯一授权。
                await self._commit_mobile_terminal(
                    item,
                    outbound,
                    turn_id=result.id,
                    client_message_id=self._verified_client_message_id(result),
                    terminal_status=result.status.value,
                    mode="live",
                )
            else:
                # 阶段2：非 mobile 保持 fire-and-forget 发布后收束。
                await self._bus.publish_outbound(outbound)
                await self._complete_message(item)
        finally:
            self._release_admission_once(item)

    async def _commit_mobile_terminal(
        self,
        item: InboundMessage,
        outbound: OutboundMessage,
        *,
        turn_id: str,
        client_message_id: str,
        terminal_status: str,
        mode: str,
    ) -> None:
        """送达 Mobile 终态并完成 handoff，以唯一闭合 span 报告结果。"""

        started = time.monotonic()
        self._observe_terminal_milestone(
            "tl:worker.terminal.start",
            session_id=item.session_key,
            turn_id=turn_id,
            client_message_id=client_message_id,
            outcome=terminal_status,
            counts=f"mode={mode}",
        )
        try:
            # 1. 终态必须先提交到 Mobile durable inbox；失败时 handoff 保留。
            if not await self._deliver_terminal(outbound):
                raise _TerminalHandoffRetainedError(
                    f"mobile terminal delivery failed turn={turn_id} handoff retained"
                )
            # 2. 只有 durable inbox 与 handoff DELETE 都成功，span 才能记 done。
            await self._complete_message(item)
        except asyncio.CancelledError:
            self._observe_terminal_milestone(
                "tl:worker.terminal.cancelled",
                session_id=item.session_key,
                turn_id=turn_id,
                client_message_id=client_message_id,
                duration_ms=(time.monotonic() - started) * 1000,
                outcome="cancelled",
                counts=f"mode={mode}",
                level=logging.WARNING,
            )
            raise
        except Exception as error:
            self._observe_terminal_milestone(
                "tl:worker.terminal.error",
                session_id=item.session_key,
                turn_id=turn_id,
                client_message_id=client_message_id,
                duration_ms=(time.monotonic() - started) * 1000,
                outcome="error",
                counts=f"mode={mode} error_type={type(error).__name__}",
                level=logging.ERROR,
            )
            raise
        self._observe_terminal_milestone(
            "tl:worker.terminal.done",
            session_id=item.session_key,
            turn_id=turn_id,
            client_message_id=client_message_id,
            duration_ms=(time.monotonic() - started) * 1000,
            outcome="delivered",
            counts=f"mode={mode}",
        )

    async def _deliver_terminal(self, outbound: OutboundMessage) -> bool:
        """以有界 worker 级退避重投同一权威 terminal。"""

        # 1. 每次调用仍由 MessageBus 拥有渠道内部重试；这里只重投同一 terminal。
        for attempt in range(len(_TERMINAL_DELIVERY_RETRY_DELAYS) + 1):
            if await self._bus.publish_outbound_awaited(outbound):
                return True
            if attempt < len(_TERMINAL_DELIVERY_RETRY_DELAYS):
                await asyncio.sleep(_TERMINAL_DELIVERY_RETRY_DELAYS[attempt])
        return False

    def _observe_terminal_milestone(
        self,
        event: str,
        *,
        session_id: str,
        turn_id: str,
        client_message_id: str,
        outcome: str = "",
        duration_ms: float | None = None,
        counts: str = "",
        level: int = logging.INFO,
    ) -> None:
        """打一条 worker terminal 观测里程碑；观测自身异常绝不覆盖业务异常。"""

        try:
            turn_milestone(
                logger,
                event,
                session_id=session_id,
                turn_id=turn_id,
                client_message_id=client_message_id,
                duration_ms=duration_ms,
                outcome=outcome,
                counts=counts,
                level=level,
            )
        except Exception as error:
            logger.error(
                "passive worker 观测失败（业务不中断）: event=%s turn=%s error=%s",
                event,
                turn_id,
                error,
            )

    def _verified_client_message_id(self, result: TurnResult) -> str:
        """从唯一 userMessage item 的已验证 metadata 取 client_message_id。

        阶段1：扫描全部 userMessage item 的 data.metadata.client_message_id；
        阶段2：多个不同非空值即内部身份冲突，fail-loud，绝不静默选中；
        阶段3：唯一非空值返回，缺失返回空串由调用方按 channel 语义处理。
        """

        values: set[str] = set()
        for entry in result.items:
            if entry.kind is not TurnItemKind.USER_MESSAGE:
                continue
            data = entry.data
            raw_metadata = data.get("metadata")
            if not isinstance(raw_metadata, dict):
                raise ValueError(f"turn userMessage metadata 无效: {entry.id}")
            metadata = cast(dict[str, object], raw_metadata)
            raw = metadata.get("client_message_id")
            if raw is not None and not isinstance(raw, str):
                raise ValueError(
                    f"turn userMessage client_message_id 必须是字符串: {entry.id}"
                )
            if isinstance(raw, str) and raw:
                values.add(raw)
        if len(values) > 1:
            raise RuntimeError(
                f"同一 turn 存在多个 userMessage client_message_id: "
                f"{sorted(values)}"
            )
        return next(iter(values), "")

    async def _redeliver_terminal(
        self,
        item: InboundMessage,
        record: TurnRecord,
    ) -> None:
        """恢复路径：用权威 turn 终态投影重投递，durable delivered 后才 ACK。"""

        result = TurnResult.from_record(record)
        outbound = self._terminal_outbound(item, result)
        await self._commit_mobile_terminal(
            item,
            outbound,
            turn_id=record.id,
            client_message_id=self._verified_client_message_id(result),
            terminal_status=record.status.value,
            mode="recovery",
        )

    def _matched_mobile_turn(self, item: InboundMessage) -> TurnRecord | None:
        """以 turns.items_json 的 client_message_id 唯一匹配为恢复 owner。"""

        client_message_id = item.metadata.get("client_message_id")
        if not isinstance(client_message_id, str) or not client_message_id:
            return None
        return self._legacy_loop.session_manager.control_store.find_turn_by_client_message_id(
            item.session_key,
            client_message_id,
        )

    def _terminal_outbound(
        self,
        item: InboundMessage,
        result: TurnResult,
    ) -> OutboundMessage:
        """把权威 turn 终态投影为出站消息；正常与恢复分支共用同一投影。

        completed/failed/recovered 一律从已验证 userMessage item 贯通
        client_message_id 到 outbound metadata；缺失（mobile）或 userMessage
        多值冲突 fail-fast。assistant metadata 只是冗余投影，不拥有该身份。
        """

        verified_cmid = self._verified_client_message_id(result)
        if result.status is TurnStatus.COMPLETED:
            assistant = next(
                entry
                for entry in reversed(result.items)
                if entry.kind.value == "assistantMessage"
            )
            data = assistant.data
            metadata = dict(cast(dict[str, Any], data.get("metadata", {})))
            if verified_cmid:
                metadata["client_message_id"] = verified_cmid
            elif item.channel == "mobile" and item.handoff_id is not None:
                # durable handoff 链必须在 channel/gateway 贯通身份，缺失即 fail-fast。
                raise RuntimeError(
                    f"mobile completed turn 缺少已验证 client_message_id: "
                    f"{result.id}"
                )
            return OutboundMessage(
                channel=item.channel,
                chat_id=item.chat_id,
                content=result.final_response or "",
                thinking=cast(str | None, data.get("thinking")),
                reply_to=cast(str | None, data.get("replyTo")),
                media=list(cast(list[str], data.get("media", []))),
                metadata=metadata,
                control_turn_id=result.interaction_id,
                execution_attempt_id=result.id,
                session_message_id=cast(str | None, data.get("sessionMessageId")),
                terminal_status=TurnTerminalStatus.COMPLETED,
            )
        metadata: dict[str, Any] = {}
        if verified_cmid:
            metadata["client_message_id"] = verified_cmid
        elif item.channel == "mobile" and item.handoff_id is not None:
            raise RuntimeError(
                f"mobile terminal 缺少已验证 client_message_id: {result.id}"
            )
        terminal_status = TurnTerminalStatus(result.status.value)
        if result.status in (TurnStatus.INTERRUPTED, TurnStatus.CANCELLED):
            return OutboundMessage(
                channel=item.channel,
                chat_id=item.chat_id,
                content="本轮已中断。",
                control_turn_id=result.interaction_id,
                execution_attempt_id=result.id,
                metadata=metadata,
                terminal_status=terminal_status,
            )
        if result.status is not TurnStatus.FAILED:
            raise ValueError(f"不支持的 terminal 状态: {result.status.value}")
        return OutboundMessage(
            channel=item.channel,
            chat_id=item.chat_id,
            content="处理消息时出错，请稍后再试。",
            control_turn_id=result.interaction_id,
            execution_attempt_id=result.id,
            metadata=metadata,
            terminal_status=terminal_status,
        )

    async def _complete_message(self, item: InboundMessage) -> None:
        """完成 durable inbound 确认，并恰一次释放 mobile session admission。"""

        try:
            await self._bus.complete_inbound(item)
        finally:
            self._release_admission_once(item)

    def _release_admission_once(self, item: InboundMessage) -> None:
        """恰一次释放本轮取得的 mobile session admission，不留重复释放身份。"""

        admission_id = item.session_admission_id
        if admission_id is None:
            return
        item.session_admission_id = None
        self._legacy_loop.session_manager.release_admission(admission_id)

    def stop(self) -> None:
        self._running = False

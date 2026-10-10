from __future__ import annotations

from agent.plugin_composition.models import MODEL_CATALOG, ModelAvailability

import asyncio
import hashlib
from collections.abc import Callable, Mapping
from datetime import UTC, datetime, timedelta
from typing import cast

from core.common.file_io import run_file_io

from agent.plugin_composition import Context
from agent.plugin_composition.bindings import BINDINGS
from agent.plugin_composition.messages import (
    MESSAGE_CATALOG,
    MessageReader,
    OwnerRecord,
)
from plugins.timer.contract import TIMERS, TimerReceipt, TimerStatus
from agent.plugin_contracts import Message, body_to_dict
from agent.plugin_contracts.models import (
    MODEL_SELECTION as MODEL_SELECTION,
    ModelSelection as ModelSelection,
)

from ._boundary import (
    ALL_TOOLS,
    DELIVERY_READ,
    DELIVERY_SENDERS,
    SEMANTIC_INTEREST,
    TOOLS,
    WAKE_TOOLS_VIEW,
    SinkValue,
    ToolView,
)
from .admission import Admission, Duties
from .api import DRIFT_WAKE, EVENTMAIL_WAKE, Config, DeliveryTarget
from .legacy_rules import read_archived_rules
from .messages import recent_context
from .request import TOOLS as WAKE_TOOLS, WAKE_PROGRAM, Request
from .source import Pointer, Source
from .state import WakeState, WakeStateReader


class DashboardView:
    """Expose Wake's durable rows and original Message flow as a read-only view."""

    def __init__(
        self,
        state: WakeStateReader,
        read_flow: Callable[[str], tuple[OwnerRecord, Request, MessageReader] | None],
        read_message: Callable[[str, str], Message | None],
        delivery_status: Callable[[str, str], Mapping[str, object] | None],
    ) -> None:
        self._state = state
        self._read_flow = read_flow
        self._read_message = read_message
        self._delivery_status = delivery_status

    def list_attempts(self, limit: int, *, offset: int = 0) -> tuple[Mapping[str, object], ...]:
        return self._state.list_attempts(limit, offset=offset)

    def count_attempts(self) -> int:
        return self._state.count_attempts()

    def get_attempt(self, attempt_id: str) -> Mapping[str, object] | None:
        row = self._state.get_attempt(attempt_id)
        if row is None:
            return None
        return {**row, "flow": self.flow(attempt_id)}

    def list_runs(self, limit: int, *, offset: int = 0) -> tuple[Mapping[str, object], ...]:
        return self._state.list_runs(limit, offset=offset)

    def count_runs(self) -> int:
        return self._state.count_runs()

    def get_run(self, run_id: str) -> Mapping[str, object] | None:
        row = self._state.get_run(run_id)
        if row is None:
            return None
        return {**row, "flow": self.flow(run_id)}

    def flow(self, flow_id: str) -> Mapping[str, object] | None:
        found = self._read_flow(flow_id)
        if found is None:
            return None
        pointer_row, request, reader = found
        pointer = Pointer.model_validate(dict(pointer_row.value))
        messages = tuple(self._message(message) for message in reader.snapshot())
        # Wake's request lives in an internal session. The notification is
        # deliberately written to the target session, so read that catalog.
        notification = self._read_message(request.target.session_id, request.notification_id)
        delivery: dict[str, object] = {
            "message_id": request.notification_id,
            "channel": request.target.channel,
            "recipient": request.target.recipient,
            "status": "not_published",
            "receipt": None,
        }
        if notification is not None:
            status = self._delivery_status(notification.message_id, request.target.channel)
            if status is None:
                delivery["status"] = "not_prepared"
            else:
                delivery.update(status)
        return {
            "flow_id": flow_id,
            "pointer": {"version": pointer_row.version, **pointer.model_dump(mode="json")},
            "request": {
                **request.model_dump(mode="json", exclude={"program_binding", "tools", "rules", "history", "events"}),
                "session_id": request.session_id,
                "input_id": request.input_id,
                "notification_id": request.notification_id,
            },
            "messages": list(messages),
            "delivery": delivery,
        }

    @staticmethod
    def _message(message: Message) -> Mapping[str, object]:
        body = body_to_dict(message.body)
        parts = body.get("parts", ()) if isinstance(body, dict) else ()
        text = "\n".join(
            str(part.get("value"))
            for part in parts
            if isinstance(part, dict) and part.get("kind") == "text" and isinstance(part.get("value"), str)
        ) if isinstance(parts, list) else ""
        return {
            "message_id": message.message_id,
            "session_id": message.session_id,
            "seq": message.seq,
            "recorded_at": message.recorded_at.isoformat(),
            "author": message.author,
            "source": message.source,
            "body": body,
            "text": text,
        }


class Runtime:
    """Timer 与来源变化只提供检查机会；原请求提交后由 Source 恢复真实执行。"""

    def __init__(self, ctx: Context, config: Config, *, now: Callable[[], datetime] = lambda: datetime.now(UTC)):
        self.ctx, self.config, self.now = ctx, self._current_config(config), now
        self.state = WakeState(ctx.data_root / "wake.sqlite3")
        # apply/follow 拥有异步初始化；构造函数不在共享 loop 执行 SQL。
        self.source = Source(ctx, self.state, now=now)
        self.duties = Duties(ctx.require(EVENTMAIL_WAKE), ctx.require(DRIFT_WAKE), self.state, ctx.require(SEMANTIC_INTEREST))
        self.changed = asyncio.Event()

    @staticmethod
    def _current_config(config: Config) -> Config:
        """把跨 generation 的配置值重建为当前 Wake 模块的类型。"""
        target = config.delivery
        if type(config) is Config and (target is None or type(target) is DeliveryTarget):
            return config
        return Config.model_validate(config.model_dump(mode="python"))

    async def capture(self, flow_id: str, admission: Admission, now: datetime) -> Request | None:
        """先固定归档程序、工具、出站目标与上下文，随后才允许领取领域条目。"""
        owner = admission.owner
        if owner is None:
            return None
        ctx, target = self.ctx, self.config.delivery
        if target is None:
            return None
        alert = await ctx.require(EVENTMAIL_WAKE).peek_alert(now) if owner == "alert" else None
        if owner == "alert" and alert is None:
            return None
        bindings = ctx.require(BINDINGS)
        sink: SinkValue = {
            "name": target.channel,
            "address": target.recipient,
            "binding_id": ctx.require(DELIVERY_SENDERS).bind(target.channel, bindings),
        }
        metadata = ctx.require(MESSAGE_CATALOG).reader(target.session_id).metadata()
        model = ctx.require(MODEL_SELECTION).read_saved(metadata if metadata is not None else {})
        catalog = ctx.require(TOOLS)
        allowed = set(self.config.investigation_tools) if owner == "content" else set()
        view: ToolView = catalog.view(
            *ctx.require(WAKE_TOOLS_VIEW).refs,
            *(ref for ref in ctx.require(ALL_TOOLS)().refs if ref.name in allowed),
        )
        names = (*WAKE_TOOLS[owner], *(ref.name for ref in view.refs
                 if ref.name in allowed and ref.name not in WAKE_TOOLS[owner]))
        return Request(
            flow_id=flow_id,
            owner=owner,
            now=now,
            timezone=self.config.timezone,
            target=target,
            sink=sink,
            program_binding=bindings.bind(WAKE_PROGRAM, {}),
            tools={
                name: await catalog.bind_scoped(view.select(name), bindings)
                for name in names
            },
            snapshot_seq=admission.pool.snapshot_seq,
            items=tuple(dict(item) for item in admission.pool.items),
            proposals=tuple(dict(item) for item in admission.proposals),
            alert_ref=None if alert is None else cast(dict[str, str], dict(alert)),
            model_id=model.model_id, reasoning_effort=model.reasoning_effort,
            rules=read_archived_rules(ctx.data_root) or "",
            history=recent_context(ctx.require(MESSAGE_CATALOG), ctx.require(DELIVERY_READ),
                                   target=target.session_id, now=now),
            events=tuple(dict(item) for item in (await ctx.require(EVENTMAIL_WAKE).active_context(now))))

    async def follow(self) -> None:
        """先恢复原请求，再独立运行到期检查与五分钟池维护。"""
        async with self.ctx.runtime_scope():
            recovered_at = self.now()
            await run_file_io(lambda: self.state.close_interrupted_attempts(recovered_at))
            for flow_id in self.source.pending():
                _ = await self._run(flow_id)
        # 两条循环共用 Duties 的维护锁，维护不因模型或目标发送排队而停顿。
        async with asyncio.TaskGroup() as group:
            if self.config.delivery is not None:
                _ = group.create_task(self._due(), name="wake:due")
            _ = group.create_task(self._maintenance(), name="wake:maintenance")

    def _blocked(self, *, require_interest: bool = True) -> str | None:
        if require_interest:
            reason = self.ctx.require(SEMANTIC_INTEREST).status()
            if reason is not None:
                return reason
        catalog = self.ctx.require(MODEL_CATALOG).snapshot()
        model_id = catalog.role_bindings.get("default")
        if model_id is None or catalog.model(model_id).availability != ModelAvailability.AVAILABLE:
            return "默认聊天模型不可用"
        return None

    async def _due(self) -> None:
        while True:
            self.changed.clear()
            async with self.ctx.runtime_scope():
                now = self.now()
                alert = await self.ctx.require(EVENTMAIL_WAKE).alert_deadline(now)
                blocked = self._blocked(require_interest=alert is None or alert > now)
            if blocked is not None:
                retry = now + timedelta(seconds=30)
                if alert is not None and alert > now:
                    retry = min(retry, alert)
                receipt = await self._wait(retry, changed=True)
                if receipt.status == TimerStatus.FIRED:
                    identity = self._attempt_id(receipt)
                    try:
                        await self._finish(identity, "admission_rejected", None,
                                           f"职责检查未开始：{blocked}")
                    except BaseException as error:
                        await self._close_error(identity, None, error)
                        raise
                continue
            async with self.ctx.runtime_scope():
                deadline = await self.duties.deadline(self.now())
            if deadline is None:
                _ = await self.changed.wait()
                continue
            receipt = await self._wait(deadline, changed=True)
            if receipt.status == TimerStatus.CANCELLED:
                continue
            flow_id = self._attempt_id(receipt)
            owner = None
            try:
                async with self.ctx.runtime_scope():
                    now = self.now()
                    watermark = await self.ctx.require(EVENTMAIL_WAKE).mail_watermark()
                    await run_file_io(lambda: self.state.set_attempt_mail_watermark(
                        attempt_id=flow_id, mail_watermark=watermark))
                    admission = await self.duties.check(now)
                    owner = admission.owner
                    original = await self.capture(flow_id, admission, now)
                    if original is None:
                        outcome = "admission_rejected" if owner is not None else (
                            "content_insufficient" if admission.pool.due_count or admission.pool.expired_count else "no_due")
                    else:
                        await self.source.accept(original)
                        result = await self._run(flow_id)
                        assert result is not None
                        outcome = result
                    await self._finish(flow_id, outcome, owner, admission.detail)
            except BaseException as error:
                await self._close_error(flow_id, owner, error)
                raise

    async def _run(self, flow_id: str) -> str | None:
        """来源循环关闭时撤销并排空自己接纳的 Task，保留原 Input 供新进程恢复。"""
        task = await self.source.start(flow_id)
        if task is None:
            return None
        try:
            return cast(str, await task.join())
        except asyncio.CancelledError:
            task.cancel()
            while not task.done:
                try:
                    _ = await task.join()
                except asyncio.CancelledError:
                    pass
            raise

    async def _maintenance(self) -> None:
        while True:
            now = self.now()
            deadline = await run_file_io(lambda: self.state.next_maintenance_deadline(
                now, interval=timedelta(minutes=5)))
            receipt = await self._wait(deadline, changed=False)
            if receipt.status == TimerStatus.CANCELLED:
                continue
            flow_id = self._attempt_id(receipt)
            try:
                async with self.ctx.runtime_scope():
                    blocked = self._blocked()
                    if blocked is not None:
                        await self._finish(flow_id, "admission_rejected", None,
                                           f"池维护未开始：{blocked}")
                        continue
                    watermark = await self.ctx.require(EVENTMAIL_WAKE).mail_watermark()
                    await run_file_io(lambda: self.state.set_attempt_mail_watermark(
                        attempt_id=flow_id, mail_watermark=watermark))
                    pool = await self.duties.maintain(self.now())
                    await self._finish(flow_id,
                        "content_insufficient" if pool.due_count or pool.expired_count else "no_due",
                        "content" if pool.due_count or pool.expired_count else None,
                        pool.detail + "；maintenance_only=1")
            except BaseException as error:
                await self._close_error(flow_id, None, error)
                raise

    @staticmethod
    def _attempt_id(receipt: TimerReceipt) -> str:
        return hashlib.sha256((receipt.timer_id + "\n" + receipt.deadline.isoformat() + "\n" +
                               receipt.settled_at.isoformat()).encode()).hexdigest()[:32]

    def _begin(self, receipt: TimerReceipt) -> str:
        identity = self._attempt_id(receipt)
        self.state.begin_attempt(attempt_id=identity, timer_id=receipt.timer_id,
            scheduled_for=receipt.deadline, fired_at=receipt.settled_at)
        return identity

    async def _finish(self, identity: str, outcome: str, owner: str | None, detail: str) -> None:
        """终态提交和连接关闭完成后才返回；时间在 loop 固定。"""
        completed_at = self.now()
        await run_file_io(lambda: self.state.finish_attempt(
            attempt_id=identity, outcome=outcome, owner=owner, detail=detail, completed_at=completed_at))

    async def _close_error(self, identity: str, owner: str | None, error: BaseException) -> None:
        """只闭合尚在检查的 attempt，保留已提交终态和原始失败。"""
        completed_at = self.now()
        outcome = "cancelled_after_fire" if isinstance(error, asyncio.CancelledError) else "failed"
        detail = ("Timer 已触发，原消息与领域回执留待恢复" if outcome == "cancelled_after_fire"
                  else f"{type(error).__name__}: {error}")

        def close() -> None:
            attempt = self.state.get_attempt(identity)
            if attempt is None:
                raise RuntimeError("实际触发的 Wake attempt 不存在")
            if attempt["outcome"] == "checking":
                self.state.finish_attempt(attempt_id=identity, outcome=outcome, owner=owner,
                                          detail=detail, completed_at=completed_at)

        try:
            await run_file_io(close)
        except BaseException as recovery_error:
            if isinstance(error, asyncio.CancelledError) and isinstance(recovery_error, asyncio.CancelledError):
                raise error  # 两次取消均已排空，不能把纯取消升级为程序失败。
            raise BaseExceptionGroup("Wake 检查与诊断结算均失败", [error, recovery_error]) from None

    async def _wait(self, deadline: datetime, *, changed: bool) -> TimerReceipt:
        """提示只撤回尚未触发的 Timer；取消时已触发回执仍留下耐久诊断。"""
        handle = self.ctx.require(TIMERS).schedule(deadline)
        timer = asyncio.create_task(handle.result())
        hint = asyncio.create_task(self.changed.wait()) if changed else None
        receipt: TimerReceipt | None = None
        error: BaseException | None = None
        # 1. 实际 fire 先落盘；取消可能发生在物理提交完成而 await 尚未返回时。
        try:
            if hint is not None:
                done, _ = await asyncio.wait((timer, hint), return_when=asyncio.FIRST_COMPLETED)
                receipt = await handle.cancel() if hint in done else timer.result()
            else:
                receipt = await timer
            if receipt.status == TimerStatus.FIRED:
                fired = receipt
                _ = await run_file_io(lambda: self._begin(fired))
        except asyncio.CancelledError as cancelled:
            error = cancelled
            try:
                receipt = await handle.cancel()
            except asyncio.CancelledError:
                pass  # close 会在独立排空任务中取回同一 Timer 的实际回执。
            except BaseException as cancel_error:
                error = BaseExceptionGroup("Wake 取消与 Timer 回执均失败", [error, cancel_error])
        except BaseException as failure:
            error = failure

        # 2. 排空原 Timer 和所有等待者；清理失败不遮盖 SQL/取消的原始错误。
        async def close() -> None:
            nonlocal receipt
            waiters = tuple(waiter for waiter in (timer, hint) if waiter is not None)
            for waiter in waiters:
                _ = waiter.cancel()
            for waiter in waiters:
                _ = await asyncio.gather(waiter, return_exceptions=True)
            if receipt is None:
                receipt = await handle.cancel()
            await handle.cleanup()

        closing = asyncio.create_task(close())
        while not closing.done():
            try:
                await asyncio.shield(closing)
            except asyncio.CancelledError as cancelled:
                if error is None:
                    error = cancelled
                elif not isinstance(error, asyncio.CancelledError):
                    error = BaseExceptionGroup("Wake 检查失败后取消", [error, cancelled])
            except Exception:
                break  # 从 closing.result 取回实际 cleanup 错误。
        try:
            closing.result()
        except BaseException as cleanup_error:
            if not (isinstance(error, asyncio.CancelledError) and isinstance(cleanup_error, asyncio.CancelledError)):
                error = (cleanup_error if error is None else
                         BaseExceptionGroup("Wake 检查与 Timer 清理均失败", [error, cleanup_error]))

        # 3. 本阶段尚未读领域水位；实际 fire 必须留下耐久诊断后才退出。
        if error is not None:
            if receipt is not None and receipt.status == TimerStatus.FIRED:
                fired = receipt
                completed_at = self.now()
                outcome = "cancelled_after_fire" if isinstance(error, asyncio.CancelledError) else "failed"
                detail = ("Timer 已触发，职责检查尚未开始" if outcome == "cancelled_after_fire"
                          else f"{type(error).__name__}: {error}")

                def settle() -> None:
                    identity = self._begin(fired)
                    self.state.finish_attempt(attempt_id=identity, outcome=outcome, owner=None,
                                              detail=detail, completed_at=completed_at)

                try:
                    await run_file_io(settle)
                except BaseException as recovery_error:
                    if isinstance(error, asyncio.CancelledError) and isinstance(recovery_error, asyncio.CancelledError):
                        raise error
                    raise BaseExceptionGroup("Wake Timer 与诊断结算均失败", [error, recovery_error]) from None
            raise error
        assert receipt is not None
        return receipt

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Callable, Mapping
from datetime import UTC, datetime, timedelta
from typing import cast

from agent.plugin_contracts.timer import TimerReceipt, TimerStatus
from agent.plugin_contracts.session_selection import read_session_model_selection
from agent.plugin_composition import Context
from agent.plugin_composition.bindings import BINDINGS
from agent.plugin_composition.messages import MESSAGE_CATALOG
from agent.plugin_composition.timers import TIMERS
from plugins.akasha.interest import SEMANTIC_INTEREST
from agent.plugin_contracts.delivery_api import Sink
from plugins.delivery.history import DELIVERY_READ
from agent.plugin_contracts.delivery import DELIVERY_SENDERS
from plugins.akasha.message_plugin import AKASHA_TOOLS
from plugins.standard_web.plugin import STANDARD_WEB_TOOLS
from agent.plugin_contracts.tools import TOOLS, ToolView
from agent.plugin_composition.messages import MessageReader, OwnerRecord
from agent.plugin_contracts import Message
from agent.plugin_contracts import body_to_dict

from .admission import Admission, Duties
from .api import Config, DRIFT_WAKE, EVENTMAIL_WAKE
from .legacy_rules import read_archived_rules
from .messages import recent_context
from .request import Request, TOOLS as WAKE_TOOLS, WAKE_PROGRAM, WAKE_TOOLS_VIEW
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
        self.ctx, self.config, self.now = ctx, config, now
        self.state = WakeState(ctx.data_root / "wake.sqlite3")
        # Runtime owns creation and schema validation. Dashboard readers only
        # open this already-initialized file read-only.
        self.state.initialize()
        self.source = Source(ctx, self.state, now=now)
        self.duties = Duties(ctx.require(EVENTMAIL_WAKE), ctx.require(DRIFT_WAKE), self.state, ctx.require(SEMANTIC_INTEREST))
        self.changed = asyncio.Event()

    def dashboard_view(self) -> DashboardView:
        """Build a dashboard view from narrow read-only callbacks."""
        catalog = self.ctx.require(MESSAGE_CATALOG)
        history = self.ctx.require(DELIVERY_READ)

        def read_message(session_id: str, message_id: str) -> Message | None:
            return catalog.reader(session_id).get(message_id)

        return DashboardView(
            self.state.read_only(), self.source.read, read_message, history.status,
        )

    def capture(self, flow_id: str, admission: Admission, now: datetime) -> Request | None:
        """先固定归档程序、工具、出站目标与上下文，随后才允许领取领域条目。"""
        owner = admission.owner
        if owner is None:
            return None
        ctx, target = self.ctx, self.config.delivery
        if target is None:
            return None
        alert = ctx.require(EVENTMAIL_WAKE).peek_alert(now) if owner == "alert" else None
        if owner == "alert" and alert is None:
            return None
        bindings = ctx.require(BINDINGS)
        sink = Sink(name=target.channel, address=target.recipient,
            binding_id=ctx.require(DELIVERY_SENDERS).bind(target.channel, bindings))
        metadata = ctx.require(MESSAGE_CATALOG).reader(target.session_id).metadata()
        model = read_session_model_selection(metadata if metadata is not None else {})
        view = ToolView.combine(
            ctx.require(WAKE_TOOLS_VIEW),
            ctx.require(AKASHA_TOOLS),
            ctx.require(STANDARD_WEB_TOOLS),
        )
        return Request(
            flow_id=flow_id,
            owner=owner,
            now=now,
            timezone=self.config.timezone,
            target=target,
            sink=sink,
            program_binding=bindings.bind(WAKE_PROGRAM, {}),
            tools={
                name: ctx.require(TOOLS).bind(view.select(name), bindings)
                for name in WAKE_TOOLS[owner]
            },
            snapshot_seq=admission.pool.snapshot_seq,
            items=tuple(dict(item) for item in admission.pool.items),
            proposals=tuple(dict(item) for item in admission.proposals),
            alert_ref=None if alert is None else cast(dict[str, str], dict(alert)),
            model_id=model.model_ref or None, reasoning_effort=model.reasoning_effort or None,
            rules=read_archived_rules(ctx.data_root) or "",
            history=recent_context(ctx.require(MESSAGE_CATALOG), ctx.require(DELIVERY_READ),
                                   target=target.session_id, now=now),
            events=tuple(dict(item) for item in ctx.require(EVENTMAIL_WAKE).active_context(now)))

    async def follow(self) -> None:
        """先恢复原请求，再独立运行到期检查与五分钟池维护。"""
        async with self.ctx.runtime_scope():
            self.state.initialize()
            _ = self.state.close_interrupted_attempts(self.now())
            for flow_id in self.source.pending():
                _ = await self._run(flow_id)
        # 两条循环共用 Duties 的维护锁，维护不因模型或目标发送排队而停顿。
        async with asyncio.TaskGroup() as group:
            if self.config.delivery is not None:
                _ = group.create_task(self._due(), name="wake:due")
            _ = group.create_task(self._maintenance(), name="wake:maintenance")

    async def _due(self) -> None:
        while True:
            self.changed.clear()
            async with self.ctx.runtime_scope():
                deadline = self.duties.deadline(self.now())
            if deadline is None:
                _ = await self.changed.wait()
                continue
            receipt = await self._wait(deadline, changed=True)
            if receipt.status == TimerStatus.CANCELLED:
                continue
            flow_id = self._begin(receipt)
            owner = None
            try:
                async with self.ctx.runtime_scope():
                    now = self.now()
                    self.state.set_attempt_mail_watermark(attempt_id=flow_id,
                        mail_watermark=self.ctx.require(EVENTMAIL_WAKE).mail_watermark())
                    admission = await self.duties.check(now)
                    owner = admission.owner
                    original = self.capture(flow_id, admission, now)
                    if original is None:
                        outcome = "admission_rejected" if owner is not None else (
                            "content_insufficient" if admission.pool.due_count or admission.pool.expired_count else "no_due")
                    else:
                        self.source.accept(original)
                        result = await self._run(flow_id)
                        assert result is not None
                        outcome = result
                    self.state.finish_attempt(attempt_id=flow_id, outcome=outcome, owner=owner,
                        detail=admission.detail, completed_at=self.now())
            except asyncio.CancelledError:
                self.state.finish_attempt(attempt_id=flow_id, outcome="cancelled_after_fire", owner=owner,
                    detail="Timer 已触发，原消息与领域回执留待恢复", completed_at=self.now())
                raise
            except Exception as error:
                # 本层只闭合本次 Timer 诊断；原错误继续上抛，未完成来源仍由原记录恢复。
                self.state.finish_attempt(attempt_id=flow_id, outcome="failed", owner=owner,
                    detail=f"{type(error).__name__}: {error}", completed_at=self.now())
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
            deadline = self.state.next_maintenance_deadline(self.now(), interval=timedelta(minutes=5))
            receipt = await self._wait(deadline, changed=False)
            if receipt.status == TimerStatus.CANCELLED:
                continue
            flow_id = self._begin(receipt)
            try:
                async with self.ctx.runtime_scope():
                    self.state.set_attempt_mail_watermark(attempt_id=flow_id,
                        mail_watermark=self.ctx.require(EVENTMAIL_WAKE).mail_watermark())
                    pool = await self.duties.maintain(self.now())
                    self.state.finish_attempt(attempt_id=flow_id,
                        outcome="content_insufficient" if pool.due_count or pool.expired_count else "no_due",
                        owner="content" if pool.due_count or pool.expired_count else None,
                        detail=pool.detail + "；maintenance_only=1", completed_at=self.now())
            except asyncio.CancelledError:
                self.state.finish_attempt(attempt_id=flow_id, outcome="cancelled_after_fire", owner=None,
                    detail="Timer 已触发，池维护被取消", completed_at=self.now())
                raise
            except Exception as error:
                self.state.finish_attempt(attempt_id=flow_id, outcome="failed", owner=None,
                    detail=f"{type(error).__name__}: {error}", completed_at=self.now())
                raise

    def _begin(self, receipt: TimerReceipt) -> str:
        identity = hashlib.sha256((receipt.timer_id + "\n" + receipt.deadline.isoformat() + "\n" +
                                   receipt.settled_at.isoformat()).encode()).hexdigest()[:32]
        self.state.begin_attempt(attempt_id=identity, timer_id=receipt.timer_id,
            scheduled_for=receipt.deadline, fired_at=receipt.settled_at)
        return identity

    async def _wait(self, deadline: datetime, *, changed: bool) -> TimerReceipt:
        """提示只撤回尚未触发的 Timer；取消时已触发回执仍留下耐久诊断。"""
        handle = self.ctx.require(TIMERS).schedule(deadline)
        timer = asyncio.create_task(handle.result())
        hint = asyncio.create_task(self.changed.wait()) if changed else None
        receipt: TimerReceipt | None = None
        try:
            if hint is not None:
                done, _ = await asyncio.wait((timer, hint), return_when=asyncio.FIRST_COMPLETED)
                receipt = await handle.cancel() if hint in done else timer.result()
            else:
                receipt = await timer
            if receipt.status == TimerStatus.FIRED:
                _ = self._begin(receipt)
            return receipt
        except asyncio.CancelledError:
            receipt = await handle.cancel()
            if receipt.status == TimerStatus.FIRED:
                identity = self._begin(receipt)
                self.state.finish_attempt(attempt_id=identity, outcome="cancelled_after_fire", owner=None,
                    detail="Timer 已触发，职责检查尚未开始", completed_at=self.now())
            raise
        finally:
            async def close() -> None:
                waiters = tuple(waiter for waiter in (timer, hint) if waiter is not None)
                for waiter in waiters:
                    _ = waiter.cancel()
                for waiter in waiters:
                    _ = await asyncio.gather(waiter, return_exceptions=True)
                await handle.cleanup()

            closing = asyncio.create_task(close())
            cancelled = False
            try:
                while not closing.done():
                    try:
                        await asyncio.shield(closing)
                    except asyncio.CancelledError:
                        cancelled = True
                try:
                    closing.result()
                except asyncio.CancelledError:
                    cancelled = True
            except Exception as error:
                if receipt is not None and receipt.status == TimerStatus.FIRED:
                    identity = self._begin(receipt)
                    attempt = self.state.get_attempt(identity)
                    assert attempt is not None
                    if attempt["outcome"] == "checking":
                        self.state.finish_attempt(attempt_id=identity, outcome="failed", owner=None,
                            detail=f"Timer cleanup failed: {error}", completed_at=self.now())
                raise
            if cancelled:
                if receipt is not None and receipt.status == TimerStatus.FIRED:
                    self.state.finish_attempt(attempt_id=self._begin(receipt), outcome="cancelled_after_fire", owner=None,
                        detail="Timer 已触发，职责检查尚未开始", completed_at=self.now())
                raise asyncio.CancelledError

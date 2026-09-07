from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import cast

from agent.control.timer import TimerReceipt, TimerStatus
from agent.model_runtime.session_selection import read_session_model_selection
from agent.plugin_composition import Context
from agent.plugin_composition.bindings import BINDINGS
from agent.plugin_composition.messages import MESSAGE_CATALOG
from agent.plugin_composition.timers import TIMERS
from plugins.akasha.interest import SEMANTIC_INTEREST
from plugins.delivery.api import Sink
from plugins.delivery.history import DELIVERY_READ
from plugins.delivery.senders import DELIVERY_SENDERS
from plugins.tools.plugin import TOOLS

from .admission import Admission, Duties
from .api import Config, DRIFT_WAKE, EVENTMAIL_WAKE
from .legacy_rules import read_archived_rules
from .messages import recent_context
from .request import Request, TOOLS as WAKE_TOOLS, WAKE_PROGRAM
from .source import Source
from .state import WakeState


class Runtime:
    """Timer 与来源变化只提供检查机会；原请求提交后由 Source 恢复真实执行。"""

    def __init__(self, ctx: Context, config: Config, *, now: Callable[[], datetime] = lambda: datetime.now(UTC)):
        self.ctx, self.config, self.now = ctx, config, now
        self.state = WakeState(ctx.data_root / "wake.sqlite3")
        self.source = Source(ctx, self.state, now=now)
        self.duties = Duties(ctx.require(EVENTMAIL_WAKE), ctx.require(DRIFT_WAKE), self.state, ctx.require(SEMANTIC_INTEREST))
        self.changed = asyncio.Event()

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
        return Request(flow_id=flow_id, owner=owner, now=now, timezone=self.config.timezone,
            target=target, sink=sink, program_binding=bindings.bind(WAKE_PROGRAM, {}),
            tools={name: ctx.require(TOOLS).bind(name, bindings) for name in WAKE_TOOLS[owner]},
            snapshot_seq=admission.pool.snapshot_seq, items=tuple(dict(item) for item in admission.pool.items),
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
            accepted = False
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
                        accepted = True
                        result = await self._run(flow_id)
                        assert result is not None
                        outcome = result
                    self.state.finish_attempt(attempt_id=flow_id, outcome=outcome, owner=owner,
                        detail=admission.detail, completed_at=self.now())
            except asyncio.CancelledError:
                self.state.finish_attempt(attempt_id=flow_id, outcome="delivery_unknown" if accepted else "cancelled_after_fire", owner=owner,
                    detail="Timer 已触发，原消息与领域回执留待恢复", completed_at=self.now())
                raise
            except Exception as error:
                # 本层只闭合本次 Timer 诊断；原错误继续上抛，未完成来源仍由原记录恢复。
                self.state.finish_attempt(attempt_id=flow_id, outcome="delivery_unknown" if accepted else "failed", owner=owner,
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

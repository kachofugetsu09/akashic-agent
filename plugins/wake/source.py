from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime, timedelta

from pydantic import BaseModel, ConfigDict

from agent.plugin_composition import Context
from agent.plugin_composition.bindings import BINDINGS
from agent.plugin_composition.messages import MESSAGE_CATALOG, MESSAGE_WRITERS, OWNER_STATE, SESSION_ADMISSION
from agent.plugin_composition.tasks import TASKS, Task, TaskSlot
from plugins.content.plugin import check_text
from plugins.delivery.plugin import DELIVERY
from plugins.models.selection import check_selection
from session.log import MessageReader, OwnerRecord, OwnerTransaction, SessionAttributes
from session.message import ContentPart, Input, Message, Output

from .api import EVENTMAIL_WAKE, EVENTMAIL_DELIVERY, DRIFT_WAKE, DRIFT_DELIVERY
from .content import (_candidate_id, _content_candidates, _datetime, _mapping,
                      _message_with_source_links, _selected_content_refs, _string)
from .messages import decision, finished, screened_candidates
from .request import Phase, Request, Stage, WAKE_PROGRAM, check_phase, check_request, read_request, retryable
from .selection import propose_content, propose_drift
from .state import WakeState
from .tools import Alert, Screen, Share, Skip


class Pointer(BaseModel):
    """来源只保存原 Input 的位置与领域/发送是否完成，不复制程序进度。"""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    session_id: str
    input_id: str
    settled: bool = False


class Source:
    """Wake 拥有选择、通知和领域确认；普通 Message/Task/程序拥有实际执行。"""

    def __init__(self, ctx: Context, state: WakeState,
                 *, now: Callable[[], datetime] = lambda: datetime.now(UTC)):
        self.ctx, self.state, self.now = ctx, state, now

    def read(self, flow_id: str) -> tuple[OwnerRecord, Request, MessageReader] | None:
        row = self.ctx.require(OWNER_STATE).open(self.ctx).read("flow:" + flow_id)
        if row is None:
            return None
        pointer = Pointer.model_validate(dict(row.value))
        reader = self.ctx.require(MESSAGE_CATALOG).reader(pointer.session_id)
        request = read_request(reader.snapshot())
        if (request.flow_id, request.session_id, request.input_id) != (flow_id, pointer.session_id, pointer.input_id):
            raise ValueError("Wake 恢复指针与原 Input 不一致")
        return row, request, reader

    def pending(self) -> tuple[str, ...]:
        return tuple(key.removeprefix("flow:") for key, row in self.ctx.require(OWNER_STATE).open(self.ctx).list()
                     if key.startswith("flow:") and not Pointer.model_validate(dict(row.value)).settled)

    def accept(self, request: Request) -> None:
        """原选择与恢复指针同事务提交，先于任何 EventMail/Drift 领取。"""
        ctx = self.ctx
        _ = ctx.require(SESSION_ADMISSION).ensure(ctx, request.session_id,
            SessionAttributes(visibility="internal", learning="excluded"))
        writer = ctx.require(MESSAGE_WRITERS).bind(ctx, author="wake", source="wake", body_types=(Input,),
            content={"wake.request": check_request})(request.session_id)
        pointer = Pointer(session_id=request.session_id, input_id=request.input_id)
        def commit(tx: OwnerTransaction) -> None:
            key = "flow:" + request.flow_id
            previous = tx.read(key)
            if previous is None:
                _ = tx.save(key, pointer.model_dump(), expected_version=None)
            _ = tx.append(writer, request.input_id, Input((ContentPart("wake.request", request.model_dump(mode="json")),)))
        try:
            ctx.require(OWNER_STATE).open(ctx).transact(commit)
        finally:
            writer.expire()

    async def start(self, flow_id: str) -> Task | None:
        def admit(slot: TaskSlot) -> Task | None:
            if slot.current is not None:
                return slot.current
            found = self.read(flow_id)
            if found is None:
                raise KeyError(flow_id)
            if Pointer.model_validate(dict(found[0].value)).settled:
                return None
            return slot.start(lambda task: self.run(task, found[1], found[2]))
        return await self.ctx.require(TASKS).open(self.ctx).admit(("flow", flow_id), admit)

    async def run(self, task: Task, request: Request, reader: MessageReader) -> str:
        if request.owner == "content":
            return await self._content(task, request, reader)
        if request.owner == "drift":
            return await self._drift(task, request, reader)
        return await self._alert(task, request, reader)

    async def _phase(self, task: Task, request: Request, reader: MessageReader,
                     stage: Stage, data: Mapping[str, object]) -> Message:
        """每阶段先保存实际 Input；终态存在便不再打开模型或工具。"""
        terminal = finished(reader, request, stage)
        if terminal is not None:
            return terminal
        ctx = self.ctx
        await ctx.require(DELIVERY).open(ctx).wait_idle(request.sink.name, request.sink.address)
        if not task.active:
            raise asyncio.CancelledError
        writer = ctx.require(MESSAGE_WRITERS).bind(ctx, author="wake", source="wake", body_types=(Input,),
            content={"wake.phase": check_phase, "model.selection": check_selection, "text": check_text})(request.session_id)
        try:
            phase = Phase(input_id=request.input_id, stage=stage)
            _ = writer.append(request.phase_id(stage), Input((ContentPart("wake.phase", phase.model_dump(mode="json")),
                ContentPart("text", json.dumps({"stage": stage, "data": dict(data)}, ensure_ascii=False)),
                ContentPart("model.selection", {"model_id": request.model_id, "reasoning_effort": request.reasoning_effort}))))
        finally:
            writer.expire()
        async with ctx.require(BINDINGS).open(request.program_binding, WAKE_PROGRAM) as (program, _):
            return await program(task, reader, request)

    def _settled(self, request: Request, reader: MessageReader) -> None:
        """未进入模型的纯业务决定也正常关闭输入，然后推进唯一来源指针。"""
        ctx = self.ctx
        messages = reader.snapshot()
        if isinstance(messages[-1].body, Input):
            writer = ctx.require(MESSAGE_WRITERS).bind(ctx, author="wake", source="wake", body_types=(Output,), content={})(request.session_id)
            try:
                _ = writer.append("wake-close:" + request.flow_id, Output((), "quiet"))
            finally:
                writer.expire()
        def commit(tx: OwnerTransaction) -> None:
            key = "flow:" + request.flow_id
            row = tx.read(key)
            assert row is not None
            pointer = Pointer.model_validate(dict(row.value))
            if not pointer.settled:
                _ = tx.save(key, pointer.model_copy(update={"settled": True}).model_dump(), expected_version=row.version)
        ctx.require(OWNER_STATE).open(ctx).transact(commit)

    def _record(self, request: Request, action: str, detail: str) -> None:
        self.state.record_decision(run_id=request.flow_id, decision=action, detail=detail, completed_at=self.now())

    async def _content(self, task: Task, request: Request, reader: MessageReader) -> str:
        """固定池快照先初筛，领取原批次后调查；成功送达后才结算所选条目。"""
        domain = self.ctx.require(EVENTMAIL_WAKE)
        proposal = propose_content(request.items, now=request.now)
        if proposal is None:
            raise ValueError("Wake 原 Content 快照没有到期候选")
        all_candidates = _content_candidates(proposal)
        allowed = {_candidate_id(_mapping(item.get("ref"), "Content ref")) for item in all_candidates}
        screen: Screen | None = None
        if proposal.decision == "select":
            _ = await self._phase(task, request, reader, "screen", {"candidates": screened_candidates(request)})
            value = decision(reader, request, "screen")
            if isinstance(value, Screen) and all(item.candidate_id in allowed for item in value.items):
                screen = value
        self.state.record_screen(run_id=request.flow_id, owner="content", candidates_seen=len(all_candidates),
            screening=() if screen is None else tuple(item.model_dump() for item in screen.items), started_at=request.now)

        # 1. 即使初筛失败也领取这份原批次，再按原业务规则延期，避免反复初筛同一池。
        wanted = allowed if screen is None else {item.candidate_id for item in screen.items}
        refs = tuple(_mapping(item.get("ref"), "Content ref") for item in all_candidates
                     if _candidate_id(_mapping(item.get("ref"), "Content ref")) in wanted)
        selected = domain.selection(request.accepted)
        if selected is None:
            claimed = domain.select_batch(refs, request.snapshot_seq, request.accepted, request.now)
            if claimed.get("selected") is not True:
                self._record(request, "defer", "原 Content 批次已经变化，领取被拒绝")
                self._settled(request, reader)
                return "admission_rejected"
            selected = domain.selection(request.accepted)
            if selected is None:
                raise ValueError("Content 领取成功却缺少领域回执")
        self.state.commit_content_admission(request.items)
        token = _string(selected.get("selection_token"), "Content selection_token")
        status = _string(selected.get("status"), "Content status")
        if status == "selected":
            if proposal.decision == "decline":
                self._record(request, "skip", "来源明确要求等待内容变化")
                self._change_content(token, "await_change")
            elif screen is None:
                self._record(request, "defer", "初筛没有提交有效候选")
                action = "invalidated" if retryable(finished(reader, request, "screen")) is False else "defer"
                self._change_content(token, action)
            else:
                _ = await self._phase(task, request, reader, "investigate", {"candidates": screened_candidates(request, screen)})
                value = decision(reader, request, "investigate")
                if isinstance(value, Share):
                    try:
                        chosen = _selected_content_refs(selected, value.items)
                    except ValueError:
                        self._record(request, "defer", "调查分享引用了原批次之外的候选")
                        self._change_content(token, "defer")
                    else:
                        self._record(request, "share", value.message)
                        self._change_content(token, "ready_for_delivery", refs=chosen)
                elif isinstance(value, Skip):
                    self._record(request, "skip", value.reason)
                    self._change_content(token, "release")
                else:
                    self._record(request, "defer", "调查没有提交唯一有效决定")
                    action = "invalidated" if retryable(finished(reader, request, "investigate")) is False else "defer"
                    self._change_content(token, action)
        return await self._finish_domain(task, request, reader, "investigate")

    def _change_content(self, token: str, action: str, *, refs: Sequence[Mapping[str, object]] | None = None) -> None:
        result = self.ctx.require(EVENTMAIL_WAKE).transition(token, action,
            not_before=self.now() + timedelta(minutes=5) if action == "defer" else None, selected_refs=refs)
        if result.get("changed") is not True:
            raise RuntimeError("原 Content 领取没有提交预期变化")

    async def _drift(self, task: Task, request: Request, reader: MessageReader) -> str:
        domain = self.ctx.require(DRIFT_WAKE)
        proposal = propose_drift(request.proposals)
        if proposal is None:
            raise ValueError("Wake 原 Drift 快照没有到期候选")
        self.state.record_screen(run_id=request.flow_id, owner="drift", candidates_seen=1,
                                 screening=({"payload": dict(proposal.payload)},), started_at=request.now)
        receipt = self.ctx.require(DRIFT_DELIVERY).lookup(request.accepted)
        if receipt is not None and receipt.get("status") != "selected":
            return await self._finish_domain(task, request, reader, "drift")
        selected = domain.selection(request.accepted)
        if selected is None:
            claim = domain.select(proposal.ref, request.accepted, request.now)
            if claim.get("selected") is not True:
                # 原 state_version 已消耗时不能再领取；已有决定保留其真实原因。
                previous = self.state.get_run(request.flow_id)
                assert previous is not None
                if previous["decision"] is None:
                    self._record(request, "defer", "原 Drift 职责领取被拒绝")
                self._settled(request, reader)
                return "admission_rejected"
            selected = domain.selection(request.accepted)
            if selected is None:
                raise ValueError("Drift 领取成功却缺少领域回执")
        if proposal.decision == "decline":
            action = "defer" if selected.get("next_due") is not None else "await_change"
            self._record(request, "skip", "来源明确拒绝本轮 Drift")
        else:
            _ = await self._phase(task, request, reader, "drift", {"duty": dict(proposal.payload)})
            value = decision(reader, request, "drift")
            if isinstance(value, Share) and not value.items:
                action = "ready_for_delivery"
                self._record(request, "share", value.message)
            elif isinstance(value, Skip):
                action = "await_change"
                self._record(request, "skip", value.reason)
            else:
                action = ("invalidated" if retryable(finished(reader, request, "drift")) is False else
                          "defer" if selected.get("next_due") is not None else "await_change")
                self._record(request, "defer", "Drift 没有提交唯一有效决定")
        result = domain.transition(_string(selected.get("selection_token"), "Drift token"), action)
        if result.get("changed") is not True:
            raise RuntimeError("原 Drift 领取没有提交预期变化")
        if action != "ready_for_delivery":
            self._settled(request, reader)
            return "deferred" if action == "defer" else "model_skip"
        return await self._finish_domain(task, request, reader, "drift")

    async def _finish_domain(self, task: Task, request: Request, reader: MessageReader, stage: Stage) -> str:
        domain = self.ctx.require(EVENTMAIL_DELIVERY if request.owner == "content" else DRIFT_DELIVERY)
        selected = domain.lookup(request.accepted)
        if selected is None:
            raise ValueError("Wake 完成缺少原领域领取")
        if selected.get("status") not in {"ready_for_delivery", "delivered", "settled"}:
            self._settled(request, reader)
            return "deferred" if selected.get("status") == "deferred" else "model_skip"
        value = decision(reader, request, stage)
        if not isinstance(value, Share):
            raise ValueError("待送达领域状态缺少实际 share ToolResult")
        text = value.message
        if request.owner == "content":
            payloads = {_candidate_id(_mapping(item["ref"], "Content ref")): _mapping(item["payload"], "Content payload")
                        for item in request.items}
            text = _message_with_source_links(text, {"source_refs": [payloads[name] for name in value.items]})
        if not await self._notify(task, request, text):
            return "delivery_unknown"
        result = domain.settle(_string(selected.get("selection_token"), "selection_token"), request.notification_id)
        if result.get("settled") is not True:
            raise RuntimeError("Wake 真实送达后的领域确认未提交")
        self._settled(request, reader)
        return "shared"

    async def _notify(self, task: Task, request: Request, text: str, *, before_start: Callable[[], str | None] | None = None) -> bool:
        """通知正文与原 Sink 一起提交；未知或拒绝回执不冒充领域已送达。"""
        target, sink = request.target, request.sink
        ctx = self.ctx
        delivery = ctx.require(DELIVERY).open(ctx)
        reader = ctx.require(MESSAGE_CATALOG).reader(target.session_id)
        message = reader.get(request.notification_id)
        if message is None:
            if not task.active:
                raise asyncio.CancelledError
            writer = ctx.require(MESSAGE_WRITERS).bind(ctx, author="wake", source="wake", body_types=(Output,),
                content={"text": check_text})(target.session_id)
            try:
                message, _ = delivery.publish(writer, request.notification_id, Output((ContentPart("text", text),), "complete"), (sink,))
            finally:
                writer.expire()
        selected = delivery.prepare(reader, message, (sink,))
        if selected.sinks != (sink.name,):
            raise ValueError("Wake 原通知目的地不一致")
        receipt = await delivery.send(message.message_id, sink.name, before_start=before_start)
        return receipt.status == "delivered"

    async def _alert(self, task: Task, request: Request, reader: MessageReader) -> str:
        """告警始终匹配原 envelope，领取后的过期与确认也核对实际发送状态。"""
        domain = self.ctx.require(EVENTMAIL_WAKE)
        ref = request.alert_ref
        assert ref is not None
        selected = domain.select_alert(request.accepted, request.now, item_ref=ref)
        if selected is None:
            return await self._finish_old_alert(request, reader)
        self.state.record_screen(run_id=request.flow_id, owner="alert", candidates_seen=1,
            screening=({"payload": _mapping(selected.get("payload"), "Alert payload")},), started_at=request.now)
        delivery = self.ctx.require(DELIVERY).open(self.ctx)
        has_delivery = delivery.selection(request.notification_id) is not None
        if not has_delivery and domain.change_alert(ref, request.accepted, "expire", self.now()):
            previous = self.state.get_run(request.flow_id)
            assert previous is not None
            if previous["decision"] is None:
                self._record(request, "skip", "告警在发送前已过期")
            self._settled(request, reader)
            return "model_skip"
        _ = await self._phase(task, request, reader, "alert", {"alert": dict(selected)})
        value = decision(reader, request, "alert")
        if not isinstance(value, Alert):
            action = "skip" if retryable(finished(reader, request, "alert")) is False else "defer"
            self._record(request, action, "告警没有提交唯一有效 share_alert")
            _ = domain.change_alert(ref, request.accepted, action, self.now(),
                                    not_before=self.now() + timedelta(minutes=5) if action == "defer" else None)
            self._settled(request, reader)
            return "model_skip" if action == "skip" else "deferred"
        self._record(request, "share", value.message)
        # 模型期间来源可能已经换版；不得发送旧版本的新通知或关闭新 projection。
        if domain.alert_status(ref["source_id"], ref["event_id"], mail_id=ref["mail_id"]) != "selected":
            return await self._finish_old_alert(request, reader)
        def before_start() -> str | None:
            if domain.alert_status(ref["source_id"], ref["event_id"], mail_id=ref["mail_id"]) != "selected":
                return "原告警版本已结束"
            expires_at = selected.get("expires_at")
            if expires_at is not None and _datetime(expires_at) <= self.now():
                return "告警在发送前已过期"
            return None

        if not await self._notify(task, request, value.message, before_start=before_start):
            receipt = delivery.receipt(request.notification_id, request.sink.name)
            if receipt is not None and receipt.status == "rejected":
                if domain.change_alert(ref, request.accepted, "expire", self.now()):
                    self._settled(request, reader)
                    return "model_skip"
                if domain.alert_status(ref["source_id"], ref["event_id"], mail_id=ref["mail_id"]) != "selected":
                    return await self._finish_old_alert(request, reader)
            return "delivery_unknown"
        changed = domain.change_alert(ref, request.accepted, "deliver", self.now())
        status = domain.alert_status(ref["source_id"], ref["event_id"], mail_id=ref["mail_id"])
        if not changed and status not in {"delivered", "superseded"}:
            raise RuntimeError("Alert 已送达但原领取无法确认")
        self._settled(request, reader)
        return "shared"

    async def _finish_old_alert(self, request: Request, reader: MessageReader) -> str:
        """已失效版本只撤回未开始发送；已有未知效果仍恢复原回执，不结算新版本。"""
        delivery = self.ctx.require(DELIVERY).open(self.ctx)
        selected = delivery.selection(request.notification_id)
        if selected is not None:
            for sink in selected.sinks:
                cancelled = await delivery.cancel_prepared(request.notification_id, sink, "原告警版本已结束")
                if not cancelled:
                    result = await delivery.send(request.notification_id, sink)
                    if result.status == "unknown":
                        return "delivery_unknown"
        self._settled(request, reader)
        return "model_skip"

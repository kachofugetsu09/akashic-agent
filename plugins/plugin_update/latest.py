"""显式候选程序复用普通工具、Task 和 Message，不启动后台裁判。"""
from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from agent.plugin_composition import Context
from agent.plugin_composition.messages import OWNER_STATE, OwnerTransaction
from agent.plugin_composition.plugin_updates import PLUGIN_UPDATES
from agent.plugin_composition.tasks import TASKS, Task, TaskSlot
from agent.plugin_contracts import ContentPart, body_to_dict, json_value

from .inputs import CallSource, Result, MODEL_SETTINGS
from .tool import Request
from .validation import PLUGIN_VALIDATION


class LatestInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    update_id: str = Field(min_length=1)
    action: Literal["run", "status", "revert"]


class LatestRequest(LatestInput):
    session_id: str = Field(min_length=1)


class LatestReference(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    update_id: str
    session_id: str


class LatestCall(BaseModel):
    """只保存调用接纳时的固定关联；执行状态由原 Task、Message 和 journal 拥有。"""

    model_config = ConfigDict(extra="forbid", strict=True)
    update_id: str
    session_id: str
    candidate_id: str
    handle: str


class Latest:
    """发起 Agent 显式启动、观察或撤销同一更新。"""

    idempotent = False

    def __init__(self, ctx: Context):
        self._ctx = ctx

    def _install(self, identity: str, session_id: str) -> Request:
        record = self._ctx.require(OWNER_STATE).open(self._ctx).read(identity)
        if record is None:
            raise ValueError("没有对应的插件安装请求")
        install = Request.model_validate(json_value(record.value))
        if install.session_id != session_id:
            raise PermissionError("更新不属于发起 session")
        return install

    async def prepare(self, arguments: Mapping[str, object], source: CallSource | None = None) -> Mapping[str, object] | str:
        try:
            request = LatestInput.model_validate(json_value(arguments))
        except ValidationError as error:
            return str(error)
        if source is None or not source.messages:
            return "plugin_latest 需要真实发起 Message"
        session_id = source.messages[-1].session_id
        try:
            self._install(request.update_id, session_id)
        except PermissionError as error:
            return str(error)
        return LatestRequest(**request.model_dump(), session_id=session_id).model_dump(mode="json")

    async def _status(self, identity: str, session_id: str) -> Result:
        """只读固定调用关联和真实过程；Task 消失不等于成功，未知不重跑。"""
        self._install(identity, session_id)
        ctx = self._ctx
        updates = ctx.require(PLUGIN_UPDATES)
        status = updates.read(ctx, identity)
        if status is None:
            raise ValueError("更新没有候选收据")
        record = ctx.require(OWNER_STATE).open(ctx).read("latest-update:" + identity)
        call = None if record is None else LatestCall.model_validate(json_value(record.value))
        if call is not None and (call.update_id != identity or call.session_id != session_id
                                 or call.candidate_id != status.candidate_id):
            raise PermissionError("调用不属于该更新授权")
        task = await ctx.require(TASKS).open(ctx).admit(("latest", identity), lambda slot: slot.current)
        if task is not None and (call is None or task.handle != call.handle):
            raise RuntimeError("活动 Task 与调用记录不一致")
        messages = updates.messages(ctx, identity, "plugin-validation:" + identity)
        text = json.dumps({
            "update": asdict(status),
            "call": None if call is None else call.model_dump(mode="json"),
            "task": None if task is None else {"handle": task.handle, "active": task.active, "done": task.done},
            "messages": [{"message_id": message.message_id, "body": body_to_dict(message.body)} for message in messages],
        }, ensure_ascii=False)
        return Result("success", (ContentPart("text", text),))

    async def invoke(self, key: str, arguments: Mapping[str, object]) -> Result:
        """接纳后立即返回；普通调用正常结束并释放 Scope 后才请求 publication。"""
        request = LatestRequest.model_validate(json_value(arguments))
        ctx = self._ctx
        identity = request.update_id
        install = self._install(identity, request.session_id)
        updates = ctx.require(PLUGIN_UPDATES)
        if request.action == "status":
            return await self._status(identity, request.session_id)
        if request.action == "revert":
            record = ctx.require(OWNER_STATE).open(ctx).read("latest-update:" + identity)
            if record is not None:
                call = LatestCall.model_validate(json_value(record.value))
                status = updates.read(ctx, identity)
                if (status is None or call.update_id != identity or call.session_id != request.session_id
                        or call.candidate_id != status.candidate_id):
                    raise PermissionError("调用不属于该更新授权")
            await updates.discard(ctx, identity, reason="发起 Agent 已 revert")
            return await self._status(identity, request.session_id)

        # 1. 同一更新只有一个接纳记录；崩溃或工具恢复没有第二次执行权。
        store = ctx.require(OWNER_STATE).open(ctx)
        run_key = "latest-call:" + key
        update_key = "latest-update:" + identity
        if store.read(run_key) is not None or store.read(update_key) is not None:
            raise RuntimeError("已有 latest 调用只能查询，不得重跑")
        status = updates.read(ctx, identity)
        if status is None or not status.ready or status.error or status.candidate_id is None:
            raise RuntimeError("更新没有已授权的确切候选")
        candidate_id = status.candidate_id
        publish = updates.publication(ctx, identity)

        async def program(_task: Task) -> object:
            settings_source = ctx.require(MODEL_SETTINGS).read_source()
            async with updates.open_validation(ctx, identity) as scope:
                scope.require(MODEL_SETTINGS).use_source(settings_source)
                return await scope.require(PLUGIN_VALIDATION).run(identity, install.install)

        def accept(slot: TaskSlot) -> tuple[Task, LatestCall]:
            task = slot.start(program)
            call = LatestCall(update_id=identity, session_id=request.session_id,
                              candidate_id=candidate_id, handle=task.handle)
            def save(tx: OwnerTransaction) -> None:
                _ = tx.save(update_key, call.model_dump(mode="json"), expected_version=None)
                _ = tx.save(run_key, {"update_id": identity, "session_id": request.session_id}, expected_version=None)
            store.transact(save)
            return task, call

        task, call = await ctx.require(TASKS).open(ctx).admit(("latest", identity), accept)

        # 2. join 等待真实 Task 的 finally 和 Scope 释放；失败与取消沿原 owner 报错。
        async def finish() -> None:
            _ = await task.join()
            publish()

        _ = await ctx.spawn(finish(), name="latest-publication:" + task.handle)
        return Result("success", (ContentPart("text", json.dumps(call.model_dump(mode="json"), ensure_ascii=False)),))

    async def query(self, key: str) -> Result | None:
        record = self._ctx.require(OWNER_STATE).open(self._ctx).read("latest-call:" + key)
        if record is None:
            return None
        request = LatestReference.model_validate(json_value(record.value))
        return await self._status(request.update_id, request.session_id)

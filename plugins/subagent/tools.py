from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from typing import Literal, cast
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from agent.plugin_composition import Context
from agent.plugin_composition.bindings import BINDINGS
from plugins.delivery.api import Sink
from plugins.tools.api import CallSource, InvalidArguments, Result
from plugins.tools.plugin import TOOLS
from session.message import ContentPart, Input
from session.message_codec import json_value

from .request import PROFILE_TOOLS, Request, SpawnInput
from .runtime import SUBAGENT_PROGRAM, SubagentBusy, Subagents, completion


class Prepared(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    task: str = Field(min_length=1)
    request: Request


class ManageInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    action: Literal["list", "cancel"]
    job_id: str = ""


class Spawn:
    """固定请求与工具权限；相同效果 key 只接纳一次，不复制最终正文。"""

    idempotent = True

    def __init__(self, ctx: Context, targets: Mapping[str, str], senders: Mapping[str, str]):
        self.ctx = ctx
        self.targets = targets
        self.senders = senders
        self.jobs = Subagents(ctx)

    async def prepare(self, arguments: Mapping[str, object], source: CallSource | None = None) -> Mapping[str, object]:
        # 1. 只有实际父调用能建立回传关系，模型参数不能伪造父 Session。
        try:
            args = SpawnInput.model_validate(dict(arguments))
        except ValidationError as error:
            raise InvalidArguments(str(error)) from error
        if not args.task.strip():
            raise InvalidArguments("子任务不能为空")
        if source is None:
            raise InvalidArguments("spawn 需要已提交的父消息调用")
        origin = None
        for message in reversed(source.messages):
            if isinstance(message.body, Input):
                parts = [part for part in message.body.parts if part.kind == "channel.origin"]
                if parts:
                    origin = cast(dict[str, str], json_value(parts[0].value))
                break
        if args.run_in_background and origin is None:
            raise InvalidArguments("后台子任务需要原会话渠道")
        sink = None
        if args.run_in_background:
            assert origin is not None
            if origin["channel"] not in self.senders:
                raise InvalidArguments("后台子任务的原渠道没有发送能力")
            sink = Sink(name=origin["channel"], binding_id=self.senders[origin["channel"]], address=origin["chat_id"])
        job_id = uuid4().hex
        task_dir = self.ctx.workspace_root("subagent-runs") / job_id
        names = PROFILE_TOOLS[args.profile]
        if set(names) - self.targets.keys():
            raise InvalidArguments("当前 profile 缺少已安装工具")
        bindings = self.ctx.require(BINDINGS)
        fixed: dict[str, str] = {}
        # 2. 目录、网络设置与实际工具实现都在 prepared 时固定，恢复不重选配置。
        for name in names:
            configuration: Mapping[str, object] | None = None
            if name in {"write_file", "edit_file"}:
                configuration = {"allowed_dir": str(task_dir)}
            elif name in {"shell", "write_stdin", "task_stop"}:
                configuration = {"owner_key": "subagent:" + job_id, "working_dir": str(task_dir),
                                 "allow_network": args.profile == "general"}
            if configuration is None:
                fixed[name] = self.targets[name]
            else:
                async with bindings.open(self.targets[name], TOOLS) as (tools, _):
                    fixed[name] = tools.bind(name, bindings, configuration=configuration)
        request = Request(job_id=job_id, label=(args.label or args.task[:30]).strip(), profile=args.profile,
            background=args.run_in_background, retry_count=args.retry_count,
            parent_session_id=source.messages[-1].session_id, parent_message_id=source.call_ref.message_id,
            parent_part_index=source.call_ref.part_index,
            origin=origin, sink=sink, program_binding=bindings.bind(SUBAGENT_PROGRAM, {}), tools=fixed)
        return Prepared(task=args.task, request=request).model_dump()

    async def invoke(self, key: str, arguments: Mapping[str, object]) -> Result:
        prepared = Prepared.model_validate(json_value(arguments))
        try:
            self.jobs.accept(key, prepared.request, prepared.task)
        except SubagentBusy as error:
            return Result("error", (ContentPart("text", str(error)),))
        if not prepared.request.background:
            task = await self.jobs.start(key)
            if task is not None:
                try:
                    _ = await task.join()
                except asyncio.CancelledError:
                    _ = await self.jobs.cancel(prepared.request.job_id)
                    raise
        result = await self.query(key)
        if result is None:
            raise RuntimeError("同步子任务尚未结算")
        return result

    async def query(self, key: str) -> Result | None:
        found = self.jobs.read(key)
        if found is None:
            return None
        record, request, reader = found
        if request.background:
            return Result("success", (ContentPart("text", f"已创建后台任务「{request.label}」（job_id={request.job_id}）。完成后会回传当前会话。"),))
        if not record.value["settled"]:
            return None
        outcome = self.jobs.outcome(reader)
        if outcome is None:
            raise ValueError("子任务结算记录缺少消息终态")
        return Result("success" if outcome[0] == "completed" else "error",
                      (ContentPart("text", completion(request, outcome, limit=100_000)),))


class Manage:
    idempotent = True

    def __init__(self, ctx: Context):
        self.jobs = Subagents(ctx)

    async def prepare(self, arguments: Mapping[str, object], source: CallSource | None = None) -> Mapping[str, object]:
        try:
            args = ManageInput.model_validate(dict(arguments))
        except ValidationError as error:
            raise InvalidArguments(str(error)) from error
        if args.action == "cancel" and not args.job_id:
            raise InvalidArguments("取消子任务需要 job_id")
        return args.model_dump()

    async def invoke(self, key: str, arguments: Mapping[str, object]) -> Result:
        args = ManageInput.model_validate(json_value(arguments))
        value: dict[str, object]
        if args.action == "list":
            jobs = self.jobs.jobs()
            value = {"running_count": len(jobs), "jobs": jobs}
        else:
            cancelled = await self.jobs.cancel(args.job_id)
            value = {"job_id": args.job_id, "status": "cancel_requested" if cancelled else "not_found"}
        return Result("success", (ContentPart("text", json.dumps(value, ensure_ascii=False)),))

    async def query(self, key: str) -> Result | None:
        return None

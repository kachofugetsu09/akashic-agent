from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable, Mapping
from typing import Literal, cast
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from agent.plugin_composition import Context
from pathlib import Path
from plugins.ledger.contract import BINDINGS
from plugins.working_directory.contract import WORKING_DIRECTORY
from .inputs import CallSource, Result, TOOL_BIND_SAVED
from plugins.ledger.contract import ContentPart, Input, Message
from plugins.ledger.contract import json_value
from plugins.models.contract import ChatModelSelection, ModelCatalogSnapshot

from .request import PROFILE_TOOLS, Request, SpawnInput, SinkInput
from .contract import SUBAGENT_PROGRAM
from .runtime import SubagentBusy, Subagents, completion


class Prepared(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    task: str = Field(min_length=1)
    request: Request


class ManageInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    action: Literal["list", "cancel", "models"] = Field(
        description="list: 列出运行中的子任务；cancel: 取消指定 job_id 的子任务；models: 查看子任务可选用的连接渠道与模型列表"
    )
    job_id: str = ""
    connection: str = Field(default="", description="action=models 时可选：按连接名称或 connection_id 过滤该渠道下的模型")


def list_available_models(
    snapshot: ModelCatalogSnapshot,
    *,
    connection: str = "",
) -> dict[str, object]:
    """Project available connections and chat models for model-facing discovery before spawn."""
    wanted_conn = connection.strip()
    avail_conns = [c for c in snapshot.connections if c.availability == "available"]
    if wanted_conn:
        exact = [c for c in avail_conns if c.connection_id == wanted_conn or c.name == wanted_conn]
        if not exact:
            exact = [c for c in avail_conns if c.connection_id.lower() == wanted_conn.lower() or c.name.lower() == wanted_conn.lower()]
        avail_conns = exact
    conn_by_id = {c.connection_id: c for c in avail_conns}
    default_id = snapshot.role_bindings.get("default")
    connections_out: list[dict[str, object]] = []
    for conn in avail_conns:
        models_out = [
            {
                "model": m.model,
                "model_id": m.model_id,
                "connection": conn.name,
                "connection_id": conn.connection_id,
                "default_reasoning_effort": m.default_reasoning_effort,
                "supported_reasoning_efforts": list(m.capabilities.supported_reasoning_efforts),
                "is_default": m.model_id == default_id,
            }
            for m in snapshot.models
            if m.kind == "chat" and m.availability == "available" and m.connection_id == conn.connection_id
        ]
        connections_out.append({
            "connection": conn.name,
            "connection_id": conn.connection_id,
            "driver_id": conn.driver_id,
            "models": models_out,
        })
    default_model = next(
        (
            {
                "model": m.model,
                "model_id": m.model_id,
                "connection": conn_by_id[m.connection_id].name,
                "connection_id": m.connection_id,
            }
            for m in snapshot.models
            if m.model_id == default_id and m.connection_id in conn_by_id
        ),
        None,
    )
    return {
        "default_model": default_model,
        "connections": connections_out,
    }


def select_model(
    snapshot: ModelCatalogSnapshot,
    *,
    model: str | None = None,
    connection: str | None = None,
    reasoning_effort: str | None = None,
) -> ChatModelSelection | str:
    """Resolve an explicit per-spawn model/channel choice against available chat models."""
    wanted_model = (model or "").strip()
    wanted_conn = (connection or "").strip()
    wanted_effort = (reasoning_effort or "").strip() or None
    if not wanted_model and not wanted_conn:
        if wanted_effort is not None:
            return "指定 reasoning_effort 时必须同时指定 model 或 connection"
        return ChatModelSelection()

    avail_conns = [c for c in snapshot.connections if c.availability == "available"]
    conn_by_id = {c.connection_id: c for c in avail_conns}
    matched_conns = avail_conns
    if wanted_conn:
        exact = [c for c in avail_conns if c.connection_id == wanted_conn or c.name == wanted_conn]
        if not exact:
            exact = [c for c in avail_conns if c.connection_id.lower() == wanted_conn.lower() or c.name.lower() == wanted_conn.lower()]
        if not exact:
            options = "、".join(f"{c.name} ({c.connection_id})" for c in avail_conns) or "无可用连接"
            return f"未找到可用连接渠道「{wanted_conn}」；可选连接：{options}"
        if len(exact) > 1:
            options = "、".join(f"{c.name} ({c.connection_id})" for c in exact)
            return f"连接渠道「{wanted_conn}」匹配多个连接，请用 connection_id 明确指定：{options}"
        matched_conns = exact

    allowed_conn_ids = {c.connection_id for c in matched_conns}
    candidates = [
        m for m in snapshot.models
        if m.kind == "chat" and m.availability == "available" and m.connection_id in allowed_conn_ids
    ]
    if not candidates:
        return f"连接渠道「{matched_conns[0].name}」下没有可用的对话模型" if matched_conns else "当前没有可用的连接渠道"

    if wanted_model:
        exact_m = [m for m in candidates if m.model_id == wanted_model or m.model == wanted_model]
        if not exact_m:
            exact_m = [m for m in candidates if m.model_id.lower() == wanted_model.lower() or m.model.lower() == wanted_model.lower()]
        if not exact_m:
            options = "、".join(f"{m.model}（{conn_by_id[m.connection_id].name}）" for m in candidates)
            return f"未找到可用模型「{wanted_model}」；可选模型：{options}"
        if len(exact_m) > 1:
            options = "、".join(f"{m.model}（connection={conn_by_id[m.connection_id].name} / {m.connection_id}，model_id={m.model_id}）" for m in exact_m)
            return f"模型「{wanted_model}」在多个连接渠道下同名，请同时指定 connection 或直接传 model_id：{options}"
        chosen = exact_m[0]
    else:
        default_id = snapshot.role_bindings.get("default")
        default_in_conn = next((m for m in candidates if m.model_id == default_id), None)
        if default_in_conn is not None:
            chosen = default_in_conn
        elif len(candidates) == 1:
            chosen = candidates[0]
        else:
            options = "、".join(m.model for m in candidates)
            return f"连接渠道「{matched_conns[0].name}」有多个可用模型，请同时指定 model：{options}"

    efforts = chosen.capabilities.supported_reasoning_efforts
    if wanted_effort is not None and efforts and wanted_effort not in efforts:
        return f"模型「{chosen.model}」不支持思考强度「{wanted_effort}」；可选：{'、'.join(efforts)}"
    effort = wanted_effort if wanted_effort is not None else chosen.default_reasoning_effort
    return ChatModelSelection(chosen.model_id, effort)


# 子任务与外部 Runner 共用的初始目录：显式 cwd 按父会话目录解析，省略时继承父会话目录，都没有时用 fallback。
async def resolve_cwd(ctx: Context, session_id: str, requested: str | None, *, fallback: Path) -> str:
    raw = (requested or "").strip()
    with ctx.borrow(WORKING_DIRECTORY) as directories:
        parent = None if directories is None else directories.snapshot(session_id).path
        if directories is not None and raw:
            target, _ = await directories.resolve_target(session_id, raw, legacy_base=parent or str(ctx.runtime.workspace))
            return await directories.check_directory(target)
    if not raw:
        return parent or str(fallback.expanduser().resolve())
    candidate = (Path(parent) if parent is not None else ctx.runtime.workspace) / Path(raw).expanduser()
    if not candidate.resolve().is_dir():
        raise ValueError(f"工作目录不存在或不是目录: {candidate.resolve()}")
    return str(candidate.resolve())


class Spawn:
    """固定请求与工具权限；相同效果 key 只接纳一次，不复制最终正文。"""

    idempotent = True

    def __init__(self, ctx: Context, targets: Mapping[str, str], senders: Mapping[str, str], jobs: Subagents,
                 *, model: Callable[[tuple[Message, ...], SpawnInput], Awaitable[ChatModelSelection | str]]):
        self.ctx = ctx
        self.targets = targets
        self.senders = senders
        self.jobs = jobs
        self.model = model

    async def prepare(self, arguments: Mapping[str, object], source: CallSource | None = None) -> Mapping[str, object] | str:
        # 1. 只有实际父调用能建立回传关系，模型参数不能伪造父 Session。
        try:
            args = SpawnInput.model_validate(dict(arguments))
        except ValidationError as error:
            return str(error)
        if not args.task.strip():
            return '子任务不能为空'
        if source is None:
            return 'spawn 需要已提交的父消息调用'
        origin = None
        for message in reversed(source.messages):
            if isinstance(message.body, Input):
                parts = [part for part in message.body.parts if part.kind == "channel.origin"]
                if parts:
                    origin = cast(dict[str, str], json_value(parts[0].value))
                break
        if args.run_in_background and origin is None:
            return '后台子任务需要原会话渠道'
        sink = None
        if args.run_in_background:
            assert origin is not None
            if origin["channel"] not in self.senders:
                return '后台子任务的原渠道没有发送能力'
            sink = SinkInput(name=origin["channel"], binding_id=self.senders[origin["channel"]], address=origin["chat_id"])
        job_id = uuid4().hex
        task_dir = self.ctx.workspace_root("subagent-runs") / job_id
        parent_session_id = source.messages[-1].session_id
        try:
            cwd = await resolve_cwd(self.ctx, parent_session_id, args.cwd, fallback=task_dir)
        except ValueError as error:
            return str(error)
        names = tuple(name for name in PROFILE_TOOLS[args.profile] if name in self.targets)
        if not names:
            return '当前 profile 缺少已安装工具'
        bindings = self.ctx.require(BINDINGS)
        fixed: dict[str, str] = {}
        # 2. 目录设置与实际工具实现都在 prepared 时固定，恢复不重选配置。
        # 参考 DSH：子任务继承父工作目录（或显式 cwd）并在目标目录直接读写执行，不再把 read/write 锁死在 subagent-runs。
        for name in names:
            configuration: Mapping[str, object] | None = None
            if name in {"read_file", "list_dir", "write_file", "edit_file"}:
                configuration = {"allowed_dir": None}
            elif name in {"shell", "write_stdin", "task_stop"}:
                configuration = {"owner_key": "subagent:" + job_id, "working_dir": cwd}
            if configuration is None:
                fixed[name] = self.targets[name]
            else:
                fixed[name] = await self.ctx.require(TOOL_BIND_SAVED)(
                    bindings,
                    self.targets[name],
                    configuration=configuration,
                )
        # 3. 模型在父调用里解析并随请求固定：调用方显式指定 model/connection 优先，否则按设置（如跟随父对话）解析；恢复时不重选。
        model = await self.model(source.messages, args)
        if isinstance(model, str):
            return model
        request = Request(job_id=job_id, label=(args.label or args.task[:30]).strip(), profile=args.profile,
            background=args.run_in_background, retry_count=args.retry_count,
            parent_session_id=parent_session_id, parent_message_id=source.call_ref.message_id,
            parent_part_index=source.call_ref.part_index,
            origin=origin, sink=sink, program_binding=bindings.bind(SUBAGENT_PROGRAM, {}), tools=fixed,
            model_id=model.model_id, reasoning_effort=model.reasoning_effort, cwd=cwd)
        return Prepared(task=args.task, request=request).model_dump()

    async def invoke(self, key: str, arguments: Mapping[str, object]) -> Result:
        prepared = Prepared.model_validate(json_value(arguments))
        try:
            await self.jobs.accept(key, prepared.request, prepared.task)
        except SubagentBusy as error:
            return Result("error", (ContentPart("text", str(error)),))
        if not prepared.request.background:
            task = await self.jobs.start(key)
            if task is not None:
                try:
                    _ = await task.join()
                except asyncio.CancelledError:
                    caller = asyncio.current_task()
                    assert caller is not None
                    # 子 Task 已撤权时不能再把 join 的取消记成用户 pause。
                    if caller.cancelling() and task.active:
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
        outcome = await self.jobs.outcome(reader)
        if outcome is None:
            raise ValueError("子任务结算记录缺少消息终态")
        return Result("success" if outcome[0] == "completed" else "error",
                      (ContentPart("text", completion(request, outcome, limit=self.jobs.result_chars)),))


class Manage:
    idempotent = True

    def __init__(self, jobs: Subagents, *, catalog: Callable[[], ModelCatalogSnapshot] | None = None):
        self.jobs = jobs
        self.catalog = catalog

    async def prepare(self, arguments: Mapping[str, object], source: CallSource | None = None) -> Mapping[str, object] | str:
        try:
            args = ManageInput.model_validate(dict(arguments))
        except ValidationError as error:
            return str(error)
        if args.action == "cancel" and not args.job_id:
            return '取消子任务需要 job_id'
        return args.model_dump()

    async def invoke(self, key: str, arguments: Mapping[str, object]) -> Result:
        args = ManageInput.model_validate(json_value(arguments))
        value: dict[str, object]
        if args.action == "list":
            jobs = self.jobs.jobs()
            value = {"running_count": len(jobs), "jobs": jobs}
        elif args.action == "models":
            if self.catalog is None:
                return Result("error", (ContentPart("text", "当前环境未挂载模型目录"),))
            value = list_available_models(self.catalog(), connection=args.connection)
        else:
            cancelled = await self.jobs.cancel(args.job_id)
            value = {"job_id": args.job_id, "status": "cancel_requested" if cancelled else "not_found"}
        return Result("success", (ContentPart("text", json.dumps(value, ensure_ascii=False)),))

    async def query(self, key: str) -> Result | None:
        return None

"""助手经 `self session ...` 开正式会话、交接原始问题、跨会话对话；与用户在界面上的同名动作一一对应。

边界（见 docs/decisions/0168-agent-sessions-through-self.md）：
- 自动输入由 `self-cli` 作者写入对话来源，不冒充用户，也不走 /stop、/abandon 文本快路径；
- 目标正在回复时不抢占：send 返回 busy，steer 暂存到下一步，用户停止后丢弃不再唤醒；
- 跨会话往返带剩余跳数，从用户原话起最多 MAX_HOPS 次自动往返，禁止发给自己。
"""
from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, replace
import json
from typing import cast
from uuid import NAMESPACE_URL, uuid4, uuid5

from agent.plugin_composition import Context
from core.common.file_io import run_file_io
from plugins.context.contract import MATERIALS_V4
from plugins.ledger.contract import (
    MESSAGE_CATALOG,
    MESSAGE_WRITERS,
    OWNER_STATE,
    SESSION_ADMIN,
    SESSION_ADMISSION,
    ContentPart,
    ContentReferences,
    Control,
    Input,
    Message,
    MessageConflict,
    MessageReader,
    OwnerTransaction,
    Output,
    SessionAttributes,
)
from plugins.operations.contract import (
    OPERATIONS_V1,
    OperationRequest,
    OperationResult,
    OperationSpec,
    OptionSpec,
)
from plugins.sources.contract import SourceBusy, GuardedSourceSession as SourceSession
from plugins.ui.contract import hidden_part, part_display_key

AUTOMATION_AUTHOR = "self-cli"
CROSS_KIND = "cross_session"
HANDOFF_SOURCE = "handoff"
MAX_HOPS = 3
_HANDOFF_PREFIX = "handoff:"
_TEXT_LIMIT = 16_000


@dataclass(frozen=True, slots=True)
class SessionPorts:
    """conversation 交给会话操作的窄入口：按作者打开来源、放弃、暂存插话。"""

    open: Callable[[str], SourceSession]
    open_automated: Callable[[str], SourceSession]
    abandon: Callable[[str, str], Awaitable[Message]]
    stage: Callable[[str, str, Input], Awaitable[bool]]
    withdraw: Callable[[str, str], bool]
    check_text: Callable[[ContentPart], ContentReferences]


def check_cross(part: ContentPart) -> ContentReferences:
    value = part.value
    if not isinstance(value, Mapping):
        raise ValueError("cross_session 必须是对象")
    row = cast(Mapping[str, object], value)
    if set(row) != {"from_session_id", "root", "hops"} or not isinstance(row["hops"], int) \
            or not all(isinstance(row[key], str) and row[key] for key in ("from_session_id", "root")):
        raise ValueError("cross_session 字段无效")
    return ContentReferences()


def _stable_id(req: OperationRequest, kind: str) -> str:
    seed = f"{req.actor}:{req.session_id}:{req.request_id}:{kind}"
    return (uuid5(NAMESPACE_URL, seed) if req.request_id else uuid4()).hex


def _text(value: object, name: str, *, required: bool = False) -> str:
    text = value.strip() if isinstance(value, str) else ""
    if required and not text:
        raise ValueError(f"缺少 --{name}")
    if len(text) > _TEXT_LIMIT:
        raise ValueError(f"--{name} 超过 {_TEXT_LIMIT} 字符")
    return text


def _files(value: object) -> list[dict[str, str]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("--files 需要 JSON 数组，元素为路径字符串或 {path, role}")
    rows: list[dict[str, str]] = []
    for item in cast(list[object], value):
        row = cast(Mapping[str, object], item) if isinstance(item, Mapping) else {"path": item}
        path, role = row.get("path"), row.get("role", "")
        if not isinstance(path, str) or not path.strip() or not isinstance(role, str):
            raise ValueError("--files 元素需要非空 path")
        rows.append({"path": path.strip(), "role": role.strip()})
    return rows


def handoff_markdown(record: Mapping[str, object]) -> str:
    lines = [f"**交接任务**：{record['goal']}"]
    files = cast(list[dict[str, str]], record.get("files") or [])
    if files:
        lines.append("\n**相关文件**")
        lines.extend(f"- `{row['path']}`" + (f" — {row['role']}" if row["role"] else "") for row in files)
    for key, label in (("context", "已知背景"), ("constraints", "约束")):
        if record.get(key):
            lines.append(f"\n**{label}**\n\n{record[key]}")
    source = record.get("source_session_id")
    if isinstance(source, str) and source:
        lines.append(f"\n来源会话：[{source}](?session={source})")
    return "\n".join(lines)


def _link(session_id: str) -> str:
    return f"[打开会话](?session={session_id})"


class SessionOperations:
    """六个会话动作；每个都对应用户界面上的一个入口。"""

    def __init__(self, ctx: Context, ports: SessionPorts):
        self._ctx = ctx
        self._ports = ports
        self._catalog = ctx.require(MESSAGE_CATALOG)
        self._store = ctx.require(OWNER_STATE).open_scoped(ctx, "handoff")
        self._handoff_writer = ctx.require(MESSAGE_WRITERS).bind(
            ctx, author="handoff", source=HANDOFF_SOURCE, body_types=(Output,),
            content={"text": ports.check_text},
        )

    # 目标必须存在、不是调用者自己。
    def _target(self, req: OperationRequest) -> str:
        target = _text(req.args.get("session"), "session", required=True)
        if target == req.session_id:
            raise ValueError("不能对当前会话自己执行跨会话动作")
        if not self._catalog.exists(target):
            raise LookupError(f"会话不存在: {target}")
        return target

    # 新输入的因果片段：继承调用者最近一条自动输入的根与跳数，从用户原话开始则给满预算。
    async def _cross_part(self, req: OperationRequest) -> ContentPart:
        origin = req.session_id or f"{req.actor}:shell"
        hops, root = MAX_HOPS, req.request_id or uuid4().hex
        if req.session_id and self._catalog.exists(req.session_id):
            reader = self._catalog.reader(req.session_id)
            latest = await reader.read_async(lambda r: r.latest_input("conversation", through_seq=r.head()))
            cross = next((p for p in latest.body.parts if isinstance(p, ContentPart) and p.kind == CROSS_KIND), None) \
                if latest is not None and isinstance(latest.body, Input) else None
            if cross is not None:
                value = cast(Mapping[str, object], cross.value)
                hops, root = cast(int, value["hops"]), cast(str, value["root"])
        # 每次自动发送消耗一跳：从用户原话起依次带 2、1、0，第 MAX_HOPS+1 次拒绝。
        hops -= 1
        if hops < 0:
            raise PermissionError("跨会话自动往返预算已用尽；请把结果交给用户决定下一步")
        return ContentPart(CROSS_KIND, {"from_session_id": origin, "root": root, "hops": hops})

    # 1. session open：建正式 listed 会话，写交接卡；start 模式再提交首条自动输入。
    async def open(self, req: OperationRequest) -> OperationResult:
        goal = _text(req.args.get("goal"), "goal", required=True)
        record: dict[str, object] = {
            "goal": goal, "files": _files(req.args.get("files")),
            "context": _text(req.args.get("context"), "context"),
            "constraints": _text(req.args.get("constraints"), "constraints"),
            "source_session_id": req.session_id or "",
        }
        scope = req.args.get("scope") or {}
        if not isinstance(scope, Mapping) or not all(isinstance(v, str) for v in cast(Mapping[str, object], scope).values()):
            raise ValueError("--scope 需要 JSON 对象，如 {\"project\": \"p_x\"}")
        channel = _text(req.args.get("channel"), "channel") or (
            req.session_id.split(":", 1)[0] if req.session_id and ":" in req.session_id else "akashic")
        session_id = f"{channel}:{_stable_id(req, 'open')}"
        attributes = SessionAttributes.scoped(cast(Mapping[str, str], scope))
        # 没给标题时用目标的首行当标题，侧栏不退回交接卡原文。
        title = _text(req.args.get("title"), "title") or goal.splitlines()[0][:40]
        if len(title) > 80:
            raise ValueError("--title 最多 80 字")
        # 参数与跳数预算都在首次持久化之前判定，拒绝时不留下半开的会话。
        if req.args.get("mode") == "start":
            _ = await self._cross_part(req)
        _ = await self._ctx.require(SESSION_ADMISSION).ensure_async(self._ctx, session_id, attributes)
        await self._write_handoff(session_id, record)
        if title:
            with self._ctx.borrow(SESSION_ADMIN) as admin:
                if admin is not None:
                    _ = await admin.set_title_if_unset(session_id, title)
        data: dict[str, object] = {"session_id": session_id, "url": f"?session={session_id}", "mode": req.args.get("mode")}
        if req.args.get("mode") == "start":
            text = "请按本会话的交接卡开始处理；遇到需要用户决定的事先停下说明。"
            accepted = await self._accept(session_id, _stable_id(req, "start"), text, req)
            data["input_id"] = accepted.message_id
        return OperationResult(True, f"已开会话 {session_id}，{_link(session_id)}", data=data)

    # 交接记录与展示卡同一事务提交；同一会话重放只接受相同内容。
    async def _write_handoff(self, session_id: str, record: Mapping[str, object]) -> None:
        writer = self._handoff_writer(session_id)
        self._store.check_access(writer)

        def save(tx: OwnerTransaction) -> None:
            current = tx.read(_HANDOFF_PREFIX + session_id)
            if current is not None:
                if dict(current.value) != dict(record):
                    raise MessageConflict("该会话已有不同的交接记录")
                return
            _ = tx.append(writer, f"handoff:{session_id}", Output((ContentPart("text", handoff_markdown(record)),), "complete"))
            _ = tx.save(_HANDOFF_PREFIX + session_id, record, expected_version=None)

        await self._store.transact_async(save)

    def handoff(self, session_id: str) -> Mapping[str, object] | None:
        current = self._store.read(_HANDOFF_PREFIX + session_id)
        return None if current is None else current.value

    async def _accept(
        self, session_id: str, message_id: str, text: str, req: OperationRequest, extra: tuple[ContentPart, ...] = (),
    ) -> Message:
        # 1. 重放：同一稳定 ID 已提交就返回原消息，不再判忙、不重算跳数（正文须与首次一致）。
        reader = self._catalog.reader(session_id)
        existing = await reader.read_async(lambda r: r.get(message_id))
        if existing is not None:
            return existing
        # 2. 空闲判定与提交在同一次准入里：用户刚开始的回复不会被自动输入撤权。
        body = Input((ContentPart("text", text), *extra, await self._cross_part(req)))
        try:
            return await self._ports.open_automated(session_id).accept(message_id, body, when_idle=True)
        except SourceBusy as error:
            raise MessageConflict("目标会话正在回复；可改用 session steer 在下一步插话，或稍后再发") from error

    # 2. session send：目标空闲时提交自动输入并唤醒回复；可有界等待回复正文。
    async def send(self, req: OperationRequest) -> OperationResult:
        target = self._target(req)
        if req.args.get("retry"):
            reader = self._catalog.reader(target)
            latest = await reader.read_async(lambda r: r.latest_input("conversation", through_seq=r.head()))
            if latest is None:
                raise LookupError("目标会话没有可重试的输入")
            # 重试也消耗跳数；用户主动停止或放弃过的输入不由助手代为恢复。
            _ = await self._cross_part(req)
            if req.actor != "user" and await reader.read_async(lambda r: stopped_after(r, latest.seq)):
                raise PermissionError("用户停止过这条输入；是否继续由用户决定")
            message = await self._ports.open(target).resume(_stable_id(req, "retry"), latest.message_id)
            return OperationResult(True, f"已请求重试 {target}", data={"session_id": target, "message_id": message.message_id})
        text = _text(req.args.get("text"), "text", required=True)
        accepted = await self._accept(target, _stable_id(req, "send"), text, req, _send_parts(req))
        data: dict[str, object] = {"session_id": target, "message_id": accepted.message_id, "seq": accepted.seq}
        wait = req.args.get("wait") or 0
        if isinstance(wait, int) and wait > 0:
            data["reply"] = await self._wait_reply(target, accepted.seq, min(wait, 600))
        return OperationResult(True, f"已发送到 {target}，{_link(target)}", data=data)

    async def _wait_reply(self, session_id: str, after_seq: int, seconds: int) -> str | None:
        reader = self._catalog.reader(session_id)
        loop = asyncio.get_running_loop()
        deadline = loop.time() + seconds
        while loop.time() < deadline:
            rows = await reader.read_async(lambda r: r.read(after_seq=after_seq, source="conversation"))
            for row in rows:
                if isinstance(row.body, Output) and row.body.finish != "continue":
                    return "\n".join(cast(str, p.value) for p in row.body.parts
                                     if isinstance(p, ContentPart) and p.kind == "text")
            await asyncio.sleep(1.0)
        return None

    # 3. session steer：目标正在回复时暂存到它的下一步；不在回复时返回 staged=false。
    async def steer(self, req: OperationRequest) -> OperationResult:
        target = self._target(req)
        text = _text(req.args.get("text"), "text", required=True)
        message_id = _stable_id(req, "steer")
        body = Input((ContentPart("text", text), await self._cross_part(req)))
        staged = await self._ports.stage(target, message_id, body)
        summary = f"已暂存插话 {message_id}" if staged else "目标会话不在回复中，请改用 session send"
        return OperationResult(True, summary, data={"session_id": target, "message_id": message_id, "staged": staged})

    # 4. session withdraw：撤回尚未写入的插话。
    async def withdraw(self, req: OperationRequest) -> OperationResult:
        target = self._target(req)
        message_id = _text(req.args.get("message_id"), "message_id", required=True)
        withdrawn = self._ports.withdraw(target, message_id)
        return OperationResult(True, "已撤回" if withdrawn else "插话已写入或不存在", data={"withdrawn": withdrawn})

    # 5. session stop / 6. session abandon：沿用来源自己的停止与放弃语义。
    async def stop(self, req: OperationRequest) -> OperationResult:
        target = self._target(req)
        message = await self._ports.open(target).pause(_stable_id(req, "stop"), wait=False)
        return OperationResult(True, f"已停止 {target} 当前回复", data={"message_id": message.message_id})

    async def abandon(self, req: OperationRequest) -> OperationResult:
        target = self._target(req)
        message = await self._ports.abandon(target, _stable_id(req, "abandon"))
        return OperationResult(True, f"已放弃 {target} 未完成的工作", data={"message_id": message.message_id})


# 与界面发送同样的附件与引用片段；校验由对话来源的 artifact_ref / reply_ref owner 完成。
def _send_parts(req: OperationRequest) -> tuple[ContentPart, ...]:
    attachments = req.args.get("attachments") or []
    if not isinstance(attachments, list) or not all(isinstance(item, str) and item for item in cast(list[object], attachments)):
        raise ValueError("--attachments 需要附件 ID 的 JSON 数组（attachment add 返回的 artifact_id）")
    parts = tuple(ContentPart("artifact_ref", item) for item in cast(list[str], attachments))
    reply = req.args.get("reply_to")
    return parts + ((ContentPart("reply_ref", reply),) if isinstance(reply, str) and reply else ())


def stopped_after(reader: MessageReader, seq: int) -> bool:
    """用户在 seq 之后停止或放弃过本来源的回合（之后又恢复也算）：自动插话随之作废，不再唤醒。"""
    return any(
        isinstance(row.body, Control) and row.body.action in ("pause", "abandon")
        for row in reader.read(after_seq=seq, source="conversation")
    )


_SESSION = OptionSpec("session", "目标会话 ID", required=True)


def _specs(ops: SessionOperations) -> tuple[OperationSpec, ...]:
    return (
        OperationSpec(
            operation_id="conversation.session_open", path=("session", "open"),
            summary="开一个用户可在侧栏接手的正式会话，并写好交接卡（目标、文件、背景、约束）",
            handler=ops.open, mirrors=("ws:akashic_clients/session.create",),
            options=(
                OptionSpec("goal", "要做什么、预期产出", required=True),
                OptionSpec("title", "会话标题"),
                OptionSpec("files", "JSON 数组：路径字符串或 {path, role}", value_type="json"),
                OptionSpec("context", "已核实的背景、决定、踩过的坑"),
                OptionSpec("constraints", "边界与约束"),
                OptionSpec("scope", "JSON 对象：会话维度，如 {\"project\": \"p_x\"}", value_type="json"),
                OptionSpec("channel", "会话渠道前缀；默认与当前会话相同，网页侧栏为 akashic"),
                OptionSpec("mode", "prepare 只备好等用户；start 立即开始处理", choices=("prepare", "start"), default="prepare"),
            ),
        ),
        OperationSpec(
            operation_id="conversation.session_send", path=("session", "send"),
            summary="给另一个会话发一条消息（以 self-cli 作者身份，不冒充用户）；目标正在回复时返回 busy",
            handler=ops.send, mirrors=("ws:akashic_clients/message.send",),
            options=(
                _SESSION, OptionSpec("text", "消息正文"),
                OptionSpec("attachments", "JSON 数组：随消息发送的附件 ID（attachment add 返回）", value_type="json"),
                OptionSpec("reply_to", "引用回复目标会话里的一条消息 ID"),
                OptionSpec("wait", "最多等待回复的秒数（0 不等）", value_type="int", default=0),
                OptionSpec("retry", "重试目标会话最近一条失败的输入", value_type="bool"),
            ),
        ),
        OperationSpec(
            operation_id="conversation.session_steer", path=("session", "steer"),
            summary="在目标会话正在进行的回复的下一步插话；用户停止后作废",
            handler=ops.steer, mirrors=("ws:akashic_clients/message.steer",),
            options=(_SESSION, OptionSpec("text", "插话正文", required=True)),
        ),
        OperationSpec(
            operation_id="conversation.session_withdraw", path=("session", "withdraw"),
            summary="撤回尚未写入的插话", handler=ops.withdraw,
            mirrors=("ws:akashic_clients/message.steer.withdraw",),
            options=(_SESSION, OptionSpec("message_id", "session steer 返回的 message_id", required=True)),
        ),
        OperationSpec(
            operation_id="conversation.session_stop", path=("session", "stop"),
            summary="停止目标会话当前回复（等同用户发 /stop）", handler=ops.stop,
            mirrors=("slash:stop",), options=(_SESSION,),
        ),
        OperationSpec(
            operation_id="conversation.session_abandon", path=("session", "abandon"),
            summary="放弃目标会话未完成的工作并结算无回执的工具调用（等同 /abandon）",
            handler=ops.abandon, mirrors=("slash:abandon",), options=(_SESSION,),
        ),
    )


async def provide_session_operations(ctx: Context, ports: SessionPorts) -> None:
    """软挂载：有 operations 时注册六个会话动作；有上下文材料时把交接卡交给新会话的模型。"""
    _ = await ctx.provide(part_display_key(CROSS_KIND), hidden_part)
    ops = SessionOperations(ctx, ports)
    # 处理函数用会话插件自己的 Context 接纳会话、写入来源，因此绑定到它的执行入口。
    specs = tuple(replace(spec, handler=ctx.entrypoint(spec.handler)) for spec in _specs(ops))

    async def register(child: Context) -> None:
        operations = child.require(OPERATIONS_V1)
        for spec in specs:
            _ = await operations.register(child, spec)

    async def materials(child: Context) -> None:
        async def prepare(snapshot: tuple[Message, ...], _source: str) -> dict[str, object]:
            record = await run_file_io(lambda: ops.handoff(snapshot[-1].session_id)) if snapshot else None
            if record is None:
                return {}
            text = "本会话由另一个会话交接而来，用户可能只说「继续」。交接内容：\n" + json.dumps(record, ensure_ascii=False)
            return {"reminders": ({"name": "session-handoff", "priority": 380, "replay": False, "text": text},)}

        _ = await child.require(MATERIALS_V4).register(child, name="session-handoff", kind="context", prepare=prepare)

    _ = await ctx.inject((OPERATIONS_V1,), register, name="session-operations")
    _ = await ctx.inject((MATERIALS_V4,), materials, name="session-handoff-materials")

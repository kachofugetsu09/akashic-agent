"""Chat-surface queries owned by the models plugin."""
from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import asdict, dataclass
from typing import cast
from urllib.parse import quote

from core.common.file_io import run_file_io
from plugins.ledger.contract import ContentPart, Control, Input, Message, MessageCatalog, Output
from plugins.models.contract import (
    ChatModelSelection,
    ModelCallStats,
    ModelCatalogSnapshot,
    ModelError,
    ModelUnavailableError,
)
from plugins.ui.contract import PluginUiRpcInvalidRequest

from .selection import read_saved


# 聊天界面经插件界面查询读模型数据；宿主不再持有模型目录与会话选择的投影。
@dataclass(frozen=True)
class ChatUiQueries:
    catalog: Callable[[], Awaitable[ModelCatalogSnapshot]]
    selection: Callable[[ChatModelSelection], Awaitable[ChatModelSelection]]
    messages: MessageCatalog
    call_stats: Callable[[str], ModelCallStats]
    message_tokens: Callable[[str, str], Awaitable[int | None]]
    icon: Callable[[str], Awaitable[str | None]]

    async def __call__(self, method: str, payload: dict[str, object], *,
                       session_id: str | None, turn_id: str | None) -> dict[str, object]:
        if method == "usage.message":
            message_id = payload.get("message_id")
            if set(payload) != {"message_id"} or not session_id or not isinstance(message_id, str) or not message_id:
                raise PluginUiRpcInvalidRequest("用量查询需要会话与消息")
            return {"output_tokens": await self.message_tokens(session_id, message_id)}
        if method == "chat.models":
            if payload:
                raise PluginUiRpcInvalidRequest("模型列表查询不接受参数")
            return await self.chat_models(session_id)
        if method == "call.latest":
            if payload or not session_id:
                raise PluginUiRpcInvalidRequest("最近调用查询需要会话且不接受参数")
            return {"call": await run_file_io(lambda: self._latest_call(session_id))}
        if method == "call.stats":
            call_id = payload.get("call_id")
            if set(payload) != {"call_id"} or not isinstance(call_id, str) or not call_id:
                raise PluginUiRpcInvalidRequest("调用统计查询需要 call_id")
            try:
                return asdict(await run_file_io(lambda: self.call_stats(call_id)))
            except KeyError as error:
                raise PluginUiRpcInvalidRequest("模型调用记录不存在") from error
        raise PluginUiRpcInvalidRequest(f"未知查询: {method}")

    # 当前目录与会话已保存的选择；选择指向的模型暂不可用时照常返回，由界面说明原因。
    async def chat_models(self, session_id: str | None) -> dict[str, object]:
        saved = ChatModelSelection()
        if session_id:
            saved = read_saved(await run_file_io(lambda: self._metadata(session_id)))
            try:
                saved = await self.selection(saved)
            except (RuntimeError, TimeoutError) as error:
                if not ModelError.matches(error, ModelUnavailableError):
                    raise
        snapshot = await self.catalog()
        icons = {item.connection_id: await self.icon(item.connection_id) for item in snapshot.connections}
        return {
            "generationId": snapshot.revision,
            "defaultRuntime": str(snapshot.role_bindings.get("default", "")),
            "sessionOverride": saved.model_id or "",
            "sessionSelection": {"modelRef": saved.model_id or "", "reasoningEffort": saved.reasoning_effort or ""},
            "runtimes": project_chat_runtimes(snapshot, icons),
            "unavailableRuntimes": project_unavailable_chat_runtimes(snapshot),
        }

    # 会话最近一次执行里最后一条带模型事实的回复对应的调用统计；本次没有、尾部找不到或记录已清理时为空。
    def _latest_call(self, session_id: str) -> dict[str, object] | None:
        reader = self.messages.reader(session_id)
        with reader.read_snapshot():
            page = reader.read_tail(limit=_LATEST_CALL_WINDOW)
        for message in reversed(page.messages):
            # 最近一次执行（新 Input 或 resume 起算）以失败、停止收尾或尚无模型产出时为空，不回退到更早的调用。
            if isinstance(message.body, (Input, Control)):
                return None
            call_id = _model_call_id(message)
            if call_id is None:
                continue
            try:
                return asdict(self.call_stats(call_id))
            except KeyError:
                return None
        return None

    def _metadata(self, session_id: str) -> Mapping[str, object]:
        reader = self.messages.reader(session_id)
        with reader.read_snapshot():
            return reader.metadata() or {}


_LATEST_CALL_WINDOW = 50


def _model_call_id(message: Message) -> str | None:
    if not isinstance(message.body, Output):
        return None
    for part in message.body.parts:
        raw: object = part.value if isinstance(part, ContentPart) and part.kind == "model.facts" else None
        if isinstance(raw, Mapping):
            call_id = cast(Mapping[str, object], raw).get("call_record_id")
            return call_id if isinstance(call_id, str) and call_id else None
    return None


# 连接停用或驱动缺席的模型只读展示，不把本地可用误称远端认证成功。
def project_unavailable_chat_runtimes(snapshot: ModelCatalogSnapshot) -> list[dict[str, str]]:
    connections = {item.connection_id: item for item in snapshot.connections}
    return [
        {"id": model.model_id, "model": model.model, "sourceName": connections[model.connection_id].name,
         "availability": model.availability}
        for model in snapshot.models
        if model.kind == "chat" and model.availability != "available"
    ]


# 可用对话模型投影为聊天界面的条目；图标由连接的驱动声明。
def project_chat_runtimes(snapshot: ModelCatalogSnapshot, icons: Mapping[str, str | None]) -> list[dict[str, object]]:
    roles_by_model: dict[str, list[str]] = {}
    for role, model_id in snapshot.role_bindings.items():
        roles_by_model.setdefault(model_id, []).append(role)
    connections = {item.connection_id: item for item in snapshot.connections}
    result: list[dict[str, object]] = []
    for model in snapshot.models:
        if model.kind != "chat" or model.availability != "available":
            continue
        connection = connections[model.connection_id]
        capabilities, sources = model.capabilities, model.capability_sources
        icon = icons.get(connection.connection_id)
        result.append({
            "id": model.model_id,
            "provider": connection.driver_id,
            "model": model.model,
            "reasoningEffort": model.default_reasoning_effort or "",
            "supportedReasoningEfforts": list(capabilities.supported_reasoning_efforts),
            "sourceId": connection.connection_id,
            "sourceName": connection.name,
            "icon": "" if icon is None else "data:image/svg+xml," + quote(icon),
            "contextWindow": capabilities.context_window or 0,
            "maxOutputTokens": capabilities.max_output_tokens or 0,
            "inputModalities": list(capabilities.input_modalities),
            "capabilitySources": {
                "contextWindow": sources.context_window,
                "maxOutputTokens": sources.max_output_tokens,
                "inputModalities": sources.input_modalities,
            },
            "roles": sorted(roles_by_model.get(model.model_id, [])),
        })
    return result

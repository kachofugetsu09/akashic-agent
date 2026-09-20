"""Akashic clients 的窄服务合同。

这个模块只描述客户端插件消费的结构能力。宿主可以用任意实现提供这些
Protocol；客户端插件不导入 bootstrap、Core runtime 或其他插件实现。
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator, Awaitable, Callable, Mapping
from pathlib import Path
from typing import Protocol, runtime_checkable

from agent.plugin_composition.channels import AttachmentRef
from agent.plugin_composition.messages import InvalidPage, MessageConflict
from agent.plugin_composition.models import ChatModelSelection, ModelCatalogSnapshot, ModelCallStats
from agent.plugin_composition.message_view import MessageDisplayReader
from agent.plugin_composition.model_settings_http import ModelControlUnavailable
from agent.plugin_contracts.message import Message


class MessagePagePort(Protocol):
    """只读消息页的字段合同；具体数据库页由宿主适配。"""

    messages: tuple[Message, ...]
    attachments: Mapping[str, tuple[AttachmentRef, ...]]
    bindings: Mapping[str, Mapping[str, object]]
    through_seq: int
    has_more: bool


class SessionEntryPort(Protocol):
    """会话目录条目的只读字段合同。"""

    session_id: str
    created_at: object
    updated_at: object
    message_count: int
    head_seq: int
    first_message: Message | None


class SessionPagePort(Protocol):
    """会话目录页的只读字段合同。"""

    items: tuple[SessionEntryPort, ...]
    total: int
    next_cursor: tuple[str, str] | None


class TurnStartedEvent(Protocol):
    session_key: str
    channel: str
    chat_id: str
    content: str
    timestamp: object
    turn_id: str
    control_turn_id: str
    client_message_id: str


class StreamDeltaReadyEvent(Protocol):
    session_key: str
    channel: str
    chat_id: str
    turn_id: str
    content_delta: str
    thinking_delta: str


class TurnOutputCompletedEvent(Protocol):
    session_key: str
    channel: str
    chat_id: str
    turn_id: str
    client_message_id: str


class ToolCallStartedEvent(Protocol):
    session_key: str
    channel: str
    chat_id: str
    iteration: int
    call_id: str
    tool_name: str
    arguments: dict[str, object]
    turn_id: str


class ToolCallCompletedEvent(Protocol):
    session_key: str
    channel: str
    chat_id: str
    iteration: int
    call_id: str
    tool_name: str
    arguments: dict[str, object]
    final_arguments: dict[str, object]
    status: str
    result_preview: str
    runtime_provenance: dict[str, str]
    turn_id: str


class AttachmentStorePort(Protocol):
    """客户端临时上传与 Mobile 分片共享的文件 owner。"""

    root: Path
    def create_path(self, prefix: str, suffix: str) -> Path: ...
    def create_persistent_path(self, prefix: str, suffix: str) -> Path: ...
    def create_staging_path(self, *, prefix: str = ".upload-", suffix: str = ".part") -> Path: ...
    def publish_staging(self, staging: Path, *, prefix: str, suffix: str) -> Path: ...
    def write_bytes(self, data: bytes, *, prefix: str, suffix: str) -> Path: ...


class ArtifactReadLeasePort(Protocol):
    ref: AttachmentRef
    async def read_bytes(self, *, max_bytes: int) -> bytes: ...
    async def read_chunk(self, *, offset: int, max_bytes: int) -> bytes: ...
    async def aclose(self) -> None: ...


@runtime_checkable
class ArtifactStorePort(Protocol):
    """Core artifact owner 的只读/导入投影。"""

    async def import_bytes(self, data: bytes, *, kind: object, filename: str | None, media_type: str | None) -> AttachmentRef: ...
    def resolve_refs(self, artifact_ids: tuple[str, ...]) -> tuple[AttachmentRef, ...]: ...
    async def acquire(self, ref: AttachmentRef) -> ArtifactReadLeasePort: ...


class MessageReaderPort(Protocol):
    @property
    def session_id(self) -> str: ...
    def head(self) -> int: ...
    def follow(self, *, after_seq: int = -1) -> AsyncGenerator[Message, None]: ...
    def metadata(self) -> Mapping[str, object] | None: ...
    def get(self, message_id: str) -> Message | None: ...
    def read_tail(self, *, before_seq: int | None, through_seq: int | None, limit: int) -> MessagePagePort: ...
    def read_page(self, *, after_seq: int = -1, through_seq: int | None = None, limit: int = 50) -> MessagePagePort: ...


class MessageCatalogPort(Protocol):
    def reader(self, session_id: str) -> MessageReaderPort: ...
    def sessions(self, *, prefix: str, visibility: str, after: tuple[str, str] | None, limit: int) -> SessionPagePort: ...


ReplyStatusPort = Callable[[str], AsyncGenerator[dict[str, object], None]]
ModelCatalogReader = Callable[[], Awaitable[ModelCatalogSnapshot]]
ModelSelectionReader = Callable[[Mapping[str, object]], Awaitable[ChatModelSelection]]
ModelStatsReader = Callable[[str], Awaitable[ModelCallStats]]


class MobileUiProvider(Protocol):
    def catalog(self) -> dict[str, object]: ...
    def asset(self, plugin_id: str, plugin_revision: str, kind: str, sha256: str) -> dict[str, object]: ...
    async def query(self, plugin_id: str, plugin_revision: str, method: str, payload: dict[str, object], *, session_id: str | None, turn_id: str | None) -> dict[str, object]: ...


class MobileUiPluginUnavailable(RuntimeError): ...
class MobileUiQueryOverloaded(RuntimeError): ...
class MobileUiQueryTimeout(RuntimeError): ...
class MobileUiRpcExecutionError(RuntimeError): ...
class MobileUiRpcInvalidRequest(ValueError): ...
class MobileUiStaleRevision(RuntimeError): ...
class ModelCatalogUnavailable(RuntimeError): ...


class RuntimeInspectionError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class RuntimeInspectionService(Protocol):
    async def list_documents(self) -> dict[str, object]: ...
    async def get_document(self, document_id: str) -> dict[str, object]: ...
    async def list_jobs(self) -> dict[str, object]: ...
    async def get_job(self, job_id: str) -> dict[str, object]: ...
    async def list_capabilities(self) -> dict[str, object]: ...
    async def get_mcp(self, owner_id: str, server_name: str) -> dict[str, object]: ...


class WebUiProvider(Protocol):
    async def bootstrap(self) -> bytes: ...
    async def state(self) -> dict[str, str]: ...


class ModelRpcInvoker(Protocol):
    async def invoke_rpc(self, method: str, params: Mapping[str, object]) -> object: ...


class MobilePairingAdminPort(Protocol):
    def create_offer(self) -> dict[str, object]: ...
    def pending_claim(self, pairing_id: str) -> dict[str, object] | None: ...
    def approve(self, pairing_id: str, confirmation_code: str) -> dict[str, object]: ...


def turn_milestone(logger: logging.Logger, event: str, **fields: object) -> None:
    """记录不含正文的客户端时序观察；权威状态仍由宿主 owner 保存。"""
    logger.info(event, extra={"event": event, **fields})


def default_chat_model_id(snapshot: ModelCatalogSnapshot) -> str:
    """读取已声明的 default 绑定，不为客户端编造模型。"""
    return str(snapshot.role_bindings.get("default", ""))


def project_chat_runtimes(snapshot: ModelCatalogSnapshot) -> list[dict[str, object]]:
    """把公共模型目录投影为既有 Web/Mobile DTO。"""
    roles_by_model: dict[str, list[str]] = {}
    for role, model_id in snapshot.role_bindings.items():
        roles_by_model.setdefault(model_id, []).append(role)
    connections = {item.connection_id: item for item in snapshot.connections}
    result: list[dict[str, object]] = []
    for model in snapshot.models:
        if model.kind.value != "chat" or model.availability.value != "available":
            continue
        connection = connections[model.connection_id]
        capabilities = model.capabilities
        sources = model.capability_sources
        result.append({
            "id": model.model_id,
            "provider": connection.driver_id,
            "catalogProvider": connection.driver_id,
            "model": model.model,
            "reasoningEffort": model.default_reasoning_effort or "",
            "supportedReasoningEfforts": list(capabilities.supported_reasoning_efforts),
            "sourceId": connection.connection_id,
            "sourceName": connection.name,
            "contextWindow": capabilities.context_window or 0,
            "maxOutputTokens": capabilities.max_output_tokens or 0,
            "inputModalities": list(capabilities.input_modalities),
            "capabilitySource": sources.context_window,
            "capabilitySources": {
                "contextWindow": sources.context_window,
                "maxOutputTokens": sources.max_output_tokens,
                "inputModalities": sources.input_modalities,
            },
            "roles": sorted(roles_by_model.get(model.model_id, [])),
        })
    return result


__all__ = [
    "ArtifactReadLeasePort", "ArtifactStorePort", "AttachmentStorePort",
    "InvalidPage", "MessageCatalogPort", "MessageConflict", "MessageDisplayReader",
    "MessagePagePort", "MessageReaderPort", "SessionEntryPort", "SessionPagePort", "Message",
    "StreamDeltaReadyEvent", "ToolCallCompletedEvent", "ToolCallStartedEvent", "TurnOutputCompletedEvent",
    "TurnStartedEvent",
    "ModelCatalogReader", "ModelCatalogSnapshot", "ModelCallStats", "ModelControlUnavailable",
    "ModelSelectionReader", "ModelStatsReader", "MobilePairingAdminPort", "RuntimeInspectionError", "RuntimeInspectionService",
    "default_chat_model_id", "project_chat_runtimes", "turn_milestone",
]

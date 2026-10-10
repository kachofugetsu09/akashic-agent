"""Akashic clients 的窄服务合同。

这个模块只描述客户端插件消费的结构能力。宿主可以用任意实现提供这些
Protocol；客户端插件不导入 bootstrap、Core runtime 或其他插件实现。
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, Awaitable, Callable, Mapping
from pathlib import Path
from typing import Literal, Protocol, runtime_checkable

from agent.plugin_composition.channels import AttachmentRef
from agent.plugin_composition.message_view import MessageDisplayReader
from agent.plugin_composition.messages import (
    InvalidPage,
    MessageConflict,
    SessionAttributes,
    SessionDeleteResult,
    SessionTitleResult,
)
from agent.plugin_composition.model_settings_http import ModelControlUnavailable
from agent.plugin_composition.models import (
    ChatModelSelection,
    ModelCatalogSnapshot,
)
from agent.plugin_composition.ui import (
    WebUiProvider as WebUiProvider,
)
from agent.plugin_composition.ui_slots import (
    PluginUiPluginUnavailable,  # noqa: F401 - 显式再导出给本插件消费者。
    PluginUiQueryOverloaded,  # noqa: F401 - 显式再导出给本插件消费者。
    PluginUiQueryTimeout,  # noqa: F401 - 显式再导出给本插件消费者。
    PluginUiRpcExecutionError,  # noqa: F401 - 显式再导出给本插件消费者。
    PluginUiRpcInvalidRequest,  # noqa: F401 - 显式再导出给本插件消费者。
    PluginUiStaleRevision,  # noqa: F401 - 显式再导出给本插件消费者。
)
from agent.plugin_contracts.message import Message
from agent.plugin_contracts.ui import (
    PluginUiProvider as PluginUiProvider,
)


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
    title: str | None


class SessionPagePort(Protocol):
    """会话目录页的只读字段合同。"""

    items: tuple[SessionEntryPort, ...]
    total: int
    next_cursor: tuple[str, str] | None


class AttachmentStorePort(Protocol):
    """客户端临时上传与 旧上传分片共享的文件 owner。"""

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
    def attributes(self) -> SessionAttributes: ...
    @property
    def deleted(self) -> bool: ...
    @property
    def title(self) -> str | None: ...
    @property
    def session_id(self) -> str: ...
    def head(self) -> int: ...
    def follow(self, *, after_seq: int = -1) -> AsyncGenerator[Message, None]: ...
    def metadata(self) -> Mapping[str, object] | None: ...
    def get(self, message_id: str) -> Message | None: ...
    def read_tail(self, *, before_seq: int | None, through_seq: int | None, limit: int) -> MessagePagePort: ...
    def read_page(self, *, after_seq: int = -1, through_seq: int | None = None, limit: int = 50) -> MessagePagePort: ...


class SessionAdminPort(Protocol):
    """用户显式的会话数据管理操作；软删/恢复与标题覆盖，没有消息或物理删除权限。"""

    async def set_deleted(self, session_key: str, *, deleted: bool) -> SessionDeleteResult: ...
    async def set_title(self, session_key: str, title: str | None) -> SessionTitleResult: ...


class MessageCatalogPort(Protocol):
    def reader(self, session_id: str) -> MessageReaderPort: ...
    def sessions(self, *, prefix: str, visibility: str, after: tuple[str, str] | None, limit: int) -> SessionPagePort: ...
    def follow(self, *, poll_interval: float | None = None,
               prefix: str = "", visibility: Literal["listed", "internal"] | None = None,
               ) -> AsyncGenerator[Mapping[str, int], None]: ...


ReplyStatusPort = Callable[[str], AsyncGenerator[dict[str, object], None]]
# 返回 None 表示回复插件未加载，无法判断哪些会话在运行。
ActiveSessionsPort = Callable[[], Awaitable[frozenset[str] | None]]
ActiveSessionsFollowPort = Callable[[], AsyncGenerator[frozenset[str] | None, None]]
ModelCatalogReader = Callable[[], Awaitable[ModelCatalogSnapshot]]
ModelSelectionReader = Callable[[Mapping[str, object]], Awaitable[ChatModelSelection]]




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


def default_chat_model_id(snapshot: ModelCatalogSnapshot) -> str:
    """读取已声明的 default 绑定，不为客户端编造模型。"""
    return str(snapshot.role_bindings.get("default", ""))


def project_unavailable_chat_runtimes(snapshot: ModelCatalogSnapshot) -> list[dict[str, str]]:
    """只读展示连接停用或驱动缺席，不把本地可用误称远端认证成功。"""
    connections = {item.connection_id: item for item in snapshot.connections}
    return [
        {
            "id": model.model_id,
            "model": model.model,
            "sourceName": connections[model.connection_id].name,
            "availability": model.availability.value,
        }
        for model in snapshot.models
        if model.kind.value == "chat" and model.availability.value != "available"
    ]


def project_chat_runtimes(snapshot: ModelCatalogSnapshot) -> list[dict[str, object]]:
    """把公共模型目录投影为既有 Web DTO。"""
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
    "MessagePagePort", "MessageReaderPort", "SessionAdminPort",
    "SessionEntryPort", "SessionPagePort", "Message",
    "ModelCatalogReader", "ModelCatalogSnapshot", "ModelControlUnavailable",
    "ModelSelectionReader", "RuntimeInspectionError", "RuntimeInspectionService",
    "default_chat_model_id", "project_chat_runtimes",
]

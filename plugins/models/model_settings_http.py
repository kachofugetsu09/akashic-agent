from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import asdict
from typing import Annotated, Any, Literal, Protocol

from fastapi import APIRouter, HTTPException, Request
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    RootModel,
    ValidationError,
    model_validator,
)

from agent.plugin_composition.model_settings_http import ModelControlUnavailable
from agent.plugin_composition.models import (
    MODEL_CATALOG,
    MODEL_CALL_STATS,
    AuthenticationError,
    ChatModelSelection,
    CapabilitySources,
    DriverUnavailableError,
    DiscoveredModel,
    ModelCallStats,
    ModelCapabilities,
    ModelCatalogSnapshot,
    ModelError,
    ModelTimeoutError,
    ModelUnavailableError,
    QuotaError,
    RateLimitError,
    RevisionConflictError,
    TransportError,
)
from agent.plugin_composition.model import ServiceKey
from plugins.gateway.contract import RpcMethod

from .settings import (
    AddConnection,
    AddModel,
    CancelConnectionAuth,
    CreateConnectionWithModel,
    DisableConnection,
    FinishConnectionAuth,
    ModelChange,
    MODEL_SETTINGS,
    SetDefaultModel,
    RemoveModel,
    SettingsReceipt,
    StartConnectionAuth,
    SyncModels,
    UpdateConnection,
    UpdateModel,
    VerifyModel,
)
from .selection import MODEL_SELECTION

# 模型失败值由标准异常承载；未知程序错误由映射入口原样抛出。
# Only discover/command accept an HTTPException as an RPC error envelope.
_MODEL_ERRORS = (ModelControlUnavailable, RuntimeError, TimeoutError, ValueError)
_HTTP_MODEL_ERRORS = (HTTPException, *_MODEL_ERRORS)


class ModelControl(Protocol):
    async def probe_embedding(self, model: str, expected_revision: int, *, connection: AddConnection | None = None, connection_id: str | None = None) -> DiscoveredModel: ...

    async def call_stats(self, call_id: str) -> ModelCallStats: ...

    async def catalog(self) -> ModelCatalogSnapshot: ...

    async def discover(
        self, connection: AddConnection
    ) -> tuple[DiscoveredModel, ...]: ...

    async def discover_saved(self, connection_id: str, expected_revision: int) -> tuple[DiscoveredModel, ...]: ...

    async def apply(self, command: ModelChange) -> SettingsReceipt: ...

    async def read_saved(self, metadata: Mapping[str, object]) -> ChatModelSelection: ...


class _ServiceResolver(Protocol):
    """Expose only the request or runtime scope's declared services."""

    def require(self, key: ServiceKey[Any]) -> Any: ...


class BoundModelControl:
    """Resolve Models services through the caller's current scope."""

    def __init__(self, resolver: _ServiceResolver) -> None:
        self._resolver = resolver

    async def call_stats(self, call_id: str) -> ModelCallStats:
        return self._resolver.require(MODEL_CALL_STATS)(call_id)

    async def catalog(self) -> ModelCatalogSnapshot:
        return self._resolver.require(MODEL_CATALOG).snapshot()

    async def discover(
        self,
        connection: AddConnection,
    ) -> tuple[DiscoveredModel, ...]:
        return await self._resolver.require(MODEL_SETTINGS).discover(connection)

    async def discover_saved(self, connection_id: str, expected_revision: int) -> tuple[DiscoveredModel, ...]:
        return await self._resolver.require(MODEL_SETTINGS).discover_saved(connection_id, expected_revision)

    async def probe_embedding(self, model: str, expected_revision: int, *, connection: AddConnection | None = None, connection_id: str | None = None) -> DiscoveredModel:
        return await self._resolver.require(MODEL_SETTINGS).probe_embedding(model, expected_revision, connection=connection, connection_id=connection_id)

    async def apply(self, command: ModelChange) -> SettingsReceipt:
        return await self._resolver.require(MODEL_SETTINGS).apply(command)

    async def read_saved(self, metadata: Mapping[str, object]) -> ChatModelSelection:
        return self._resolver.require(MODEL_SELECTION).read_saved(metadata)


class _Payload(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class ConnectionInput(_Payload):
    expected_revision: int = Field(ge=0)
    connection_id: str = Field(min_length=1, max_length=128)
    name: str = Field(min_length=1, max_length=128)
    driver_id: str = Field(min_length=1, max_length=128)
    endpoint: str = Field(min_length=1, max_length=2048)
    auth_identity: str = Field(min_length=1, max_length=128)
    credential: dict[str, str]
    driver_config: dict[str, JsonValue] = Field(default_factory=dict)


class EmbeddingProbePayload(_Payload):
    expected_revision: int = Field(ge=0)
    model: str = Field(min_length=1, max_length=256)
    connection: ConnectionInput | None = None
    connection_id: str | None = Field(default=None, min_length=1, max_length=128)

    @model_validator(mode="after")
    def check_connection(self) -> EmbeddingProbePayload:
        if (self.connection is None) == (self.connection_id is None):
            raise ValueError("请选择已有连接或填写新连接，两者不能同时使用。")
        if self.connection is not None and self.connection.expected_revision != self.expected_revision:
            raise ValueError("连接与试算的配置版本不同。")
        return self


class AddConnectionPayload(ConnectionInput):
    type: Literal["add_connection"]


class UpdateConnectionPayload(_Payload):
    type: Literal["update_connection"]
    expected_revision: int = Field(ge=0)
    connection_id: str = Field(min_length=1, max_length=128)
    name: str = Field(min_length=1, max_length=128)
    endpoint: str | None = Field(default=None, min_length=1, max_length=2048)
    auth_identity: str = Field(min_length=1, max_length=128)
    credential: dict[str, str] | None = None
    driver_config: dict[str, JsonValue] | None = None


class DisableConnectionPayload(_Payload):
    type: Literal["disable_connection"]
    expected_revision: int = Field(ge=0)
    connection_id: str = Field(min_length=1, max_length=128)


class CapabilitiesPayload(_Payload):
    context_window: int | None = Field(default=None, gt=0)
    max_output_tokens: int | None = Field(default=None, gt=0)
    input_modalities: list[str] = Field(default_factory=lambda: ["text"])
    supports_tool_calls: bool | None = None
    supports_parallel_tool_calls: bool | None = None
    supported_reasoning_efforts: list[str] = Field(default_factory=list)
    embedding_dimensions: int | None = Field(default=None, gt=0)
    embedding_normalization: str | None = None


class CapabilitySourcesPayload(_Payload):
    context_window: str = "unknown"
    max_output_tokens: str = "unknown"
    input_modalities: str = "unknown"
    tool_calls: str = "unknown"
    parallel_tool_calls: str = "unknown"
    reasoning_efforts: str = "unknown"
    embedding_dimensions: str = "unknown"
    embedding_normalization: str = "unknown"


class ModelInput(_Payload):
    expected_revision: int = Field(ge=0)
    model_id: str = Field(min_length=1, max_length=128)
    connection_id: str = Field(min_length=1, max_length=128)
    kind: Literal["chat", "embedding"]
    model: str = Field(min_length=1, max_length=256)
    capabilities: CapabilitiesPayload
    capability_sources: CapabilitySourcesPayload
    default_reasoning_effort: str | None = Field(default=None, max_length=32)
    driver_config: dict[str, JsonValue] = Field(default_factory=dict)
    make_default_embedding: bool = False
    discovery_owned: bool = False

    @model_validator(mode="after")
    def check_default_kind(self) -> ModelInput:
        if self.make_default_embedding and self.kind != "embedding":
            raise ValueError("只有向量模型可以设为默认向量模型。")
        return self


class AddModelPayload(ModelInput):
    type: Literal["add_model"]


class SetDefaultPayload(_Payload):
    type: Literal["set_default"]
    expected_revision: int = Field(ge=0)
    role: Literal["default", "fast", "agent", "vision"] | None
    model_id: str = Field(min_length=1, max_length=128)
    verify_embedding: bool = False

    @model_validator(mode="after")
    def check_verification_kind(self) -> SetDefaultPayload:
        if self.verify_embedding and self.role is not None:
            raise ValueError("向量重验只能用于默认向量模型。")
        return self


class SavedDiscoveryPayload(_Payload):
    expected_revision: int = Field(ge=0)
    connection_id: str = Field(min_length=1, max_length=128)


class VerifyModelPayload(_Payload):
    type: Literal["verify_model"]
    expected_revision: int = Field(ge=0)
    model_id: str = Field(min_length=1, max_length=128)


class RemoveModelPayload(_Payload):
    type: Literal["remove_model"]
    expected_revision: int = Field(ge=0)
    model_id: str = Field(min_length=1, max_length=128)


class UpdateModelPayload(_Payload):
    type: Literal["update_model"]
    expected_revision: int = Field(ge=0)
    model_id: str = Field(min_length=1, max_length=128)
    context_window: int | None = Field(default=None, gt=0)
    max_output_tokens: int | None = Field(default=None, gt=0)
    image_input: bool


class SyncModelsPayload(_Payload):
    type: Literal["sync_models"]
    expected_revision: int = Field(ge=0)
    connection_id: str = Field(min_length=1, max_length=128)


class StartAuthPayload(_Payload):
    type: Literal["start_auth"]
    driver_id: str = Field(min_length=1, max_length=128)
    connection_id: str = Field(min_length=1, max_length=128)
    input: dict[str, str] = Field(default_factory=dict)


class FinishAuthPayload(_Payload):
    type: Literal["finish_auth"]
    expected_revision: int = Field(ge=0)
    attempt_id: str = Field(min_length=1, max_length=256)


class CancelAuthPayload(_Payload):
    type: Literal["cancel_auth"]
    attempt_id: str = Field(min_length=1, max_length=256)


class CreateConnectionWithModelPayload(_Payload):
    type: Literal["create_connection_with_model"]
    connection: ConnectionInput
    model: ModelInput


CommandPayload = Annotated[
    AddConnectionPayload
    | UpdateConnectionPayload
    | DisableConnectionPayload
    | AddModelPayload
    | VerifyModelPayload
    | RemoveModelPayload
    | UpdateModelPayload
    | SetDefaultPayload
    | SyncModelsPayload
    | StartAuthPayload
    | FinishAuthPayload
    | CancelAuthPayload
    | CreateConnectionWithModelPayload,
    Field(discriminator="type"),
]


class CallStatsParams(_Payload):
    """Parameters for the read-only call statistics RPC."""

    call_id: str = Field(min_length=1, max_length=256)


class EmptyParams(_Payload):
    """Parameters for an RPC that takes no business input."""


class CommandParams(RootModel[CommandPayload]):
    """Keep the command wire body unchanged while using RpcMethod."""


def _validation_detail(error: ValueError | ValidationError) -> object:
    if isinstance(error, ValidationError):
        return error.errors(include_input=False, include_context=False)
    return [{"type": "json_invalid", "msg": "JSON 无效"}]


async def _read_payload[Payload: BaseModel](
    request: Request, payload_type: type[Payload],
) -> Payload:
    """Validate JSON without reflecting credentials or validation context."""
    try:
        return payload_type.model_validate(await request.json())
    except (ValueError, ValidationError) as error:
        raise HTTPException(status_code=422, detail=_validation_detail(error)) from error


def _rpc_ok(body: Mapping[str, object]) -> dict[str, object]:
    return {"status": 200, "body": dict(body)}


def _rpc_error(error: HTTPException) -> dict[str, object]:
    return {
        "status": error.status_code,
        "body": {"detail": error.detail},
    }


def _http_error(error: Exception, *, operation: str) -> HTTPException:
    """Map model owner failures to the long-standing HTTP contract."""

    if isinstance(error, HTTPException):
        return error
    if operation == "call_stats" and isinstance(error, KeyError):
        return HTTPException(status_code=404, detail="模型调用不存在")
    if ModelError.matches(error, RevisionConflictError):
        return HTTPException(status_code=409, detail=str(error))
    if ModelError.matches(error, AuthenticationError):
        return HTTPException(status_code=401, detail=str(error))
    if ModelError.matches(error, RateLimitError):
        return HTTPException(status_code=429, detail=str(error))
    if ModelError.matches(error, QuotaError):
        return HTTPException(status_code=402, detail=str(error))
    if (ModelError.matches(error, DriverUnavailableError) or isinstance(error, ModelControlUnavailable)):
        return HTTPException(status_code=503, detail=str(error))
    if ModelError.matches(error, ModelUnavailableError):
        return HTTPException(status_code=409, detail=str(error))
    if ModelError.matches(error, ModelTimeoutError):
        return HTTPException(status_code=504, detail=str(error))
    if ModelError.matches(error, TransportError):
        return HTTPException(status_code=502, detail=str(error))
    if (ModelError.matches(error) or isinstance(error, ValueError)):
        return HTTPException(status_code=422, detail=str(error))
    raise error


async def _call_stats_body(control: ModelControl, call_id: str) -> dict[str, object]:
    return asdict(await control.call_stats(call_id))


async def _catalog_body(control: ModelControl) -> dict[str, object]:
    return _catalog_payload(await control.catalog())


async def _discover_body(
    control: ModelControl,
    payload: ConnectionInput,
) -> dict[str, object]:
    models = await control.discover(_add_connection(payload))
    return {"models": [_discovered_payload(model) for model in models]}


async def _discover_saved_body(
    control: ModelControl, payload: SavedDiscoveryPayload,
) -> dict[str, object]:
    models = await control.discover_saved(payload.connection_id, payload.expected_revision)
    return {"models": [_discovered_payload(model) for model in models]}


async def _command_body(
    control: ModelControl,
    payload: CommandPayload,
) -> dict[str, object]:
    receipt = await control.apply(_command(payload))
    return _receipt_payload(receipt)


def create_model_settings_router(
    control: ModelControl,
    *,
    prefix: str,
) -> APIRouter:
    """Expose the models plugin's provider-neutral HTTP contract."""

    router = APIRouter(prefix=prefix)

    @router.get("/calls/{call_id}")
    async def call_stats(call_id: str) -> dict[str, object]:
        try:
            return await _call_stats_body(control, call_id)
        except (KeyError, ModelControlUnavailable) as error:
            raise _http_error(error, operation="call_stats") from error

    @router.get("/catalog")
    async def catalog() -> dict[str, object]:
        try:
            return await _catalog_body(control)
        except ModelControlUnavailable as error:
            raise _http_error(error, operation="catalog") from error

    @router.post("/discover")
    async def discover(request: Request) -> dict[str, object]:
        payload = await _read_payload(request, ConnectionInput)
        try:
            return await _discover_body(control, payload)
        except _HTTP_MODEL_ERRORS as error:
            raise _http_error(error, operation="discover") from error

    @router.post("/discover_saved")
    async def discover_saved(request: Request) -> dict[str, object]:
        payload = await _read_payload(request, SavedDiscoveryPayload)
        try:
            return await _discover_saved_body(control, payload)
        except _MODEL_ERRORS as error:
            raise _http_error(error, operation="discover") from error

    @router.post("/probe_embedding")
    async def probe_embedding(request: Request) -> dict[str, object]:
        payload = await _read_payload(request, EmbeddingProbePayload)
        try:
            return await _embedding_probe_body(control, payload)
        except _MODEL_ERRORS as error:
            raise _http_error(error, operation="discover") from error

    @router.post("/command")
    async def command(request: Request) -> dict[str, object]:
        payload = (await _read_payload(request, CommandParams)).root
        try:
            return await _command_body(control, payload)
        except _HTTP_MODEL_ERRORS as error:
            raise _http_error(error, operation="command") from error

    return router


def _rpc_method[Payload: BaseModel](
    payload_type: type[Payload],
    invoke: Callable[[Payload], Awaitable[dict[str, object]]],
    *,
    errors: tuple[type[Exception], ...],
    operation: str,
) -> RpcMethod:
    """Adapt one typed operation, preserving its exact error boundary."""
    async def handle(params: BaseModel) -> object:
        assert isinstance(params, payload_type)
        try:
            return _rpc_ok(await invoke(params))
        except errors as error:
            return _rpc_error(_http_error(error, operation=operation))

    return RpcMethod(payload_type, handle)


def rpc_methods(control: ModelControl) -> dict[str, RpcMethod]:
    """Publish model HTTP operations as plugin-owned RPC methods."""
    return {
        "models/call_stats": _rpc_method(
            CallStatsParams, lambda params: _call_stats_body(control, params.call_id),
            errors=(KeyError, ModelControlUnavailable), operation="call_stats",
        ),
        "models/catalog": _rpc_method(
            EmptyParams, lambda params: _catalog_body(control),
            errors=(ModelControlUnavailable,), operation="catalog",
        ),
        "models/discover": _rpc_method(
            ConnectionInput, lambda params: _discover_body(control, params),
            errors=_HTTP_MODEL_ERRORS, operation="discover",
        ),
        "models/discover_saved": _rpc_method(
            SavedDiscoveryPayload, lambda params: _discover_saved_body(control, params),
            errors=_MODEL_ERRORS, operation="discover",
        ),
        "models/probe_embedding": _rpc_method(
            EmbeddingProbePayload, lambda params: _embedding_probe_body(control, params),
            errors=_MODEL_ERRORS, operation="discover",
        ),
        "models/command": _rpc_method(
            CommandParams, lambda params: _command_body(control, params.root),
            errors=_HTTP_MODEL_ERRORS, operation="command",
        ),
    }


def _command(payload: CommandPayload) -> ModelChange:
    if isinstance(payload, AddConnectionPayload):
        return _add_connection(payload)
    if isinstance(payload, UpdateConnectionPayload):
        return UpdateConnection(
            expected_revision=payload.expected_revision,
            connection_id=payload.connection_id,
            name=payload.name,
            endpoint=payload.endpoint,
            auth_identity=payload.auth_identity,
            credential=payload.credential,
            driver_config=payload.driver_config,
        )
    if isinstance(payload, DisableConnectionPayload):
        return DisableConnection(payload.expected_revision, payload.connection_id)
    if isinstance(payload, AddModelPayload):
        return _add_model(payload)
    if isinstance(payload, VerifyModelPayload):
        return VerifyModel(expected_revision=payload.expected_revision, model_id=payload.model_id)
    if isinstance(payload, RemoveModelPayload):
        return RemoveModel(payload.expected_revision, payload.model_id)
    if isinstance(payload, UpdateModelPayload):
        return UpdateModel(
            expected_revision=payload.expected_revision,
            model_id=payload.model_id,
            context_window=payload.context_window,
            max_output_tokens=payload.max_output_tokens,
            image_input=payload.image_input,
        )
    if isinstance(payload, SetDefaultPayload):
        return SetDefaultModel(
            payload.expected_revision,
            payload.role,
            payload.model_id,
            payload.verify_embedding,
        )
    if isinstance(payload, SyncModelsPayload):
        return SyncModels(payload.expected_revision, payload.connection_id)
    if isinstance(payload, StartAuthPayload):
        return StartConnectionAuth(
            payload.driver_id,
            payload.connection_id,
            payload.input,
        )
    if isinstance(payload, FinishAuthPayload):
        return FinishConnectionAuth(payload.expected_revision, payload.attempt_id)
    if isinstance(payload, CancelAuthPayload):
        return CancelConnectionAuth(payload.attempt_id)
    if isinstance(payload, CreateConnectionWithModelPayload):
        return CreateConnectionWithModel(
            connection=_add_connection(payload.connection),
            model=_add_model(payload.model),
        )
    raise AssertionError(f"unhandled command payload: {type(payload).__name__}")


def _add_connection(payload: ConnectionInput) -> AddConnection:
    return AddConnection(
        expected_revision=payload.expected_revision,
        connection_id=payload.connection_id,
        name=payload.name,
        driver_id=payload.driver_id,
        endpoint=payload.endpoint,
        auth_identity=payload.auth_identity,
        credential=payload.credential,
        driver_config=payload.driver_config,
    )


async def _embedding_probe_body(control: ModelControl, payload: EmbeddingProbePayload) -> dict[str, object]:
    result = await control.probe_embedding(
        payload.model, payload.expected_revision,
        connection=None if payload.connection is None else _add_connection(payload.connection),
        connection_id=payload.connection_id,
    )
    return {"model": _discovered_payload(result), "revision": payload.expected_revision}


def _add_model(payload: ModelInput) -> AddModel:
    return AddModel(
        expected_revision=payload.expected_revision,
        model_id=payload.model_id,
        connection_id=payload.connection_id,
        kind=payload.kind,
        model=payload.model,
        capabilities=ModelCapabilities(
            **{
                **payload.capabilities.model_dump(),
                "input_modalities": tuple(payload.capabilities.input_modalities),
                "supported_reasoning_efforts": tuple(
                    payload.capabilities.supported_reasoning_efforts
                ),
            }
        ),
        capability_sources=CapabilitySources(**payload.capability_sources.model_dump()),
        make_default_embedding=payload.make_default_embedding,
        discovery_owned=payload.discovery_owned,
        default_reasoning_effort=payload.default_reasoning_effort,
        driver_config=payload.driver_config,
    )


def _capabilities_payload(capabilities: ModelCapabilities) -> dict[str, object]:
    return {
        "contextWindow": capabilities.context_window,
        "maxOutputTokens": capabilities.max_output_tokens,
        "inputModalities": list(capabilities.input_modalities),
        "supportsToolCalls": capabilities.supports_tool_calls,
        "supportsParallelToolCalls": capabilities.supports_parallel_tool_calls,
        "supportedReasoningEfforts": list(capabilities.supported_reasoning_efforts),
        "embeddingDimensions": capabilities.embedding_dimensions,
        "embeddingNormalization": capabilities.embedding_normalization,
    }


def _capability_sources_payload(sources: CapabilitySources) -> dict[str, object]:
    return {
        "contextWindow": sources.context_window,
        "maxOutputTokens": sources.max_output_tokens,
        "inputModalities": sources.input_modalities,
        "toolCalls": sources.tool_calls,
        "parallelToolCalls": sources.parallel_tool_calls,
        "reasoningEfforts": sources.reasoning_efforts,
        "embeddingDimensions": sources.embedding_dimensions,
        "embeddingNormalization": sources.embedding_normalization,
    }


def _catalog_payload(snapshot: ModelCatalogSnapshot) -> dict[str, object]:
    return {
        "revision": snapshot.revision,
        "connections": [
            {
                "id": item.connection_id,
                "name": item.name,
                "driverId": item.driver_id,
                "authIdentity": item.auth_identity,
                "availability": item.availability,
            }
            for item in snapshot.connections
        ],
        "models": [
            {
                "id": item.model_id,
                "connectionId": item.connection_id,
                "kind": item.kind,
                "model": item.model,
                "defaultReasoningEffort": item.default_reasoning_effort,
                "availability": item.availability,
                "capabilities": _capabilities_payload(item.capabilities),
                "capabilitySources": _capability_sources_payload(item.capability_sources),
            }
            for item in snapshot.models
        ],
        "roleBindings": {
            role: model_id for role, model_id in snapshot.role_bindings.items()
        },
        "defaultEmbeddingModelId": snapshot.default_embedding_model_id,
    }


def _discovered_payload(model: DiscoveredModel) -> dict[str, object]:
    """Project an unsaved provider model without inventing a registry ID."""

    return {
        "kind": model.kind if model.kind is not None else None,
        "model": model.model,
        "defaultReasoningEffort": model.default_reasoning_effort,
        "capabilities": _capabilities_payload(model.capabilities),
        "capabilitySources": _capability_sources_payload(model.capability_sources),
        "driverConfig": _json_value(model.driver_config),
    }


def _receipt_payload(receipt: SettingsReceipt) -> dict[str, object]:
    return {
        "revision": receipt.revision,
        "status": receipt.status,
        "attemptId": receipt.attempt_id,
        "challenge": _json_value(receipt.challenge),
    }


def _json_value(value: object) -> object:
    """Restore frozen domain JSON to ordinary HTTP response containers."""

    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    return value


__all__ = [
    "CallStatsParams",
    "CommandParams",
    "ConnectionInput",
    "EmptyParams",
    "ModelControl",
    "create_model_settings_router",
    "rpc_methods",
]

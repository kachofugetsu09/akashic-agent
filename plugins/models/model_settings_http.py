from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
from typing import Annotated, Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    RootModel,
    ValidationError,
)

from agent.plugin_composition.model_settings_http import (
    ModelControl,
    ModelControlUnavailable,
)
from agent.plugin_composition.models import (
    AddConnection,
    AddModel,
    AuthenticationError,
    CancelConnectionAuth,
    CapabilitySources,
    CreateConnectionWithModel,
    DisableConnection,
    DriverUnavailableError,
    DiscoveredModel,
    FinishConnectionAuth,
    ModelCapabilities,
    ModelCatalogSnapshot,
    ModelChange,
    ModelError,
    ModelKind,
    ModelTimeoutError,
    ModelUnavailableError,
    QuotaError,
    RateLimitError,
    RevisionConflictError,
    SetDefaultModel,
    SettingsReceipt,
    StartConnectionAuth,
    SyncModels,
    TransportError,
    UpdateConnection,
)
from agent.plugin_composition.rpc import RpcMethod


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


class AddModelPayload(ModelInput):
    type: Literal["add_model"]


class SetDefaultPayload(_Payload):
    type: Literal["set_default"]
    expected_revision: int = Field(ge=0)
    role: Literal["default", "fast", "agent", "vision"] | None
    model_id: str = Field(min_length=1, max_length=128)


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
    if isinstance(error, RevisionConflictError):
        return HTTPException(status_code=409, detail=str(error))
    if isinstance(error, AuthenticationError):
        return HTTPException(status_code=401, detail=str(error))
    if isinstance(error, RateLimitError):
        return HTTPException(status_code=429, detail=str(error))
    if isinstance(error, QuotaError):
        return HTTPException(status_code=402, detail=str(error))
    if isinstance(error, (ModelControlUnavailable, DriverUnavailableError)):
        return HTTPException(status_code=503, detail=str(error))
    if isinstance(error, ModelUnavailableError):
        return HTTPException(status_code=409, detail=str(error))
    if isinstance(error, ModelTimeoutError):
        return HTTPException(status_code=504, detail=str(error))
    if isinstance(error, TransportError):
        return HTTPException(status_code=502, detail=str(error))
    if isinstance(error, (ModelError, ValueError)):
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
    if payload.driver_id != "openai-compatible":
        raise HTTPException(
            status_code=422,
            detail="模型预览仅支持 openai-compatible",
        )
    models = await control.discover(_add_connection(payload))
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
    prefix: str = "/api/chat/model-settings",
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
        try:
            payload = ConnectionInput.model_validate(await request.json())
        except (ValueError, ValidationError) as error:
            raise HTTPException(status_code=422, detail=_validation_detail(error)) from error
        try:
            return await _discover_body(control, payload)
        except (
            HTTPException,
            AuthenticationError,
            RateLimitError,
            QuotaError,
            ModelControlUnavailable,
            DriverUnavailableError,
            ModelUnavailableError,
            ModelTimeoutError,
            TransportError,
            ModelError,
            ValueError,
        ) as error:
            raise _http_error(error, operation="discover") from error

    @router.post("/command")
    async def command(request: Request) -> dict[str, object]:
        try:
            payload = CommandParams.model_validate(await request.json()).root
        except (ValueError, ValidationError) as error:
            raise HTTPException(status_code=422, detail=_validation_detail(error)) from error
        try:
            return await _command_body(control, payload)
        except (
            HTTPException,
            RevisionConflictError,
            AuthenticationError,
            RateLimitError,
            QuotaError,
            ModelControlUnavailable,
            DriverUnavailableError,
            ModelUnavailableError,
            ModelTimeoutError,
            TransportError,
            ModelError,
            ValueError,
        ) as error:
            raise _http_error(error, operation="command") from error

    return router


def rpc_methods(control: ModelControl) -> dict[str, RpcMethod]:
    """Publish model HTTP operations as plugin-owned RPC methods."""

    async def call_stats(params: BaseModel) -> object:
        assert isinstance(params, CallStatsParams)
        try:
            return _rpc_ok(await _call_stats_body(control, params.call_id))
        except (KeyError, ModelControlUnavailable) as error:
            return _rpc_error(_http_error(error, operation="call_stats"))

    async def catalog(params: BaseModel) -> object:
        assert isinstance(params, EmptyParams)
        try:
            return _rpc_ok(await _catalog_body(control))
        except ModelControlUnavailable as error:
            return _rpc_error(_http_error(error, operation="catalog"))

    async def discover(params: BaseModel) -> object:
        assert isinstance(params, ConnectionInput)
        try:
            return _rpc_ok(await _discover_body(control, params))
        except (
            HTTPException,
            AuthenticationError,
            RateLimitError,
            QuotaError,
            ModelControlUnavailable,
            DriverUnavailableError,
            ModelUnavailableError,
            ModelTimeoutError,
            TransportError,
            ModelError,
            ValueError,
        ) as error:
            return _rpc_error(_http_error(error, operation="discover"))

    async def command(params: BaseModel) -> object:
        assert isinstance(params, CommandParams)
        try:
            return _rpc_ok(await _command_body(control, params.root))
        except (
            HTTPException,
            RevisionConflictError,
            AuthenticationError,
            RateLimitError,
            QuotaError,
            ModelControlUnavailable,
            DriverUnavailableError,
            ModelUnavailableError,
            ModelTimeoutError,
            TransportError,
            ModelError,
            ValueError,
        ) as error:
            return _rpc_error(_http_error(error, operation="command"))

    return {
        "models/call_stats": RpcMethod(CallStatsParams, call_stats),
        "models/catalog": RpcMethod(EmptyParams, catalog),
        "models/discover": RpcMethod(ConnectionInput, discover),
        "models/command": RpcMethod(CommandParams, command),
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
    if isinstance(payload, SetDefaultPayload):
        return SetDefaultModel(
            payload.expected_revision,
            payload.role,
            payload.model_id,
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


def _add_model(payload: ModelInput) -> AddModel:
    return AddModel(
        expected_revision=payload.expected_revision,
        model_id=payload.model_id,
        connection_id=payload.connection_id,
        kind=ModelKind(payload.kind),
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
        default_reasoning_effort=payload.default_reasoning_effort,
        driver_config=payload.driver_config,
    )


def _catalog_payload(snapshot: ModelCatalogSnapshot) -> dict[str, object]:
    return {
        "revision": snapshot.revision,
        "connections": [
            {
                "id": item.connection_id,
                "name": item.name,
                "driverId": item.driver_id,
                "authIdentity": item.auth_identity,
                "availability": item.availability.value,
            }
            for item in snapshot.connections
        ],
        "models": [
            {
                "id": item.model_id,
                "connectionId": item.connection_id,
                "kind": item.kind.value,
                "model": item.model,
                "defaultReasoningEffort": item.default_reasoning_effort,
                "availability": item.availability.value,
                "capabilities": {
                    "contextWindow": item.capabilities.context_window,
                    "maxOutputTokens": item.capabilities.max_output_tokens,
                    "inputModalities": list(item.capabilities.input_modalities),
                    "supportsToolCalls": item.capabilities.supports_tool_calls,
                    "supportsParallelToolCalls": (
                        item.capabilities.supports_parallel_tool_calls
                    ),
                    "supportedReasoningEfforts": list(
                        item.capabilities.supported_reasoning_efforts
                    ),
                    "embeddingDimensions": (item.capabilities.embedding_dimensions),
                    "embeddingNormalization": (
                        item.capabilities.embedding_normalization
                    ),
                },
                "capabilitySources": {
                    "contextWindow": item.capability_sources.context_window,
                    "maxOutputTokens": item.capability_sources.max_output_tokens,
                    "inputModalities": item.capability_sources.input_modalities,
                    "toolCalls": item.capability_sources.tool_calls,
                    "parallelToolCalls": (item.capability_sources.parallel_tool_calls),
                    "reasoningEfforts": item.capability_sources.reasoning_efforts,
                    "embeddingDimensions": (
                        item.capability_sources.embedding_dimensions
                    ),
                    "embeddingNormalization": (
                        item.capability_sources.embedding_normalization
                    ),
                },
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
        "kind": model.kind.value,
        "model": model.model,
        "defaultReasoningEffort": model.default_reasoning_effort,
        "capabilities": {
            "contextWindow": model.capabilities.context_window,
            "maxOutputTokens": model.capabilities.max_output_tokens,
            "inputModalities": list(model.capabilities.input_modalities),
            "supportsToolCalls": model.capabilities.supports_tool_calls,
            "supportsParallelToolCalls": model.capabilities.supports_parallel_tool_calls,
            "supportedReasoningEfforts": list(
                model.capabilities.supported_reasoning_efforts
            ),
            "embeddingDimensions": model.capabilities.embedding_dimensions,
            "embeddingNormalization": model.capabilities.embedding_normalization,
        },
        "capabilitySources": {
            "contextWindow": model.capability_sources.context_window,
            "maxOutputTokens": model.capability_sources.max_output_tokens,
            "inputModalities": model.capability_sources.input_modalities,
            "toolCalls": model.capability_sources.tool_calls,
            "parallelToolCalls": model.capability_sources.parallel_tool_calls,
            "reasoningEfforts": model.capability_sources.reasoning_efforts,
            "embeddingDimensions": model.capability_sources.embedding_dimensions,
            "embeddingNormalization": model.capability_sources.embedding_normalization,
        },
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

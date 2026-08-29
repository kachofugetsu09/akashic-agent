from __future__ import annotations

from typing import Annotated, Literal, Protocol

from fastapi import APIRouter, HTTPException, Request
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    TypeAdapter,
    ValidationError,
)

from agent.plugin_composition import (
    AddConnection,
    AddModel,
    AuthenticationError,
    CancelConnectionAuth,
    CapabilitySources,
    CreateConnectionWithModel,
    DisableConnection,
    DriverUnavailableError,
    FinishConnectionAuth,
    ModelCapabilities,
    ModelCatalogSnapshot,
    ModelChange,
    ModelError,
    ModelKind,
    ModelTimeoutError,
    ModelRole,
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
from agent.plugins.model_control import ModelControlUnavailable


class ModelControl(Protocol):
    async def catalog(self) -> ModelCatalogSnapshot: ...

    async def apply(self, command: ModelChange) -> SettingsReceipt: ...


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

_COMMAND_ADAPTER = TypeAdapter(CommandPayload)


def create_model_settings_router(control: ModelControl) -> APIRouter:
    """Expose provider-neutral model catalog and settings commands over HTTP."""

    router = APIRouter(prefix="/api/chat/model-settings")

    @router.get("/catalog")
    async def catalog() -> dict[str, object]:
        try:
            return _catalog_payload(await control.catalog())
        except ModelControlUnavailable as error:
            raise HTTPException(status_code=503, detail=str(error)) from error

    @router.post("/command")
    async def command(request: Request) -> dict[str, object]:
        try:
            payload = _COMMAND_ADAPTER.validate_python(await request.json())
        except (ValueError, ValidationError) as error:
            detail = (
                error.errors(include_input=False, include_context=False)
                if isinstance(error, ValidationError)
                else [{"type": "json_invalid", "msg": "JSON 无效"}]
            )
            raise HTTPException(status_code=422, detail=detail) from error
        try:
            receipt = await control.apply(_command(payload))
        except RevisionConflictError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        except AuthenticationError as error:
            raise HTTPException(status_code=401, detail=str(error)) from error
        except RateLimitError as error:
            raise HTTPException(status_code=429, detail=str(error)) from error
        except QuotaError as error:
            raise HTTPException(status_code=402, detail=str(error)) from error
        except (ModelControlUnavailable, DriverUnavailableError) as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        except ModelUnavailableError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        except ModelTimeoutError as error:
            raise HTTPException(status_code=504, detail=str(error)) from error
        except TransportError as error:
            raise HTTPException(status_code=502, detail=str(error)) from error
        except (ModelError, ValueError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        return _receipt_payload(receipt)

    return router


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
            None if payload.role is None else ModelRole(payload.role),
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
            role.value: model_id for role, model_id in snapshot.role_bindings.items()
        },
        "defaultEmbeddingModelId": snapshot.default_embedding_model_id,
    }


def _receipt_payload(receipt: SettingsReceipt) -> dict[str, object]:
    return {
        "revision": receipt.revision,
        "status": receipt.status,
        "attemptId": receipt.attempt_id,
        "challenge": receipt.challenge,
    }


__all__ = ["ModelControl", "create_model_settings_router"]

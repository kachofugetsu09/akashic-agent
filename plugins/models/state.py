from __future__ import annotations

import hashlib
import json
import math
import secrets
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import asdict
from types import MappingProxyType
from typing import Any, AsyncIterator, Mapping, Sequence, cast

from agent.plugin_composition import (
    AddConnection,
    AddModel,
    BoundChatModel,
    BoundEmbeddingModel,
    BoundModelDescriptor,
    CancelConnectionAuth,
    ChatModelSelection,
    ConnectionDescriptor,
    Context,
    DisableConnection,
    DriverConnection,
    DriverConnectionDescriptor,
    DriverChatModel,
    DriverEmbeddingModel,
    DriverUnavailableError,
    Effect,
    EmbeddingResult,
    EmbeddingSpaceDescriptor,
    FinishConnectionAuth,
    LLMResponse,
    MODEL_DRIVERS,
    ModelAvailability,
    ModelCatalogSnapshot,
    ModelChange,
    ModelDescriptor,
    ModelDriverDefinition,
    ModelExecution,
    ModelKind,
    ModelRequest,
    ModelRole,
    ModelUnavailableError,
    SetDefaultModel,
    SettingsReceipt,
    SnapshotSealing,
    StartConnectionAuth,
    SyncModels,
    UpdateConnection,
    ValidatedChatModelSelection,
    lease_current_runtime_snapshot,
)

from .store import ModelsStore, StoredConnection, StoredModel, StoredSnapshot


class _BoundChat:
    def __init__(
        self,
        descriptor: BoundModelDescriptor,
        driver: DriverChatModel,
    ) -> None:
        self._descriptor = descriptor
        self._driver = driver

    @property
    def descriptor(self) -> BoundModelDescriptor:
        return self._descriptor

    async def complete(self, request: ModelRequest) -> LLMResponse:
        continuation = request.continuation
        if continuation is not None and continuation.binding_id != self._descriptor.binding_id:
            raise ModelUnavailableError("continuation 不属于当前 model binding")
        return await self._driver.complete(request)

    def estimate_context_tokens(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]] = (),
    ) -> int:
        return self._driver.estimate_context_tokens(messages, tools)

    def estimate_appended_message_tokens(
        self,
        messages: Sequence[Mapping[str, Any]],
    ) -> int:
        return self._driver.estimate_appended_message_tokens(messages)

    @property
    def max_tool_schemas(self) -> int | None:
        return self._driver.max_tool_schemas


class _BoundEmbedding:
    def __init__(
        self,
        descriptor: EmbeddingSpaceDescriptor,
        driver: DriverEmbeddingModel,
    ) -> None:
        self._descriptor = descriptor
        self._driver = driver

    @property
    def descriptor(self) -> EmbeddingSpaceDescriptor:
        return self._descriptor

    async def embed(self, texts: Sequence[str]) -> EmbeddingResult:
        result = await self._driver.embed(texts)
        if any(len(vector) != self._descriptor.dimensions for vector in result.vectors):
            raise ModelUnavailableError("embedding 返回维度与绑定空间不一致")
        return result


class _Execution:
    def __init__(
        self,
        state: ModelsState,
        model_id: str | None,
        reasoning_effort: str | None,
        chat: Mapping[ModelRole, BoundChatModel],
        embedding: BoundEmbeddingModel | None,
    ) -> None:
        self.state = state
        self.model_id = model_id
        self.reasoning_effort = reasoning_effort
        self._chat = MappingProxyType(dict(chat))
        self._embedding = embedding

    def chat(self, role: ModelRole) -> BoundChatModel:
        try:
            return self._chat[role]
        except KeyError as exc:
            raise ModelUnavailableError(f"模型角色不可用: {role.value}") from exc

    def embedding(self) -> BoundEmbeddingModel:
        if self._embedding is None:
            raise ModelUnavailableError("尚未配置默认 embedding 模型")
        return self._embedding


_CURRENT_EXECUTION: ContextVar[_Execution | None] = ContextVar(
    "models_current_execution",
    default=None,
)


class _DriversView:
    def __init__(self, state: ModelsState) -> None:
        self._state = state

    async def register(
        self,
        ctx: Context,
        definition: ModelDriverDefinition,
    ) -> Effect:
        return await self._state.register_driver(ctx, definition)


class _ChatModelsView:
    def __init__(self, state: ModelsState) -> None:
        self._state = state

    def execution(
        self,
        *,
        model_id: str | None = None,
        reasoning_effort: str | None = None,
    ):
        return self._state.execution(model_id, reasoning_effort)


class _EmbeddingsView:
    def __init__(self, state: ModelsState) -> None:
        self._state = state

    def bind(self, *, model_id: str | None = None):
        return self._state.embedding_scope(model_id)


class _CatalogView:
    def __init__(self, state: ModelsState) -> None:
        self._state = state

    def snapshot(self) -> ModelCatalogSnapshot:
        return self._state.catalog_snapshot()

    def validate_chat_selection(
        self,
        selection: ChatModelSelection,
    ) -> ValidatedChatModelSelection:
        return self._state.validate_chat_selection(selection)


class _SettingsView:
    def __init__(self, state: ModelsState) -> None:
        self._state = state

    async def apply(self, command: ModelChange) -> SettingsReceipt:
        return await self._state.apply_change(command)


class _MemoryCredential:
    def __init__(
        self,
        connection_id: str,
        auth_identity: str,
        payload: Mapping[str, str],
    ) -> None:
        self.connection_id = connection_id
        self.auth_identity = auth_identity
        self._payload = dict(payload)

    async def read(self) -> Mapping[str, str]:
        return MappingProxyType(dict(self._payload))

    async def refresh(self, payload: Mapping[str, str]) -> None:
        self._payload = dict(payload)

    @asynccontextmanager
    async def exclusive(self) -> AsyncIterator[None]:
        yield


class ModelsState:
    """Own one model revision store and one Root-local frozen driver registry."""

    def __init__(self, store: ModelsStore, *, root_instance_token: object) -> None:
        self.store = store
        self.root_instance_token = root_instance_token
        self._driver_registrations: dict[str, ModelDriverDefinition] = {}
        self._drivers: Mapping[str, ModelDriverDefinition] = MappingProxyType({})
        self.sealed = False
        self.drivers = _DriversView(self)
        self.chat_models = _ChatModelsView(self)
        self.embeddings = _EmbeddingsView(self)
        self.catalog = _CatalogView(self)
        self.settings = _SettingsView(self)
        self._auth_attempts: dict[str, tuple[str, str, Mapping[str, Any]]] = {}

    async def register_driver(
        self,
        ctx: Context,
        definition: ModelDriverDefinition,
    ) -> Effect:
        """Register one driver as an Effect of its provider Fiber."""

        if (
            ctx.root_instance_token is not self.root_instance_token
            or ctx.require(MODEL_DRIVERS) is not self.drivers
        ):
            raise RuntimeError("model driver 与 MODEL_DRIVERS 不属于同一个 Root")
        if not definition.driver_id.strip() or not definition.contract_version.strip():
            raise ValueError("model driver identity 不能为空")

        def setup():
            if self.sealed:
                raise RuntimeError("model driver registry 已封印")
            if definition.driver_id in self._driver_registrations:
                raise ValueError(f"model driver 重复注册: {definition.driver_id}")
            self._driver_registrations[definition.driver_id] = definition

            def cleanup() -> None:
                current = self._driver_registrations.get(definition.driver_id)
                if current is definition:
                    del self._driver_registrations[definition.driver_id]

            return cleanup

        return await ctx.effect(setup, label=f"model-driver:{definition.driver_id}")

    async def seal(self, _event: SnapshotSealing) -> None:
        """Freeze registrations after checking committed config readability."""

        if self.sealed:
            raise RuntimeError("model driver registry 重复封印")
        snapshot = self.store.read_snapshot()
        for connection in (() if snapshot is None else snapshot.connections.values()):
            if not connection.enabled:
                continue
            definition = self._driver_registrations.get(connection.driver_id)
            if definition is None:
                continue
            await definition.open(
                _driver_connection_descriptor(connection),
                self.store.credential_handle(
                    connection.connection_id,
                    connection.auth_identity,
                ),
            )
        self._drivers = MappingProxyType(dict(self._driver_registrations))
        self.sealed = True

    def catalog_snapshot(self) -> ModelCatalogSnapshot:
        snapshot = self._snapshot_or_empty()
        connections = tuple(
            ConnectionDescriptor(
                connection_id=item.connection_id,
                name=item.name,
                driver_id=item.driver_id,
                endpoint=item.endpoint,
                auth_identity=item.auth_identity,
                availability=self._availability(item),
            )
            for item in snapshot.connections.values()
        )
        models = tuple(self._model_descriptor(snapshot, item) for item in snapshot.models.values())
        return ModelCatalogSnapshot(
            revision=snapshot.revision,
            connections=connections,
            models=models,
            role_bindings={ModelRole(role): model_id for role, model_id in snapshot.role_bindings.items()},
            default_embedding_model_id=snapshot.default_embedding_model_id,
        )

    def validate_chat_selection(
        self,
        selection: ChatModelSelection,
    ) -> ValidatedChatModelSelection:
        if selection.reasoning_effort and selection.model_id is None:
            raise ValueError("推理强度必须绑定显式模型")
        if selection.model_id is None:
            return ValidatedChatModelSelection(None, None)
        snapshot = self._snapshot_required()
        model = snapshot.models.get(selection.model_id)
        if model is None or model.kind is not ModelKind.CHAT or not model.enabled:
            raise ModelUnavailableError(f"聊天模型不可用: {selection.model_id}")
        connection = snapshot.connections[model.connection_id]
        if self._availability(connection) is not ModelAvailability.AVAILABLE:
            raise ModelUnavailableError(f"聊天模型连接不可用: {selection.model_id}")
        efforts = model.capabilities.supported_reasoning_efforts
        if selection.reasoning_effort and efforts and selection.reasoning_effort not in efforts:
            raise ValueError(f"模型不支持推理强度: {selection.reasoning_effort}")
        return ValidatedChatModelSelection(selection.model_id, selection.reasoning_effort)

    @asynccontextmanager
    async def execution(
        self,
        model_id: str | None,
        reasoning_effort: str | None,
    ) -> AsyncIterator[ModelExecution]:
        existing = _CURRENT_EXECUTION.get()
        if existing is not None:
            if existing.state is not self:
                raise RuntimeError("同一执行不能绑定两个 models Service")
            if existing.model_id != model_id or existing.reasoning_effort != reasoning_effort:
                raise RuntimeError("嵌套 model execution 选择冲突")
            yield existing
            return
        selection = self.validate_chat_selection(ChatModelSelection(model_id, reasoning_effort))
        lease = lease_current_runtime_snapshot()
        try:
            snapshot = self._snapshot_required()
            execution = await self._build_execution(
                lease.snapshot.snapshot_id,
                snapshot,
                selection.model_id,
                selection.reasoning_effort,
            )
            token = _CURRENT_EXECUTION.set(execution)
            try:
                yield execution
            finally:
                _CURRENT_EXECUTION.reset(token)
        finally:
            await lease.release()

    @asynccontextmanager
    async def embedding_scope(
        self,
        model_id: str | None,
    ) -> AsyncIterator[BoundEmbeddingModel]:
        if _CURRENT_EXECUTION.get() is not None:
            raise RuntimeError("Turn 内必须使用当前 ModelExecution.embedding()")
        lease = lease_current_runtime_snapshot()
        try:
            snapshot = self._snapshot_required()
            selected = model_id or snapshot.default_embedding_model_id
            if selected is None:
                raise ModelUnavailableError("尚未配置默认 embedding 模型")
            bound = await self._bind_embedding(
                lease.snapshot.snapshot_id,
                snapshot,
                selected,
            )
            yield bound
        finally:
            await lease.release()

    async def _build_execution(
        self,
        plugin_snapshot_id: str,
        snapshot: StoredSnapshot,
        explicit_model_id: str | None,
        reasoning_effort: str | None,
    ) -> _Execution:
        chat: dict[ModelRole, BoundChatModel] = {}
        opened: dict[str, DriverConnection] = {}
        for role in ModelRole:
            model_id = snapshot.role_bindings.get(role.value)
            binding_role = role.value
            if explicit_model_id is not None and role in {ModelRole.DEFAULT, ModelRole.AGENT}:
                model_id = explicit_model_id
            if model_id is None:
                if role is ModelRole.DEFAULT:
                    raise ModelUnavailableError("尚未配置 default 聊天模型")
                model_id = snapshot.role_bindings.get(ModelRole.DEFAULT.value)
                binding_role = ModelRole.DEFAULT.value
            if model_id is None:
                continue
            effort = (
                reasoning_effort
                if explicit_model_id
                and role in {ModelRole.DEFAULT, ModelRole.AGENT}
                else snapshot.role_reasoning_efforts.get(binding_role)
                or snapshot.models[model_id].default_reasoning_effort
            )
            chat[role] = await self._bind_chat(
                plugin_snapshot_id,
                snapshot,
                model_id,
                role,
                effort,
                opened,
            )
        embedding = None
        if snapshot.default_embedding_model_id is not None:
            embedding = await self._bind_embedding(
                plugin_snapshot_id,
                snapshot,
                snapshot.default_embedding_model_id,
                opened,
            )
        return _Execution(self, explicit_model_id, reasoning_effort, chat, embedding)

    async def _bind_chat(
        self,
        plugin_snapshot_id: str,
        snapshot: StoredSnapshot,
        model_id: str,
        role: ModelRole,
        effort: str | None,
        opened: dict[str, DriverConnection],
    ) -> BoundChatModel:
        model = snapshot.models.get(model_id)
        if model is None or model.kind is not ModelKind.CHAT or not model.enabled:
            raise ModelUnavailableError(f"聊天模型不可用: {model_id}")
        connection = snapshot.connections[model.connection_id]
        definition, driver = await self._open_driver(connection, opened)
        capability_digest = _capability_digest(model)
        descriptor = BoundModelDescriptor(
            binding_id=_binding_id(
                plugin_snapshot_id,
                snapshot.revision,
                definition.contract_version,
                connection,
                model,
                effort,
                capability_digest,
            ),
            plugin_snapshot_id=plugin_snapshot_id,
            model_revision=snapshot.revision,
            model_id=model.model_id,
            connection_id=connection.connection_id,
            driver_id=connection.driver_id,
            driver_contract_version=definition.contract_version,
            auth_identity=connection.auth_identity,
            model=model.model,
            role=role,
            reasoning_effort=effort,
            capabilities=model.capabilities,
            capability_sources=model.capability_sources,
            capability_digest=capability_digest,
        )
        return _BoundChat(
            descriptor,
            driver.bind_chat(descriptor, model.driver_config),
        )

    async def _bind_embedding(
        self,
        plugin_snapshot_id: str,
        snapshot: StoredSnapshot,
        model_id: str,
        opened: dict[str, DriverConnection] | None = None,
    ) -> BoundEmbeddingModel:
        model = snapshot.models.get(model_id)
        if model is None or model.kind is not ModelKind.EMBEDDING or not model.enabled:
            raise ModelUnavailableError(f"embedding 模型不可用: {model_id}")
        dimensions = model.capabilities.embedding_dimensions
        if dimensions is None or dimensions <= 0:
            raise ModelUnavailableError(f"embedding 模型缺少 dimensions: {model_id}")
        connection = snapshot.connections[model.connection_id]
        definition, driver = await self._open_driver(connection, opened or {})
        descriptor = EmbeddingSpaceDescriptor(
            plugin_snapshot_id=plugin_snapshot_id,
            model_revision=snapshot.revision,
            model_id=model.model_id,
            connection_id=connection.connection_id,
            driver_id=connection.driver_id,
            driver_contract_version=definition.contract_version,
            auth_identity=connection.auth_identity,
            connection_fingerprint=_connection_fingerprint(connection),
            model=model.model,
            dimensions=dimensions,
            normalization=model.capabilities.embedding_normalization or "none",
            capability_digest=_capability_digest(model),
        )
        return _BoundEmbedding(
            descriptor,
            driver.bind_embedding(descriptor, model.driver_config),
        )

    async def _open_driver(
        self,
        connection: StoredConnection,
        opened: dict[str, DriverConnection],
    ) -> tuple[ModelDriverDefinition, DriverConnection]:
        if not connection.enabled:
            raise ModelUnavailableError(f"模型连接已禁用: {connection.connection_id}")
        definition = self._drivers.get(connection.driver_id)
        if definition is None:
            raise DriverUnavailableError(f"model driver 不可用: {connection.driver_id}")
        driver = opened.get(connection.connection_id)
        if driver is None:
            driver = await definition.open(
                _driver_connection_descriptor(connection),
                self.store.credential_handle(connection.connection_id, connection.auth_identity),
            )
            opened[connection.connection_id] = driver
        return definition, driver

    async def apply_change(self, command: ModelChange) -> SettingsReceipt:
        """Keep the exact driver generation alive across settings network I/O."""

        lease = lease_current_runtime_snapshot()
        try:
            root = lease.snapshot.composition_root
            if root is None or root.context.root_instance_token is not self.root_instance_token:
                raise RuntimeError("model settings 不属于当前 runtime snapshot")
            return await self._apply_change(command)
        finally:
            await lease.release()

    async def _apply_change(self, command: ModelChange) -> SettingsReceipt:
        if not self.sealed:
            raise RuntimeError("models settings 只能使用已发布 snapshot")
        if isinstance(command, AddConnection):
            await self._probe_new_connection(command)
            revision = self.store.add_connection(command)
        elif isinstance(command, UpdateConnection):
            await self._probe_updated_connection(command)
            revision = self.store.update_connection(command)
        elif isinstance(command, DisableConnection):
            revision = self.store.disable_connection(command)
        elif isinstance(command, AddModel):
            await self._check_model(command)
            revision = self.store.add_model(command)
        elif isinstance(command, SetDefaultModel):
            revision = self.store.set_default(command)
        elif isinstance(command, SyncModels):
            revision = await self._sync_models(command)
        elif isinstance(command, StartConnectionAuth):
            return await self._start_auth(command)
        elif isinstance(command, FinishConnectionAuth):
            return await self._finish_auth(command)
        elif isinstance(command, CancelConnectionAuth):
            return await self._cancel_auth(command)
        else:
            raise TypeError(f"不支持的 ModelChange: {type(command).__name__}")
        return SettingsReceipt(revision=revision, status="committed")

    async def _sync_models(self, command: SyncModels) -> int:
        """Discover outside SQLite, then publish one catalog revision with CAS."""

        snapshot = self._snapshot_required()
        connection = snapshot.connections.get(command.connection_id)
        if connection is None or not connection.enabled:
            raise ModelUnavailableError(f"模型连接不可用: {command.connection_id}")
        definition = self._driver_required(connection.driver_id)
        if definition.discover is None:
            raise ValueError(f"driver 不支持模型发现: {connection.driver_id}")
        discovered = await definition.discover(
            _driver_connection_descriptor(connection),
            self.store.credential_handle(
                connection.connection_id,
                connection.auth_identity,
            ),
        )
        return self.store.sync_models(
            command.expected_revision,
            connection.connection_id,
            discovered,
        )

    async def _probe_new_connection(self, command: AddConnection) -> None:
        definition = self._driver_required(command.driver_id)
        descriptor = DriverConnectionDescriptor(
            connection_id=command.connection_id,
            name=command.name,
            driver_id=command.driver_id,
            endpoint=command.endpoint,
            auth_identity=command.auth_identity,
            config=command.driver_config,
        )
        credential = _MemoryCredential(command.connection_id, command.auth_identity, command.credential)
        if definition.probe is not None:
            await definition.probe(descriptor, credential)
        else:
            await definition.open(descriptor, credential)

    async def _probe_updated_connection(self, command: UpdateConnection) -> None:
        snapshot = self._snapshot_required()
        existing = snapshot.connections.get(command.connection_id)
        if existing is None:
            raise ModelUnavailableError(f"模型连接不存在: {command.connection_id}")
        definition = self._driver_required(existing.driver_id)
        descriptor = DriverConnectionDescriptor(
            connection_id=existing.connection_id,
            name=command.name,
            driver_id=existing.driver_id,
            endpoint=command.endpoint,
            auth_identity=command.auth_identity,
            config=(
                command.driver_config
                if command.driver_config is not None
                else existing.driver_config
            ),
        )
        credential = (
            _MemoryCredential(existing.connection_id, command.auth_identity, command.credential)
            if command.credential is not None
            else self.store.credential_handle(existing.connection_id, command.auth_identity)
        )
        if definition.probe is not None:
            await definition.probe(descriptor, credential)
        else:
            await definition.open(descriptor, credential)

    async def _check_model(self, command: AddModel) -> None:
        snapshot = self._snapshot_required()
        connection = snapshot.connections.get(command.connection_id)
        if connection is None:
            raise ModelUnavailableError(f"模型连接不存在: {command.connection_id}")
        definition, driver = await self._open_driver(connection, {})
        model = StoredModel.from_command(command)
        if command.kind is ModelKind.CHAT:
            descriptor = self._temporary_chat_descriptor(snapshot, connection, model, definition)
            _ = driver.bind_chat(descriptor, model.driver_config)
        else:
            descriptor = self._temporary_embedding_descriptor(snapshot, connection, model, definition)
            _ = driver.bind_embedding(descriptor, model.driver_config)

    async def _start_auth(self, command: StartConnectionAuth) -> SettingsReceipt:
        definition = self._driver_required(command.driver_id)
        if definition.start_auth is None:
            raise ValueError(f"driver 不支持登录: {command.driver_id}")
        result = await definition.start_auth(dict(command.input))
        attempt_id = secrets.token_urlsafe(18)
        state, challenge = _auth_state_and_challenge(result)
        self._auth_attempts[attempt_id] = (
            command.driver_id,
            command.connection_id,
            state,
        )
        return SettingsReceipt(
            revision=self._snapshot_or_empty().revision,
            status="pending",
            attempt_id=attempt_id,
            challenge=cast(Mapping[str, Any] | None, challenge),
        )

    async def _finish_auth(self, command: FinishConnectionAuth) -> SettingsReceipt:
        attempt = self._auth_attempts.get(command.attempt_id)
        if attempt is None:
            raise ValueError(f"auth attempt 不存在: {command.attempt_id}")
        driver_id, _connection_id, state = attempt
        definition = self._driver_required(driver_id)
        if definition.finish_auth is None:
            raise ValueError(f"driver 不支持完成登录: {driver_id}")
        result = await definition.finish_auth(state)
        if str(result.get("status") or "") != "complete":
            next_state, challenge = _auth_state_and_challenge(result)
            self._auth_attempts[command.attempt_id] = (
                driver_id,
                _connection_id,
                next_state,
            )
            return SettingsReceipt(
                revision=self._snapshot_or_empty().revision,
                status="pending",
                attempt_id=command.attempt_id,
                challenge=cast(Mapping[str, Any] | None, challenge),
            )
        connection = _auth_connection_fields(result)
        current = self.store.read_snapshot()
        existing = None if current is None else current.connections.get(_connection_id)
        if existing is None:
            change: AddConnection | UpdateConnection = AddConnection(
                expected_revision=command.expected_revision,
                connection_id=_connection_id,
                name=connection["name"],
                driver_id=driver_id,
                endpoint=connection["endpoint"],
                auth_identity=connection["auth_identity"],
                credential=connection["credential"],
                driver_config=connection["driver_config"],
            )
            await self._probe_new_connection(change)
            revision = self.store.add_connection(change)
        else:
            if existing.driver_id != driver_id:
                raise ValueError("auth driver 与已有 connection 不一致")
            change = UpdateConnection(
                expected_revision=command.expected_revision,
                connection_id=_connection_id,
                name=connection["name"],
                endpoint=connection["endpoint"],
                auth_identity=connection["auth_identity"],
                credential=connection["credential"],
                driver_config=connection["driver_config"],
            )
            await self._probe_updated_connection(change)
            revision = self.store.update_connection(change)
        del self._auth_attempts[command.attempt_id]
        return SettingsReceipt(revision=revision, status="committed")

    async def _cancel_auth(self, command: CancelConnectionAuth) -> SettingsReceipt:
        attempt = self._auth_attempts.pop(command.attempt_id, None)
        if attempt is None:
            raise ValueError(f"auth attempt 不存在: {command.attempt_id}")
        driver_id, _connection_id, state = attempt
        definition = self._driver_required(driver_id)
        if definition.cancel_auth is not None:
            await definition.cancel_auth(state)
        return SettingsReceipt(
            revision=self._snapshot_or_empty().revision,
            status="cancelled",
            attempt_id=command.attempt_id,
        )

    def _driver_required(self, driver_id: str) -> ModelDriverDefinition:
        definition = self._drivers.get(driver_id)
        if definition is None:
            raise DriverUnavailableError(f"model driver 不可用: {driver_id}")
        return definition

    def _snapshot_required(self) -> StoredSnapshot:
        snapshot = self.store.read_snapshot()
        if snapshot is None:
            raise ModelUnavailableError("尚未配置任何模型")
        return snapshot

    def _snapshot_or_empty(self) -> StoredSnapshot:
        return self.store.read_snapshot() or StoredSnapshot.empty()

    def _availability(self, connection: StoredConnection) -> ModelAvailability:
        if not connection.enabled:
            return ModelAvailability.DISABLED
        if connection.driver_id not in self._drivers:
            return ModelAvailability.DRIVER_UNAVAILABLE
        return ModelAvailability.AVAILABLE

    def _model_descriptor(self, snapshot: StoredSnapshot, model: StoredModel) -> ModelDescriptor:
        connection = snapshot.connections[model.connection_id]
        availability = self._availability(connection)
        if not model.enabled:
            availability = ModelAvailability.DISABLED
        return ModelDescriptor(
            model_id=model.model_id,
            connection_id=model.connection_id,
            kind=model.kind,
            model=model.model,
            default_reasoning_effort=model.default_reasoning_effort,
            capabilities=model.capabilities,
            capability_sources=model.capability_sources,
            availability=availability,
        )

    def _temporary_chat_descriptor(
        self,
        snapshot: StoredSnapshot,
        connection: StoredConnection,
        model: StoredModel,
        definition: ModelDriverDefinition,
    ) -> BoundModelDescriptor:
        digest = _capability_digest(model)
        return BoundModelDescriptor(
            binding_id="settings-probe",
            plugin_snapshot_id="settings-probe",
            model_revision=snapshot.revision,
            model_id=model.model_id,
            connection_id=connection.connection_id,
            driver_id=connection.driver_id,
            driver_contract_version=definition.contract_version,
            auth_identity=connection.auth_identity,
            model=model.model,
            role=ModelRole.DEFAULT,
            reasoning_effort=None,
            capabilities=model.capabilities,
            capability_sources=model.capability_sources,
            capability_digest=digest,
        )

    def _temporary_embedding_descriptor(
        self,
        snapshot: StoredSnapshot,
        connection: StoredConnection,
        model: StoredModel,
        definition: ModelDriverDefinition,
    ) -> EmbeddingSpaceDescriptor:
        dimensions = model.capabilities.embedding_dimensions
        if dimensions is None or dimensions <= 0:
            raise ValueError("embedding dimensions 必须大于 0")
        return EmbeddingSpaceDescriptor(
            plugin_snapshot_id="settings-probe",
            model_revision=snapshot.revision,
            model_id=model.model_id,
            connection_id=connection.connection_id,
            driver_id=connection.driver_id,
            driver_contract_version=definition.contract_version,
            auth_identity=connection.auth_identity,
            connection_fingerprint=_connection_fingerprint(connection),
            model=model.model,
            dimensions=dimensions,
            normalization=model.capabilities.embedding_normalization or "none",
            capability_digest=_capability_digest(model),
        )


def _driver_connection_descriptor(connection: StoredConnection) -> DriverConnectionDescriptor:
    return DriverConnectionDescriptor(
        connection_id=connection.connection_id,
        name=connection.name,
        driver_id=connection.driver_id,
        endpoint=connection.endpoint,
        auth_identity=connection.auth_identity,
        config=connection.driver_config,
    )


def _capability_digest(model: StoredModel) -> str:
    payload = {
        "capabilities": asdict(model.capabilities),
        "sources": asdict(model.capability_sources),
        "driver_config": _plain_json(model.driver_config),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:20]


def _connection_fingerprint(connection: StoredConnection) -> str:
    payload = {
        "endpoint": connection.endpoint,
        "config": _plain_json(connection.driver_config),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:20]


def _plain_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_json(item) for item in value]
    return value


def _binding_id(
    plugin_snapshot_id: str,
    revision: int,
    contract_version: str,
    connection: StoredConnection,
    model: StoredModel,
    effort: str | None,
    capability_digest: str,
) -> str:
    value = "\0".join(
        (
            plugin_snapshot_id,
            str(revision),
            connection.driver_id,
            contract_version,
            connection.connection_id,
            connection.auth_identity,
            model.model_id,
            effort or "",
            capability_digest,
        )
    )
    return hashlib.sha256(value.encode()).hexdigest()[:24]


def _auth_connection_fields(result: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the provider-neutral fields returned by a completed login."""

    fields: dict[str, Any] = {}
    for name in ("name", "endpoint", "auth_identity"):
        value = result.get(name)
        if not isinstance(value, str) or not value.strip():
            raise RuntimeError(f"driver auth 缺少 {name}")
        fields[name] = value.strip()
    credential = result.get("credential")
    if not isinstance(credential, Mapping) or not credential:
        raise RuntimeError("driver auth 缺少 credential")
    if not all(isinstance(key, str) and isinstance(value, str) for key, value in credential.items()):
        raise RuntimeError("driver auth credential 必须只含 string")
    driver_config = result.get("driver_config", {})
    if not isinstance(driver_config, Mapping):
        raise RuntimeError("driver auth driver_config 必须是 object")
    fields["credential"] = cast(Mapping[str, str], credential)
    fields["driver_config"] = cast(Mapping[str, Any], driver_config)
    return fields


def _auth_state_and_challenge(
    result: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Mapping[str, Any] | None]:
    state = result.get("state")
    if not isinstance(state, Mapping):
        raise RuntimeError("driver auth 必须返回 private state object")
    challenge = result.get("challenge")
    if challenge is not None and not isinstance(challenge, Mapping):
        raise RuntimeError("driver auth challenge 必须是 object")
    return (
        cast(Mapping[str, Any], _freeze_json(state)),
        (
            None
            if challenge is None
            else cast(Mapping[str, Any], _freeze_json(challenge))
        ),
    )


def _freeze_json(value: Any) -> Any:
    active: set[int] = set()

    def freeze(item: Any) -> Any:
        if isinstance(item, Mapping):
            identity = id(item)
            if identity in active:
                raise ValueError("auth state 不允许循环引用")
            active.add(identity)
            try:
                frozen: dict[str, Any] = {}
                for key, nested in item.items():
                    if not isinstance(key, str):
                        raise TypeError("auth state key 必须是 string")
                    frozen[key] = freeze(nested)
                return MappingProxyType(frozen)
            finally:
                active.remove(identity)
        if isinstance(item, (list, tuple)):
            identity = id(item)
            if identity in active:
                raise ValueError("auth state 不允许循环引用")
            active.add(identity)
            try:
                return tuple(freeze(nested) for nested in item)
            finally:
                active.remove(identity)
        if isinstance(item, float) and not math.isfinite(item):
            raise ValueError("auth state number 必须是有限值")
        if item is None or isinstance(item, (str, int, float, bool)):
            return item
        raise TypeError(f"auth state 不是 JSON value: {type(item).__name__}")

    return freeze(value)

"""Small real Models fixture for material execution tests."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent.plugin_composition import (
    CompositionRoot,
    EmbeddingResult,
    LLMResponse,
    ModelRequest,
)
from plugins.models.contract import (
    BoundModelDescriptor,
    CapabilitySources,
    DriverConnection,
    DriverConnectionDescriptor,
    EmbeddingSpaceDescriptor,
    ModelCapabilities,
)
from plugins.models.contract import EMBEDDINGS
from plugins.models.contract import (
    CHAT_MODELS,
    MODEL_DRIVERS,
    ModelDriverDefinition,
)
from plugins.models.settings import (
    AddConnection,
    AddModel,
    CreateConnectionWithModel,
    MODEL_SETTINGS,
    SetDefaultModel,
)
from plugins.models.state import ModelsState
from plugins.models.store import ModelsStore

_DRIVER_ID = "material-fixture"
_CONNECTION_ID = "material-fixture-connection"
_CHAT_MODEL_ID = "material-fixture-chat"
_EMBEDDING_MODEL_ID = "material-fixture-embedding"


@dataclass(slots=True)
class MaterialModelDriver:
    """Record real model bindings while optionally holding external calls."""

    release: asyncio.Event | None = None
    block_external_calls: bool = False
    entered: asyncio.Event = field(default_factory=asyncio.Event)
    opened: list[DriverConnectionDescriptor] = field(default_factory=list)
    chat_bindings: list[BoundModelDescriptor] = field(default_factory=list)
    embedding_bindings: list[EmbeddingSpaceDescriptor] = field(default_factory=list)
    chat_requests: list[ModelRequest] = field(default_factory=list)
    embedding_requests: list[tuple[EmbeddingSpaceDescriptor, tuple[str, ...]]] = (
        field(default_factory=list)
    )
    closed: int = 0

    def definition(self) -> ModelDriverDefinition:
        """Return the one fixture-only driver registered in the real root."""

        return ModelDriverDefinition(
            driver_id=_DRIVER_ID,
            contract_version="fixture-v1",
            open=self.open,
            probe=self.probe,
            probe_embedding=self.probe_embedding,
        )

    async def open(
        self,
        descriptor: DriverConnectionDescriptor,
        _credential: object,
    ) -> DriverConnection:
        """Open no network resource and expose the two normal driver binders."""

        self.opened.append(descriptor)
        return DriverConnection(
            bind_chat=self.bind_chat,
            bind_embedding=self.bind_embedding,
            close=self.close,
        )

    async def probe(
        self,
        _descriptor: DriverConnectionDescriptor,
        _credential: object,
    ) -> None:
        """Accept the fixture connection without making an external request."""

    async def probe_embedding(
        self,
        _descriptor: DriverConnectionDescriptor,
        _credential: object,
        model: str,
    ):
        """Return the same measured space that the saved embedding model declares."""

        if model != "fixture-embedding":
            raise ValueError(f"unexpected fixture embedding model: {model}")
        return _embedding_discovery()

    def bind_chat(
        self,
        descriptor: BoundModelDescriptor,
        _config: Mapping[str, Any],
    ) -> _FixtureChat:
        """Bind a chat model that records calls through the ordinary Models path."""

        self.chat_bindings.append(descriptor)
        return _FixtureChat(self, descriptor)

    def bind_embedding(
        self,
        descriptor: EmbeddingSpaceDescriptor,
        _config: Mapping[str, Any],
    ) -> _FixtureEmbedding:
        """Bind an embedding model that returns its declared vector dimension."""

        self.embedding_bindings.append(descriptor)
        return _FixtureEmbedding(self, descriptor)

    async def wait_if_blocked(self) -> None:
        """Let a test deterministically pause a provider call after it starts."""

        self.entered.set()
        if self.block_external_calls and self.release is not None:
            await self.release.wait()

    async def close(self) -> None:
        """Record cleanup through the real driver-scope Effect."""

        self.closed += 1

    def reset_records(self) -> None:
        """Discard setup probes so a test observes only its own calls."""

        self.entered.clear()
        self.opened.clear()
        self.chat_bindings.clear()
        self.embedding_bindings.clear()
        self.chat_requests.clear()
        self.embedding_requests.clear()
        self.closed = 0


class _FixtureChat:
    """A minimal DriverChatModel with deterministic, non-network output."""

    max_tool_schemas = None

    def __init__(
        self,
        driver: MaterialModelDriver,
        descriptor: BoundModelDescriptor,
    ) -> None:
        self._driver = driver
        self._descriptor = descriptor

    async def complete(self, request: ModelRequest) -> LLMResponse:
        self._driver.chat_requests.append(request)
        await self._driver.wait_if_blocked()
        return LLMResponse(content=f"fixture:{self._descriptor.model_id}")

    def estimate_context_tokens(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]] = (),
    ) -> int:
        return len(messages) + len(tools)

    def estimate_appended_message_tokens(
        self,
        messages: Sequence[Mapping[str, Any]],
    ) -> int:
        return len(messages)


class _FixtureEmbedding:
    """A minimal DriverEmbeddingModel that respects its saved descriptor."""

    def __init__(
        self,
        driver: MaterialModelDriver,
        descriptor: EmbeddingSpaceDescriptor,
    ) -> None:
        self._driver = driver
        self._descriptor = descriptor

    async def embed(self, texts: Sequence[str]) -> EmbeddingResult:
        saved = tuple(texts)
        self._driver.embedding_requests.append((self._descriptor, saved))
        await self._driver.wait_if_blocked()
        vector = tuple(float(index + 1) for index in range(self._descriptor.dimensions))
        return EmbeddingResult(vectors=tuple(vector for _ in saved))


@dataclass(frozen=True, slots=True)
class MaterialModels:
    """The real root, state, store, views, and controlled external driver."""

    root: CompositionRoot
    state: ModelsState
    store: ModelsStore
    driver: MaterialModelDriver

    @property
    def chat_models(self):
        return self.state.chat_models

    @property
    def embeddings(self):
        return self.state.embeddings

    @property
    def settings(self):
        return self.state.settings


@asynccontextmanager
async def material_models(
    tmp_path: Path,
    *,
    release: asyncio.Event | None = None,
) -> AsyncGenerator[MaterialModels]:
    """Mount real Models services with one saved chat and embedding model."""

    root = CompositionRoot("material-model-fixture")
    store = ModelsStore(tmp_path / "model-registry.sqlite3", tmp_path / "model-backups")
    driver = MaterialModelDriver(release=release)
    states: list[ModelsState] = []

    async def mount_models(ctx) -> None:
        store.initialize()
        state = ModelsState(store, context=ctx)
        states.append(state)
        _ = await ctx.effect(lambda: store.close, label="material-model-store")
        _ = await ctx.provide(MODEL_DRIVERS, state.drivers)
        _ = await ctx.provide(CHAT_MODELS, state.chat_models)
        _ = await ctx.provide(EMBEDDINGS, state.embeddings)
        _ = await ctx.provide(MODEL_SETTINGS, state.settings)

    async def mount_driver(ctx) -> None:
        _ = await ctx.require(MODEL_DRIVERS).register(ctx, driver.definition())

    await root.mount(mount_models, name="models")
    state = states[0]
    await root.mount(mount_driver, name="material-driver", inject=(MODEL_DRIVERS,))
    try:
        await _save_models(state)
        driver.block_external_calls = True
        driver.reset_records()
        yield MaterialModels(root=root, state=state, store=store, driver=driver)
    finally:
        await root.dispose()


async def _save_models(state: ModelsState) -> None:
    """Use the real settings transaction and store schema for fixture data."""

    connection = AddConnection(
        expected_revision=0,
        connection_id=_CONNECTION_ID,
        name="Material fixture",
        driver_id=_DRIVER_ID,
        endpoint="https://fixture.invalid",
        auth_identity="fixture-credential",
        credential={"token": "fixture"},
    )
    chat = AddModel(
        expected_revision=0,
        model_id=_CHAT_MODEL_ID,
        connection_id=_CONNECTION_ID,
        kind='chat',
        model="fixture-chat",
        capabilities=ModelCapabilities(
            context_window=8_192,
            max_output_tokens=1_024,
            supports_tool_calls=True,
            supports_parallel_tool_calls=True,
        ),
        capability_sources=CapabilitySources(
            context_window="fixture",
            max_output_tokens="fixture",
            tool_calls="fixture",
            parallel_tool_calls="fixture",
        ),
    )
    receipt = await state.settings.apply(CreateConnectionWithModel(connection, chat))
    receipt = await state.settings.apply(
        SetDefaultModel(
            expected_revision=receipt.revision,
            role="default",
            model_id=_CHAT_MODEL_ID,
        )
    )
    await state.settings.apply(
        AddModel(
            expected_revision=receipt.revision,
            model_id=_EMBEDDING_MODEL_ID,
            connection_id=_CONNECTION_ID,
            kind='embedding',
            model="fixture-embedding",
            capabilities=_embedding_capabilities(),
            capability_sources=_embedding_sources(),
            make_default_embedding=True,
        )
    )


def _embedding_capabilities() -> ModelCapabilities:
    """Return the actual persisted dimensional contract for fixture vectors."""

    return ModelCapabilities(
        embedding_dimensions=3,
        embedding_normalization="unit",
        supports_tool_calls=False,
        supports_parallel_tool_calls=False,
    )


def _embedding_sources() -> CapabilitySources:
    """Mark fixture dimensions as probe evidence, as the real settings path requires."""

    return CapabilitySources(
        embedding_dimensions="probe",
        embedding_normalization="probe",
    )


def _embedding_discovery():
    """Return a probe result accepted by the real embedding settings boundary."""

    from plugins.models.contract import DiscoveredModel

    return DiscoveredModel(
        kind='embedding',
        model="fixture-embedding",
        capabilities=_embedding_capabilities(),
        capability_sources=_embedding_sources(),
    )

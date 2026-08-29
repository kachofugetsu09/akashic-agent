from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
import asyncio

import pytest

from agent.plugin_composition import (
    BoundModelDescriptor,
    CapabilitySources,
    LLMResponse,
    ModelCapabilities,
    ModelRequest,
    ModelRole,
)
from agent.plugin_composition.models import ModelUnavailableError
from agent.tools.vision import ReadImageVisionTool
from tests.model_plugin_fakes import bind_test_model_snapshot


class _VisionModel:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.requests: list[ModelRequest] = []

    async def complete(self, request: ModelRequest) -> LLMResponse:
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        return LLMResponse(content="一只猫")


class _BoundModel:
    def __init__(
        self,
        responder: _VisionModel,
        *,
        role: ModelRole,
        revision: int,
    ) -> None:
        self.responder = responder
        self._descriptor = BoundModelDescriptor(
            binding_id=f"vision:{revision}:{role.value}",
            plugin_snapshot_id="test-plugin-snapshot",
            model_revision=revision,
            model_id=f"{role.value}-{revision}",
            connection_id="vision-connection",
            driver_id="vision-driver",
            driver_contract_version="1",
            auth_identity="vision-test",
            model=f"{role.value}-{revision}",
            role=role,
            reasoning_effort=None,
            capabilities=ModelCapabilities(input_modalities=("text", "image")),
            capability_sources=CapabilitySources(input_modalities="test"),
            capability_digest=f"capabilities-{revision}",
        )

    @property
    def descriptor(self) -> BoundModelDescriptor:
        return self._descriptor

    async def complete(self, request: ModelRequest) -> LLMResponse:
        assert isinstance(request, ModelRequest)
        assert not hasattr(request, "model")
        assert not hasattr(request, "provider")
        return await self.responder.complete(request)


class _ChatModels:
    def __init__(self, model: _VisionModel) -> None:
        self.model = model
        self.roles: list[ModelRole] = []
        self.execution_calls = 0
        self.execution_exits = 0
        self.revision = 1
        self.current: object | None = None
        self.owner: asyncio.Task[object] | None = None
        self.executions: list[object] = []

    @asynccontextmanager
    async def execution(self, **_selection: object):
        if self.current is not None:
            if asyncio.current_task() is not self.owner:
                raise RuntimeError("model execution 不能由子 task 继承")
            yield self.current
            return
        self.execution_calls += 1
        facade = self
        revision = self.revision

        class _Execution:
            def __init__(self) -> None:
                self.models = {
                    role: _BoundModel(facade.model, role=role, revision=revision)
                    for role in (ModelRole.AGENT, ModelRole.VISION)
                }

            def chat(self, role: ModelRole) -> _BoundModel:
                facade.roles.append(role)
                return self.models[role]

        execution = _Execution()
        self.current = execution
        self.owner = asyncio.current_task()
        self.executions.append(execution)
        try:
            yield execution
        finally:
            self.current = None
            self.owner = None
            self.execution_exits += 1


@pytest.mark.asyncio
async def test_vision_tool_uses_turn_vision_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = tmp_path / "image.png"
    image.write_bytes(b"fixture")
    monkeypatch.setattr(
        "agent.tools.vision._encode_image_data_uri",
        lambda _path: "data:image/png;base64,AA==",
    )
    model = _VisionModel()
    chat_models = _ChatModels(model)

    async with bind_test_model_snapshot(object(), chat_models=chat_models):
        async with chat_models.execution() as execution:
            agent = execution.chat(ModelRole.AGENT)
            chat_models.revision = 2
            result = await ReadImageVisionTool().execute(str(image), "图里有什么？")
            vision = execution.chat(ModelRole.VISION)

    assert result == "一只猫"
    assert chat_models.execution_calls == 1
    assert chat_models.execution_exits == 1
    assert chat_models.roles == [
        ModelRole.AGENT,
        ModelRole.VISION,
        ModelRole.VISION,
    ]
    assert agent.descriptor.plugin_snapshot_id == vision.descriptor.plugin_snapshot_id
    assert agent.descriptor.model_revision == vision.descriptor.model_revision == 1
    assert len(model.requests) == 1
    request = model.requests[0]
    assert request.max_output_tokens == 2048
    assert request.disable_reasoning is True
    content = request.messages[0]["content"]
    assert content[0] == {"type": "text", "text": "图里有什么？"}
    assert content[1]["image_url"]["url"] == "data:image/png;base64,AA=="

    async with bind_test_model_snapshot(object(), chat_models=chat_models):
        await ReadImageVisionTool().execute(str(image), "下一轮")
    assert chat_models.execution_calls == 2
    latest = chat_models.executions[-1].models[ModelRole.VISION]  # type: ignore[attr-defined]
    assert latest.descriptor.model_revision == 2


@pytest.mark.asyncio
async def test_vision_tool_preserves_public_model_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = tmp_path / "image.png"
    image.write_bytes(b"fixture")
    monkeypatch.setattr(
        "agent.tools.vision._encode_image_data_uri",
        lambda _path: "data:image/png;base64,AA==",
    )
    chat_models = _ChatModels(_VisionModel(ModelUnavailableError("vision missing")))

    async with bind_test_model_snapshot(object(), chat_models=chat_models):
        result = await ReadImageVisionTool().execute(str(image), "describe")

    assert result == "调用视觉模型失败：vision missing"


@pytest.mark.asyncio
async def test_vision_tool_does_not_hide_internal_image_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = tmp_path / "image.png"
    image.write_bytes(b"fixture")

    def fail(_path: Path) -> str:
        raise AssertionError("internal-marker")

    monkeypatch.setattr("agent.tools.vision._encode_image_data_uri", fail)

    with pytest.raises(AssertionError, match="internal-marker"):
        await ReadImageVisionTool().execute(str(image), "describe")


@pytest.mark.asyncio
async def test_vision_tool_rejects_execution_without_turn_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = tmp_path / "image.png"
    image.write_bytes(b"fixture")
    monkeypatch.setattr(
        "agent.tools.vision._encode_image_data_uri",
        lambda _path: "data:image/png;base64,AA==",
    )

    with pytest.raises(RuntimeError, match="exact Turn snapshot"):
        await ReadImageVisionTool().execute(str(image), "describe")


@pytest.mark.asyncio
async def test_vision_tool_rejects_inherited_child_task(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = tmp_path / "image.png"
    image.write_bytes(b"fixture")
    monkeypatch.setattr(
        "agent.tools.vision._encode_image_data_uri",
        lambda _path: "data:image/png;base64,AA==",
    )
    chat_models = _ChatModels(_VisionModel())

    async with bind_test_model_snapshot(object(), chat_models=chat_models):
        async with chat_models.execution():
            child = asyncio.create_task(
                ReadImageVisionTool().execute(str(image), "child")
            )
            with pytest.raises(RuntimeError, match="exact Turn snapshot"):
                await child

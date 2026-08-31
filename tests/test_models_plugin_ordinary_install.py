from __future__ import annotations

import asyncio
import hashlib
import os
import shutil
import sqlite3
import subprocess
import sys
import threading
import time
from contextlib import closing
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import pytest
import uvicorn
from fastapi.testclient import TestClient

from agent.plugin_composition import (
    AddConnection,
    AddModel,
    CancelConnectionAuth,
    CHAT_MODELS,
    EMBEDDINGS,
    MODEL_CATALOG,
    CapabilitySources,
    CreateConnectionWithModel,
    DiscoveredModel,
    ModelCapabilities,
    ModelKind,
    ModelAvailability,
    ModelRequest,
    ModelRole,
    ModelUnavailableError,
    FinishConnectionAuth,
    StartConnectionAuth,
    SetDefaultModel,
    SyncModels,
    UpdateConnection,
)
from agent.tools.vision import ReadImageVisionTool
from agent.plugins.install import install_git_plugin, uninstall_plugin
from agent.plugins.dashboard_host import PluginDashboardHost
from agent.plugins.manager import PluginManager
from agent.plugins.model_control import RuntimeModelControl
from agent.plugins.snapshot import bind_runtime_snapshot, reset_runtime_snapshot
from bootstrap.chat_api import create_chat_app
from bootstrap.web_runtime import chat_socket_path
from bootstrap.web_shell import create_web_shell_app
from bus.event_bus import EventBus
from infra.channels.web_chat_channel import WebChatChannel


class _RepositoryModelsImportBlocker:
    """Make repository-local models imports fail during the ordinary-plugin gate."""

    def find_spec(
        self,
        fullname: str,
        path: object = None,
        target: object = None,
    ) -> None:
        _ = path, target
        if fullname == "plugins.models" or fullname.startswith("plugins.models."):
            raise ModuleNotFoundError(f"repository plugin import blocked: {fullname}")
        return None


def _manager(tmp_path: Path) -> PluginManager:
    return PluginManager(
        plugin_dirs=[Path("plugins/shell_ui")],
        event_bus=EventBus(),
        tool_registry=None,
        workspace=tmp_path / "workspace",
        installed_cache_root=tmp_path / "home" / "cache",
    )


def _commit(repo: Path) -> None:
    for args in (
        ("init",),
        ("config", "user.name", "test"),
        ("config", "user.email", "test@example.com"),
        ("add", "."),
        ("commit", "-m", "init"),
    ):
        result = subprocess.run(
            ("git", *args),
            cwd=repo,
            capture_output=True,
            text=True,
            env=os.environ.copy(),
        )
        assert result.returncode == 0, result.stderr


def _write_fake_driver(repo: Path) -> None:
    repo.mkdir(parents=True)
    (repo / "akashic.plugin.toml").write_text(
        "schema_version = 1\n"
        'name = "fake-model-driver"\n'
        'version = "1.0.0"\n'
        "api_version = 3\n"
        'entrypoint = "plugin.py"\n',
        encoding="utf-8",
    )
    (repo / "plugin.py").write_text(
        "from agent.plugin_composition import (\n"
        "  MODEL_DRIVERS, DriverConnection, EmbeddingResult, LLMResponse,\n"
        "  CapabilitySources, DiscoveredModel, ModelCapabilities, ModelContinuation,\n"
        "  ModelDriverDefinition, ModelKind,\n"
        ")\n"
        "api_version = 3\n"
        "name = 'fake-model-driver'\n"
        "version = '1.0.0'\n"
        "inject = (MODEL_DRIVERS,)\n"
        "workspace_roots = ()\n"
        "workspace_files = ()\n"
        "opened_configs = []\n"
        "chat_requests = []\n"
        "cancel_calls = []\n"
        "finish_started = None\n"
        "finish_continue = None\n"
        "class Chat:\n"
        "  def __init__(self, descriptor): self.descriptor = descriptor\n"
        "  async def complete(self, request):\n"
        "    chat_requests.append(request)\n"
        "    return LLMResponse(content='ok', continuation=ModelContinuation(\n"
        "      self.descriptor.binding_id, {'step': 1}))\n"
        "  def estimate_context_tokens(self, messages, tools=()):\n"
        "    return len(messages) + len(tools)\n"
        "  def estimate_appended_message_tokens(self, messages):\n"
        "    return len(messages)\n"
        "  @property\n"
        "  def max_tool_schemas(self): return 64\n"
        "class Embedding:\n"
        "  def __init__(self, descriptor): self.descriptor = descriptor\n"
        "  async def embed(self, texts):\n"
        "    return EmbeddingResult(tuple(\n"
        "      ((1.0, 0.0) if text == 'wrong' else (1.0, 0.0, 0.0))\n"
        "      for text in texts))\n"
        "async def open_driver(connection, credential):\n"
        "  opened_configs.append(dict(connection.config))\n"
        "  secret = await credential.read()\n"
        "  assert secret['access_token'] == 'secret'\n"
        "  return DriverConnection(lambda d, c: Chat(d), lambda d, c: Embedding(d))\n"
        "async def discover(connection, credential):\n"
        "  await credential.read()\n"
        "  return (DiscoveredModel(kind=ModelKind.CHAT, model='fake-chat-wire',\n"
        "    capabilities=ModelCapabilities(context_window=8192),\n"
        "    capability_sources=CapabilitySources(context_window='fake-catalog'),\n"
        "    driver_config={'catalog': 'refreshed'}),)\n"
        "async def start_auth(input):\n"
        "  state = {'poll': 0}\n"
        "  if input.get('block') == '1': state['block'] = True\n"
        "  return {'state': state, 'challenge': {'code': {'value': 'abc'}}}\n"
        "async def finish_auth(state):\n"
        "  if state.get('block'):\n"
        "    finish_started.set()\n"
        "    await finish_continue.wait()\n"
        "  poll = state['poll']\n"
        "  if poll < 2:\n"
        "    return {'status': 'pending', 'state': {'poll': poll + 1},\n"
        "            'challenge': {'poll': poll + 1}}\n"
        "  return {'status': 'complete', 'name': 'OAuth',\n"
        "          'endpoint': 'https://oauth.example.test/v1',\n"
        "          'auth_identity': 'oauth-account',\n"
        "          'credential': {'driver': 'api_key', 'access_token': 'secret'},\n"
        "          'driver_config': {}}\n"
        "async def cancel_auth(state):\n"
        "  cancel_calls.append(dict(state))\n"
        "  if len(cancel_calls) == 1: raise RuntimeError('temporary cancel failure')\n"
        "async def apply(ctx, config):\n"
        "  drivers = ctx.require(MODEL_DRIVERS)\n"
        "  await drivers.register(ctx, ModelDriverDefinition(\n"
        "    driver_id='fake', contract_version='1', open=open_driver, discover=discover,\n"
        "    start_auth=start_auth, finish_auth=finish_auth, cancel_auth=cancel_auth))\n",
        encoding="utf-8",
    )


async def _configure_and_call(manager: PluginManager, workspace: Path) -> None:
    snapshot = manager.current_snapshot
    assert snapshot is not None and snapshot.composition_root is not None
    root = snapshot.composition_root
    control = RuntimeModelControl(manager.snapshot_store)
    with pytest.raises(ModelUnavailableError, match="维度"):
        await control.apply(
            CreateConnectionWithModel(
                connection=AddConnection(
                    expected_revision=0,
                    connection_id="failed-connection",
                    name="Failed",
                    driver_id="fake",
                    endpoint="https://example.test/v1",
                    auth_identity="failed-account",
                    credential={"driver": "api_key", "access_token": "secret"},
                ),
                model=AddModel(
                    expected_revision=0,
                    model_id="failed-embedding",
                    connection_id="failed-connection",
                    kind=ModelKind.EMBEDDING,
                    model="fake-embedding-wire",
                    capabilities=ModelCapabilities(embedding_dimensions=2),
                    capability_sources=CapabilitySources(embedding_dimensions="test"),
                ),
            )
        )
    failed_catalog = await control.catalog()
    assert failed_catalog.revision == 0
    assert failed_catalog.connections == () and failed_catalog.models == ()
    assert (workspace / "model-registry.sqlite3").is_file()
    revision = (
        await control.apply(
            AddConnection(
                expected_revision=0,
                connection_id="fake-connection",
                name="Fake",
                driver_id="fake",
                endpoint="https://example.test/v1",
                auth_identity="fake-account",
                credential={"driver": "api_key", "access_token": "secret"},
                driver_config={"nested": {"mode": "old"}},
            )
        )
    ).revision
    revision = (
        await control.apply(
            AddModel(
                expected_revision=revision,
                model_id="fake-chat",
                connection_id="fake-connection",
                kind=ModelKind.CHAT,
                model="fake-chat-wire",
                capabilities=ModelCapabilities(
                    context_window=4096,
                    max_output_tokens=512,
                    supports_tool_calls=True,
                ),
                capability_sources=CapabilitySources(
                    context_window="test",
                    max_output_tokens="test",
                    tool_calls="test",
                ),
            )
        )
    ).revision
    revision = (
        await control.apply(
            SetDefaultModel(
                expected_revision=revision,
                role=ModelRole.DEFAULT,
                model_id="fake-chat",
            )
        )
    ).revision
    with pytest.raises(ModelUnavailableError, match="维度"):
        await control.apply(
            AddModel(
                expected_revision=revision,
                model_id="wrong-embedding",
                connection_id="fake-connection",
                kind=ModelKind.EMBEDDING,
                model="fake-embedding-wire",
                capabilities=ModelCapabilities(embedding_dimensions=2),
                capability_sources=CapabilitySources(embedding_dimensions="test"),
            )
        )
    assert (await control.catalog()).revision == revision
    revision = (
        await control.apply(
            AddModel(
                expected_revision=revision,
                model_id="fake-embedding",
                connection_id="fake-connection",
                kind=ModelKind.EMBEDDING,
                model="fake-embedding-wire",
                capabilities=ModelCapabilities(
                    embedding_dimensions=3,
                    embedding_normalization="none",
                ),
                capability_sources=CapabilitySources(
                    embedding_dimensions="test",
                    embedding_normalization="test",
                ),
            )
        )
    ).revision
    revision = (
        await control.apply(
            SetDefaultModel(
                expected_revision=revision,
                role=None,
                model_id="fake-embedding",
            )
        )
    ).revision
    assert revision == 5

    synced = await control.apply(
        SyncModels(expected_revision=revision, connection_id="fake-connection")
    )
    revision = synced.revision
    assert revision == 5
    synced_model = root.context.require(MODEL_CATALOG).snapshot().model("fake-chat")
    assert synced_model.capabilities.context_window == 4096

    with pytest.raises(ValueError, match="image-capable"):
        await control.apply(
            SetDefaultModel(
                expected_revision=revision,
                role=ModelRole.VISION,
                model_id="fake-chat",
            )
        )
    revision = (
        await control.apply(
            AddModel(
                expected_revision=revision,
                model_id="fake-vision",
                connection_id="fake-connection",
                kind=ModelKind.CHAT,
                model="fake-vision-wire",
                capabilities=ModelCapabilities(input_modalities=("text", "image")),
                capability_sources=CapabilitySources(input_modalities="test"),
            )
        )
    ).revision
    revision = (
        await control.apply(
            SetDefaultModel(
                expected_revision=revision,
                role=ModelRole.VISION,
                model_id="fake-vision",
            )
        )
    ).revision
    assert revision == 7

    lease = await manager._snapshot_store.acquire()
    token = bind_runtime_snapshot(lease)
    try:
        chat_models = root.context.require(CHAT_MODELS)
        async with chat_models.execution() as execution:
            chat = execution.chat(ModelRole.DEFAULT)
            response = await chat.complete(
                ModelRequest(messages=({"role": "user", "content": "hi"},))
            )
            assert response.content == "ok"
            assert response.continuation is not None
            assert response.continuation.binding_id == chat.descriptor.binding_id
            vision = execution.chat(ModelRole.VISION)
            image = workspace / "vision.png"
            image.write_bytes(b"fixture")
            with patch(
                "agent.tools.vision._encode_image_data_uri",
                return_value="data:image/png;base64,AA==",
            ):
                assert (
                    await ReadImageVisionTool().execute(str(image), "describe") == "ok"
                )
            assert (
                vision.descriptor.plugin_snapshot_id
                == chat.descriptor.plugin_snapshot_id
            )
            assert (
                vision.descriptor.model_revision
                == chat.descriptor.model_revision
                == revision
            )
            driver_generation = manager.generation("fake-model-driver@ordinary-test")
            assert driver_generation is not None
            request = driver_generation.instance.module.chat_requests[-1]
            assert isinstance(request, ModelRequest)
            assert request.messages[0]["content"][1]["type"] == "image_url"
        embeddings = root.context.require(EMBEDDINGS)
        described = embeddings.describe()
        async with chat_models.execution():
            async with embeddings.bind() as embedding:
                assert embedding.descriptor.identity == described.identity
            with pytest.raises(ModelUnavailableError, match="不可用"):
                async with embeddings.bind(model_id="another-embedding"):
                    pass
        async with embeddings.bind() as embedding:
            assert embedding.descriptor.identity == described.identity
            result = await embedding.embed(("hello",))
            assert result.vectors == ((1.0, 0.0, 0.0),)
            with pytest.raises(ModelUnavailableError, match="维度"):
                await embedding.embed(("wrong",))

        async with chat_models.execution() as parent_execution:

            async def inherited_chat_child() -> None:
                async with root.context.runtime_scope():
                    async with chat_models.execution():
                        pass

            with pytest.raises(RuntimeError, match="不能由子 task 继承"):
                await asyncio.create_task(inherited_chat_child())

            async def independent_child() -> object:
                async with root.context.runtime_scope():
                    async with chat_models.independent_execution() as execution:
                        return execution

            child_execution = await asyncio.create_task(independent_child())
            assert child_execution is not parent_execution

        child_ready = asyncio.Event()
        child_continue = asyncio.Event()

        async with chat_models.execution():

            async def inherited_child() -> None:
                child_ready.set()
                await child_continue.wait()
                async with embeddings.bind():
                    pass

            child = asyncio.create_task(inherited_child())
            await child_ready.wait()
        child_continue.set()
        with pytest.raises(RuntimeError, match="不能由子 task 继承"):
            await child
    finally:
        reset_runtime_snapshot(token)
        await lease.release()

    # Core 给后台操作绑定同一 Root 的短 lease；退出后不长期占用 generation。
    async with root.context.runtime_scope():
        async with root.context.require(EMBEDDINGS).bind() as embedding:
            assert embedding.descriptor.identity == described.identity

    updated = await control.apply(
        UpdateConnection(
            expected_revision=revision,
            connection_id="fake-connection",
            name="Fake",
            auth_identity="fake-account",
            endpoint=None,
            driver_config={},
        )
    )
    revision = updated.revision
    driver_generation = manager.generation("fake-model-driver@ordinary-test")
    assert driver_generation is not None
    assert driver_generation.instance.module.opened_configs[-1] == {}
    with closing(sqlite3.connect(workspace / "model-registry.sqlite3")) as connection:
        endpoint = connection.execute(
            "SELECT base_url FROM model_connections WHERE id = 'fake-connection'"
        ).fetchone()
    assert endpoint == ("https://example.test/v1",)

    started = await control.apply(
        StartConnectionAuth(
            driver_id="fake",
            connection_id="oauth-connection",
        )
    )
    assert started.attempt_id is not None and started.challenge is not None
    with pytest.raises(TypeError):
        started.challenge["code"]["value"] = "changed"  # type: ignore[index]
    for expected_poll in (1, 2):
        pending = await control.apply(
            FinishConnectionAuth(
                expected_revision=revision,
                attempt_id=started.attempt_id,
            )
        )
        assert pending.status == "pending"
        assert pending.challenge == {"poll": expected_poll}
    completed = await control.apply(
        FinishConnectionAuth(
            expected_revision=revision,
            attempt_id=started.attempt_id,
        )
    )
    assert completed.status == "committed"
    assert completed.revision == revision + 1

    cancel_started = await control.apply(
        StartConnectionAuth(driver_id="fake", connection_id="cancel-connection")
    )
    assert cancel_started.attempt_id is not None
    cancel = CancelConnectionAuth(attempt_id=cancel_started.attempt_id)
    with pytest.raises(RuntimeError, match="temporary cancel failure"):
        await control.apply(cancel)
    cancelled = await control.apply(cancel)
    assert cancelled.status == "cancelled"
    assert driver_generation.instance.module.cancel_calls == [
        {"poll": 0},
        {"poll": 0},
    ]

    driver_module = driver_generation.instance.module
    driver_module.finish_started = asyncio.Event()
    driver_module.finish_continue = asyncio.Event()
    racing = await control.apply(
        StartConnectionAuth(
            driver_id="fake",
            connection_id="cancel-during-finish",
            input={"block": "1"},
        )
    )
    assert racing.attempt_id is not None
    finishing = asyncio.create_task(
        control.apply(FinishConnectionAuth(revision, racing.attempt_id))
    )
    await driver_module.finish_started.wait()
    cancelling = asyncio.create_task(
        control.apply(CancelConnectionAuth(racing.attempt_id))
    )
    await asyncio.sleep(0)
    driver_module.finish_continue.set()
    with pytest.raises(ValueError, match="已取消"):
        await finishing
    assert (await cancelling).status == "cancelled"
    assert all(
        connection.connection_id != "cancel-during-finish"
        for connection in (await control.catalog()).connections
    )


def _exercise_public_model_control(manager: PluginManager, tmp_path: Path) -> None:
    """Cross 2236 and a real UDS before changing one installed-plugin binding."""

    workspace = tmp_path / "workspace"
    socket_path = chat_socket_path(workspace)
    socket_path.parent.mkdir(parents=True, exist_ok=True)
    control = RuntimeModelControl(manager.snapshot_store)
    chat_app = create_chat_app(
        workspace=workspace,
        channel=WebChatChannel(),
        model_control=cast(Any, control),
    )
    server = uvicorn.Server(
        uvicorn.Config(
            chat_app,
            uds=str(socket_path),
            log_level="critical",
            access_log=False,
            ws="none",
        )
    )
    thread = threading.Thread(
        target=lambda: asyncio.run(server.serve()),
        name="ordinary-model-control-uds",
        daemon=True,
    )
    thread.start()
    deadline = time.monotonic() + 5
    while not socket_path.is_socket() and thread.is_alive():
        if time.monotonic() >= deadline:
            break
        time.sleep(0.01)
    assert socket_path.is_socket()
    try:
        shell = create_web_shell_app(tmp_path / "config.toml", workspace)
        with TestClient(shell) as client:
            before = client.get("/api/settings/model/catalog")
            retired_memory = client.post(
                "/api/settings/memory",
                headers={
                    "Origin": "http://testserver",
                    "X-Akasic-CSRF": "1",
                },
                json={
                    "enabled": True,
                    "embedding_model_id": "fake-embedding",
                },
            )
            changed = client.post(
                "/api/settings/model/command",
                headers={
                    "Origin": "http://testserver",
                    "X-Akasic-CSRF": "1",
                },
                json={
                    "type": "set_default",
                    "expected_revision": 9,
                    "role": "fast",
                    "model_id": "fake-chat",
                },
            )
        assert before.status_code == 200
        assert before.json()["revision"] == 9
        assert before.json()["models"][0]["id"] == "fake-chat"
        assert retired_memory.status_code == 404
        assert not (tmp_path / "config.toml").exists()
        assert changed.status_code == 200
        assert changed.json()["revision"] == 10
    finally:
        server.should_exit = True
        thread.join(timeout=5)
    assert not thread.is_alive()


@pytest.mark.asyncio
async def test_models_plugin_installs_and_runs_without_builtin_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    models_repo = tmp_path / "models-repo"
    shutil.copytree(Path("plugins/models"), models_repo)
    shutil.rmtree(models_repo / "__pycache__", ignore_errors=True)
    _commit(models_repo)
    driver_repo = tmp_path / "driver-repo"
    _write_fake_driver(driver_repo)
    _commit(driver_repo)

    models_install = install_git_plugin(
        workspace=tmp_path / "workspace",
        source=str(models_repo),
        marketplace="ordinary-test",
        plugins_home=tmp_path / "home",
    )
    driver_install = install_git_plugin(
        workspace=tmp_path / "workspace",
        source=str(driver_repo),
        marketplace="ordinary-test",
        plugins_home=tmp_path / "home",
    )
    for module_name in tuple(sys.modules):
        if module_name == "plugins.models" or module_name.startswith("plugins.models."):
            monkeypatch.delitem(sys.modules, module_name)
    monkeypatch.setattr(
        sys,
        "meta_path",
        [_RepositoryModelsImportBlocker(), *sys.meta_path],
    )
    before_repo_modules = {
        name for name in sys.modules if name.startswith("plugins.models")
    }
    manager = _manager(tmp_path)
    await manager.load_all()

    generation = manager.generation("models@ordinary-test")
    assert generation is not None
    assert generation.source_type == "installed"
    assert generation.plugin_dir == models_install.installed_path
    contract_bytes = Path(
        "packages/akashic-models-ui-v1/contract.json"
    ).read_bytes()
    assert dict(generation.instance.web_contract_digests) == {
        "models.connection-types.v1": hashlib.sha256(contract_bytes).hexdigest()
    }
    assert {
        name for name in sys.modules if name.startswith("plugins.models")
    } == before_repo_modules
    package = generation.instance.module.__package__
    assert package
    installed_modules = [
        module
        for module_name, module in sys.modules.items()
        if module_name == package or module_name.startswith(f"{package}.")
    ]
    assert installed_modules
    for module in installed_modules:
        module_file = module.__file__
        if module_file is not None:
            assert (
                Path(module_file)
                .resolve()
                .is_relative_to(models_install.installed_path)
            )
    driver_generation = manager.generation("fake-model-driver@ordinary-test")
    assert driver_generation is not None
    assert driver_generation.source_type == "installed"
    assert driver_generation.plugin_dir == driver_install.installed_path
    driver_module_file = driver_generation.instance.module.__file__
    assert driver_module_file is not None
    assert (
        Path(driver_module_file).resolve().is_relative_to(driver_install.installed_path)
    )
    dashboard_host = PluginDashboardHost(core_routes=())
    snapshot = manager.current_snapshot
    assert snapshot is not None
    dashboard_host.prepare_initial_snapshot(snapshot)
    manager.bind_dashboard_preparer(
        dashboard_host.prepare_snapshot,
        validation_releaser=dashboard_host.release_validation,
    )
    assert [
        route.path
        for binding in snapshot.dashboard_bindings
        for route in binding.routes  # type: ignore[attr-defined]
    ] == [
        "/api/dashboard/models/catalog",
        "/api/dashboard/models/command",
    ]
    await _configure_and_call(manager, tmp_path / "workspace")
    await asyncio.to_thread(_exercise_public_model_control, manager, tmp_path)

    registry = tmp_path / "workspace/model-registry.sqlite3"
    registry_before = registry.read_bytes()
    manifest = models_repo / "akashic.plugin.toml"
    manifest.write_text(
        manifest.read_text(encoding="utf-8").replace(
            'version = "1.0.0"', 'version = "1.0.1"'
        ),
        encoding="utf-8",
    )
    plugin_source = models_repo / "plugin.py"
    plugin_source.write_text(
        plugin_source.read_text(encoding="utf-8").replace(
            'version = "1.0.0"', 'version = "1.0.1"'
        ),
        encoding="utf-8",
    )
    _commit(models_repo)
    upgraded = install_git_plugin(
        workspace=tmp_path / "workspace",
        source=str(models_repo),
        marketplace="ordinary-test",
        plugins_home=tmp_path / "home",
        stage_candidate=True,
    )
    _ = await manager.reconcile_changed()
    status = manager.candidate_status()
    assert status["candidate_plugin_id"] == "models@ordinary-test"
    assert status["candidate_state"] == "latest_ready", status
    assert registry.read_bytes() == registry_before
    _ = await manager.switch_ready("models@ordinary-test")
    current = manager.generation("models@ordinary-test")
    assert current is not None and current.plugin_dir == upgraded.installed_path
    snapshot = manager.current_snapshot
    assert snapshot is not None and snapshot.composition_root is not None
    catalog = snapshot.composition_root.context.require(MODEL_CATALOG).snapshot()
    assert catalog.revision == 10
    assert registry.read_bytes() == registry_before
    await manager.terminate_all()

    reloaded = _manager(tmp_path)
    await reloaded.load_all()
    snapshot = reloaded.current_snapshot
    assert snapshot is not None and snapshot.composition_root is not None
    catalog = snapshot.composition_root.context.require(MODEL_CATALOG).snapshot()
    assert catalog.revision == 10
    assert catalog.role_bindings[ModelRole.DEFAULT] == "fake-chat"
    assert catalog.role_bindings[ModelRole.FAST] == "fake-chat"
    assert catalog.role_bindings[ModelRole.VISION] == "fake-vision"
    assert catalog.default_embedding_model_id == "fake-embedding"
    await reloaded.terminate_all()

    _ = uninstall_plugin(
        "fake-model-driver@ordinary-test",
        workspace=tmp_path / "workspace",
        plugins_home=tmp_path / "home",
    )
    without_driver = _manager(tmp_path)
    await without_driver.load_all()
    snapshot = without_driver.current_snapshot
    assert snapshot is not None and snapshot.composition_root is not None
    catalog = snapshot.composition_root.context.require(MODEL_CATALOG).snapshot()
    assert catalog.revision == 10
    assert all(
        model.availability is ModelAvailability.DRIVER_UNAVAILABLE
        for model in catalog.models
    )
    await without_driver.terminate_all()

    _ = install_git_plugin(
        workspace=tmp_path / "workspace",
        source=str(driver_repo),
        marketplace="ordinary-test",
        plugins_home=tmp_path / "home",
    )
    restored = _manager(tmp_path)
    await restored.load_all()
    snapshot = restored.current_snapshot
    assert snapshot is not None and snapshot.composition_root is not None
    assert all(
        model.availability is ModelAvailability.AVAILABLE
        for model in snapshot.composition_root.context.require(MODEL_CATALOG)
        .snapshot()
        .models
    )
    await restored.terminate_all()

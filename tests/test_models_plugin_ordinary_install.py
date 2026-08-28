from __future__ import annotations

import asyncio
import ast
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from agent.plugin_composition import (
    AddConnection,
    AddModel,
    CHAT_MODELS,
    EMBEDDINGS,
    MODEL_CATALOG,
    MODEL_SETTINGS,
    CapabilitySources,
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
from agent.plugins.install import install_git_plugin, uninstall_plugin
from agent.plugins.manager import PluginManager
from agent.plugins.snapshot import bind_runtime_snapshot, reset_runtime_snapshot
from bus.event_bus import EventBus


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
        plugin_dirs=[],
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
        "class Chat:\n"
        "  def __init__(self, descriptor): self.descriptor = descriptor\n"
        "  async def complete(self, request):\n"
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
        "  return {'state': {'poll': 0}, 'challenge': {'code': {'value': 'abc'}}}\n"
        "async def finish_auth(state):\n"
        "  poll = state['poll']\n"
        "  if poll < 2:\n"
        "    return {'status': 'pending', 'state': {'poll': poll + 1},\n"
        "            'challenge': {'poll': poll + 1}}\n"
        "  return {'status': 'complete', 'name': 'OAuth',\n"
        "          'endpoint': 'https://oauth.example.test/v1',\n"
        "          'auth_identity': 'oauth-account',\n"
        "          'credential': {'driver': 'api_key', 'access_token': 'secret'},\n"
        "          'driver_config': {}}\n"
        "async def apply(ctx, config):\n"
        "  drivers = ctx.require(MODEL_DRIVERS)\n"
        "  await drivers.register(ctx, ModelDriverDefinition(\n"
        "    driver_id='fake', contract_version='1', open=open_driver, discover=discover,\n"
        "    start_auth=start_auth, finish_auth=finish_auth))\n",
        encoding="utf-8",
    )


async def _configure_and_call(manager: PluginManager) -> None:
    snapshot = manager.current_snapshot
    assert snapshot is not None and snapshot.composition_root is not None
    root = snapshot.composition_root
    settings = root.context.require(MODEL_SETTINGS)
    revision = (
        await settings.apply(
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
        await settings.apply(
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
        await settings.apply(
            SetDefaultModel(
                expected_revision=revision,
                role=ModelRole.DEFAULT,
                model_id="fake-chat",
            )
        )
    ).revision
    revision = (
        await settings.apply(
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
        await settings.apply(
            SetDefaultModel(
                expected_revision=revision,
                role=None,
                model_id="fake-embedding",
            )
        )
    ).revision
    assert revision == 5

    synced = await settings.apply(
        SyncModels(expected_revision=revision, connection_id="fake-connection")
    )
    revision = synced.revision
    assert revision == 5
    synced_model = root.context.require(MODEL_CATALOG).snapshot().model("fake-chat")
    assert synced_model.capabilities.context_window == 4096

    lease = await manager._snapshot_store.acquire()
    token = bind_runtime_snapshot(lease)
    try:
        chat_models = root.context.require(CHAT_MODELS)
        async with chat_models.execution() as execution:
            chat = execution.chat(ModelRole.DEFAULT)
            response = await chat.complete(ModelRequest(messages=({"role": "user", "content": "hi"},)))
            assert response.content == "ok"
            assert response.continuation is not None
            assert response.continuation.binding_id == chat.descriptor.binding_id
        embeddings = root.context.require(EMBEDDINGS)
        described = embeddings.describe()
        async with chat_models.execution() as execution:
            current = execution.embedding()
            async with embeddings.bind() as embedding:
                assert embedding is current
                assert embedding.descriptor.identity == described.identity
            with pytest.raises(RuntimeError, match="选择冲突"):
                async with embeddings.bind(model_id="another-embedding"):
                    pass
        async with embeddings.bind() as embedding:
            assert embedding.descriptor.identity == described.identity
            result = await embedding.embed(("hello",))
            assert result.vectors == ((1.0, 0.0, 0.0),)
            with pytest.raises(ModelUnavailableError, match="维度"):
                await embedding.embed(("wrong",))

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

    updated = await settings.apply(
        UpdateConnection(
            expected_revision=revision,
            connection_id="fake-connection",
            name="Fake",
            endpoint="https://example.test/v1",
            auth_identity="fake-account",
            driver_config={},
        )
    )
    revision = updated.revision
    driver_generation = manager.generation("fake-model-driver@ordinary-test")
    assert driver_generation is not None
    assert driver_generation.instance.module.opened_configs[-1] == {}

    started = await settings.apply(
        StartConnectionAuth(
            driver_id="fake",
            connection_id="oauth-connection",
        )
    )
    assert started.attempt_id is not None and started.challenge is not None
    with pytest.raises(TypeError):
        started.challenge["code"]["value"] = "changed"  # type: ignore[index]
    for expected_poll in (1, 2):
        pending = await settings.apply(
            FinishConnectionAuth(
                expected_revision=revision,
                attempt_id=started.attempt_id,
            )
        )
        assert pending.status == "pending"
        assert pending.challenge == {"poll": expected_poll}
    completed = await settings.apply(
        FinishConnectionAuth(
            expected_revision=revision,
            attempt_id=started.attempt_id,
        )
    )
    assert completed.status == "committed"
    assert completed.revision == revision + 1


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
    before_repo_modules = {name for name in sys.modules if name.startswith("plugins.models")}
    manager = _manager(tmp_path)
    await manager.load_all()

    generation = manager.generation("models@ordinary-test")
    assert generation is not None
    assert generation.source_type == "installed"
    assert generation.plugin_dir == models_install.installed_path
    assert {name for name in sys.modules if name.startswith("plugins.models")} == before_repo_modules
    package = generation.instance.module.__package__
    assert package
    installed_modules = [
        module
        for module_name, module in sys.modules.items()
        if module_name == package or module_name.startswith(f"{package}.")
    ]
    assert installed_modules
    assert all(
        Path(module.__file__).resolve().is_relative_to(models_install.installed_path)
        for module in installed_modules
        if getattr(module, "__file__", None)
    )
    driver_generation = manager.generation("fake-model-driver@ordinary-test")
    assert driver_generation is not None
    assert driver_generation.source_type == "installed"
    assert driver_generation.plugin_dir == driver_install.installed_path
    assert Path(driver_generation.instance.module.__file__).resolve().is_relative_to(
        driver_install.installed_path
    )
    settings_lease = await manager._snapshot_store.acquire()
    settings_token = bind_runtime_snapshot(settings_lease)
    try:
        await _configure_and_call(manager)
    finally:
        reset_runtime_snapshot(settings_token)
        await settings_lease.release()
    await manager.terminate_all()

    reloaded = _manager(tmp_path)
    await reloaded.load_all()
    snapshot = reloaded.current_snapshot
    assert snapshot is not None and snapshot.composition_root is not None
    catalog = snapshot.composition_root.context.require(MODEL_CATALOG).snapshot()
    assert catalog.revision == 7
    assert catalog.role_bindings[ModelRole.DEFAULT] == "fake-chat"
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
    assert catalog.revision == 7
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
        for model in snapshot.composition_root.context.require(MODEL_CATALOG).snapshot().models
    )
    await restored.terminate_all()


def test_models_plugin_does_not_import_repository_plugin_package() -> None:
    for path in Path("plugins/models").glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith("plugins.models")
            elif isinstance(node, ast.Import):
                assert all(not alias.name.startswith("plugins.models") for alias in node.names)

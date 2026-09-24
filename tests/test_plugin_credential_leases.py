from pathlib import Path
import shutil

import pytest

from tests.fixtures.plugin_workspace import initialize_plugin_workspace

from agent.plugin_composition import CredentialRef, ServiceKey
from agent.plugin_composition.bindings import BINDINGS
from agent.plugin_composition.credentials import CREDENTIALS
from agent.plugins.manager import PluginManager
from agent.plugin_composition.config_input import CONFIG_INPUT, load_config, save_config, save_credential
from bus.event_bus import EventBus
from session.log import MessageLog


PROBE = ServiceKey("test.credential_reader")
MODULE = '''from pydantic import BaseModel, ConfigDict
from agent.plugin_composition import CREDENTIALS, CredentialRef, ServiceKey
api_version = 3
name = "secret_reader"
version = "1.0.0"
inject = (CREDENTIALS,)
class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")
    token: CredentialRef
async def apply(ctx):
    config = Config.model_validate(ctx.config)
    class Reader:
        async def read(self):
            async with ctx.require(CREDENTIALS).open(ctx, {"token": config.token}) as client:
                return client.credential(config.token)
        async def undeclared(self):
            async with ctx.require(CREDENTIALS).open(ctx, {"other": CredentialRef(("other",))}):
                raise AssertionError("undeclared credential admitted")
    await ctx.provide(ServiceKey("test.credential_reader"), Reader())
    await ctx.provide(ServiceKey("test.credential_context"), ctx)
'''


def environment(tmp_path):
    source = tmp_path / "plugins/secret_reader"
    source.mkdir(parents=True)
    (source / "plugin.py").write_text(MODULE)
    config = tmp_path / "workspace/plugin-data/secret_reader-builtin" / CONFIG_INPUT
    config.parent.mkdir(parents=True)
    save_config(config.parent, {"token": save_credential(config.parent, "fixture-private-token")})
    log = MessageLog(tmp_path / "sessions.db")
    initialize_plugin_workspace(tmp_path / "workspace")
    host = PluginManager([source.parent], event_bus=EventBus(), workspace=tmp_path / "workspace",
                         installed_cache_root=tmp_path / "home/cache", message_log=log)
    return source, config, log, host


@pytest.mark.asyncio
async def test_credential_binding_rejects_config_drift_after_restart(tmp_path):
    source, config, log, host = environment(tmp_path)
    try:
        await host.load_all()
        root = host.live_root
        assert root is not None
        ctx = root.context
        reader = ctx.require(PROBE)
        original_ctx = ctx.require(ServiceKey("test.credential_context"))
        async with original_ctx.runtime_scope():
            assert await reader.read() == "fixture-private-token"
            with pytest.raises(RuntimeError, match="frozen plugin"):
                await reader.undeclared()
            reference = ctx.require(BINDINGS).bind(PROBE, {})
            generation = host.generation("secret_reader")
            assert generation is not None
            assert generation.config_projection["token"] == load_config(config.parent)[0]["token"]
            assert "fixture-private-token" not in str(generation.config_projection)
            async with ctx.require(CREDENTIALS).open(original_ctx, {"token": original_ctx.config["token"]}) as client:
                assert client.credential(original_ctx.config["token"]) == "fixture-private-token"
            with pytest.raises(RuntimeError, match="已关闭"):
                client.credential(original_ctx.config["token"])
        for path in host._archive.path.rglob("*"):
            if path.is_file():
                assert b"fixture-private-token" not in path.read_bytes()
    finally:
        await host.terminate_all()
        log.close()
    save_config(config.parent, {"token": save_credential(config.parent, "replacement-token")})
    log = MessageLog(tmp_path / "sessions.db")
    host = PluginManager([source.parent], event_bus=EventBus(), workspace=tmp_path / "workspace",
                         installed_cache_root=tmp_path / "home/cache", message_log=log)
    try:
        await host.load_all()
        root = host.live_root
        assert root is not None
        bindings = root.context.require(BINDINGS)
        async with bindings.open(reference, PROBE) as (reader, metadata):
            assert metadata == {}
            with pytest.raises(RuntimeError, match="config revision 已漂移"):
                await reader.read()
    finally:
        await host.terminate_all()
        log.close()


@pytest.mark.asyncio
async def test_local_update_keeps_credential_access_on_formal_owner(tmp_path):
    source, config, log, host = environment(tmp_path)
    try:
        await host.load_all()
        root = host.live_root
        assert root is not None
        old_context = root.context.require(ServiceKey("test.credential_context"))
        async with old_context.runtime_scope():
            assert await old_context.require(PROBE).read() == "fixture-private-token"
            async with old_context.require(CREDENTIALS).open(
                old_context, {"token": old_context.config["token"]}
            ) as client:
                assert client.credential(old_context.config["token"]) == "fixture-private-token"
            with pytest.raises(RuntimeError, match="已关闭"):
                client.credential(old_context.config["token"])
        (source / "plugin.py").write_text(MODULE + "\nmarker = 'updated'\n")
        result = await host.reconcile_changed()
        assert result[0]["publication_state"] == "active"
        assert host.live_root is root
        new_context = root.context.require(ServiceKey("test.credential_context"))
        assert new_context is not old_context
        async with new_context.runtime_scope():
            assert await new_context.require(PROBE).read() == "fixture-private-token"
            with pytest.raises(RuntimeError, match="frozen plugin"):
                await new_context.require(PROBE).undeclared()
        assert "fixture-private-token" not in config.read_text()
    finally:
        await host.terminate_all()
        log.close()


@pytest.mark.asyncio
async def test_local_update_rejects_restored_legacy_config(tmp_path):
    source, config, log, host = environment(tmp_path)
    try:
        await host.load_all()
        (config.parent / "config.local.toml").write_text('token="old-secret"')
        (source / "plugin.py").write_text(MODULE + "\nmarker = 'candidate'\n")
        root = host.live_root
        with pytest.raises(RuntimeError, match="升级"):
            await host.reconcile_changed()
        assert host.live_root is root
        assert (config.parent / "config.local.toml").read_text() == 'token="old-secret"'
    finally:
        await host.terminate_all()
        log.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("shared_directory", [False, True])
async def test_local_install_never_copies_unrelated_historical_credentials(tmp_path, shared_directory):
    """本地安装只归档目标代码，历史凭据仍在原 owner 私有目录。"""
    from agent.plugins.install import install_git_plugin
    from tests.test_plugin_install import _commit, _write_v3_plugin

    source, config, log, host = environment(tmp_path)
    (config.parent / "notes.txt").write_text("preserved history")
    try:
        await host.load_all()
        root = host.live_root
        assert root is not None
        credential_context = root.context.require(ServiceKey("test.credential_context"))
        async with credential_context.runtime_scope():
            reference = root.context.require(BINDINGS).bind(PROBE, {})
        assert reference
    finally:
        await host.terminate_all()

    if shared_directory:
        # 当前插件可以将同名字段用作普通配置；私有凭据仍不进入副本。
        (source / "plugin.py").write_text(MODULE.replace('token: CredentialRef', 'token: str'))
        save_config(config.parent, {"token": "public-new-setting"})
    else:
        shutil.rmtree(source)

    plain = tmp_path / "plain-source"
    _write_v3_plugin(plain, name="plain", module_source='''
from pydantic import BaseModel
api_version = 3
name = "plain"
version = "1.0.0"
class Config(BaseModel):
    label: str = "ordinary"
async def apply(ctx):
    Config.model_validate(ctx.config)
''')
    _commit(plain)
    install_git_plugin(workspace=tmp_path / "workspace", source=str(plain), marketplace="lab",
                       plugins_home=tmp_path / "installed")
    plain_config = tmp_path / "workspace/plugin-data/plain-lab/config.input.json"
    plain_config.parent.mkdir(parents=True, exist_ok=True)
    save_config(plain_config.parent, {"label": "public config"})
    host = PluginManager([source.parent], event_bus=EventBus(), workspace=tmp_path / "workspace",
                         installed_cache_root=tmp_path / "installed/cache", message_log=log)
    try:
        await host.load_all()
        root = host.live_root
        assert root is not None
        (plain / "plugin.py").write_text((plain / "plugin.py").read_text() + '\nmarker="candidate"\n')
        _commit(plain)
        before = tuple(log._connection.iterdump())
        accepted = await host.install(
            source=str(plain), marketplace="lab", ref_name="", sparse_paths=[],
            update_id="plain-update",
        )
        assert accepted.selection == "selected"
        operation = host._operation
        assert operation is not None
        await operation.task
        assert host.live_root is root
        assert (config.parent / "notes.txt").read_text() == "preserved history"
        for path in host._archive.path.rglob("*"):
            if path.is_file():
                assert b"fixture-private-token" not in path.read_bytes(), path
        assert tuple(log._connection.iterdump()) == before
        assert "fixture-private-token" not in config.read_text()
    finally:
        await host.terminate_all()
        log.close()

from pathlib import Path
import shutil

import pytest

from agent.plugin_composition import CredentialRef, ServiceKey
from agent.plugin_composition.bindings import BINDINGS, Bindings
from agent.plugin_composition.credentials import CREDENTIALS
from agent.plugins.manager import PluginManager
from agent.plugins.snapshot import RuntimeSnapshotStore, lease_runtime_snapshot
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
    host = PluginManager([source.parent], event_bus=EventBus(), workspace=tmp_path / "workspace",
                         installed_cache_root=tmp_path / "home/cache", message_log=log)
    return source, config, log, host


@pytest.mark.asyncio
async def test_credential_binding_uses_current_reader_after_config_change(tmp_path):
    source, config, log, host = environment(tmp_path)
    try:
        await host.load_all()
        async with lease_runtime_snapshot(host.snapshot_store) as snapshot:
            ctx = snapshot.composition_root.context
            reader = ctx.require(PROBE)
            assert await reader.read() == "fixture-private-token"
            with pytest.raises(RuntimeError, match="frozen plugin"):
                await reader.undeclared()
            reference = ctx.require(BINDINGS).bind(PROBE, {})
            generation = snapshot.generations["secret_reader"]
            assert generation.config_projection["token"] == load_config(config.parent)[0]["token"]
            assert "fixture-private-token" not in str(generation.config_projection)
            original_ctx = ctx.require(ServiceKey("test.credential_context"))
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
        snapshot = host.current_snapshot
        assert snapshot is not None and snapshot.composition_root is not None
        bindings = Bindings(log, host._archive, snapshot.composition_root)
        async with bindings.open(reference, PROBE) as (reader, metadata):
            assert metadata == {}
            assert await reader.read() == "replacement-token"
    finally:
        await host.terminate_all()
        log.close()


@pytest.mark.asyncio
async def test_candidate_registers_credential_consumer_but_cannot_read_secret(tmp_path):
    source, config, log, host = environment(tmp_path)
    try:
        await host.load_all()
        (source / "plugin.py").write_text(MODULE + "\nmarker = 'candidate'\n")
        prepared = await host.prepare_candidate("secret_reader")
        assert prepared is not None
        latest = prepared.runtime_snapshot
        assert latest is not None and latest is not host.current_snapshot
        candidate_store = RuntimeSnapshotStore()
        candidate_store.install(latest)
        try:
            async with lease_runtime_snapshot(candidate_store):
                reader = latest.composition_root.context.require(PROBE)
                with pytest.raises(RuntimeError, match="candidate 验证期"):
                    await reader.read()
                ctx = latest.composition_root.context.require(ServiceKey("test.credential_context"))
                assert not (ctx.data_root / "config.local.toml").exists()
        finally:
            await candidate_store.close()
        assert "fixture-private-token" not in config.read_text()
    finally:
        await host.terminate_all()
        log.close()


@pytest.mark.asyncio
async def test_candidate_rejects_restored_plaintext_backup(tmp_path):
    source, config, log, host = environment(tmp_path)
    try:
        await host.load_all()
        (config.parent / "config.local.toml.before-setup.bak").write_text('token="old-secret"')
        (source / "plugin.py").write_text(MODULE + "\nmarker = 'candidate'\n")
        with pytest.raises(RuntimeError, match="升级"):
            await host.prepare_candidate("secret_reader")
        assert (config.parent / "config.local.toml.before-setup.bak").read_text() == 'token="old-secret"'
    finally:
        await host.terminate_all()
        log.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("shared_directory", [False, True])
async def test_business_validation_never_copies_historical_credentials(tmp_path, shared_directory):
    """候选验证不复制历史 binding 的秘密或重新打开正式服务。"""
    from agent.plugins.install import install_git_plugin
    from tests.test_plugin_install import _commit, _write_v3_plugin

    source, config, log, host = environment(tmp_path)
    (config.parent / "notes.txt").write_text("preserved history")
    try:
        await host.load_all()
        async with lease_runtime_snapshot(host.snapshot_store) as snapshot:
            reference = snapshot.composition_root.context.require(BINDINGS).bind(PROBE, {})
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
        (plain / "plugin.py").write_text((plain / "plugin.py").read_text() + '\nmarker="candidate"\n')
        _commit(plain)
        result, _ = await host.install_candidate(source=str(plain), marketplace="lab", ref_name="", sparse_paths=[])
        if shared_directory:
            save_config(config.parent, {"token": save_credential(config.parent, "fixture-private-token")})
        before = tuple(log._connection.iterdump())
        async with host.open_validation(result.update_id) as scope:
            validation = next(iter(host._validation_hosts.values()))
            copied = validation.workspace / "plugin-data/secret_reader-builtin"
            assert load_config(validation.workspace / "plugin-data/plain-lab")[0] == {"label": "public config"}
            if shared_directory:
                assert (copied / CONFIG_INPUT).is_file()
                assert (copied / "notes.txt").read_text() == "preserved history"
                async with scope.require(BINDINGS).open(reference, PROBE) as (reader, _):
                    with pytest.raises(RuntimeError, match="candidate 验证期"):
                        await reader.read()
            else:
                assert not copied.exists()
                with pytest.raises(RuntimeError, match="当前 runtime scope 不提供服务"):
                    async with scope.require(BINDINGS).open(reference, PROBE):
                        pytest.fail("removed historical provider was reopened")
            for path in validation.workspace.rglob("*"):
                if path.is_file():
                    assert b"fixture-private-token" not in path.read_bytes(), path
        assert tuple(log._connection.iterdump()) == before
        assert "fixture-private-token" not in config.read_text()
    finally:
        await host.terminate_all()
        log.close()

"""用改造前 Core 生成真实命令选择，再验证升级读取不改权威事实。"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[1]
BASE = "bdcc1660"
OWNER = '''from agent.plugin_composition import ServiceKey
from COMMAND_CONTRACT import COMMANDS, CommandDefinition, CommandResult
api_version = 3
name = "command_owner"
version = "1.0.0"
inject = (COMMANDS,)
READY = ServiceKey("scenario.command.ready")
async def apply(ctx):
    def recover(call):
        with (ctx.data_root / "recoveries").open("a") as file:
            file.write("recovered\\n")
        return CommandResult("success", "saved command recovered")
    await ctx.require(COMMANDS).register(ctx, CommandDefinition(
        "probe", "recover saved command", lambda call: CommandResult("success", "new"), recover=recover))
    await ctx.provide(READY, True)
'''


def snapshot(database: Path) -> dict[str, list[tuple[object, ...]]]:
    """读取完整原消息、绑定和 owner_state，不把行数当内容证据。"""
    with sqlite3.connect(database) as connection:
        assert connection.execute("PRAGMA integrity_check").fetchone() == ("ok",)
        return {name: connection.execute(f"SELECT * FROM {name} ORDER BY 1").fetchall()
                for name in ("messages", "bindings", "owner_records", "message_bindings", "attachments",
                             "attachment_imports", "message_attachments", "channel_identities",
                             "channel_identity_migrations", "inbound_handoffs")}


async def seed_old_resources(workspace: Path, log) -> None:
    """由旧发行代码发布附件、路由和待处理交接，不伪造数据库行。"""
    from contextlib import closing
    from dataclasses import asdict
    from agent.plugin_composition.channels import AttachmentKind
    from infra.channels.artifacts import ChannelAttachmentArtifactStore
    from session.artifact_store import ArtifactStore
    from session.identities import ChannelIdentities
    from session.inbound_store import InboundHandoffStore
    from session.message import ContentPart, ContentReferences, Input

    with closing(ArtifactStore(workspace / "sessions.db")) as metadata:
        artifacts = ChannelAttachmentArtifactStore(workspace=workspace, metadata_store=metadata)
        ref = await artifacts.import_bytes(b"old attachment bytes", kind=AttachmentKind.FILE,
            filename="old.txt", media_type="text/plain")
    (workspace / "artifact-id").write_text(ref.artifact_id)
    writer = log.writer("saved", author="human", source="scenario", body_types=(Input,),
        content={"attachment": lambda part: ContentReferences(artifact_ids=(ref.artifact_id,))})
    writer.append("saved-attachment", Input((ContentPart("attachment", asdict(ref)),)))
    with closing(ChannelIdentities(workspace / "sessions.db")) as identities:
        identities.remember("probe", "provider", "recipient")
    with closing(InboundHandoffStore(workspace / "sessions.db")) as inbounds:
        inbounds.reserve_inbound_handoff(handoff_id="pending", dedupe_key="old-pending", channel="probe",
            sender="sender", chat_id="room", session_key="saved", content="not yet accepted",
            timestamp="2026-10-10T00:00:00+00:00", media_json="[]", created_at="2026-10-10T00:00:00+00:00",
            metadata_json=json.dumps({"durable_inbound": True, "durable_handoff_id": "pending",
                "provider_message_id": "provider-pending", "durable_attachment_refs": [asdict(ref)]}))


async def read_current_resources(workspace: Path, root) -> None:
    """新 Ledger 只通过公开窄端口读取旧附件、身份和待处理输入。"""
    from plugins.ledger.contract import CHANNEL_ATTACHMENT_READ, CHANNEL_IDENTITY, INPUT_CUSTODY
    attachments = root.context.require(CHANNEL_ATTACHMENT_READ)
    ref, = attachments.resolve_refs(((workspace / "artifact-id").read_text(),))
    lease = await attachments.acquire(ref)
    try:
        assert await lease.read_bytes(max_bytes=1024) == b"old attachment bytes"
    finally:
        await lease.aclose()
    assert root.context.require(CHANNEL_IDENTITY).resolve("probe", "provider") == "recipient"
    custody = root.context.require(INPUT_CUSTODY)
    key = dict(channel="probe", session_key="saved", provider_message_id="provider-pending")
    assert custody.has_pending_durable_inbound(**key)
    assert custody.pending_durable_attachment_refs(**key) == (ref,)


def artifact_files(workspace: Path) -> dict[str, str]:
    return {str(path.relative_to(workspace)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (workspace / "uploads").rglob("*") if path.is_file()}


async def run(directory: Path, *, old: bool) -> None:
    """在独立进程中使用实际 provider、SQLite 与 OwnerCall。"""
    from agent.plugin_composition import ServiceKey
    from agent.plugins.manager import PluginManager
    from agent.plugins.selection import PluginSelection
    if old:
        from agent.plugin_composition.bindings import BINDINGS
        from session.log import MessageLog, SessionAttributes
        from session.message import ContentPart, ContentReferences, Input
    else:
        from plugins.ledger.contract import BINDINGS, SessionAttributes, ContentPart, ContentReferences, Input
        from plugins.ledger.log import MessageLog

    workspace = directory / "workspace"
    if old:
        workspace.mkdir()
        PluginSelection(workspace).initialize()
    log = MessageLog(workspace / "sessions.db")
    if old:
        from bus.event_bus import EventBus
        host = PluginManager([directory / "providers"], workspace=workspace,
                             installed_cache_root=directory / "home/cache", message_log=log,
                             event_bus=EventBus())
    else:
        host = PluginManager([directory / "providers"], workspace=workspace,
                             installed_cache_root=directory / "home/cache")
    if old:
        service = ServiceKey("core.commands")
    else:
        from plugins.commands.contract import COMMANDS as service
    try:
        await host.load_all()
        if not old:
            await host.install(source=str(directory / "providers/ledger"), marketplace="lab",
                               ref_name="", sparse_paths=[], update_id="add-ledger")
            await host._operation.task
        root = host.live_root
        assert root is not None
        assert root.context.require(ServiceKey("scenario.command.ready")) is True
        bindings = root.context.require(BINDINGS)
        if old:
            # 1. 旧服务从真实登记闭包生成 hash 与选择，不手工伪造 descriptor。
            context = root._service_provider(service)[0]
            async with context.runtime_scope():
                identity = context.require(service).freeze().bind(bindings, "/probe")
            assert identity is not None
            descriptor = log.read_binding(identity)
            assert descriptor["service"] == "core.commands"
            payload = json.dumps(descriptor, sort_keys=True, ensure_ascii=False,
                                 separators=(",", ":"))
            assert hashlib.sha256(payload.encode()).hexdigest() == identity
            (directory / "binding-id").write_text(identity)
            log.ensure_session("saved", SessionAttributes())
            writer = log.writer("saved", author="human", source="scenario", body_types=(Input,),
                                content={"text": lambda part: ContentReferences(binding_ids=(identity,))})
            writer.append("saved-input", Input((ContentPart("text", "/probe"),)))
            log.owner("scenario").transact(lambda transaction: transaction.save(
                "admitted-command", {"binding_id": identity, "message_id": "saved-input"},
                expected_version=None))
            await seed_old_resources(workspace, log)
        else:
            await read_current_resources(workspace, root)
            # 2. 当前 provider 接管恢复，旧运行 key 没有重新登记。
            assert root.context.get(ServiceKey("core.commands")) is None
            identity = (directory / "binding-id").read_text()
            async with root.context.open_service(service), bindings.open(identity, service) as (selected, metadata):
                assert metadata == {"name": "probe"}
                result = await selected.freeze().execute(
                    "/probe", session_key="saved", channel="scenario", chat_id="room",
                    sender="human", recover=True)
                assert result is not None and result.result.text == "saved command recovered"
            assert bindings.describe(identity, service) == {"name": "probe"}
            assert log.read_binding(identity)["service"] == "core.commands"
            try:
                bindings.describe(identity, ServiceKey("unrelated.service"))
            except ValueError:
                pass
            else:
                raise AssertionError("历史 service 解码不能接受无关 provider")
    finally:
        await host.terminate_all()
        log.close()


def main() -> None:
    if len(sys.argv) == 4 and sys.argv[1] == "--child":
        directory = Path(sys.argv[2])
        old = sys.argv[3] == "old"
        sys.path.insert(0, str(directory / "old" if old else ROOT))
        asyncio.run(run(directory, old=old))
        return
    sys.path.insert(0, str(ROOT))
    with tempfile.TemporaryDirectory(prefix="akashic-binding-upgrade-") as temporary:
        directory = Path(temporary)
        (directory / "config.toml").write_text("[runtime]\n")
        # 1. 归档准确旧基线，旧进程无法从当前 checkout 导入 Core。
        archive = directory / "old.tar"
        with archive.open("wb") as output:
            subprocess.run(["git", "archive", BASE, "agent", "bootstrap", "bus", "core",
                            "infra", "session", "utils", "plugins/commands"], cwd=ROOT, stdout=output, check=True)
        old = directory / "old"
        old.mkdir()
        with tarfile.open(archive) as source:
            source.extractall(old, filter="data")
        providers = directory / "providers"
        shutil.copytree(old / "plugins/commands", providers / "commands")
        owner = providers / "command_owner"
        owner.mkdir()
        (owner / "plugin.py").write_text(OWNER.replace("COMMAND_CONTRACT", "agent.plugin_composition.commands"))
        script = directory / "scenario.py"
        shutil.copyfile(__file__, script)
        environment = {**os.environ, "HOME": str(directory / "home"), "PYTHONPATH": "",
                       "AKASHIC_PLUGIN_HOME": str(directory / "home"), "AKASHIC_PLUGIN_DISTRIBUTION": ""}
        subprocess.run([sys.executable, str(script), "--child", str(directory), "old"],
                       cwd=directory, env=environment, check=True)
        database = directory / "workspace/sessions.db"
        before = snapshot(database)
        files = artifact_files(directory / "workspace")
        assert files
        # 栈内 journal schema 也已升级；走实际迁移入口，不绕过校验或删除旧账本。
        from agent.migrations.runner import MigrationRunner
        MigrationRunner(repo_root=ROOT, config_path=directory / "config.toml",
                        workspace=directory / "workspace", plugin_dirs=[ROOT / "plugins/ledger"]).run()
        assert snapshot(database) == before
        assert artifact_files(directory / "workspace") == files
        # 2. 同一 workspace 只升级源码，原绑定和消息没有数据管理写入。
        shutil.rmtree(providers / "commands")
        shutil.copytree(ROOT / "plugins/commands", providers / "commands", ignore=shutil.ignore_patterns("__pycache__"))
        (owner / "plugin.py").write_text(OWNER.replace("COMMAND_CONTRACT", "plugins.commands.contract"))
        for name in ("ledger", "channels"):
            shutil.copytree(ROOT / "plugins" / name, providers / name,
                            ignore=shutil.ignore_patterns("__pycache__"))
        ledger = providers / "ledger"
        subprocess.run(["git", "init", "-q", str(ledger)], check=True)
        subprocess.run(["git", "-C", str(ledger), "add", "."], check=True)
        subprocess.run(["git", "-C", str(ledger), "-c", "user.name=Scenario", "-c",
                        "user.email=scenario@example.invalid", "commit", "-qm", "ledger"], check=True)
        environment["PYTHONPATH"] = str(ROOT)
        subprocess.run([sys.executable, str(script), "--child", str(directory), "new"],
                       cwd=directory, env=environment, check=True)
        assert snapshot(database) == before
        assert artifact_files(directory / "workspace") == files
        assert (directory / "workspace/plugin-data/command_owner-builtin/recoveries").read_text() == "recovered\n"
        print(json.dumps({"old_core_generated": True, "current_provider_recovered_once": True,
                          "no_old_runtime_key": True, "binding_message_owner_rows_unchanged": True, "attachments_readable": True,
                          "identity_and_pending_input_readable": True, "all_source_rows_and_files_unchanged": True}))


if __name__ == "__main__":
    main()

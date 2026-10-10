"""真实 Ledger/Channel 的提交后崩溃与插件换代验收。"""
import asyncio
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PROBE = '''import asyncio
from datetime import UTC, datetime
from agent.plugin_composition import ServiceKey
from plugins.channels.contract import (CHANNELS, CHANNEL_INPUT_V2, ChannelDefinition,
    ChannelCapability, InboundIdentity, ChannelReady, StopReceipt, RawInbound,
    ChannelInboundMessage)
from plugins.ledger.contract import (MESSAGE_WRITERS, OWNER_STATE, SESSION_ADMISSION,
    Input, Output, ContentPart, ContentReferences, SessionAttributes)
api_version = 3
name = "ledger_probe"
version = "1.0.0"
inject = (CHANNELS, MESSAGE_WRITERS, OWNER_STATE, SESSION_ADMISSION)
async def apply(ctx):
    ready = asyncio.Event()
    adapters = []
    ctx.require(SESSION_ADMISSION).ensure(ctx, "probe:room", SessionAttributes())
    writer = ctx.require(MESSAGE_WRITERS).bind(ctx, author="scenario", source="scenario",
        body_types=(Input, Output), content={"text": lambda part: ContentReferences()})
    owner = ctx.require(OWNER_STATE).open(ctx)
    async def accept(session, identity, message):
        def commit(tx):
            write = writer(session)
            result = tx.append(write, identity, Input((ContentPart("text", message.content),)))
            tx.append(write, identity + ":output", Output((ContentPart("text", "reply:" + message.content),), "complete"))
            if tx.read(identity) is None:
                tx.save(identity, {"committed": True}, expected_version=None)
            return result
        result = owner.transact(commit)
        if (ctx.runtime.workspace / "crash").exists():
            print("COMMITTED", flush=True)
            await asyncio.Event().wait()
        return result
    await ctx.provide(CHANNEL_INPUT_V2, accept)
    class Adapter:
        def __init__(self, factory):
            self.factory = factory
            self.ports = None
            adapters.append(self)
        def attach_runtime(self, ports):
            self.ports = ports
        async def start(self):
            return ChannelReady(self.factory.binding_token)
        def open_admission(self):
            ready.set()
        def close_admission(self):
            ready.clear()
        async def stop(self):
            return StopReceipt(self.factory.binding_token, True)
        async def deliver(self, request):
            raise RuntimeError("inbound-only scenario adapter")
    await ctx.require(CHANNELS).register(ctx, ChannelDefinition("probe",
        frozenset({ChannelCapability.INBOUND, ChannelCapability.DURABLE_INBOUND}),
        Adapter, InboundIdentity.PROVIDER_MESSAGE_ID))
    async def receive(identity):
        await ready.wait()
        ports = adapters[-1].ports
        raw = RawInbound(identity, ChannelInboundMessage("probe", "sender", "room", identity,
            datetime.now(UTC), {"durable_inbound": True, "durable_handoff_id": identity,
            "provider_message_id": identity, "session_key_override": "probe:room"}), "sender", "room")
        assert await ports.durable_inbound.reserve(raw)
        assert await ports.ingress.admit(raw)
    await ctx.provide(ServiceKey("scenario.receive"), receive)
'''

async def child(folder, phase):
    from agent.plugin_composition import ServiceKey
    from agent.plugins.manager import PluginManager
    from plugins.ledger.contract import MESSAGE_CATALOG
    host = PluginManager([folder / 'sources'], workspace=folder / 'workspace',
                         installed_cache_root=folder / 'home/cache')
    try:
        await host.load_all()
        await host.start_runtime()
        root = host.live_root
        assert root is not None
        async def receive(identity):
            async with root.context.open_service(ServiceKey('scenario.receive')) as port:
                await port(identity)
        if phase == 'crash':
            await receive('first')
            (folder / 'workspace/crash').touch()
            await receive('second')
        else:
            catalog = root.context.require(MESSAGE_CATALOG)
            async with asyncio.timeout(10):
                while True:
                    with sqlite3.connect(folder / 'workspace/sessions.db') as db:
                        if db.execute('SELECT count(*) FROM inbound_handoffs').fetchone()[0] == 0:
                            break
                    await asyncio.sleep(.01)
            await receive('third')
            before = catalog.reader('probe:room').snapshot()
            observer = host._active_generations['observer'].fiber
            # 安装同一真实 Ledger 的新 generation；数据和无关 Fiber 必须保持。
            await host.install(source=str(folder / 'ledger'), marketplace='lab', ref_name='',
                               sparse_paths=[], update_id='ledger-replace')
            await host.wait_idle()
            assert host.read_update('ledger-replace').state == 'active'
            assert host._active_generations['observer'].fiber is observer
            assert root.context.require(MESSAGE_CATALOG).reader('probe:room').snapshot() == before
            await receive('fourth')
            await host.uninstall('ledger@lab')
            await host.wait_idle()
            assert host._active_generations['observer'].fiber is observer
            print('RECOVERED', flush=True)
    finally:
        await host.terminate_all()

async def parent(folder):
    from agent.plugins.selection import PluginSelection
    from agent.plugins.install import install_git_plugin
    workspace = folder / 'workspace'
    workspace.mkdir()
    PluginSelection(workspace).initialize()
    os.environ.update(HOME=str(folder / 'home'), AKASHIC_PLUGIN_HOME=str(folder / 'home'), AKASHIC_PLUGIN_DISTRIBUTION='')
    for name in ('channels', 'onboarding', 'ledger_invariants'):
        shutil.copytree(ROOT / 'plugins' / name, folder / 'sources' / name, ignore=shutil.ignore_patterns('__pycache__'))
    ledger = folder / 'ledger'
    shutil.copytree(ROOT / 'plugins/ledger', ledger, ignore=shutil.ignore_patterns('__pycache__'))
    subprocess.run(['git', 'init', '-q', str(ledger)], check=True)
    subprocess.run(['git', '-C', str(ledger), 'add', '.'], check=True)
    subprocess.run(['git', '-C', str(ledger), '-c', 'user.name=Scenario', '-c', 'user.email=scenario@example.invalid', 'commit', '-qm', 'first'], check=True)
    install_git_plugin(workspace=workspace, source=str(ledger), marketplace='lab', plugins_home=folder / 'home')
    probe = folder / 'sources/ledger_probe'
    probe.mkdir()
    (probe / 'plugin.py').write_text(PROBE)
    observer = folder / 'sources/observer'
    observer.mkdir()
    (observer / 'plugin.py').write_text('api_version=3\nname="observer"\nversion="1.0.0"\nasync def apply(ctx):\n    pass\n')
    command = [sys.executable, __file__, str(folder)]
    process = await asyncio.create_subprocess_exec(*command, 'crash', stdout=asyncio.subprocess.PIPE)
    try:
        async with asyncio.timeout(20):
            line = await process.stdout.readline()
            assert line == b'COMMITTED\n', line
    finally:
        if process.returncode is None:
            process.kill()
        await process.wait()
    with sqlite3.connect(workspace / 'sessions.db') as db:
        before = db.execute('SELECT * FROM messages ORDER BY seq').fetchall()
        assert len(before) == 4
    (workspace / 'crash').unlink()
    (ledger / 'plugin.py').write_text((ledger / 'plugin.py').read_text() + '\n# replacement scenario\n')
    subprocess.run(['git', '-C', str(ledger), 'add', '.'], check=True)
    subprocess.run(['git', '-C', str(ledger), '-c', 'user.name=Scenario', '-c', 'user.email=scenario@example.invalid', 'commit', '-qm', 'second'], check=True)
    process = await asyncio.create_subprocess_exec(*command, 'recover')
    async with asyncio.timeout(40):
        assert await process.wait() == 0
    with sqlite3.connect(workspace / 'sessions.db') as db:
        assert db.execute('PRAGMA integrity_check').fetchone() == ('ok',)
        after = db.execute('SELECT * FROM messages ORDER BY seq').fetchall()
        assert after[:4] == before and len(after) == 8
    print(json.dumps({'commit_before_kill': True, 'recovery_no_duplicate': True, 'four_rounds': True,
                      'ledger_replaced_and_removed': True, 'observer_unchanged': True, 'rows_preserved': True}))

if len(sys.argv) > 1:
    asyncio.run(child(Path(sys.argv[1]), sys.argv[2]))
else:
    with tempfile.TemporaryDirectory(prefix='akashic-ledger-runtime-') as folder:
        asyncio.run(parent(Path(folder)))

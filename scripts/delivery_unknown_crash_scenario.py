"""真实外部文件已写出后 kill，重启不把未知非幂等发送伪装成成功或重发。"""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import shutil
import sqlite3
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PROBE = '''import asyncio
from contextlib import asynccontextmanager
from agent.plugin_composition import ServiceKey
from plugins.delivery.contract import DELIVERY_SENDERS, DELIVERY_GUARDED_START
from plugins.ledger.contract import (BINDINGS, MESSAGE_CATALOG, MESSAGE_WRITERS,
    SESSION_ADMISSION, SessionAttributes, Output, ContentPart, ContentReferences)
api_version = 3
name = "sender-probe"
version = "1.0.0"
inject = (DELIVERY_SENDERS, DELIVERY_GUARDED_START, BINDINGS, MESSAGE_CATALOG,
          MESSAGE_WRITERS, SESSION_ADMISSION)
async def apply(ctx):
    class Sender:
        idempotent = False
        async def send(self, key, address, message):
            with (ctx.data_root / "external-effect").open("a") as file:
                file.write(message.message_id + "\\n")
            print("EXTERNAL_SENT", flush=True)
            await asyncio.Event().wait()
        async def query(self, key, address):
            return None
    @asynccontextmanager
    async def open_sender():
        yield Sender()
    await ctx.require(DELIVERY_SENDERS).register(ctx, name="probe", idempotent=False, open=open_sender)
    async def run(first):
        delivery = ctx.require(DELIVERY_GUARDED_START).open(ctx)
        if first:
            ctx.require(SESSION_ADMISSION).ensure(ctx, "scenario", SessionAttributes())
            writer = ctx.require(MESSAGE_WRITERS).bind(ctx, author="assistant", source="scenario",
                body_types=(Output,), content={"text": lambda value: ContentReferences()})("scenario")
            message = writer.append("saved-output", Output((ContentPart("text", "already sent"),), "complete"))
            binding = ctx.require(DELIVERY_SENDERS).bind("probe", ctx.require(BINDINGS))
            await delivery.prepare_async(ctx.require(MESSAGE_CATALOG).reader("scenario"), message,
                ({"name": "probe", "binding_id": binding, "address": "recipient"},))
        receipt = await delivery.send("saved-output", "probe")
        return {"status": receipt.status, "error": receipt.error}
    await ctx.provide(ServiceKey("scenario.send"), ctx.entrypoint(run))
'''


async def child(directory: Path, first: bool) -> None:
    from agent.plugin_composition import ServiceKey
    from agent.plugins.manager import PluginManager
    host = PluginManager([directory / "sources"], workspace=directory / "workspace",
        installed_cache_root=directory / "home/cache")
    try:
        await host.load_all()
        assert all(item.load_error is None for item in host._active_generations.values()), [(name, str(item.load_error)) for name, item in host._active_generations.items()]
        result = await host.live_root.context.require(ServiceKey("scenario.send"))(first)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    finally:
        await host.terminate_all()


async def run(directory: Path) -> dict[str, object]:
    """两个独立进程使用同一真实 Ledger/Delivery；对端是不可查询的文件发送者。"""
    from agent.plugins.selection import PluginSelection
    from agent.plugins.bundles import set_plugin_choice
    workspace = directory / "workspace"
    workspace.mkdir()
    PluginSelection(workspace).initialize()
    set_plugin_choice(workspace, "channels", enabled=False)
    for name in ("ledger", "delivery", "channels"):
        shutil.copytree(ROOT / "plugins" / name, directory / "sources" / name,
                        ignore=shutil.ignore_patterns("__pycache__"))
    probe = directory / "sources/sender-probe"
    probe.mkdir()
    (probe / "plugin.py").write_text(PROBE)
    environment = {**os.environ, "HOME": str(directory / "home"), "AKASHIC_PLUGIN_HOME": str(directory / "home"),
        "AKASHIC_PLUGIN_DISTRIBUTION": "", "PYTHONPATH": str(ROOT)}
    async def start(mode):
        return await asyncio.create_subprocess_exec(sys.executable, __file__, "--child", str(directory), mode,
            env=environment, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    first = await start("first")
    try:
        async with asyncio.timeout(20):
            while True:
                line = await first.stdout.readline()
                if line.strip() == b"EXTERNAL_SENT":
                    break
                if not line:
                    raise AssertionError((await first.stderr.read()).decode())
        first.kill()
        await first.communicate()
    finally:
        if first.returncode is None:
            first.kill()
            await first.communicate()
    with sqlite3.connect(workspace / "sessions.db") as db:
        original = db.execute("SELECT * FROM messages ORDER BY seq").fetchall()
        rows = [json.loads(row[0]) for row in db.execute("SELECT value FROM owner_records WHERE key LIKE 'delivery:%'")]
        assert len(rows) == 1 and rows[0]["phase"] == "started", rows
    second = await start("recover")
    try:
        output, errors = await asyncio.wait_for(second.communicate(), 20)
        assert second.returncode == 0, (output, errors)
        receipt = json.loads(output.splitlines()[-1])
        assert receipt["status"] == "failed" and "可能已送达" in receipt["error"], receipt
    finally:
        if second.returncode is None:
            second.kill()
            await second.communicate()
    assert (workspace / "plugin-data/sender-probe-builtin/external-effect").read_text() == "saved-output\n"
    with sqlite3.connect(workspace / "sessions.db") as db:
        assert db.execute("SELECT * FROM messages ORDER BY seq").fetchall() == original
        assert db.execute("PRAGMA integrity_check").fetchone() == ("ok",)
    return {"killed_after_external_send": True, "recovered_status": "failed", "send_count": 1,
            "source_messages_unchanged": True}


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--child":
        asyncio.run(child(Path(sys.argv[2]), sys.argv[3] == "first"))
    else:
        with tempfile.TemporaryDirectory(prefix="akashic-delivery-crash-") as temporary:
            print(json.dumps(asyncio.run(run(Path(temporary)))))

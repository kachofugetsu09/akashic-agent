"""真实插件组合中验证追加检查、事务回滚、停用与重新挂载。"""
from __future__ import annotations

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
PROBE = '''from agent.plugin_composition import ServiceKey
from plugins.ledger.contract import (MESSAGE_WRITERS, OWNER_STATE, Input, Output,
    Control, ToolCall, ToolResult, CallRef)
api_version=3
name="probe"
version="1.0.0"
inject=(MESSAGE_WRITERS, OWNER_STATE)
async def apply(ctx):
    write=ctx.require(MESSAGE_WRITERS).bind(ctx, author="probe", source="probe",
        body_types=(Input, Output, Control), content={})
    owner=ctx.require(OWNER_STATE).open(ctx)
    async def submit(identity, body, transaction=False):
        writer=write("room")
        if transaction:
            def commit(tx):
                tx.save(identity, {"commit": True}, expected_version=None)
                return tx.append(writer, identity, body)
            return await owner.transact_async(commit)
        return await writer.append_async(identity, body)
    await ctx.provide(ServiceKey("scenario.submit"), submit)
'''


async def run(base: Path) -> None:
    from agent.plugins.manager import PluginManager
    from agent.plugins.bundles import set_plugin_choice
    from agent.plugin_composition import ServiceKey
    from plugins.ledger.contract import MESSAGE_CATALOG, Control, Input, Output, MessageConflict

    workspace, sources = base / "workspace", base / "sources"
    workspace.mkdir()
    from agent.plugins.selection import PluginSelection
    PluginSelection(workspace).initialize()
    os.environ.update(HOME=str(base / "home"), AKASHIC_PLUGIN_HOME=str(base / "home"),
                      AKASHIC_PLUGIN_DISTRIBUTION="", AKASHIC_EXTRA_PLUGIN_DIRS="")
    for name in ("ledger",):
        shutil.copytree(ROOT / "plugins" / name, sources / name, ignore=shutil.ignore_patterns("__pycache__"))
    (sources / "probe").mkdir()
    (sources / "probe/plugin.py").write_text(PROBE)
    from agent.plugins.install import install_git_plugin
    checks = base / "checks"
    shutil.copytree(ROOT / "plugins/ledger_invariants", checks, ignore=shutil.ignore_patterns("__pycache__"))
    subprocess.run(["git", "init", "-q", str(checks)], check=True)
    subprocess.run(["git", "-C", str(checks), "add", "."], check=True)
    subprocess.run(["git", "-C", str(checks), "-c", "user.name=Scenario", "-c",
                    "user.email=scenario@example.invalid", "commit", "-qm", "checks"], check=True)
    install_git_plugin(workspace=workspace, source=str(checks), marketplace="lab", plugins_home=base / "home")
    host = PluginManager([sources], workspace=workspace, installed_cache_root=base / "home/cache")
    try:
        await host.load_all()
        await host.start_runtime()
        root = host.live_root
        assert root is not None
        async def submit(identity, body, transaction=False):
            async with root.context.open_service(ServiceKey("scenario.submit")) as port:
                return await port(identity, body, transaction)
        # 1. 无 Input 的 Output 合法；已闭合前缀不能再次 abandon。
        first = await submit("first", Output((), "complete"))
        reader = root.context.require(MESSAGE_CATALOG).reader("room")
        baseline = reader.snapshot()
        for identity, body in (("future", Control("abandon", first.seq)), ("closed", Control("abandon", first.seq))):
            try:
                await submit(identity, body, True)
            except (MessageConflict, ValueError):
                pass
            else:
                raise AssertionError(f"无效控制已提交: {identity}")
            assert reader.snapshot() == baseline
            with sqlite3.connect(workspace / "sessions.db") as db:
                assert db.execute("SELECT count(*) FROM owner_records WHERE key=?", (identity,)).fetchone() == (0,)
        # 2. 真实损坏的 next_seq 即使没撞 UNIQUE，也不能把新消息插到既有前缀中。
        await submit("second", Input(()))
        with sqlite3.connect(workspace / "sessions.db") as db:
            db.execute("UPDATE sessions SET next_seq=5 WHERE key='room'")
        await submit("gap", Input(()))
        with sqlite3.connect(workspace / "sessions.db") as db:
            db.execute("UPDATE sessions SET next_seq=3 WHERE key='room'")
        try:
            await submit("bad-seq", Input(()))
        except MessageConflict:
            pass
        else:
            raise AssertionError("损坏序号未被追加检查阻止")
        with sqlite3.connect(workspace / "sessions.db") as db:
            db.execute("UPDATE sessions SET next_seq=6 WHERE key='room'")
        before = reader.snapshot()
        ledger = host._active_generations["ledger"].fiber
        # 3. 只撤回检查，不重启 Ledger；重新挂载后规则恢复。
        set_plugin_choice(workspace, "ledger_invariants@lab", enabled=False)
        await host.reconcile_changed()
        assert host._active_generations["ledger"].fiber is ledger
        await submit("without-check", Control("abandon", first.seq))
        await host.install(source=str(checks), marketplace="lab", ref_name="", sparse_paths=[], update_id="checks-return")
        await host.wait_idle()
        assert host.read_update("checks-return").state == "active"
        try:
            await submit("with-check", Control("abandon", first.seq))
        except MessageConflict:
            pass
        else:
            raise AssertionError("检查没有随重新挂载恢复")
        assert reader.snapshot()[:len(before)] == before
        assert host._active_generations["ledger"].fiber is ledger
    finally:
        await host.terminate_all()
    print(json.dumps({"transaction_rollback": True, "sequence_corruption_rejected": True,
                      "disable_and_reload": True, "ledger_unchanged": True}))


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="akashic-ledger-checks-") as base:
        asyncio.run(run(Path(base)))

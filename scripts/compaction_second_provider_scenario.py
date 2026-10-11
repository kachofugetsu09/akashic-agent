"""真实压缩生成摘要后，用独立归档替换摘要查询 owner。"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
import sqlite3
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CONSUMER = '''from agent.plugin_composition import ServiceKey
from plugins.compaction.contract import COMPACTION_SUMMARIES
api_version = 3
name = "summary-reader"
version = "1.0.0"
async def apply(ctx):
    with (ctx.data_root / "applies").open("a") as stream:
        stream.write("apply\\n")
    async def read(session):
        with ctx.borrow(COMPACTION_SUMMARIES) as lookup:
            if lookup is None:
                raise LookupError("summary provider unavailable")
            row = lookup.head(session)
            if row is None:
                raise LookupError(session)
            selected = lookup.resolve({"record_ref": row.reference, "session_id": session}, session_id=session)
            return {"reference": selected.reference, "source_message_ids": selected.source_message_ids,
                    "content": selected.content}
    await ctx.provide(ServiceKey("scenario.summary.read"), ctx.entrypoint(read))
'''


def add_reader(sources: Path) -> None:
    source = sources / "summary-reader"
    source.mkdir()
    (source / "plugin.py").write_text(CONSUMER)


async def replace(host, folder: Path) -> None:
    """使用实际已发布摘要导出只读副本；原库的每行保持相同。"""
    from agent.plugin_composition import ServiceKey
    from agent.plugins.bundles import set_plugin_choice
    from scripts.artifact_provider_scenario import repository

    root = host.live_root
    read = root.context.require(ServiceKey("scenario.summary.read"))
    before = await read("test:room")
    consumer = host.generation("summary-reader").fiber
    source = folder / "archive-source"
    repository(ROOT / "examples/summary_archive", source)
    with sqlite3.connect(folder / "workspace/sessions.db") as db:
        records = [json.loads(row[0]) for row in db.execute(
            "SELECT value FROM owner_records WHERE owner='plugin:compaction' AND key LIKE 'summary:%'")]
        original = db.execute("SELECT * FROM messages ORDER BY seq").fetchall()
        owned = db.execute("SELECT * FROM owner_records ORDER BY owner,key").fetchall()
    fields = ("version", "reference", "session_id", "generation", "parent", "source_message_ids",
              "summary_message_ids", "omitted_message_ids", "content")
    data = folder / "workspace/plugin-data/summary-archive-lab"
    data.mkdir(parents=True)
    (data / "summaries.json").write_text(json.dumps([{key: row[key] for key in fields} for row in records]))
    set_plugin_choice(folder / "workspace", "compaction", enabled=False)
    await host.reconcile_disabled_and_drain("compaction")
    await host.install(source=str(source), marketplace="lab", ref_name="", sparse_paths=[], update_id="archive")
    await host.wait_idle()
    assert host.read_update("archive").state == "active"
    assert await read("test:room") == before
    assert host.generation("summary-reader").fiber is consumer
    assert (folder / "workspace/plugin-data/summary-reader-builtin/applies").read_text() == "apply\n"
    with sqlite3.connect(folder / "workspace/sessions.db") as db:
        assert db.execute("SELECT * FROM messages ORDER BY seq").fetchall() == original
        assert db.execute("SELECT * FROM owner_records ORDER BY owner,key").fetchall() == owned
    await host.uninstall("summary-archive@lab")
    await host.wait_idle()
    try:
        await read("test:room")
    except LookupError:
        pass
    else:
        raise AssertionError("卸载后不应读到已退役 provider")
    await host.install(source=str(source), marketplace="lab", ref_name="", sparse_paths=[], update_id="reopen-archive")
    await host.wait_idle()
    assert await read("test:room") == before


if __name__ == "__main__":
    from scripts.compaction_workflow import run
    with tempfile.TemporaryDirectory(prefix="akashic-second-summary-") as directory:
        result = asyncio.run(run(Path(directory), "short_tail", second_provider=True))
        print(json.dumps({**result, "independent_summary_lookup": True, "consumer_apply_delta": 0,
                          "original_rows_unchanged": True, "uninstall_reopen": True}))

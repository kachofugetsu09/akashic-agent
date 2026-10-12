"""隔离加载真实插件，验证非法合并被拒绝，文件冲突后原草稿自动重试。"""
from __future__ import annotations

import asyncio
from contextlib import closing
import json
import logging
from pathlib import Path
import shutil
import sqlite3
import sys
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plugins.markdown_memory.store import MarkdownProfileStore
from agent.plugin_composition.config_input import save_config
from plugins.content.plugin import check_text
from plugins.ledger.contract import ContentPart, Output
from tests.test_default_reply import application


def receipts(path: Path) -> list[tuple[str, str, str]]:
    with closing(sqlite3.connect(path)) as connection:
        return connection.execute(
            "SELECT source_ref, kind, payload FROM consolidation_writes ORDER BY source_ref, kind",
        ).fetchall()


async def run(folder: Path) -> None:
    """通过模型 HTTP 边界、磁盘冲突与正式 follower 观察拒绝、恢复和原文保全。"""
    # 1. 准备一次性记忆及达到整理门槛的真实 receipt，不操作正式 workspace。
    memory = folder / "workspace/memory"
    database = memory / "markdown-profile-writes.db"
    store = MarkdownProfileStore(memory, database)
    topic = memory / "topics/editor.md"
    original = "---\ndescription: 编辑器约定\n---\n- 使用 helix\n- 用 helix 编辑代码\n"
    topic.write_text(original)
    protected = {doc: store.read(doc) for doc in ("MEMORY.md", "SELF.md")}
    for index in range(5):
        reference = f"seed-{index}"
        store.write_draft(reference, {"version": 3, "scope": None, "ops": []},
                          session_key="seed", generation=index)
        store.apply(reference)
    initial_receipts = receipts(database)
    conflict = asyncio.Event()
    retried = asyncio.Event()
    pruned = asyncio.Event()
    important = "- 保留重要的协作约定。"
    unimportant = "- 临时线索：" + "x" * 90
    requests: list[str] = []
    failure: list[Exception] = []

    class ConflictLog(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            if "保留草稿稍后重试" in record.getMessage():
                conflict.set()
            if "超额剪枝已完成" in record.getMessage():
                pruned.set()

    # 2. 真实模型端口先返回非法空行，修正后返回有效 merge；恢复后的新整理无操作。
    async def serve(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            header = (await reader.readuntil(b"\r\n\r\n")).decode()
            length = int(next(row.split(":", 1)[1] for row in header.split("\r\n")
                              if row.lower().startswith("content-length:")))
            payload = json.loads(await reader.readexactly(length))
            requests.append(str(payload["messages"]))
            if len(requests) == 2:
                assert "上次草稿没有提交，校验失败" in requests[-1]
                assert topic.read_text() == original and receipts(database) == initial_receipts
                (memory / "review-conflicts").write_text("3")
            ops = [] if len(requests) > 2 else [{
                "op": "merge", "doc": "topics/editor.md",
                "olds": ["- 使用 helix", "- 用 helix 编辑代码"],
                "line": "" if len(requests) == 1 else "- 使用 helix 编辑代码",
            }]
            if "你在压缩超额的长期 Markdown 记忆" in requests[-1]:
                ops = [{"op": "prune", "doc": "MEMORY.md", "old": unimportant}]
            body = json.dumps({"text": json.dumps({"ops": ops}, ensure_ascii=False)}).encode()
            writer.write(f"HTTP/1.1 200 OK\r\nContent-Length: {len(body)}\r\nConnection: close\r\n\r\n".encode() + body)
            await writer.drain()
            if len(requests) > 2:
                retried.set()
        except (AssertionError, ValueError, KeyError, asyncio.IncompleteReadError) as error:
            failure.append(error)
            conflict.set()
            retried.set()
            pruned.set()
        finally:
            writer.close()
            await writer.wait_closed()

    server = await asyncio.start_server(serve, "127.0.0.1", 0)
    url = f"http://127.0.0.1:{server.sockets[0].getsockname()[1]}/complete"

    def configure(sources: Path) -> None:
        save_config(folder / "workspace/plugin-data/context-builtin", {
            "summary_source": ["compaction", "compaction"],
            "prompt_sources": {"markdown_memory": "markdown_memory"},
        })
        save_config(folder / "workspace/plugin-data/markdown_memory-builtin", {"memory_budget": 80})
        for name in ("markdown_memory", "runtime_inspection"):
            shutil.copytree(Path(__file__).resolve().parents[1] / "plugins" / name, sources / name,
                            ignore=shutil.ignore_patterns("__pycache__"))
        entry = sources / "test_provider/plugin.py"
        source = entry.read_text()
        start, end = source.index("    class Driver:"), source.index("    descriptor =")
        source = source[:start] + f'''    class Driver:
        max_tool_schemas = None
        def estimate_context_tokens(self, messages, tools):
            return len(str(messages)) // 4
        async def complete(self, request):
            import httpx
            from plugins.ledger.contract import json_value
            async with httpx.AsyncClient(trust_env=False) as client:
                result = await client.post({url!r}, json={{"messages": json_value(request.messages)}})
                result.raise_for_status()
                return LLMResponse(result.json()["text"])
''' + source[end:]
        source = source.replace("    class Target:", "        independent_execution = execution\n    class Target:")
        entry.write_text(source)
        # 仅在隔离副本注入磁盘写入：每次快照后、替换前修改实际文件，共三次。
        entry = sources / "markdown_memory/store.py"
        source = entry.read_text()
        hook = "            workspace.apply(ops, strict=False)"
        assert source.count(hook) == 1
        source = source.replace(hook, hook + '''
            marker = self.root / "review-conflicts"
            if marker.exists() and (remaining := int(marker.read_text())) > 0:
                path = self.root / "topics/editor.md"
                path.write_text(path.read_text() + f"- external {remaining}\\n")
                marker.write_text(str(remaining - 1))''')
        entry.write_text(source)

    handler = ConflictLog()
    logger = logging.getLogger("plugins.markdown_memory")
    previous_level = logger.level
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    try:
        async with application(folder, replying=False, compaction=True, extra_sources=configure) as (log, _host):
            await asyncio.wait_for(conflict.wait(), 10)
            if failure:
                raise failure[0]
            pending = store.pending_source_refs()
            assert len(pending) == 1 and pending[0].startswith("consolidation:")
            assert not store.is_applied(pending[0])
            external = "- external 3\n- external 2\n- external 1\n"
            assert topic.read_text() == original + external
            assert store._read_receipt(pending[0], "markdown_before_v1") is None
            # 3. 不发送任何新消息；等待正式 30 秒重试，确认同一草稿完成且人工增量保留。
            await asyncio.wait_for(retried.wait(), 40)
            if failure:
                raise failure[0]
            assert store.is_applied(pending[0]) and not store.pending_source_refs()
            assert topic.read_text() == original.replace(
                "- 使用 helix\n- 用 helix 编辑代码\n", "- 使用 helix 编辑代码\n") + external
            assert store._read_receipt(pending[0], "markdown_before_v1") == {"topics/editor.md": original + external}
            assert all(store.read(doc) == text for doc, text in protected.items())
            assert all(row in receipts(database) for row in initial_receipts)
            assert log.reader("seed").snapshot() == ()
            # 4. 模拟原生文件编辑超额；真实 Output 唤醒 follower，未到定期整理时间也能自愈。
            before_pruning = store.read("MEMORY.md").replace("## 用户偏好", "## 用户偏好\n" + important + "\n" + unimportant)
            (memory / "MEMORY.md").write_text(before_pruning)
            log.writer("pruning", author="assistant", source="programmatic", body_types=(Output,),
                       content={"text": check_text}).append("wake-pruning", Output((ContentPart("text", "检查容量"),), "complete"))
            await asyncio.wait_for(pruned.wait(), 10)
            if failure:
                raise failure[0]
            assert important in store.read("MEMORY.md") and unimportant not in store.read("MEMORY.md")
            pruning_refs = {ref for ref, kind, _ in receipts(database)
                            if ref.startswith("consolidation:budget:") and kind == "markdown_ops_applied_v1"}
            assert len(pruning_refs) == 1
            assert store._read_receipt(pruning_refs.pop(), "markdown_before_v1") == {"MEMORY.md": before_pruning}
            assert store.read("SELF.md") == protected["SELF.md"]
            print(json.dumps({"invalid_merge_rejected": True, "conflicts": 3,
                              "pending_retried_without_new_message": True,
                              "protected_files_and_receipts_preserved": True,
                              "native_overflow_pruned_with_backup": True, "model_requests": len(requests)}))
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous_level)
        server.close()
        await server.wait_closed()


if __name__ == "__main__":
    with TemporaryDirectory(prefix="akashic-markdown-recovery-") as directory:
        asyncio.run(run(Path(directory)))

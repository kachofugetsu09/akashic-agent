"""真实回复、压缩与记忆插件接线，验证新提炼超额后自动剪枝并保全消息。"""
from __future__ import annotations

import asyncio
from contextlib import closing
from datetime import UTC, datetime
import json
from pathlib import Path
import shutil
import sqlite3
import sys
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.plugin_composition.config_input import save_config
from plugins.channels.contract import CHANNEL_INPUT_V2, ChannelInboundMessage
from plugins.compaction.message_summary import HEADINGS
from plugins.content.plugin import check_text
from plugins.ledger.contract import ContentPart, Input, Output
from plugins.markdown_memory.store import MarkdownProfileStore
from plugins.runtime_inspection.contract import DOCUMENTS
from tests.test_default_reply import application


async def run(folder: Path) -> None:
    """只替换模型外端；通过真实频道输入触发压缩、提炼和随草稿提交的剪枝。"""
    first, second = "- 偏好详细示例：" + "示例" * 30, "- 偏好调试记录：" + "记录" * 30
    condensed = "- 优先用具体示例解释概念。"
    extracted = 0
    pruning = 0
    partial_sent = asyncio.Event()
    finish_stream = asyncio.Event()

    # 1. 本地 HTTP 模型分别响应压缩、提炼、剪枝与业务请求。
    async def serve(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            header = (await reader.readuntil(b"\r\n\r\n")).decode()
            length = int(next(row.split(":", 1)[1] for row in header.split("\r\n")
                              if row.lower().startswith("content-length:")))
            request = json.loads(await reader.readexactly(length))
            prompt = "\n".join(str(message["content"]) for message in request["messages"])
            nonlocal extracted, pruning
            if "你在压缩超额的长期 Markdown 记忆" in prompt:
                pruning += 1
                text = json.dumps({"ops": [
                    {"op": "prune", "doc": "MEMORY.md", "old": "1", "line": condensed},
                    {"op": "prune", "doc": "MEMORY.md", "old": "2"},
                ]}, ensure_ascii=False)
            elif "你维护 Akashic 的长期 Markdown 记忆" in prompt:
                rows = json.loads(prompt.split("本次精确来源：\n", 1)[1])
                message = next(row for row in rows if row["author"] == "user")
                text = json.dumps({"ops": [] if extracted else [
                    {"op": "add", "doc": "MEMORY.md", "section": "## 用户偏好", "line": line,
                     "message_ids": [message["message_id"]]} for line in (first, second)
                ]}, ensure_ascii=False)
                extracted += 1
            elif "[Source messages]" in prompt:
                text = "\n".join(heading + "\nRecorded facts." for heading in HEADINGS)
            else:
                text = "业务回复完成"
            if "你在压缩超额的长期 Markdown 记忆" in prompt:
                assert request["stream"] is True
                writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n")
                half = len(text) // 2
                for index, part in enumerate((text[:half], text[half:])):
                    frame = {"choices": [{"index": 0, "delta": {"content": part}, "finish_reason": None}]}
                    writer.write(("data: " + json.dumps(frame) + "\n\n").encode())
                    await writer.drain()
                    if index == 0:
                        partial_sent.set()
                        await finish_stream.wait()
                writer.write(b'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n')
            else:
                body = json.dumps({"text": text}).encode()
                writer.write(f"HTTP/1.1 200 OK\r\nContent-Length: {len(body)}\r\nConnection: close\r\n\r\n".encode() + body)
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()

    server = await asyncio.start_server(serve, "127.0.0.1", 0)
    url = f"http://127.0.0.1:{server.sockets[0].getsockname()[1]}/complete"

    def configure(sources: Path) -> None:
        for name in ("markdown_memory", "runtime_inspection"):
            shutil.copytree(Path(__file__).resolve().parents[1] / "plugins" / name, sources / name,
                            ignore=shutil.ignore_patterns("__pycache__"))
        save_config(folder / "workspace/plugin-data/context-builtin", {
            "summary_source": ["compaction", "compaction"], "prompt_sources": {"markdown_memory": "markdown_memory"},
        })
        save_config(folder / "workspace/plugin-data/markdown_memory-builtin", {"memory_budget": 90, "consolidate": False})
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
            if "你在压缩超额的长期 Markdown 记忆" in str(request.messages):
                return await streaming_driver.complete(request)
            async with httpx.AsyncClient(trust_env=False) as client:
                response = await client.post({url!r}, json={{"messages": json_value(request.messages)}})
                response.raise_for_status()
                return LLMResponse(response.json()["text"])
''' + source[end:]
        # 剪枝走正式 OpenAI-compatible 驱动；只有远端服务由本地 HTTP 替代。
        source = source.replace("    model = _BoundChat(descriptor, Driver(), store)", f'''
    from dataclasses import replace
    from plugins.models.contract import DriverConnectionDescriptor
    from plugins.openai_compatible.driver import definition
    class Credential:
        connection_id = "fixture"
        auth_identity = "fixture"
        async def read(self):
            return {{"api_key": "local-scenario"}}
        async def refresh(self, payload):
            raise AssertionError("local credential does not refresh")
        @asynccontextmanager
        async def exclusive(self):
            yield
    connection = await definition().open(DriverConnectionDescriptor(
        "fixture", "fixture", "openai-compatible", {url!r}, "fixture", {{}}), Credential())
    await ctx.effect(lambda: connection.close, label="scenario-http")
    streaming_driver = connection.bind_chat(replace(descriptor, driver_id="openai-compatible"), {{}})
    model = _BoundChat(descriptor, Driver(), store)
''')
        entry.write_text(source.replace("    class Target:", "        independent_execution = execution\n    class Target:"))

    try:
        async with application(folder, replying=True, compaction=True, output_tokens=512,
                               extra_sources=configure) as (log, host):
            # 2. 两个完整旧 Turn 足以触发压缩，每个原文批次仍能放进提炼模型窗口。
            for index in range(2):
                log.writer("test:room", author="user", source="conversation", body_types=(Input,),
                           content={"text": check_text}).append(f"old-user-{index}", Input((ContentPart("text", first + second),)))
                log.writer("test:room", author="assistant", source="conversation", body_types=(Output,),
                           content={"text": check_text}).append(f"old-output-{index}", Output((ContentPart("text", "x" * 18000),), "complete"))
            database = folder / "workspace/sessions.db"
            with closing(sqlite3.connect(database)) as connection:
                original = connection.execute("SELECT * FROM messages ORDER BY seq").fetchall()
            memory = folder / "workspace/memory"
            store = MarkdownProfileStore(memory, memory / "markdown-profile-writes.db")
            before = store.read("MEMORY.md")
            root = host.live_root
            assert root is not None
            await root.context.require(CHANNEL_INPUT_V2)("test:room", "current", ChannelInboundMessage(
                "test", "user", "room", "继续处理", datetime.now(UTC), {}))

            await asyncio.wait_for(partial_sent.wait(), 15)
            assert store.read("MEMORY.md") == before
            assert store.latest_applied("test:room") is None
            finish_stream.set()

            async def wait_for_receipt() -> None:
                while store.latest_applied("test:room") is None:
                    await asyncio.sleep(0.05)
            await asyncio.wait_for(wait_for_receipt(), 15)
            # 3. 最终档案只包含压缩结果；本批新事实和剪枝共享一条可恢复 draft。
            assert extracted >= 1 and pruning == 1
            assert condensed in store.read("MEMORY.md") and first not in store.read("MEMORY.md") and second not in store.read("MEMORY.md")
            assert root.context.require(DOCUMENTS).get_document("memory")["markdown"] == store.read("MEMORY.md")
            latest = store.latest_applied("test:room")
            assert latest is not None
            assert store._read_receipt(latest[0], "markdown_before_v1") == {"MEMORY.md": before}
            with closing(sqlite3.connect(database)) as connection:
                assert connection.execute("SELECT * FROM messages ORDER BY seq").fetchall()[:len(original)] == original
                assert connection.execute("PRAGMA integrity_check").fetchall() == [("ok",)]
            print(json.dumps({"new_facts_pruned_before_commit": True, "pruning_requests": pruning,
                              "real_driver_streamed_without_partial_commit": True,
                              "before_image_preserved": True, "original_messages_preserved": len(original)}))
    finally:
        server.close()
        await server.wait_closed()


if __name__ == "__main__":
    with TemporaryDirectory(prefix="akashic-markdown-learning-") as directory:
        asyncio.run(run(Path(directory)))

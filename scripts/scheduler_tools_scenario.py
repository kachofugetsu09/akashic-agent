"""真实定时执行验证会话内执行、历史范围、晚注册、并发目录、卸载和工具权限；模型对端为本地 HTTP。"""
from __future__ import annotations

import argparse
import asyncio
from datetime import UTC, datetime, timedelta
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.plugin_composition.config_input import save_config
from agent.plugin_composition.tasks import TASKS
from plugins.channels.contract import CHANNEL_INPUT_V2 as CHANNEL_INPUT, ChannelInboundMessage
from plugins.content.plugin import check_text
from plugins.ledger.contract import ContentPart, Control, Input, Output, ToolResult
from plugins.scheduler.schedule import ScheduledJob
from plugins.scheduler.store import JobStore, fire_key
from plugins.scheduler.tools import cancel_fires
from tests.support.delivery_sources import sources as delivery_sources
from tests.test_default_reply import application


PROVIDER = '''from contextlib import asynccontextmanager
from plugins.tools.contract import TOOLS, Result
from plugins.ledger.contract import ContentPart
api_version = 3
name = "NAME"
version = "1.0.0"
inject = (TOOLS,)
async def apply(ctx):
    catalog = ctx.require(TOOLS)
    await catalog.declare_group(ctx, description="读取隔离样本文件")
    class ReadSample:
        idempotent = True
        async def prepare(self, arguments, source=None):
            return arguments
        async def invoke(self, key, arguments):
            import asyncio
            while (ctx.data_root / "hold").exists():
                (ctx.data_root / "holding").touch()
                await asyncio.sleep(0.05)
            value = (ctx.data_root / "sample.txt").read_text()
            with (ctx.data_root / "calls.jsonl").open("a") as stream:
                stream.write(key + "\\n")
            return Result("success", (ContentPart("text", value),))
        async def query(self, key):
            return None
    @asynccontextmanager
    async def open_tool(_state):
        yield ReadSample()
    for name in NAMES:
        await catalog.register(ctx, name=name, description="读取样本",
            parameters={"type": "object", "properties": {}, "additionalProperties": False},
            open=open_tool, idempotent=True)
'''


def provider(directory: Path, name: str, names: tuple[str, ...]) -> Path:
    """建立可经正式安装链发布的本地工具插件。"""
    path = directory / name
    path.mkdir()
    (path / "plugin.py").write_text(PROVIDER.replace('"NAME"', repr(name)).replace("NAMES", repr(names)))
    subprocess.run(["git", "init", "-q", "--initial-branch=source", str(path)], check=True)
    subprocess.run(["git", "-C", str(path), "add", "plugin.py"], check=True)
    subprocess.run(["git", "-C", str(path), "-c", "user.name=Scenario", "-c",
                    "user.email=scenario@example.invalid", "commit", "-qm", "工具样本"], check=True)
    return path


def rows(workspace: Path) -> list[tuple]:
    with sqlite3.connect(workspace / "sessions.db") as connection:
        return connection.execute("SELECT * FROM messages ORDER BY session_key, seq").fetchall()


async def run(directory: Path) -> dict[str, object]:
    """从真实调度文件触发完整 ReAct、工具回执和最终送达。"""
    # 1. 只替换外部模型；HTTP 请求保留实际菜单、system 和消息。
    requests: dict[str, list[dict]] = {}
    frozen = asyncio.Event()
    release = asyncio.Event()
    handlers: set[asyncio.Task] = set()

    async def serve(reader, writer):
        task = asyncio.current_task()
        assert task is not None
        handlers.add(task)
        try:
            header = (await reader.readuntil(b"\r\n\r\n")).decode()
            length = int(next(line.split(":", 1)[1] for line in header.split("\r\n")
                              if line.lower().startswith("content-length:")))
            request = json.loads(await reader.readexactly(length))
            request["system"] += "\n".join(str(message["content"]) for message in request["messages"]
                                           if message["role"] == "system")
            if "user-ping" in str(request["messages"][-1]):
                requests.setdefault("ping", []).append(request)
                body = json.dumps({"text": "pong"}).encode()
                writer.write(f"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {len(body)}\r\nConnection: close\r\n\r\n".encode() + body)
                await writer.drain()
                return
            match = re.search(r"case:(\w+)", str(request["messages"]))
            assert match is not None, request
            case = match[1]
            calls = requests.setdefault(case, [])
            calls.append(request)
            if case == "frozen" and len(calls) == 1:
                frozen.set()
                await release.wait()
            plugin = "extra@lab" if case in {"frozen", "fresh"} else "late@lab"
            name = "read_extra_sample" if case in {"frozen", "fresh"} else "read_late_sample"
            steps: list[list[tuple[str, dict[str, object]]]] = [[("load_tools", {"plugin": plugin})]]
            if case == "followup":
                # 定时轮里再建一个原文提醒，目标沿用本次触发的会话。
                steps = [[("load_tools", {"plugin": "scheduler"})], [("tool_call", {"name": "schedule", "arguments": {
                    "action": "notify", "trigger": "after", "when": "1s", "text": "followup-notice", "timezone": "UTC"}})]]
            elif case not in {"frozen", "removed"}:
                # interrupt 一次提交两个调用：用户插话时第二个可能还没开始。
                steps.append([("tool_call", {"name": name, "arguments": {}})] * (2 if case in {"interrupt", "cancelled"} else 1))
            if case != "followup":
                steps.append([("tool_call", {"name": "memorize", "arguments": {}})])
            if len(calls) <= len(steps):
                result = {"calls": [{"id": f"{case}-{len(calls)}-{index}", "name": tool, "arguments": arguments}
                                    for index, (tool, arguments) in enumerate(steps[len(calls) - 1])]}
            else:
                result = {"text": "completed:" + case}
            if case == "failed":
                result = {"context_error": "scenario context limit"}
            body = json.dumps(result).encode()
            writer.write(f"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {len(body)}\r\nConnection: close\r\n\r\n".encode() + body)
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()
            handlers.discard(task)

    server = await asyncio.start_server(serve, "127.0.0.1", 0)
    url = f"http://127.0.0.1:{server.sockets[0].getsockname()[1]}"

    def configure(sources: Path) -> None:
        save_config(directory / "workspace/plugin-data/scheduler-builtin", {"max_output_tokens": 512})
        for name in ("scheduler", "timer", "delivery_policy"):
            shutil.copytree(ROOT / "plugins" / name, sources / name, ignore=shutil.ignore_patterns("__pycache__"))
        delivery_sources(sources)
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
            async with httpx.AsyncClient(trust_env=False, timeout=60) as client:
                response = await client.post({url!r}, json={{
                    "messages": json_value(request.messages), "system": request.system_prompt,
                    "tools": json_value(request.tools)}})
                response.raise_for_status()
                value = response.json()
                if "context_error" in value:
                    from plugins.models.contract import ContextLengthError
                    raise ContextLengthError(value["context_error"]).exception()
                return LLMResponse(value.get("text"), [ToolCall(**call) for call in value.get("calls", [])])
''' + source[end:]
        entry.write_text(source)

    workspace = directory / "workspace"
    home = directory / "home"
    home.mkdir()
    os.environ.update(HOME=str(home), AKASHIC_PLUGIN_HOME=str(home), AKASHIC_PLUGIN_DISTRIBUTION="")
    late = provider(directory, "late", ("read_late_sample", "message_push", "recall_memory", "memorize", "remember_memory", "forget_memory"))
    extra = provider(directory, "extra", ("read_extra_sample",))
    try:
        async with application(directory, replying=True, extra_sources=configure) as (log, host):
            root = host.live_root
            assert root is not None
            scheduler = host._active_generations["scheduler"].fiber
            assert scheduler.state == "active"
            scheduler_context = scheduler.context

            async def install(path: Path) -> None:
                await host.install(source=str(path), marketplace="lab", ref_name="", sparse_paths=[], update_id="add-" + path.name)
                assert host._operation is not None
                await host._operation.task
                assert host.read_update("add-" + path.name).state == "active"
                assert host._active_generations["scheduler"].fiber is scheduler
                assert scheduler.context is scheduler_context
                data = workspace / "plugin-data" / (path.name + "-lab")
                (data / "sample.txt").write_text("sample:" + path.name)

            store = JobStore(workspace / "schedules.json")

            async def schedule(case: str, context: str = "none") -> ScheduledJob:
                # 每个任务落在自己的会话里；同一会话的任务与对话串行执行。
                job = ScheduledJob(id=case, action="run", trigger="at", channel="test", chat_id=case,
                                   session_id="test:" + case, text="case:" + case, context=context,
                                   timezone="UTC", fire_at=datetime.now(UTC) - timedelta(seconds=1))
                seed(job.session_id)
                await store.add("schedule:" + case, job, case)
                return job

            async def finish(job: ScheduledJob) -> list:
                async with asyncio.timeout(30):
                    while True:
                        state = await asyncio.to_thread(store.read)
                        fire = state.fires.get(fire_key(job))
                        if fire is not None and fire.status != "pending":
                            assert fire.status == "delivered", fire
                            messages = log.reader(job.session_id).snapshot()
                            return [message for message in messages if message.source == fire.source]
                        await asyncio.sleep(0.02)

            def seed(session_id: str) -> None:
                log.writer(session_id, author="assistant", source="seed", body_types=(Output,),
                           content={"text": check_text}).append(
                    "seed:" + session_id, Output((ContentPart("text", "history-secret"),), "complete"))

            # 2. scheduler 已启动后安装 provider；真实模型加载并调用其工具。
            await install(late)
            first = await finish(await schedule("late"))
            history = await finish(await schedule("history", "session"))
            assert all("history-secret" not in str(call["messages"]) for call in requests["late"]), "context=none 不应看到会话历史"
            assert "history-secret" in str(requests["history"][0]["messages"]), "context=session 应带会话历史"
            assert any(isinstance(message.body, Output) and message.body.finish == "complete" for message in history)

            # 任务的工具还在执行时用户发来消息：对话先等任务让出，两边都完成且互不破坏。
            data = workspace / "plugin-data/late-lab"
            (data / "hold").touch()
            pending = await schedule("interrupt")
            async with asyncio.timeout(10):
                while not (data / "holding").exists():
                    await asyncio.sleep(0.02)
            await root.context.require(CHANNEL_INPUT)("test:interrupt", "u1", ChannelInboundMessage(
                "test", "user", "interrupt", "user-ping", datetime.now(UTC), {}))
            await asyncio.sleep(1)
            (data / "hold").unlink()
            interrupted = await finish(pending)
            async with asyncio.timeout(30):
                while not any(isinstance(row.body, Output) and row.source == "conversation" and "pong" in str(row.body.parts)
                              for row in log.reader("test:interrupt").snapshot()):
                    await asyncio.sleep(0.02)
            # 进行中的任务轮次不进入对话请求；完成后对话能看到它。
            assert "case:interrupt" not in str(requests["ping"][0]["messages"]), "进行中的任务不应进入对话请求"
            assert any(isinstance(message.body, Output) and message.body.finish == "complete" for message in interrupted)

            # Cancel during a tool batch, then verify the saved round closes and later input still runs.
            (data / "holding").unlink()
            (data / "hold").touch()
            cancelled = await schedule("cancelled")
            async with asyncio.timeout(10):
                while not (data / "holding").exists():
                    await asyncio.sleep(0.02)
            before_cancel = rows(workspace)
            async with scheduler_context.runtime_scope():
                await store.cancel("cancel:running", (cancelled.id,))
                await cancel_fires(store, scheduler_context.require(TASKS).open(scheduler_context), (cancelled.id,))
            (data / "hold").unlink()
            async with asyncio.timeout(10):
                while not any(isinstance(row.body, Control) and row.body.action == "abandon"
                              for row in log.reader(cancelled.session_id).snapshot()):
                    await asyncio.sleep(0.02)
            assert set(before_cancel).issubset(rows(workspace))
            await root.context.require(CHANNEL_INPUT)(cancelled.session_id, "after-cancel", ChannelInboundMessage(
                "test", "user", "cancelled", "user-ping", datetime.now(UTC), {}))
            async with asyncio.timeout(30):
                while not any(isinstance(row.body, Output) and row.source == "conversation" and "pong" in str(row.body.parts)
                              for row in log.reader(cancelled.session_id).snapshot()):
                    await asyncio.sleep(0.02)

            # A model capacity rejection must leave a saved failure, not a permanently pending round.
            failed = await schedule("failed")
            async with asyncio.timeout(10):
                while (failure := (await asyncio.to_thread(store.read)).fires.get(fire_key(failed))) is None or failure.status == "pending":
                    await asyncio.sleep(0.02)
            assert failure.status == "failed" and "scenario context limit" in str(failure.error), failure
            assert any(isinstance(row.body, Control) and row.body.action == "failure"
                       for row in log.reader(failed.session_id).snapshot())

            # Restore the crash window after saving a failure but before settling the fire.
            recovery = ScheduledJob(id="failure-recovery", action="run", trigger="at", channel="test", chat_id="recovery",
                                    session_id="test:recovery", text="case:recovery", timezone="UTC", fire_at=datetime.now(UTC))
            seed(recovery.session_id)
            key = fire_key(recovery)
            writer = log.writer(recovery.session_id, author="scheduler", source="scheduler:" + key,
                                body_types=(Input, Control), content={"text": check_text})
            admitted = writer.append("scheduler-input:" + key, Input((ContentPart("text", recovery.text),)))
            writer.append("scheduler-failure:" + key, Control("failure", admitted.seq, "saved capacity error"))
            await store.add("schedule:recovery", recovery, "recovery")
            async with asyncio.timeout(10):
                while (restored := (await asyncio.to_thread(store.read)).fires.get(key)) is None or restored.status == "pending":
                    await asyncio.sleep(0.02)
            assert restored.status == "failed" and restored.error == "saved capacity error", restored
            assert "recovery" not in requests

            # 定时轮里创建后续任务：目标沿用本次触发；原文提醒到点写入同一会话并送达。
            await finish(await schedule("followup"))
            async with asyncio.timeout(30):
                while True:
                    state = await asyncio.to_thread(store.read)
                    notice = [fire for fire in state.fires.values() if fire.job.text == "followup-notice"]
                    if notice and notice[0].status != "pending":
                        break
                    await asyncio.sleep(0.05)
            assert notice[0].status == "delivered" and notice[0].job.session_id == "test:followup", notice
            assert "sample:late" in str(first), "晚注册工具未进入定时任务"
            results = [message.body for message in first if isinstance(message.body, ToolResult)]
            assert all(result.outcome == "success" for result in results)
            loaded_text = results[0].parts[0].value
            assert isinstance(loaded_text, str)
            loaded = json.loads(loaded_text)
            # 与会话工具一致，只去掉写长期记忆的三个工具。
            assert {tool["function"]["name"] for tool in loaded["tools"]} == {"read_late_sample", "message_push", "recall_memory"}
            assert results[1].parts[0].value == "sample:late"
            before = rows(workspace)

            # 3. 第一任务已发请求后增加 provider；新任务看见它，旧任务不能获权。
            old = await schedule("frozen")
            await asyncio.wait_for(frozen.wait(), 10)
            await install(extra)
            fresh = await finish(await schedule("fresh"))
            assert "sample:extra" in str(fresh)
            release.set()
            held = await finish(old)
            assert "插件不属于当前获授工具目录" in str(held)
            assert "extra@lab" not in requests["frozen"][0]["system"]
            assert "extra@lab" in requests["fresh"][0]["system"]

            # 4. 卸载对新任务生效，历史消息与已完成工具回执不改写。
            await host.uninstall("late@lab")
            assert host._operation is not None
            await host._operation.task
            assert host._active_generations["scheduler"].fiber is scheduler
            assert scheduler.context is scheduler_context
            removed = await finish(await schedule("removed"))
            assert "插件不属于当前获授工具目录" in str(removed)
            assert "late@lab" not in requests["removed"][0]["system"]
            assert set(before).issubset(rows(workspace))
            for case, calls in requests.items():
                assert all(call["system"] == calls[0]["system"] and call["tools"] == calls[0]["tools"] for call in calls), case
            for messages in (first, history, fresh, held, removed):
                assert "model.tool_rejection" in str(messages), "禁用工具必须明确拒绝"
            for name, count in (("late", 5), ("extra", 1)):
                calls = (workspace / "plugin-data" / (name + "-lab") / "calls.jsonl").read_text().splitlines()
                assert len(calls) == count, (name, calls)
            sent = (workspace / "plugin-data/test_sender-builtin/sent.jsonl").read_text().splitlines()
            assert len(sent) == 10 and len(set(sent)) == 10
            with sqlite3.connect(workspace / "sessions.db") as connection:
                assert connection.execute("PRAGMA integrity_check").fetchone() == ("ok",)
    finally:
        release.set()
        server.close()
        await server.wait_closed()
        await asyncio.gather(*handlers)
        (directory / "requests.json").write_text(json.dumps(requests, ensure_ascii=False, indent=2))
    return {"in_session": True, "history_none_isolated": True, "history_session_included": True,
            "late_registration": True, "concurrent_fixed_views": True, "uninstall": True,
            "disabled_tools_rejected": True, "messages_preserved": True, "deliveries": 10, "user_preempts_task": True, "cancelled_round_closed": True, "followup_from_task": True,
            "model_failure_settled": True, "model_requests": {key: len(value) for key, value in requests.items()}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = asyncio.run(run(args.output))
    (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False))

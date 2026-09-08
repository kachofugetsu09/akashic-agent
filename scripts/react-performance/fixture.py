"""一次性 workspace 中的真实消息链路与测量标记。"""

import asyncio
import gc
from collections import defaultdict
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import re
import sys
import tempfile
import time
import threading
from types import ModuleType
from unittest.mock import patch

ROOT = Path(
    os.environ.get("AKASHIC_PERF_ROOT", str(Path(__file__).resolve().parents[2]))
)
sys.path.insert(0, str(ROOT))
from tests.test_default_reply import application
from tests.test_delivery_bindings import sources as delivery_sources
from agent.plugin_composition import ServiceKey
from agent.plugin_composition.channels import CHANNEL_INPUT, ChannelInboundMessage
from agent.plugins.snapshot import lease_runtime_snapshot
from session.log import MessageCatalog, MessageLog, MessageReader, MessageWriter
from session.message import (
    ContentPart,
    ContentReferences,
    Input,
    Message,
    Output,
    ToolResult,
)
from session.message_codec import encode_body

TRACE = []
READS = defaultdict(lambda: [0, 0, 0.0])
ACTIVE = False
DONE = None
PROFILE = None
probe_module = ModuleType("latency_probe_marks")


def mark(name):
    if ACTIVE:
        TRACE.append((name, time.perf_counter()))
        if name == "sender_enter":
            DONE.set()


probe_module.mark = mark
sys.modules[probe_module.__name__] = probe_module


def extras(mode):
    """在既有 fixture 的工具和发送边界加计时，不替换业务流程。"""

    def setup(sources):
        delivery_sources(sources)
        shutil.copytree(
            ROOT / "plugins/delivery_policy",
            sources / "delivery_policy",
            ignore=shutil.ignore_patterns("__pycache__"),
        )
        provider = sources / "test_provider/plugin.py"
        code = provider.read_text()
        code = "from latency_probe_marks import mark\n" + code
        code = code.replace(
            "calls.append(request)",
            'mark("provider_enter")\n            calls.append(request)',
        )
        code = code.replace(
            "if len(calls) == 1:",
            'mark("provider_return")\n            if len(calls) <= 2:',
        )
        code = code.replace(
            "async def invoke(self, key, args):",
            'async def invoke(self, key, args):\n            mark("tool_enter")',
        )
        code = code.replace(
            'return Result("success",',
            'mark("tool_return")\n            return Result("success",',
        )
        provider.write_text(code)
        sender = sources / "test_sender/plugin.py"
        sender.write_text(
            "from latency_probe_marks import mark\n"
            + sender.read_text().replace(
                "async def send(self, key, address, message):",
                'async def send(self, key, address, message):\n            mark("sender_enter")',
            )
        )

    return setup


def seed(log, count, model_store=None, model_descriptor=None):
    """只在一次性数据库填入历史与成功模型账，再推进已有发送 cursor。"""
    if count == 0:
        return
    writer = log.writer(
        "test:room",
        author="user",
        source="conversation",
        body_types=(Input,),
        content={"text": lambda part: ContentReferences()},
    )
    writer.append("history:0", Input((ContentPart("text", "history " + "x" * 504),)))
    db = log._connection
    template = dict(db.execute("SELECT * FROM messages").fetchone())
    columns = tuple(template)
    output = encode_body(
        Output((ContentPart("text", "history " + "x" * 504),), "complete")
    )
    rows = []
    model_rows = []
    for index in range(1, count):
        row = {
            **template,
            "id": f"history:{index}",
            "seq": index,
            "author": "assistant" if index % 2 else "user",
            "body": output if index % 2 else template["body"],
        }
        if index % 2 and model_store is not None:
            from dataclasses import asdict
            from plugins.models.projection import response_facts
            from agent.plugin_composition.models import LLMResponse

            identity = f"history-call:{index}"
            facts = response_facts(LLMResponse("history", call_record_id=identity), ())
            row["body"] = encode_body(
                Output((ContentPart("text", "history " + "x" * 504), facts), "complete")
            )
            model_rows.append(
                (
                    identity,
                    json.dumps(asdict(model_descriptor)),
                    "historical-fixture",
                    "success",
                )
            )
        rows.append(tuple(row[column] for column in columns))
    if model_rows:
        with model_store._connect() as connection, connection:
            connection.executemany(
                "INSERT INTO model_calls (id,binding_json,request_digest,state) VALUES (?,?,?,?)",
                model_rows,
            )
    with db:
        db.executemany(
            "INSERT INTO messages ("
            + ",".join(columns)
            + ") VALUES ("
            + ",".join("?" for _ in columns)
            + ")",
            rows,
        )
        db.execute("UPDATE sessions SET next_seq=? WHERE key=?", (count, "test:room"))
    # 历史发送已结算；测量不包含启动追赶。
    log.owner("plugin:delivery").transact(
        lambda tx: tx.save(
            'cursor:["delivery_policy","test:room"]',
            {"through_seq": count - 1},
            expected_version=None,
        )
    )


async def run(mode, count, full_history=False):
    """测入站接纳至 sender 进入，并检查真实消息与工具效果顺序。"""
    global ACTIVE, DONE
    TRACE.clear()
    READS.clear()
    DONE = asyncio.Event()
    decoded_messages = [0]
    import session.log as message_log

    original_decode = message_log._message

    def decode(row):
        result = original_decode(row)
        if ACTIVE:
            decoded_messages[0] += 1
        return result

    original_snapshot = MessageReader.snapshot
    original_write = MessageLog._write
    from plugins.models.store import ModelsStore

    created_stores = []
    original_init = ModelsStore.__init__

    def store_init(store, *args, **kwargs):
        original_init(store, *args, **kwargs)
        created_stores.append(store)

    original_thread = asyncio.to_thread
    threads_idle = asyncio.Event()
    threads_idle.set()
    thread_jobs = 0
    finished_jobs = 0

    async def to_thread(function, /, *args, **kwargs):
        nonlocal thread_jobs, finished_jobs
        thread_jobs += 1
        threads_idle.clear()
        try:
            return await original_thread(function, *args, **kwargs)
        finally:
            thread_jobs -= 1
            finished_jobs += 1
            if thread_jobs == 0:
                threads_idle.set()

    original_append = MessageWriter._append
    writes = threading.local()

    def append(writer, *args, **kwargs):
        result = original_append(writer, *args, **kwargs)
        writes.pending.append(result)
        return result

    def snapshot(reader, **kwargs):
        start = time.perf_counter()
        result = original_snapshot(reader, **kwargs)
        if ACTIVE:
            frame = sys._getframe(1)
            key = f"{Path(frame.f_code.co_filename).parent.name}/{Path(frame.f_code.co_filename).name}:{frame.f_code.co_name}"
            item = READS[key]
            item[0] += 1
            item[1] += len(result)
            item[2] += (time.perf_counter() - start) * 1000
        return result

    def write(log, callback):
        writes.pending = []
        try:
            result = original_write(log, callback)
        except BaseException:
            writes.pending = []
            raise
        for message in writes.pending:
            if isinstance(message.body, Input):
                mark("input_committed")
            elif isinstance(message.body, ToolResult):
                mark("tool_result_committed")
            elif isinstance(message.body, Output):
                mark("output_" + message.body.finish + "_committed")
        writes.pending = []
        return result

    from contextlib import ExitStack

    with (
        ExitStack() as patches,
        tempfile.TemporaryDirectory(prefix="akashic-flow-perf-") as directory,
    ):
        patches.enter_context(patch.object(asyncio, "to_thread", to_thread))
        patches.enter_context(patch.object(MessageReader, "snapshot", snapshot))
        patches.enter_context(patch.object(message_log, "_message", decode))
        patches.enter_context(patch.object(ModelsStore, "__init__", store_init))
        patches.enter_context(patch.object(MessageLog, "_write", write))
        patches.enter_context(patch.object(MessageWriter, "_append", append))
        path = Path(directory)
        async with application(
            path, replying=True, start=False, extra_sources=extras(mode)
        ) as (log, host):
            model_store = model_descriptor = None
            if full_history:
                from agent.plugin_composition import CHAT_MODELS
                from agent.plugin_composition.models import ModelRole

                async with lease_runtime_snapshot(host.snapshot_store) as runtime:
                    context = runtime.composition_root.context
                    assert len(created_stores) == 1
                    model_store = created_stores[0]
                    async with context.require(CHAT_MODELS).execution() as execution:
                        model_descriptor = execution.chat(ModelRole.AGENT).descriptor
            seed(log, count, model_store, model_descriptor)
            await host.start_runtime()
            # 排空启动追赶，让订阅进入等待；启动耗时不计入本次请求。
            for _ in range(30):
                await asyncio.sleep(0)
            while True:
                await threads_idle.wait()
                finished = finished_jobs
                await asyncio.sleep(0)
                if thread_jobs == 0 and finished_jobs == finished:
                    break
            async with lease_runtime_snapshot(host.snapshot_store) as runtime:
                root = runtime.composition_root.context
                accept = root.require(CHANNEL_INPUT)
                gc.collect()
                ACTIVE = True
                if PROFILE is not None:
                    PROFILE.enable()
                cpu_start = time.process_time()
                mark("channel_input_enter")
                message = ChannelInboundMessage(
                    "test",
                    "user",
                    "room",
                    "do the work",
                    datetime(2026, 9, 9, tzinfo=UTC),
                    {},
                )
                await accept("test:room", "measured-input", message)
                mark("channel_input_return")
            await asyncio.wait_for(DONE.wait(), 300)
            async with lease_runtime_snapshot(host.snapshot_store) as runtime:
                delivery = runtime.composition_root.context.require(
                    ServiceKey("fixture.delivery")
                )()
                await delivery.wait_idle("test", "room")
            ACTIVE = False
            if PROFILE is not None:
                PROFILE.disable()
            cpu_ms = (time.process_time() - cpu_start) * 1000
            trace = list(TRACE)
            reads = dict(READS)
            rows = log.reader("test:room").read(after_seq=count - 1)
            assert [type(row.body).__name__ for row in rows] == [
                "Input",
                "Output",
                "ToolResult",
                "Output",
                "ToolResult",
                "Output",
            ]
            assert (path / "effect.txt").read_text().splitlines() == ["once", "once"]
            sent = list((path / "workspace").rglob("sent.jsonl"))
            assert len(sent) == 1 and len(sent[0].read_text().splitlines()) == 1
            calls = runtime.composition_root.context.require(
                ServiceKey("fixture.calls")
            )
            assert len(calls) == 3
            # 比较时只归一化生成 ID；上面的断言另外固定实际效果数量和顺序。
            request_shapes = [
                [(m["role"], len(str(m.get("content", "")))) for m in call.messages]
                for call in calls
            ]
            from session.message_codec import json_value

            normalized = json.dumps(
                [json_value(call.messages) for call in calls],
                sort_keys=True,
                ensure_ascii=False,
            )
            normalized = re.sub(
                r"/tmp/akashic-flow-perf-[a-zA-Z0-9_-]+", "/tmp/FIXTURE", normalized
            )
            normalized = re.sub(r"\b[a-f0-9]{64}\b", "HASH", normalized)
            normalized = re.sub(r"\b[a-f0-9]{32}\b", "UUID", normalized)
            request_digest = hashlib.sha256(normalized.encode()).hexdigest()
    times = defaultdict(list)
    for name, timestamp in trace:
        times[name].append(timestamp)

    def elapsed(a, b):
        return round((b - a) * 1000, 3)

    start = times["channel_input_enter"][0]
    return {
        "mode": mode,
        "history": count,
        "full_history": full_history,
        "cpu_ms": round(cpu_ms, 3),
        "decoded_messages": decoded_messages[0],
        "admission_ms": elapsed(start, times["channel_input_return"][0]),
        "first_provider_ms": elapsed(start, times["provider_enter"][0]),
        "tool_round_ms": [
            elapsed(times["provider_return"][i], times["provider_enter"][i + 1])
            for i in range(2)
        ],
        "tool_result_to_provider_ms": [
            elapsed(times["tool_result_committed"][i], times["provider_enter"][i + 1])
            for i in range(2)
        ],
        "final_return_to_send_ms": elapsed(
            times["provider_return"][-1], times["sender_enter"][0]
        ),
        "final_commit_to_send_ms": elapsed(
            times["output_complete_committed"][0], times["sender_enter"][0]
        ),
        "total_ms": elapsed(start, times["sender_enter"][0]),
        "reads": reads,
        "trace": [(name, elapsed(start, stamp)) for name, stamp in trace],
        "request_shapes": request_shapes,
        "request_digest": request_digest,
    }

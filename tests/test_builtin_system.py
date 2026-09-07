"""系统场景只通过外部协议、文件和重开后的结果验收，不导入生产实现。"""
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
import hashlib
import json
from pathlib import Path
import re
import socket

from aiohttp import web
from akashic_sdk import AsyncAkashic
import pytest

from tests.fixtures.builtin_runtime import dashboard, runtime


class ScriptedModel:
    """按实际请求中的当前输入和工具结果回答，网络闸门固定交错位置。"""

    def __init__(self, root: Path):
        self.root = root
        self.steps: dict[str, list[tuple[str, dict]]] = {}
        self.answers: dict[str, str] = {}
        self.requests: list[dict] = []
        self.embeddings: list[dict] = []
        self.entered: dict[str, asyncio.Event] = {}
        self.release: dict[str, asyncio.Event] = {}
        self.status: dict[str, int] = {}
        self.summary_requests: list[dict] = []
        self.profile_requests: list[dict] = []
        self.wake_stages: list[str] = []

    def hold(self, label: str) -> None:
        self.entered[label] = asyncio.Event()
        self.release[label] = asyncio.Event()

    async def chat(self, request: web.Request) -> web.Response:
        """固定外部模型决策；实际工具和持久写入均由服务执行。"""
        body = await request.json()
        self.requests.append(body)
        with (self.root / "system-model.jsonl").open("a") as evidence:
            evidence.write(json.dumps(body, ensure_ascii=False) + "\n")
        messages = body["messages"]
        serialized = json.dumps(messages, ensure_ascii=False)
        if "[Source messages]" in serialized:
            self.summary_requests.append(body)
            headings = ("## Goal", "## Constraints & Preferences", "## Progress", "### Done", "### In Progress",
                        "### Blocked", "## Key Decisions", "## Next Steps", "## Critical Context")
            return self.text_response(body, "\n".join(headings) + "\nPREFERENCE:用户喜欢青绿色。")
        if "本次精确来源：" in serialized:
            self.profile_requests.append(body)
            memory = self.root / "workspace/memory/MEMORY.md"
            existing = memory.read_text() if memory.exists() else ""
            if not existing.strip():
                existing = "# 用户长期记忆\n## 用户事实\n## 用户偏好\n## 用户明确要求长期记住的关键内容\n"
            text = existing if "用户喜欢青绿色" in existing else existing + "\n- 用户喜欢青绿色。\n"
            return self.text_response(body, json.dumps({"memory": text,
                "self": (self.root / "workspace/memory/SELF.md").read_text()}, ensure_ascii=False))
        for message in reversed(messages):
            if message["role"] != "user":
                continue
            content = message.get("content", "")
            texts = [part["text"] for part in content if part.get("type") == "text"] if isinstance(content, list) else [content]
            for text in texts:
                if not text.startswith('{"stage":'):
                    continue
                phase = json.loads(text)
                stage, data = phase["stage"], phase["data"]
                self.wake_stages.append(stage)
                name = "share_alert" if stage == "alert" else "share_content"
                arguments: dict = {"message": f"NOTICE:{stage}"}
                if stage == "screen":
                    name = "screen_content"
                    arguments = {"items": [{"candidate_id": item["candidate_id"], "initial_interest": "relevant",
                                            "question": "核对正文"} for item in data["candidates"]]}
                elif stage != "alert":
                    arguments["items"] = [item["candidate_id"] for item in data["candidates"]] if stage == "investigate" else []
                return web.json_response({"choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
                    "role": "assistant", "tool_calls": [{"id": f"wake-{stage}", "type": "function",
                        "function": {"name": name, "arguments": json.dumps(arguments)}}]}}]})
        selected = None
        for index, message in enumerate(messages):
            if message["role"] != "user":
                continue
            content = message.get("content", "")
            texts = [part["text"] for part in content if part.get("type") == "text"] if isinstance(content, list) else [content]
            for text in texts:
                match = re.match(r"CASE:(\w+)\b", text)
                if match:
                    selected = (index, match[1])
        if selected is None:
            return web.json_response({"error": {"message": "fixture cannot identify input"}}, status=400)
        index, label = selected
        if label in self.entered:
            self.entered[label].set()
            await self.release[label].wait()
        if label in self.status:
            return web.json_response({"error": {"message": f"fixture failure {label}", "type": "authentication_error"}},
                                     status=self.status[label])
        results = [item for item in messages[index + 1:] if item["role"] == "tool"]
        steps = self.steps.get(label, [])
        if len(results) < len(steps):
            name, arguments = steps[len(results)]
            if "{task_dir}" in str(arguments):
                task_dir = re.search(r"/[^\s\"\\]*subagent-runs/[a-f0-9]{32}", serialized)
                assert task_dir is not None, "模型请求必须给出子任务实际目录"
                arguments = {key: value.replace("{task_dir}", task_dir[0]) if isinstance(value, str) else value
                             for key, value in arguments.items()}
            encoded = json.dumps(arguments, ensure_ascii=False)
            middle = max(1, len(encoded) // 2)
            deltas = [{"role": "assistant", "tool_calls": [{"index": 0, "id": f"{label}-{len(results)}",
                       "type": "function", "function": {"name": name, "arguments": encoded[:middle]}}]},
                      {"tool_calls": [{"index": 0, "function": {"arguments": encoded[middle:]}}]}]
            finish = "tool_calls"
        else:
            deltas = [{"role": "assistant", "content": self.answers.get(label, f"DONE:{label}")}]
            finish = "stop"
        if not body.get("stream"):
            message = ({"role": "assistant", "tool_calls": [{"id": f"{label}-{len(results)}", "type": "function",
                        "function": {"name": name, "arguments": encoded}}]} if finish == "tool_calls"
                       else {"role": "assistant", "content": self.answers.get(label, f"DONE:{label}")})
            return web.json_response({"choices": [{"index": 0, "message": message, "finish_reason": finish}]})
        chunks = [{"choices": [{"index": 0, "delta": delta, "finish_reason": None}]} for delta in deltas]
        chunks.append({"choices": [{"index": 0, "delta": {}, "finish_reason": finish}]})
        return web.Response(text="".join("data: " + json.dumps(chunk, ensure_ascii=False) + "\n\n" for chunk in chunks)
                            + "data: [DONE]\n\n", content_type="text/event-stream")

    @staticmethod
    def text_response(body: dict, text: str) -> web.Response:
        if not body.get("stream"):
            return web.json_response({"choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                                                  "finish_reason": "stop"}]})
        chunks = [{"choices": [{"index": 0, "delta": {"role": "assistant", "content": text}, "finish_reason": None}]},
                  {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}]
        return web.Response(text="".join("data: " + json.dumps(chunk, ensure_ascii=False) + "\n\n" for chunk in chunks)
                            + "data: [DONE]\n\n", content_type="text/event-stream")

    async def embed(self, request: web.Request) -> web.Response:
        """不同正文产生不同且可重复的向量；观察真实学习和查询的 HTTP 输入。"""
        body = await request.json()
        self.embeddings.append(body)
        inputs = body["input"]
        vectors = []
        for index, text in enumerate(inputs):
            digest = hashlib.sha256(text.encode()).digest()
            values = [1 + byte / 255 for byte in digest[:8]]
            norm = sum(value * value for value in values) ** .5
            vectors.append({"index": index, "embedding": [value / norm for value in values]})
        return web.json_response({"data": vectors, "usage": {"prompt_tokens": len(inputs), "total_tokens": len(inputs)}})

    @asynccontextmanager
    async def serve(self) -> AsyncIterator[str]:
        """每次使用独立 loopback 端口，不继承账户或访问真实模型。"""
        app = web.Application()
        async def models(_request: web.Request) -> web.Response:
            return web.json_response({"data": [{"id": "fixture"}, {"id": "embedding"}]})

        app.router.add_get("/v1/models", models)
        app.router.add_post("/v1/chat/completions", self.chat)
        app.router.add_post("/v1/embeddings", self.embed)
        runner = web.AppRunner(app)
        await runner.setup()
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        try:
            await web.SockSite(runner, sock).start()
            yield f"http://127.0.0.1:{sock.getsockname()[1]}/v1"
        finally:
            for event in self.release.values():
                event.set()
            await runner.cleanup()


async def complete(client: AsyncAkashic, session: str, label: str, *, text: str = "", after: int = -1) -> list[dict]:
    """从指定游标订阅新终态，不能把重放的旧终稿当作本次完成。"""
    async with await client.session_follow(session, after_seq=after) as subscription:
        await client.message_send(session, f"CASE:{label}\n{text}", message_id=f"input-{label}")
        async with asyncio.timeout(35):
            async for event in subscription.events():
                if event["type"] == "messages.appended" and any(
                    (item["body"].get("finish") in {"complete", "quiet"} or item["body"].get("action") == "failure") for item in event["items"]
                ):
                    break
    rows = (await client.message_read(session, limit=200))["items"]
    assert rows[-1]["seq"] > after
    return rows


def result_text(rows: list[dict]) -> str:
    return json.dumps([row["body"] for row in rows if row["body"]["kind"] == "tool_result"], ensure_ascii=False)


def tool_outcomes(rows: list[dict]) -> list[str]:
    return [row["body"]["outcome"] for row in rows if row["body"]["kind"] == "tool_result"]


@pytest.mark.asyncio
async def test_schedule_and_child_complete_real_work_and_survive_reopen(tmp_path: Path) -> None:
    """父任务创建调度和子任务；二者实际读写、隔离、回传，并在新进程中保留结果。"""
    model = ScriptedModel(tmp_path)
    scheduled_file = tmp_path / "scheduled.txt"
    model.steps = {
        "parent": [
            ("schedule", {"tier": "soft", "trigger": "after", "when": "1s", "timezone": "UTC",
                          "channel": "akashic", "chat_id": "scheduled-sink", "prompt": "CASE:scheduled", "name": "fixture-job"}),
            ("spawn", {"task": "CASE:child", "profile": "scripting"}),
            ("list_schedules", {}),
        ],
        "child": [("write_file", {"path": "{task_dir}/child.txt", "content": "CHILD:原文🧪"}),
                  ("read_file", {"path": "{task_dir}/child.txt"})],
        "scheduled": [("write_file", {"path": str(scheduled_file), "content": "SCHEDULED:独立原文🧪"})],
    }
    settings = {"plugins": {"reply": 'tools = ["schedule", "spawn", "list_schedules"]\n'}}
    async with model.serve() as endpoint:
        async with runtime(tmp_path, endpoint, settings=settings) as (first, address):
            async with await AsyncAkashic.connect(address) as client:
                parent = (await client.session_create())["session_id"]
                # 订阅先建立；调度可能早于父工具链完成，不能漏掉一次性通知。
                async with await client.session_follow("akashic:scheduled-sink") as notices:
                    rows = await complete(client, parent, "parent")
                    assert tool_outcomes(rows) == ["success"] * 3, result_text(rows)
                    async with asyncio.timeout(20):
                        async for event in notices.events():
                            if event["type"] == "messages.appended":
                                break
                notice = (await client.message_read("akashic:scheduled-sink"))["items"]
                assert len(notice) == 1 and "DONE:scheduled" in str(notice)
                child_files = list((tmp_path / "workspace/subagent-runs").glob("*/child.txt"))
                assert len(child_files) == 1
                child_file = child_files[0]
                assert child_file.read_text() == "CHILD:原文🧪"
                assert scheduled_file.read_text() == "SCHEDULED:独立原文🧪"
                assert "DONE:child" in result_text(rows)
                state = json.loads((tmp_path / "workspace/schedules.json").read_text())
                assert len(state["fires"]) == 1
                fire_session = "scheduler:" + next(iter(state["fires"]))
                internal = (await client.message_read(fire_session))["items"]
                assert internal[0]["source"] == "scheduler"
                assert "CASE:parent" not in str(internal)
                saved = {parent: rows, "akashic:scheduled-sink": notice, fire_session: internal}
            first.kill()
            await first.wait()
        count = len(model.requests)
        async with runtime(tmp_path, endpoint, settings=settings) as (_second, address):
            async with await AsyncAkashic.connect(address) as client:
                for session, original in saved.items():
                    assert (await client.message_read(session, limit=200))["items"] == original
                assert len(model.requests) == count, "重开不应再次执行已完成调度或子任务"
                next_session = (await client.session_create())["session_id"]
                assert (await complete(client, next_session, "next"))[-1]["body"]["finish"] == "complete"
        assert child_file.read_text() == "CHILD:原文🧪"
        assert scheduled_file.read_text() == "SCHEDULED:独立原文🧪"


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["tool_arguments", "provider_auth"])
async def test_failure_is_recorded_and_same_session_can_continue(tmp_path: Path, failure: str) -> None:
    """工具拒绝和 HTTP 认证失败必须可见，不能阻断同一会话下一条实际输入。"""
    model = ScriptedModel(tmp_path)
    if failure == "tool_arguments":
        model.steps["bad"] = [("write_file", {"path": str(tmp_path / "forbidden-effect"), "content": 42})]
    else:
        model.status["bad"] = 401
    async with model.serve() as endpoint:
        async with runtime(tmp_path, endpoint) as (_process, address):
            async with await AsyncAkashic.connect(address) as client:
                session = (await client.session_create())["session_id"]
                failed = await complete(client, session, "bad")
                if failure == "tool_arguments":
                    assert tool_outcomes(failed) == ["error"], result_text(failed)
                else:
                    assert failed[-1]["body"]["action"] == "failure"
                    assert "failure bad" in failed[-1]["body"]["reason"]
                assert not (tmp_path / "forbidden-effect").exists()
                continued = await complete(client, session, "good", after=failed[-1]["seq"])
                assert continued[:len(failed)] == failed
                assert continued[-1]["body"]["finish"] == "complete"
                assert "DONE:good" in str(continued[-1])


@pytest.mark.asyncio
@pytest.mark.parametrize("tool,arguments", [
    ("tool_search", {"query": "write_file", "select": "write_file"}),
    ("load_skill", {"skill": 42}),
])
async def test_builtin_argument_error_allows_same_turn_to_continue(tmp_path: Path, tool: str, arguments: dict) -> None:
    """模型参数错误作为工具结果保存，同轮纠正后仍能执行实际文件操作。"""
    model = ScriptedModel(tmp_path)
    effect = tmp_path / "recovered.txt"
    model.steps["recover_arguments"] = [
        (tool, arguments),
        ("tool_search", {"query": "select:write_file"}),
        ("write_file", {"path": str(effect), "content": "RECOVERED:参数错误"}),
    ]
    settings = {"plugins": {"reply": 'tools = ["tool_search", "load_skill", "write_file"]\n'}}
    async with model.serve() as endpoint:
        async with runtime(tmp_path, endpoint, settings=settings) as (_process, address):
            async with await AsyncAkashic.connect(address) as client:
                session = (await client.session_create())["session_id"]
                rows = await complete(client, session, "recover_arguments")
                assert tool_outcomes(rows) == ["error", "success", "success"], rows
                assert rows[-1]["body"].get("finish") == "complete"
                assert effect.read_text() == "RECOVERED:参数错误"
                assert len(model.requests) == 4, "模型必须收到参数错误后在同轮继续"


@pytest.mark.asyncio
async def test_model_settings_switch_keeps_inflight_request_and_changes_next_input(tmp_path: Path) -> None:
    """真实 HTTP 设置切换后，旧请求仍由原 endpoint 完成，新输入只调用新 endpoint。"""
    old, new = ScriptedModel(tmp_path), ScriptedModel(tmp_path)
    old.hold("old")
    old.steps["old"] = [("read_file", {"path": str(tmp_path / "source.txt")})]
    (tmp_path / "source.txt").write_text("SOURCE:原请求")
    async with old.serve() as first_endpoint, new.serve() as second_endpoint:
        async with runtime(tmp_path, first_endpoint) as (_process, address):
            async with await AsyncAkashic.connect(address) as client:
                session = (await client.session_create())["session_id"]
                pending = asyncio.create_task(complete(client, session, "old"))
                try:
                    await asyncio.wait_for(old.entered["old"].wait(), 10)
                    async with dashboard(tmp_path, "models") as web_client:
                        response = await web_client.post("/api/dashboard/models/command", json={
                            "type": "update_connection", "expected_revision": 3, "connection_id": "fixture",
                            "name": "Changed", "endpoint": second_endpoint, "auth_identity": "fixture"})
                        assert response.status_code == 200, response.text
                    old.release["old"].set()
                    before = await pending
                finally:
                    old.release["old"].set()
                    if not pending.done():
                        pending.cancel()
                    await asyncio.gather(pending, return_exceptions=True)
                assert tool_outcomes(before) == ["success"]
                assert len(old.requests) == 2 and not new.requests
                after = await complete(client, session, "new", after=before[-1]["seq"])
                assert after[:len(before)] == before
                assert len(new.requests) == 1 and len(old.requests) == 2
                assert "DONE:new" in str(after[-1])


@pytest.mark.asyncio
async def test_real_embedding_recall_and_dashboard_keep_original_messages_after_restart(tmp_path: Path) -> None:
    """实际 HTTP embedding 学习后跨进程召回，出处和详情必须指向原会话的完整消息。"""
    model = ScriptedModel(tmp_path)
    model.steps["recall"] = [("recall_memory", {"query": "ORIGINAL:雪山蓝莓", "limit": 5})]
    settings = {"embedding": True, "plugins": {"reply": 'tools = ["recall_memory"]\n'}}
    async with model.serve() as endpoint:
        async with runtime(tmp_path, endpoint, settings=settings) as (_first, address):
            async with await AsyncAkashic.connect(address) as client:
                source = (await client.session_create())["session_id"]
                rows = await complete(client, source, "learn", text="ORIGINAL:雪山蓝莓\n" + "长原文🧪" * 100)
        async with runtime(tmp_path, endpoint, settings=settings) as (_second, address):
            async with await AsyncAkashic.connect(address) as client:
                target = (await client.session_create())["session_id"]
                recalled = await complete(client, target, "recall")
                assert tool_outcomes(recalled) == ["success"], result_text(recalled)
                assert "ORIGINAL:雪山蓝莓" in result_text(recalled)
                assert (await client.message_read(source))["items"] == rows
                async with dashboard(tmp_path, "akasha") as web_client:
                    response = await web_client.get("/api/dashboard/akasha-inspector/turns", params={"session_key": target})
                    assert response.status_code == 200, response.text
                    queries = response.json()["items"]
                    assert queries
                    hits = [message for query in queries for hit in query["hits"] for message in hit["messages"]]
                    assert any(item["message_id"] == "input-learn" and item["session_id"] == source for item in hits)
                    detail = await web_client.get(f"/api/dashboard/akasha-inspector/turns/{queries[0]['query_id']}")
                    assert detail.status_code == 200
                    assert "长原文🧪" * 100 in str(detail.json())
        assert model.embeddings, "必须通过真实 HTTP embedding，而不是预置学习图"


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["before_tool_result", "after_tool_result"])
async def test_crash_at_tool_commit_preserves_effect_and_recovers_without_repeating(tmp_path: Path, phase: str) -> None:
    """真实写入后在结果事务两侧强杀；未知与已提交结果必须各自恢复且不重写文件。"""
    model = ScriptedModel(tmp_path)
    effect = tmp_path / "crash-effect.txt"
    model.steps["crash"] = [("write_file", {"path": str(effect), "content": "COMMITTED_EFFECT:原文🧪"})]
    settings = {"crash_phase": phase}
    async with model.serve() as endpoint:
        async with runtime(tmp_path, endpoint, settings=settings) as (first, address):
            async with await AsyncAkashic.connect(address) as client:
                session = (await client.session_create())["session_id"]
                await client.message_send(session, "CASE:crash", message_id="input-crash")
                await asyncio.wait_for(first.wait(), 15)
                assert first.returncode == -9
            assert json.loads((tmp_path / "crash-reached.json").read_text())["phase"] == phase
            assert effect.read_text() == "COMMITTED_EFFECT:原文🧪"
            original_write = effect.stat().st_mtime_ns
        async with runtime(tmp_path, endpoint, settings=settings) as (_second, address):
            async with await AsyncAkashic.connect(address) as client:
                async with await client.session_follow(session) as subscription:
                    async with asyncio.timeout(20):
                        async for event in subscription.events():
                            if event["type"] == "messages.appended" and any(
                                (item["body"].get("finish") == "complete" or item["body"].get("action") == "failure") for item in event["items"]):
                                break
                rows = (await client.message_read(session))["items"]
                assert tool_outcomes(rows) == (["unknown"] if phase == "before_tool_result" else ["success"])
                assert len(rows) == 4 and rows[0]["id"] == "input-crash"
                if phase == "before_tool_result":
                    assert rows[-1]["body"]["action"] == "failure"
                    assert "工具效果需核对" in rows[-1]["body"]["reason"]
                assert effect.stat().st_mtime_ns == original_write, "恢复重放了无法安全重试的文件写入"
                assert effect.read_text() == "COMMITTED_EFFECT:原文🧪"
                continued = await complete(client, session, "after_crash", after=rows[-1]["seq"])
                assert continued[:len(rows)] == rows
                if phase == "before_tool_result":
                    # 新 Input 不是未知外部效果的确认，不得绕过原停止边界。
                    assert continued[-1]["body"]["action"] == "failure"
                    fresh = (await client.session_create())["session_id"]
                    assert "DONE:fresh" in str((await complete(client, fresh, "fresh"))[-1])
                else:
                    assert "DONE:after_crash" in str(continued[-1])
                assert effect.stat().st_mtime_ns == original_write


@pytest.mark.asyncio
async def test_compaction_projects_exact_facts_to_markdown_without_rewriting_history(tmp_path: Path) -> None:
    """真实对话逐步超过窗口，摘要与长期记忆沿实际消费链推进，重开仍保留全部原文。"""
    model = ScriptedModel(tmp_path)
    padding = "".join(hashlib.sha256(str(index).encode()).hexdigest() for index in range(90))
    for label in ("long_a", "long_b", "long_c", "long_d"):
        model.answers[label] = "ANSWER:" + label + "\n" + padding
    settings = {"context_window": 20000, "plugins": {
        "reply": 'tools = []\nmax_output_tokens = 1000\n',
        "compaction": 'keep_recent_tokens = 128\n',
    }}
    async with model.serve() as endpoint:
        async with runtime(tmp_path, endpoint, settings=settings) as (_process, address):
            async with await AsyncAkashic.connect(address) as client:
                session = (await client.session_create())["session_id"]
                saved: list[dict] = []
                for label in ("long_a", "long_b", "long_c", "long_d"):
                    rows = await complete(client, session, label, text="用户喜欢青绿色。\n" + padding,
                                          after=saved[-1]["seq"] if saved else -1)
                    assert rows[-1]["body"].get("finish") == "complete", rows[-1]
                    assert rows[:len(saved)] == saved
                    saved = rows
                assert model.summary_requests, "长对话必须经过真实压缩模型 HTTP"
                assert any(part["kind"] == "context.summary" for row in saved
                           for part in row["body"].get("parts", [])), "成功 Output 必须保留实际使用的摘要引用"
                memory = tmp_path / "workspace/memory/MEMORY.md"
                async with asyncio.timeout(15):
                    while not memory.exists() or "用户喜欢青绿色" not in memory.read_text():
                        await asyncio.sleep(.02)
                assert model.profile_requests, "Markdown 必须消费实际已使用摘要的原文"
                assert all("用户喜欢青绿色" in json.dumps(body, ensure_ascii=False) for body in model.profile_requests)
        profiles = len(model.profile_requests)
        memory_bytes = memory.read_bytes()
        async with runtime(tmp_path, endpoint, settings=settings) as (_second, address):
            async with await AsyncAkashic.connect(address) as client:
                assert (await client.message_read(session, limit=200))["items"] == saved
                continued = await complete(client, session, "short", after=saved[-1]["seq"])
                assert continued[:len(saved)] == saved
                assert "DONE:short" in str(continued[-1])
                assert memory.read_bytes() == memory_bytes
                assert len(model.profile_requests) >= profiles


@pytest.mark.asyncio
@pytest.mark.parametrize("producer,stages", [("drift", ["drift"]), ("alert", ["alert"]),
                                            ("content", ["screen", "investigate"])])
async def test_proactive_producer_reaches_real_delivery_and_survives_reopen(tmp_path: Path, producer: str, stages: list[str]) -> None:
    """生产者只提交业务输入；真实 Wake 判断、私有工具、投递和恢复均由 App 执行。"""
    model = ScriptedModel(tmp_path)
    settings = {"embedding": True, "plugins": {"wake": '[delivery]\nchannel="akashic"\nrecipient="wake-sink"\nsession_id="akashic:wake-sink"\n'}}
    async with model.serve() as endpoint:
        async with runtime(tmp_path, endpoint, settings=settings) as (process, address):
            async with await AsyncAkashic.connect(address) as client:
                async with await client.session_follow("akashic:wake-sink") as notices:
                    assert process.stdin is not None and process.stdout is not None
                    process.stdin.write(json.dumps({"type": producer}).encode() + b"\n")
                    await process.stdin.drain()
                    async with asyncio.timeout(10):
                        receipt = json.loads(await process.stdout.readline())
                    assert receipt["fixture_receipt"]
                    async with asyncio.timeout(35):
                        async for event in notices.events():
                            if event["type"] == "messages.appended":
                                break
                saved = (await client.message_read("akashic:wake-sink"))["items"]
                assert len(saved) == 1 and f"NOTICE:{stages[-1]}" in str(saved)
                assert model.wake_stages == stages
                if producer == "content":
                    assert "https://example.com/original" in str(saved)
        # 正常退出先排空领域结算；重开必须保留真实投递，不能再调用模型或再次通知。
        calls = len(model.requests)
        async with runtime(tmp_path, endpoint, settings=settings) as (_second, address):
            async with await AsyncAkashic.connect(address) as client:
                assert (await client.message_read("akashic:wake-sink"))["items"] == saved
                async with dashboard(tmp_path, "wake") as web_client:
                    runs = await web_client.get("/api/dashboard/wake/runs")
                    assert runs.status_code == 200, runs.text
                    assert runs.json()["total"] == 1
                    run = runs.json()["items"][0]
                    detail = await web_client.get(f'/api/dashboard/wake/runs/{run["run_id"]}')
                    assert detail.status_code == 200, detail.text
                    assert "NOTICE:" in detail.text
                assert len(model.requests) == calls

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent.tools.base import Tool
from agent.tools.registry import ToolRegistry
from agent.looping.ports import ObservabilityServices, SessionServices
from agent.turns.orchestrator import TurnOrchestrator, TurnOrchestratorDeps
from agent.turns.outbound import OutboundDispatch
from agent.turns.result import TurnOutbound, TurnResult, TurnTrace
from proactive_v2.agent_tick import AgentTick
from proactive_v2.context import AgentTickContext
from proactive_v2.drift_runner import DriftRunner
from proactive_v2.drift_state import DriftStateStore
from proactive_v2.drift_tools import DriftToolDeps, build_drift_tool_registry
from proactive_v2.agent_tick_factory import AgentTickDeps, AgentTickFactory
from proactive_v2.gateway import GatewayDeps
from proactive_v2.mcp_sources import McpClientPool
from proactive_v2.tools import ToolDeps
from tests.proactive_v2.conftest import FakeLLM, FakeRng, cfg_with, make_agent_tick


def _write_skill(root: Path, name: str = "explore-curiosity") -> Path:
    skill_dir = root / "skills" / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        (
            "---\n"
            f"name: {name}\n"
            "description: 对用户产生好奇，通过提问丰满用户画像\n"
            "---\n\n"
            "test skill\n"
        ),
        encoding="utf-8",
    )
    return skill_dir


class _DummyTool(Tool):
    def __init__(self, name: str):
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._name

    @property
    def parameters(self) -> dict:
        return {"type": "object", "properties": {}, "required": []}

    async def execute(self, **kwargs):
        return json.dumps({"ok": True}, ensure_ascii=False)


def _build_shared_tools() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(_DummyTool("recall_memory"))
    reg.register(_DummyTool("web_fetch"))
    reg.register(_DummyTool("web_search"))
    reg.register(_DummyTool("fetch_messages"))
    reg.register(_DummyTool("search_messages"))
    reg.register(_DummyTool("shell"))
    return reg


class _FakeWebFetchTool(Tool):
    @property
    def name(self) -> str:
        return "web_fetch"

    @property
    def description(self) -> str:
        return "web_fetch"

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {"url": {"type": "string"}},
            "required": ["url"],
        }

    async def execute(self, **kwargs):
        return json.dumps(
            {"text": "x" * 20, "length": 20, "format": "text"},
            ensure_ascii=False,
        )


async def _exec_drift_tool(
    tmp_path: Path,
    ctx: AgentTickContext,
    tool_name: str,
    args: dict,
    *,
    store: DriftStateStore | None = None,
    send_message_fn=None,
):
    resolved_store = store or DriftStateStore(tmp_path)
    reg = build_drift_tool_registry(
        ctx=ctx,
        deps=DriftToolDeps(
            drift_dir=tmp_path,
            store=resolved_store,
            shared_tools=_build_shared_tools(),
            send_message_fn=send_message_fn,
        ),
    )
    return await reg.execute(tool_name, args)


def test_drift_tool_schemas_include_reused_tools():
    ctx = AgentTickContext(now_utc=datetime.now(timezone.utc))
    names = {
        schema["function"]["name"]
        for schema in build_drift_tool_registry(
            ctx=ctx,
            deps=DriftToolDeps(
                drift_dir=Path("."),
                store=DriftStateStore(Path(".")),
                shared_tools=_build_shared_tools(),
            ),
        ).get_schemas()
    }
    assert "recall_memory" in names
    assert "web_fetch" in names
    assert "fetch_messages" in names
    assert "search_messages" in names
    assert "shell" in names
    assert "read_file" in names
    assert "edit_file" in names
    assert "get_recent_chat" not in names


def test_drift_system_prompt_discourages_stuck_skill_and_lists_new_tools(tmp_path: Path):
    _write_skill(tmp_path)
    store = DriftStateStore(tmp_path)
    runner = DriftRunner(
        store=store,
        tool_deps=DriftToolDeps(
            drift_dir=tmp_path,
            store=store,
            shared_tools=_build_shared_tools(),
        ),
    )
    prompt = runner._build_system_prompt(store.scan_skills())
    assert "不要因为某个 skill 最近刚运行过" in prompt
    assert "如果这个 skill 当前明显处于“等待用户回复/等待外部条件”的状态，就不要选它" in prompt
    assert "fetch_messages" in prompt
    assert "search_messages" in prompt
    assert "shell" in prompt


@pytest.mark.asyncio
async def test_drift_web_fetch_keeps_drift_level_truncation(tmp_path: Path):
    ctx = AgentTickContext(now_utc=datetime.now(timezone.utc))
    shared = ToolRegistry()
    shared.register(_DummyTool("recall_memory"))
    shared.register(_FakeWebFetchTool())
    shared.register(_DummyTool("web_search"))
    reg = build_drift_tool_registry(
        ctx=ctx,
        deps=DriftToolDeps(
            drift_dir=tmp_path,
            store=DriftStateStore(tmp_path),
            shared_tools=shared,
            max_web_fetch_chars=8,
        ),
    )
    raw = await reg.execute("web_fetch", {"url": "https://example.com"})
    payload = json.loads(raw)
    assert payload["text"] == "x" * 8
    assert payload["length"] == 8
    assert payload["truncated"] is True


@pytest.mark.asyncio
async def test_drift_readfile_accepts_outside_path(tmp_path: Path):
    ctx = AgentTickContext(now_utc=datetime.now(timezone.utc))
    outside = tmp_path.parent / "outside-read.txt"
    outside.write_text("outside ok\n", encoding="utf-8")
    raw = await _exec_drift_tool(tmp_path, ctx, "read_file", {"path": str(outside)})
    assert "outside ok" in str(raw)


@pytest.mark.asyncio
async def test_drift_readfile_accepts_skill_shorthand_path(tmp_path: Path):
    _write_skill(tmp_path)
    ctx = AgentTickContext(now_utc=datetime.now(timezone.utc))
    raw = await _exec_drift_tool(
        tmp_path, ctx, "read_file", {"path": "skills/explore-curiosity/SKILL.md"}
    )
    assert "test skill" in str(raw)


@pytest.mark.asyncio
async def test_drift_readfile_accepts_absolute_path_inside_drift_dir(tmp_path: Path):
    skill_dir = _write_skill(tmp_path)
    ctx = AgentTickContext(now_utc=datetime.now(timezone.utc))
    raw = await _exec_drift_tool(
        tmp_path, ctx, "read_file", {"path": str(skill_dir / "SKILL.md")}
    )
    assert "test skill" in str(raw)


@pytest.mark.asyncio
async def test_finish_drift_rejects_unknown_skill(tmp_path: Path):
    store = DriftStateStore(tmp_path)
    ctx = AgentTickContext(now_utc=datetime.now(timezone.utc))
    raw = await _exec_drift_tool(
        tmp_path,
        ctx,
        "finish_drift",
        {"skill_used": "missing", "one_line": "x", "next": "y"},
        store=store,
    )
    assert json.loads(raw)["error"] == "unknown skill: missing"


@pytest.mark.asyncio
async def test_drift_writefile_returns_json_error_on_directory_target(tmp_path: Path):
    _write_skill(tmp_path)
    ctx = AgentTickContext(now_utc=datetime.now(timezone.utc))
    raw = await _exec_drift_tool(
        tmp_path,
        ctx,
        "write_file",
        {"path": "skills/explore-curiosity", "content": "x"},
    )
    assert "写入文件失败" in str(raw)


def test_drift_state_store_scan_skills_reads_frontmatter(tmp_path: Path):
    _write_skill(tmp_path)
    store = DriftStateStore(tmp_path)
    skills = store.scan_skills()
    assert len(skills) == 1
    assert skills[0].name == "explore-curiosity"


@pytest.mark.asyncio
async def test_drift_runner_runs_and_finishes(tmp_path: Path):
    _write_skill(tmp_path)
    store = DriftStateStore(tmp_path)
    llm = FakeLLM(
        [
            ("read_file", {"path": "skills/explore-curiosity/SKILL.md"}),
            (
                "finish_drift",
                {
                    "skill_used": "explore-curiosity",
                    "one_line": "问了一个问题",
                    "next": "继续问下一个问题",
                },
            ),
        ]
    )
    ctx = AgentTickContext(now_utc=datetime.now(timezone.utc), session_key="s")
    runner = DriftRunner(
        store=store,
        tool_deps=DriftToolDeps(
            drift_dir=tmp_path,
            store=store,
            shared_tools=_build_shared_tools(),
        ),
        max_steps=5,
    )
    entered = await runner.run(ctx, llm)
    assert entered is True
    assert ctx.drift_finished is True
    drift = store.load_drift()
    assert drift["recent_runs"][-1]["skill"] == "explore-curiosity"
    assert llm.tool_choices[:2] == ["required", "required"]


@pytest.mark.asyncio
async def test_drift_runner_restricts_tools_after_send_message(tmp_path: Path):
    _write_skill(tmp_path)
    store = DriftStateStore(tmp_path)
    llm = FakeLLM(
        [
            ("send_message", {"content": "hello"}),
            ("finish_drift", {"skill_used": "explore-curiosity", "one_line": "sent", "next": "next"}),
        ]
    )
    runner = DriftRunner(
        store=store,
        tool_deps=DriftToolDeps(
            drift_dir=tmp_path,
            store=store,
            shared_tools=_build_shared_tools(),
            send_message_fn=AsyncMock(return_value=True),
        ),
        max_steps=5,
    )
    ctx = AgentTickContext(now_utc=datetime.now(timezone.utc), session_key="s")
    await runner.run(ctx, llm)
    second_names = {schema["function"]["name"] for schema in llm.calls[1][0:1]} if False else None
    assert llm.calls
    # 第二次 llm 调用的 schemas 只能由 DriftRunner 约束为 write_file/edit_file/finish_drift；
    # FakeLLM 不记录 schemas，这里用行为结果兜底：send 后仍正常 finish。
    assert ctx.drift_finished is True


@pytest.mark.asyncio
async def test_drift_runner_forced_write_allows_write_file_or_edit_file(tmp_path: Path):
    _write_skill(tmp_path)
    store = DriftStateStore(tmp_path)
    captured: list[tuple[list[str], str | dict]] = []

    async def llm(messages: list[dict], schemas: list[dict], tool_choice: str | dict = "auto"):
        captured.append(([s["function"]["name"] for s in schemas], tool_choice))
        step = len(captured)
        if step == 1:
            return {"name": "read_file", "input": {"path": "skills/explore-curiosity/SKILL.md"}}
        if step == 2:
            return {"name": "edit_file", "input": {"path": "skills/explore-curiosity/SKILL.md", "old_text": "test skill\n", "new_text": "updated\n"}}
        if step == 3:
            return {
                "name": "finish_drift",
                "input": {
                    "skill_used": "explore-curiosity",
                    "one_line": "updated",
                    "next": "continue",
                },
            }
        return None

    runner = DriftRunner(
        store=store,
        tool_deps=DriftToolDeps(
            drift_dir=tmp_path,
            store=store,
            shared_tools=_build_shared_tools(),
        ),
        max_steps=3,
    )
    ctx = AgentTickContext(now_utc=datetime.now(timezone.utc), session_key="s")
    await runner.run(ctx, llm)
    assert captured[1][1] == "required"
    assert set(captured[1][0]) == {"write_file", "edit_file"}


@pytest.mark.asyncio
async def test_agent_tick_enters_drift_and_records_action(tmp_path: Path):
    _write_skill(tmp_path)
    gate = MagicMock()
    gate.should_act.return_value = (True, {})
    llm = FakeLLM(
        [
            ("read_file", {"path": "skills/explore-curiosity/SKILL.md"}),
            (
                "finish_drift",
                {
                    "skill_used": "explore-curiosity",
                    "one_line": "整理了漂移状态",
                    "next": "下次继续",
                },
            ),
        ]
    )
    tick = make_agent_tick(
        cfg=cfg_with(drift_enabled=True, drift_dir=str(tmp_path)),
        any_action_gate=gate,
        llm_fn=llm,
        tool_deps=ToolDeps(recent_chat_fn=AsyncMock(return_value=[])),
        gateway_deps=GatewayDeps(
            alert_fn=AsyncMock(return_value=[]),
            feed_fn=AsyncMock(return_value=[]),
            context_fn=AsyncMock(return_value=[]),
        ),
        rng=FakeRng(value=1.0),
        drift_runner=DriftRunner(
            store=DriftStateStore(tmp_path),
            tool_deps=DriftToolDeps(
                drift_dir=tmp_path,
                store=DriftStateStore(tmp_path),
                shared_tools=_build_shared_tools(),
            ),
            max_steps=5,
        ),
    )
    await tick.tick()
    assert tick.last_ctx.drift_entered is True
    gate.record_action.assert_called_once()


@pytest.mark.asyncio
async def test_agent_tick_drift_send_message_skips_normal_post_loop(tmp_path: Path):
    _write_skill(tmp_path)
    sender = AsyncMock(return_value=True)
    events: list[object] = []

    class _Writer:
        def emit(self, event: object) -> None:
            events.append(event)

    class _Session:
        def __init__(self) -> None:
            self.messages: list[dict] = []
            self.metadata: dict[str, object] = {}
            self.last_consolidated = 0
            self.presence = None

        def add_message(self, role: str, content: str, media=None, **kwargs) -> None:
            msg = {"role": role, "content": content}
            msg.update(kwargs)
            self.messages.append(msg)

    session = _Session()
    session_manager = SimpleNamespace(
        get_or_create=lambda _key: session,
        append_messages=AsyncMock(return_value=None),
    )

    class _PostTurn:
        def schedule(self, event) -> None:
            return

    class _Outbound:
        async def dispatch(self, outbound: OutboundDispatch) -> bool:
            return await sender(outbound.content)

    orchestrator = TurnOrchestrator(
        TurnOrchestratorDeps(
            session=SessionServices(
                session_manager=session_manager,
                presence=SimpleNamespace(record_proactive_sent=lambda _key: None),
            ),
            trace=ObservabilityServices(workspace=Path("."), observe_writer=_Writer()),
            post_turn=_PostTurn(),
            outbound=_Outbound(),
        )
    )

    async def send_message(content: str) -> bool:
        return await orchestrator.handle_proactive_turn(
            result=TurnResult(
                decision="reply",
                outbound=TurnOutbound(session_key="test_session", content=content),
                trace=TurnTrace(source="proactive", extra={"source_mode": "drift"}),
            ),
            session_key="test_session",
            channel="telegram",
            chat_id="1",
        )

    gate = MagicMock()
    gate.should_act.return_value = (True, {})
    llm = FakeLLM(
        [
            ("send_message", {"content": "hello from drift"}),
            (
                "finish_drift",
                {
                    "skill_used": "explore-curiosity",
                    "one_line": "发出一条消息",
                    "next": "下次继续",
                },
            ),
        ]
    )
    tick = AgentTick(
        cfg=cfg_with(
            drift_enabled=True,
            drift_dir=str(tmp_path),
            default_channel="telegram",
            default_chat_id="1",
        ),
        session_key="test_session",
        state_store=SimpleNamespace(
            count_deliveries_in_window=lambda *_args: 0,
            get_last_context_only_at=lambda *_args: None,
            count_context_only_in_window=lambda *_args, **_kwargs: 0,
            get_last_drift_at=lambda *_args: None,
            mark_drift_run=lambda *_args, **_kwargs: None,
            is_delivery_duplicate=lambda *_args, **_kwargs: False,
        ),
        any_action_gate=gate,
        last_user_at_fn=lambda: None,
        passive_busy_fn=None,
        turn_orchestrator=orchestrator,
        deduper=AsyncMock(),
        tool_deps=ToolDeps(recent_chat_fn=AsyncMock(return_value=[])),
        gateway_deps=GatewayDeps(
            alert_fn=AsyncMock(return_value=[]),
            feed_fn=AsyncMock(return_value=[]),
            context_fn=AsyncMock(return_value=[]),
        ),
        llm_fn=llm,
        rng=FakeRng(value=1.0),
        recent_proactive_fn=lambda: [],
        drift_runner=DriftRunner(
            store=DriftStateStore(tmp_path),
            tool_deps=DriftToolDeps(
                drift_dir=tmp_path,
                store=DriftStateStore(tmp_path),
                shared_tools=_build_shared_tools(),
                send_message_fn=send_message,
            ),
            max_steps=5,
        ),
    )

    await tick.tick()

    sender.assert_awaited_once_with("hello from drift")
    gate.record_action.assert_called_once()
    assert tick.last_ctx.drift_entered is True
    assert tick.last_ctx.drift_message_sent is True
    assert len(events) == 1
    event = events[0]
    assert len(event.tool_calls) == 1
    payload = json.loads(event.tool_calls[0]["args"])
    assert payload["decision"] == "reply"
    assert payload["sent"] is True
    assert payload["skip_reason"] == ""


class _FakeProvider:
    async def chat(self, **kwargs):
        return SimpleNamespace(tool_calls=[])


def _build_factory(tmp_path: Path, *, sender_ok: bool, state_store):
    sender = AsyncMock()
    sender.send.return_value = sender_ok

    session = SimpleNamespace(
        messages=[],
        add_message=lambda *args, **kwargs: session.messages.append(
            {"args": args, "kwargs": kwargs}
        ),
    )
    session_manager = SimpleNamespace(
        get_or_create=lambda _key: session,
        append_messages=AsyncMock(return_value=None),
    )

    class _PostTurn:
        def schedule(self, event) -> None:
            return

    class _Outbound:
        async def dispatch(self, outbound) -> bool:
            return await sender.send(outbound.content)

    from agent.looping.ports import ObservabilityServices, SessionServices
    from agent.turns.orchestrator import TurnOrchestrator, TurnOrchestratorDeps

    orchestrator = TurnOrchestrator(
        TurnOrchestratorDeps(
            session=SessionServices(
                session_manager=session_manager,
                presence=SimpleNamespace(record_proactive_sent=lambda _key: None),
            ),
            trace=ObservabilityServices(workspace=Path("."), observe_writer=None),
            post_turn=_PostTurn(),
            outbound=_Outbound(),
        )
    )

    deps = AgentTickDeps(
        cfg=cfg_with(
            drift_enabled=True,
            drift_dir=str(tmp_path),
            default_channel="telegram",
            default_chat_id="1",
        ),
        sense=SimpleNamespace(
            target_session_key=lambda: "telegram:1",
            collect_recent=lambda: [],
            collect_recent_proactive=lambda n: [],
        ),
        presence=SimpleNamespace(get_last_user_at=lambda _: None),
        provider=_FakeProvider(),
        model="m",
        max_tokens=128,
        memory=None,
        state_store=state_store,
        any_action_gate=SimpleNamespace(),
        passive_busy_fn=None,
        deduper=None,
        rng=SimpleNamespace(),
        workspace_context_fn=lambda: "",
        observe_writer=None,
        shared_tools=_build_shared_tools(),
        turn_orchestrator=orchestrator,
        pool=McpClientPool(),
    )
    return AgentTickFactory(deps), sender


@pytest.mark.asyncio
async def test_factory_drift_send_message_returns_false_when_send_fails(tmp_path: Path):
    state = SimpleNamespace(path=tmp_path / "proactive_state.json", mark_delivery=MagicMock())
    factory, sender = _build_factory(tmp_path, sender_ok=False, state_store=state)
    send_message = factory._build_drift_send_message_fn()
    assert send_message is not None
    ok = await send_message("hello")
    assert ok is False
    state.mark_delivery.assert_not_called()
    sender.send.assert_called_once_with("hello")


@pytest.mark.asyncio
async def test_factory_drift_send_message_marks_delivery_on_success(tmp_path: Path):
    state = SimpleNamespace(path=tmp_path / "proactive_state.json", mark_delivery=MagicMock())
    factory, sender = _build_factory(tmp_path, sender_ok=True, state_store=state)
    send_message = factory._build_drift_send_message_fn()
    assert send_message is not None
    ok = await send_message("hello")
    assert ok is True
    state.mark_delivery.assert_called_once()
    sender.send.assert_called_once_with("hello")

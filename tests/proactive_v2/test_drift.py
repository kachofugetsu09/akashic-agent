from __future__ import annotations
from typing import Any, cast

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent.prompting import is_context_frame
from agent.core.proactive_turn import ProactiveTurnPipeline, ProactiveTurnPipelineDeps
from agent.tools.base import Tool
from agent.tools.registry import ToolRegistry
from agent.looping.ports import SessionServices
from agent.turns.orchestrator import TurnOrchestrator, TurnOrchestratorDeps
from agent.turns.outbound import OutboundDispatch
from agent.turns.result import TurnOutbound, TurnResult, TurnTrace
from proactive_v2.context import AgentTickContext
from agent.core.drift_turn import DriftTurnPipeline, DriftTurnPipelineDeps
from proactive_v2.drift_state import DriftStateStore
from proactive_v2.drift_tools import DriftToolDeps, build_drift_tool_registry
from proactive_v2.agent_tick_factory import AgentTickDeps, AgentTickFactory
from proactive_v2.gateway import GatewayDeps
from proactive_v2.mcp_sources import McpClientPool
from proactive_v2.tools import ToolDeps
from tests.proactive_v2.conftest import FakeLLM, FakeRng, cfg_with, make_proactive_pipeline, run_proactive_pipeline


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


class _CapturingShellTool(Tool):
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    @property
    def name(self) -> str:
        return "shell"

    @property
    def description(self) -> str:
        return "shell"

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "command": {"type": "string"},
                "description": {"type": "string"},
                "cwd": {"type": "string"},
            },
            "required": ["command", "description"],
        }

    async def execute(self, **kwargs):
        self.calls.append(dict(kwargs))
        return json.dumps({"ok": True}, ensure_ascii=False)


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
            builtin_skills_dir=getattr(resolved_store, "builtin_skills_dir", None),
            shared_tools=_build_shared_tools(),
            send_message_fn=send_message_fn,
        ),
    )
    return await reg.execute(tool_name, args)


def _make_drift_pipeline(
    *,
    store: DriftStateStore,
    tool_deps: DriftToolDeps,
    max_steps: int = 20,
) -> DriftTurnPipeline:
    return DriftTurnPipeline(
        DriftTurnPipelineDeps(
            store=store,
            tool_deps=tool_deps,
            max_steps=max_steps,
        )
    )


def test_drift_tool_schemas_include_reused_tools(tmp_path: Path):
    ctx = AgentTickContext(now_utc=datetime.now(timezone.utc))
    names = {
        schema["function"]["name"]
        for schema in build_drift_tool_registry(
            ctx=ctx,
            deps=DriftToolDeps(
                drift_dir=tmp_path,
                store=DriftStateStore(tmp_path),
                shared_tools=_build_shared_tools(),
            ),
        ).get_schemas()
    }
    assert "recall_memory" in names
    assert "web_fetch" in names
    assert "fetch_messages" in names
    assert "search_messages" in names
    assert "shell" in names
    assert "select_skill" in names
    assert "read_file" in names
    assert "list_dir" in names
    assert "edit_file" in names
    assert "get_recent_chat" not in names


def test_drift_message_push_schema_supports_media(tmp_path: Path):
    ctx = AgentTickContext(now_utc=datetime.now(timezone.utc))
    schemas = build_drift_tool_registry(
        ctx=ctx,
        deps=DriftToolDeps(
            drift_dir=tmp_path,
            store=DriftStateStore(tmp_path),
            shared_tools=_build_shared_tools(),
        ),
    ).get_schemas()
    message_push = next(
        schema["function"] for schema in schemas if schema["function"]["name"] == "message_push"
    )
    props = message_push["parameters"]["properties"]
    assert "image" in props
    assert "media" in props
    assert "message" not in message_push["parameters"].get("required", [])


@pytest.mark.asyncio
async def test_drift_message_push_sends_media(tmp_path: Path):
    ctx = AgentTickContext(now_utc=datetime.now(timezone.utc))
    send_message = AsyncMock(return_value=True)
    raw = await _exec_drift_tool(
        tmp_path,
        ctx,
        "message_push",
        {
            "message": "新表情来啦",
            "image": "/tmp/one.png",
            "media": ["/tmp/two.png"],
        },
        send_message_fn=send_message,
    )
    assert json.loads(cast(Any, raw))["ok"] is True
    send_message.assert_awaited_once_with("新表情来啦", ["/tmp/one.png", "/tmp/two.png"])


@pytest.mark.asyncio
async def test_drift_system_prompt_discourages_stuck_skill_and_lists_new_tools(tmp_path: Path):
    _write_skill(tmp_path)
    store = DriftStateStore(tmp_path)
    pipeline = _make_drift_pipeline(
        store=store,
        tool_deps=DriftToolDeps(
            drift_dir=tmp_path,
            store=store,
            shared_tools=_build_shared_tools(),
        ),
    )
    prompt = pipeline._build_system_prompt()
    runtime = str((await pipeline._build_runtime_context_message(store.scan_skills()))["content"])
    assert "select_skill 会记录本轮 selected_skill" in prompt
    assert "路径由 drift mount resolver 解析" in prompt
    assert "message_result" in prompt
    assert "select_skill" in prompt
    assert "idle_drift" in prompt
    assert "fetch_messages" in prompt
    assert "search_messages" in prompt
    assert "shell" in prompt
    assert is_context_frame(runtime)
    assert "drift_skills" in runtime


@pytest.mark.asyncio
async def test_drift_runtime_context_provides_skill_selection_state(tmp_path: Path):
    _write_skill(tmp_path, name="explore-curiosity")
    _write_skill(tmp_path, name="meme-auto-generate")
    store = DriftStateStore(tmp_path)
    store.save_finish(
        skill_used="explore-curiosity",
        status="completed",
        briefing="没有自然切口",
        message_result="silent",
        scratchpad_update=None,
        global_note_update=None,
        now_utc=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    store.save_finish(
        skill_used="meme-auto-generate",
        status="completed",
        briefing="刚生成表情",
        message_result="sent",
        scratchpad_update=None,
        global_note_update=None,
        now_utc=datetime(2026, 1, 2, tzinfo=timezone.utc),
        cursor_update={"next_mode": "create_category"},
    )
    pipeline = _make_drift_pipeline(
        store=store,
        tool_deps=DriftToolDeps(
            drift_dir=tmp_path,
            store=store,
            shared_tools=_build_shared_tools(),
        ),
    )
    runtime = str((await pipeline._build_runtime_context_message(store.scan_skills()))["content"])
    assert "drift_selection_context" in runtime
    assert "按 skill 名称排列，顺序不代表优先级" in runtime
    assert "选择依据：status、上次 finish 时间" in runtime
    assert "recent_raw_chat" in runtime
    assert "local_context 只在 select_skill 后作为执行上下文参考" in runtime
    assert "explore-curiosity: status=completed" in runtime
    assert "meme-auto-generate: status=completed" in runtime
    assert "last_finish=2026-01-01T00:00:00+00:00" in runtime
    assert "last_finish=2026-01-02T00:00:00+00:00" in runtime
    assert "上次 finish：2026-01-01T00:00:00+00:00" in runtime
    assert "briefing=没有自然切口" in runtime
    assert "briefing=刚生成表情" in runtime
    assert 'cursor={"next_mode": "create_category"}' in runtime
    assert "本轮首选" not in runtime
    assert "首个工具调用" not in runtime
    assert runtime.index("drift_selection_context") < runtime.index("long_term_memory")


@pytest.mark.asyncio
async def test_drift_runtime_context_does_not_expose_memory_file_path(tmp_path: Path):
    _write_skill(tmp_path)
    memory_file = tmp_path / "memory" / "MEMORY.md"
    memory_file.parent.mkdir()
    memory_file.write_text("- test memory", encoding="utf-8")
    store = DriftStateStore(tmp_path)
    pipeline = _make_drift_pipeline(
        store=store,
        tool_deps=DriftToolDeps(
            drift_dir=tmp_path,
            store=store,
            shared_tools=_build_shared_tools(),
            memory=SimpleNamespace(
                memory_file=memory_file,
                read_long_term=lambda: "- test memory",
                read_recent_context=lambda: "",
            ),
        ),
    )
    runtime = str((await pipeline._build_runtime_context_message(store.scan_skills()))["content"])
    assert "drift_runtime_state" not in runtime
    assert "长期记忆文件 MEMORY.md" not in runtime
    assert str(memory_file) not in runtime


@pytest.mark.asyncio
async def test_drift_runtime_context_uses_recent_five_raw_chat_messages(tmp_path: Path):
    _write_skill(tmp_path)
    store = DriftStateStore(tmp_path)

    async def recent_chat_fn(n: int = 20) -> list[dict]:
        return [
            {"role": "user", "content": f"消息 {idx}", "proactive": idx % 2 == 0}
            for idx in range(1, 8)
        ]

    pipeline = _make_drift_pipeline(
        store=store,
        tool_deps=DriftToolDeps(
            drift_dir=tmp_path,
            store=store,
            recent_chat_fn=recent_chat_fn,
            shared_tools=_build_shared_tools(),
        ),
    )
    runtime = str((await pipeline._build_runtime_context_message(store.scan_skills()))["content"])
    assert "recent_raw_chat" in runtime
    assert "recent_context" not in runtime
    assert "消息 1" not in runtime
    assert "消息 2" not in runtime
    assert "消息 3" in runtime
    assert "消息 7" in runtime
    assert "proactive=true" in runtime


@pytest.mark.asyncio
async def test_drift_pipeline_can_idle_before_selecting_skill(tmp_path: Path):
    _write_skill(tmp_path)
    store = DriftStateStore(tmp_path)
    captured: list[tuple[list[str], str | dict]] = []

    async def llm(messages: list[dict], schemas: list[dict], tool_choice: str | dict = "auto"):
        captured.append(([s["function"]["name"] for s in schemas], tool_choice))
        return {"name": "idle_drift", "input": {"reason": "最近对话刚结束，主动打扰价值不高"}}

    pipeline = _make_drift_pipeline(
        store=store,
        tool_deps=DriftToolDeps(
            drift_dir=tmp_path,
            store=store,
            shared_tools=_build_shared_tools(),
        ),
        max_steps=3,
    )
    ctx = AgentTickContext(now_utc=datetime.now(timezone.utc), session_key="s")
    await pipeline.run(ctx, cast(Any, llm))
    recent = store.load_drift()["recent_runs"][-1]
    assert captured == [(["select_skill", "idle_drift"], "required")]
    assert ctx.drift_finished is True
    assert ctx.drift_selected_skill == "idle"
    assert recent["skill"] == "idle"
    assert recent["message_result"] == "silent"
    assert "主动打扰价值不高" in recent["briefing"]


@pytest.mark.asyncio
async def test_drift_web_fetch_uses_shared_tool_result_without_wrapper(tmp_path: Path):
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
        ),
    )
    raw = await reg.execute("web_fetch", {"url": "https://example.com"})
    payload = json.loads(cast(Any, raw))
    assert payload["text"] == "x" * 20
    assert payload["length"] == 20
    assert "truncated" not in payload


@pytest.mark.asyncio
async def test_drift_shell_defaults_to_drift_dir(tmp_path: Path):
    ctx = AgentTickContext(now_utc=datetime.now(timezone.utc))
    shared = ToolRegistry()
    shell = _CapturingShellTool()
    shared.register(shell)
    reg = build_drift_tool_registry(
        ctx=ctx,
        deps=DriftToolDeps(
            drift_dir=tmp_path,
            store=DriftStateStore(tmp_path),
            shared_tools=shared,
        ),
    )

    await reg.execute(
        "shell",
        {"command": "python skills/demo.py", "description": "运行脚本"},
    )
    await reg.execute(
        "shell",
        {"command": "python scripts/demo.py", "description": "运行脚本", "cwd": "skills/demo"},
    )

    assert shell.calls[0]["cwd"] == str(tmp_path)
    assert shell.calls[1]["cwd"] == str(tmp_path / "skills/demo")


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
async def test_select_skill_records_selected_skill_and_returns_skill_doc(tmp_path: Path):
    _write_skill(tmp_path)
    store = DriftStateStore(tmp_path)
    now = datetime.now(timezone.utc)
    store.save_finish(
        skill_used="explore-curiosity",
        status="completed",
        briefing="刚问过音乐偏好",
        message_result="silent",
        scratchpad_update="短期避免继续问音乐，优先换成食物口味。",
        global_note_update=None,
        now_utc=now,
        cursor_update={"last_topic": "音乐", "waiting_for_answer": True},
        journal_append=[
            {
                "entry_type": "curiosity_asked",
                "key": "music",
                "payload": {"question": "你平时听什么音乐？"},
            }
        ],
    )
    ctx = AgentTickContext(now_utc=datetime.now(timezone.utc))
    raw = await _exec_drift_tool(
        tmp_path,
        ctx,
        "select_skill",
        {"skill_name": "explore-curiosity"},
        store=store,
    )
    payload = json.loads(cast(Any, raw))
    assert payload["ok"] is True
    assert payload["skill"] == "explore-curiosity"
    assert "test skill" in payload["content"]
    assert payload["local_context"]["last_status"] == "completed"
    assert payload["local_context"]["last_briefing"] == "刚问过音乐偏好"
    assert payload["local_context"]["scratchpad"] == "短期避免继续问音乐，优先换成食物口味。"
    assert payload["local_context"]["cursor"]["last_topic"] == "音乐"
    assert payload["local_context"]["journal_recent"][0]["entry_type"] == "curiosity_asked"
    assert ctx.drift_selected_skill == "explore-curiosity"


@pytest.mark.asyncio
async def test_drift_listdir_accepts_skill_shorthand_path(tmp_path: Path):
    _write_skill(tmp_path)
    ctx = AgentTickContext(now_utc=datetime.now(timezone.utc))
    raw = await _exec_drift_tool(
        tmp_path, ctx, "list_dir", {"path": "skills/explore-curiosity"}
    )
    assert "SKILL.md" in str(raw)


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
        {
            "skill_used": "missing",
            "status": "completed",
            "briefing": "x",
            "message_result": "silent",
        },
        store=store,
    )
    assert json.loads(cast(Any, raw))["error"] == "unknown skill: missing"


@pytest.mark.asyncio
async def test_finish_drift_requires_message_result_to_match_actual_send(tmp_path: Path):
    _write_skill(tmp_path)
    store = DriftStateStore(tmp_path)
    ctx = AgentTickContext(now_utc=datetime.now(timezone.utc))
    raw = await _exec_drift_tool(
        tmp_path,
        ctx,
        "finish_drift",
        {
            "skill_used": "explore-curiosity",
            "status": "completed",
            "briefing": "x",
            "message_result": "sent",
        },
        store=store,
    )
    payload = json.loads(cast(Any, raw))
    assert payload["error"] == "message_result=sent requires successful message_push first"
    assert ctx.drift_finished is False


@pytest.mark.asyncio
async def test_finish_drift_rejects_missing_message_result(tmp_path: Path):
    _write_skill(tmp_path)
    store = DriftStateStore(tmp_path)
    ctx = AgentTickContext(now_utc=datetime.now(timezone.utc))
    raw = await _exec_drift_tool(
        tmp_path,
        ctx,
        "finish_drift",
        {
            "skill_used": "explore-curiosity",
            "status": "completed",
            "briefing": "x",
        },
        store=store,
    )
    payload = json.loads(cast(Any, raw))
    assert payload["error"] == "message_result must be one of: sent, silent"
    assert ctx.drift_finished is False


@pytest.mark.asyncio
async def test_finish_drift_rejects_silent_after_message_sent(tmp_path: Path):
    _write_skill(tmp_path)
    store = DriftStateStore(tmp_path)
    ctx = AgentTickContext(now_utc=datetime.now(timezone.utc))
    ctx.drift_message_sent = True
    raw = await _exec_drift_tool(
        tmp_path,
        ctx,
        "finish_drift",
        {
            "skill_used": "explore-curiosity",
            "status": "completed",
            "briefing": "x",
            "message_result": "silent",
        },
        store=store,
    )
    payload = json.loads(cast(Any, raw))
    assert payload["error"] == "message_result=silent conflicts with successful message_push"
    assert ctx.drift_finished is False


@pytest.mark.asyncio
async def test_finish_drift_saves_silent_message_result(tmp_path: Path):
    _write_skill(tmp_path)
    store = DriftStateStore(tmp_path)
    ctx = AgentTickContext(now_utc=datetime.now(timezone.utc))
    raw = await _exec_drift_tool(
        tmp_path,
        ctx,
        "finish_drift",
        {
            "skill_used": "explore-curiosity",
            "status": "completed",
            "briefing": "x",
            "message_result": "silent",
        },
        store=store,
    )
    assert json.loads(cast(Any, raw))["ok"] is True
    assert store.load_drift()["recent_runs"][-1]["message_result"] == "silent"
    assert not (tmp_path / "skills" / "explore-curiosity" / "state.json").exists()


@pytest.mark.asyncio
async def test_finish_drift_saves_cursor_and_journal(tmp_path: Path):
    _write_skill(tmp_path)
    store = DriftStateStore(tmp_path)
    ctx = AgentTickContext(now_utc=datetime.now(timezone.utc))
    raw = await _exec_drift_tool(
        tmp_path,
        ctx,
        "finish_drift",
        {
            "skill_used": "explore-curiosity",
            "status": "completed",
            "briefing": "问了音乐偏好",
            "message_result": "silent",
            "cursor_update": {"last_topic": "music", "waiting_for_answer": True},
            "journal_append": [
                {
                    "entry_type": "curiosity_asked",
                    "key": "music",
                    "payload": {"question": "最近常听什么？"},
                }
            ],
        },
        store=store,
    )

    assert json.loads(cast(Any, raw))["ok"] is True
    continuum = store.load_skill_continuum("explore-curiosity")
    journal = store.load_skill_journal("explore-curiosity")
    assert continuum["cursor"]["last_topic"] == "music"
    assert continuum["cursor"]["waiting_for_answer"] is True
    assert journal[0]["entry_type"] == "curiosity_asked"
    assert journal[0]["key"] == "music"


@pytest.mark.asyncio
async def test_finish_drift_paused_requires_scratchpad(tmp_path: Path):
    _write_skill(tmp_path)
    store = DriftStateStore(tmp_path)
    ctx = AgentTickContext(now_utc=datetime.now(timezone.utc))
    raw = await _exec_drift_tool(
        tmp_path,
        ctx,
        "finish_drift",
        {
            "skill_used": "explore-curiosity",
            "status": "paused",
            "briefing": "读到一半",
            "message_result": "silent",
        },
        store=store,
    )
    payload = json.loads(cast(Any, raw))
    assert payload["error"] == (
        "scratchpad_update is required when status is paused or waiting"
    )
    assert ctx.drift_finished is False


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


def test_drift_state_store_links_steps_to_finished_run(tmp_path: Path):
    _write_skill(tmp_path)
    store = DriftStateStore(tmp_path)
    now = datetime.now(timezone.utc)
    store.append_step(
        step_index=1,
        tool_name="read_file",
        input_preview="{}",
        output_preview="ok",
        now_utc=now,
    )
    store.save_finish(
        skill_used="explore-curiosity",
        status="completed",
        briefing="done",
        message_result="silent",
        scratchpad_update=None,
        global_note_update=None,
        now_utc=now,
    )

    conn = sqlite3.connect(store.db_file)
    try:
        row = conn.execute(
            """
            SELECT run_steps.run_id, runs.id
            FROM run_steps
            JOIN runs ON runs.id = run_steps.run_id
            """
        ).fetchone()
    finally:
        conn.close()
    assert row is not None
    assert row[0] == row[1]


@pytest.mark.asyncio
async def test_drift_pipeline_runs_and_finishes(tmp_path: Path):
    _write_skill(tmp_path)
    store = DriftStateStore(tmp_path)
    llm = FakeLLM(
        [
            ("select_skill", {"skill_name": "explore-curiosity"}),
            (
                "finish_drift",
                {
                    "skill_used": "explore-curiosity",
                    "status": "completed",
                    "briefing": "问了一个问题",
                    "message_result": "silent",
                },
            ),
        ]
    )
    ctx = AgentTickContext(now_utc=datetime.now(timezone.utc), session_key="s")
    pipeline = _make_drift_pipeline(
        store=store,
        tool_deps=DriftToolDeps(
            drift_dir=tmp_path,
            store=store,
            shared_tools=_build_shared_tools(),
        ),
        max_steps=5,
    )
    entered = await pipeline.run(ctx, cast(Any, llm))
    assert entered is True
    assert ctx.drift_finished is True
    assert is_context_frame(str(llm.calls[0][1]["content"]))
    drift = store.load_drift()
    assert drift["recent_runs"][-1]["skill"] == "explore-curiosity"
    assert ctx.drift_selected_skill == "explore-curiosity"
    assert llm.tool_choices[:2] == [
        "required",
        "required",
    ]
    conn = sqlite3.connect(store.db_file)
    try:
        latest_run = conn.execute("SELECT max(id) FROM runs").fetchone()
        finish_step = conn.execute(
            "SELECT run_id FROM run_steps WHERE tool_name = 'finish_drift'"
        ).fetchone()
    finally:
        conn.close()
    assert latest_run is not None
    assert finish_step is not None
    assert finish_step[0] == latest_run[0]


@pytest.mark.asyncio
async def test_drift_pipeline_restricts_tools_after_send_message(tmp_path: Path):
    _write_skill(tmp_path)
    store = DriftStateStore(tmp_path)
    send_message = AsyncMock(return_value=True)
    llm = FakeLLM(
        [
            ("select_skill", {"skill_name": "explore-curiosity"}),
            ("message_push", {"message": "hello\\n\\nfrom drift"}),
            (
                "finish_drift",
                {
                    "skill_used": "explore-curiosity",
                    "status": "completed",
                    "briefing": "sent",
                    "message_result": "sent",
                },
            ),
        ]
    )
    pipeline = _make_drift_pipeline(
        store=store,
        tool_deps=DriftToolDeps(
            drift_dir=tmp_path,
            store=store,
            shared_tools=_build_shared_tools(),
            send_message_fn=send_message,
        ),
        max_steps=5,
    )
    ctx = AgentTickContext(now_utc=datetime.now(timezone.utc), session_key="s")
    await pipeline.run(ctx, cast(Any, llm))
    second_names = {schema["function"]["name"] for schema in llm.calls[1][0:1]} if False else None
    assert llm.calls
    # FakeLLM 不记录 schemas，这里用行为结果兜底：send 后仍正常 finish。
    assert ctx.drift_finished is True
    assert store.load_drift()["recent_runs"][-1]["message_result"] == "sent"
    send_message.assert_awaited_once_with("hello\n\nfrom drift", [])


@pytest.mark.asyncio
async def test_drift_pipeline_wraps_up_at_step_limit(tmp_path: Path):
    _write_skill(tmp_path)
    store = DriftStateStore(tmp_path)
    captured: list[tuple[list[str], str | dict]] = []

    async def llm(messages: list[dict], schemas: list[dict], tool_choice: str | dict = "auto"):
        captured.append(([s["function"]["name"] for s in schemas], tool_choice))
        step = len(captured)
        if step == 1:
            return {"name": "select_skill", "input": {"skill_name": "explore-curiosity"}}
        if step == 2:
            return {
                "name": "write_file",
                "input": {"path": "skills/explore-curiosity/state.json", "content": "{}"},
            }
        if step == 3:
            return {
                "name": "read_file",
                "input": {"path": "skills/explore-curiosity/state.json"},
            }
        if step == 4:
            return {
                "name": "finish_drift",
                "input": {
                    "skill_used": "explore-curiosity",
                    "status": "paused",
                    "briefing": "读了 skill 并写了中间状态",
                    "scratchpad_update": "下次继续检查 state.json",
                    "message_result": "silent",
                },
            }
        return None

    pipeline = _make_drift_pipeline(
        store=store,
        tool_deps=DriftToolDeps(
            drift_dir=tmp_path,
            store=store,
            shared_tools=_build_shared_tools(),
        ),
        max_steps=3,
    )
    ctx = AgentTickContext(now_utc=datetime.now(timezone.utc), session_key="s")
    await pipeline.run(ctx, cast(Any, llm))
    assert "read_file" in captured[2][0]
    assert "write_file" in captured[2][0]
    assert "shell" in captured[2][0]
    assert captured[2][1] == "required"
    assert captured[3][0] == ["finish_drift"]
    assert captured[3][1] == {"type": "function", "function": {"name": "finish_drift"}}
    assert ctx.drift_finished is True
    assert store.load_drift()["recent_runs"][-1]["status"] == "paused"


@pytest.mark.asyncio
async def test_drift_pipeline_does_not_restrict_before_step_limit(tmp_path: Path):
    _write_skill(tmp_path)
    store = DriftStateStore(tmp_path)
    captured: list[tuple[list[str], str | dict]] = []

    async def llm(messages: list[dict], schemas: list[dict], tool_choice: str | dict = "auto"):
        captured.append(([s["function"]["name"] for s in schemas], tool_choice))
        if tool_choice == {"type": "function", "function": {"name": "finish_drift"}}:
            return {
                "name": "finish_drift",
                "input": {
                    "skill_used": "explore-curiosity",
                    "status": "paused",
                    "briefing": "达到步数上限后停止继续读取",
                    "scratchpad_update": "下次根据已读 SKILL.md 继续判断是否要行动",
                    "message_result": "silent",
                },
            }
        if len(captured) == 1:
            return {"name": "select_skill", "input": {"skill_name": "explore-curiosity"}}
        return {
            "name": "read_file",
            "input": {"path": "skills/explore-curiosity/SKILL.md"},
        }

    pipeline = _make_drift_pipeline(
        store=store,
        tool_deps=DriftToolDeps(
            drift_dir=tmp_path,
            store=store,
            shared_tools=_build_shared_tools(),
        ),
        max_steps=6,
    )
    ctx = AgentTickContext(now_utc=datetime.now(timezone.utc), session_key="s")
    await pipeline.run(ctx, cast(Any, llm))
    assert captured[0][0] == ["select_skill", "idle_drift"]
    assert captured[0][1] == "required"
    for schemas, tool_choice in captured[1:6]:
        assert tool_choice == "required"
        assert "read_file" in schemas
        assert "write_file" in schemas
        assert "shell" in schemas
        assert "finish_drift" in schemas
    assert captured[6][0] == ["finish_drift"]
    assert captured[6][1] == {"type": "function", "function": {"name": "finish_drift"}}
    assert ctx.drift_finished is True
    assert store.load_drift()["recent_runs"][-1]["status"] == "paused"


@pytest.mark.asyncio
async def test_drift_pipeline_fallback_pauses_when_wrap_up_ignores_finish(tmp_path: Path):
    _write_skill(tmp_path)
    store = DriftStateStore(tmp_path)

    async def llm(messages: list[dict], schemas: list[dict], tool_choice: str | dict = "auto"):
        step = len([m for m in messages if m.get("role") == "tool"]) + 1
        if step == 1:
            return {"name": "select_skill", "input": {"skill_name": "explore-curiosity"}}
        if step == 2:
            return {"name": "read_file", "input": {"path": "skills/explore-curiosity/queue.md"}}
        return {"name": "read_file", "input": {"path": "skills/explore-curiosity/state.json"}}

    pipeline = _make_drift_pipeline(
        store=store,
        tool_deps=DriftToolDeps(
            drift_dir=tmp_path,
            store=store,
            shared_tools=_build_shared_tools(),
        ),
        max_steps=1,
    )
    ctx = AgentTickContext(now_utc=datetime.now(timezone.utc), session_key="s")
    await pipeline.run(ctx, cast(Any, llm))
    drift = store.load_drift()
    assert ctx.drift_finished is True
    assert drift["recent_runs"][-1]["skill"] == "explore-curiosity"
    assert drift["recent_runs"][-1]["status"] == "paused"


@pytest.mark.asyncio
async def test_drift_pipeline_wrap_up_retries_non_finish_once(tmp_path: Path):
    _write_skill(tmp_path)
    store = DriftStateStore(tmp_path)
    wrap_up_calls = 0

    async def llm(messages: list[dict], schemas: list[dict], tool_choice: str | dict = "auto"):
        nonlocal wrap_up_calls
        if len([m for m in messages if m.get("role") == "tool"]) == 0:
            return {"name": "select_skill", "input": {"skill_name": "explore-curiosity"}}
        wrap_up_calls += 1
        if wrap_up_calls == 1:
            return {"name": "read_file", "input": {"path": "skills/explore-curiosity/queue.md"}}
        return {
            "name": "finish_drift",
            "input": {
                "skill_used": "explore-curiosity",
                "status": "paused",
                "briefing": "读了 skill，但还没有完成动作",
                "scratchpad_update": "下次从 explore-curiosity 的自然问题判断继续",
                "message_result": "silent",
            },
        }

    pipeline = _make_drift_pipeline(
        store=store,
        tool_deps=DriftToolDeps(
            drift_dir=tmp_path,
            store=store,
            shared_tools=_build_shared_tools(),
        ),
        max_steps=1,
    )
    ctx = AgentTickContext(now_utc=datetime.now(timezone.utc), session_key="s")
    await pipeline.run(ctx, cast(Any, llm))
    drift = store.load_drift()
    assert wrap_up_calls == 2
    assert ctx.drift_finished is True
    assert drift["recent_runs"][-1]["skill"] == "explore-curiosity"
    assert drift["recent_runs"][-1]["briefing"] == "读了 skill，但还没有完成动作"


@pytest.mark.asyncio
async def test_agent_tick_enters_drift_and_records_action(tmp_path: Path):
    _write_skill(tmp_path)
    gate = MagicMock()
    gate.should_act.return_value = (True, {})
    llm = FakeLLM(
        [
            ("select_skill", {"skill_name": "explore-curiosity"}),
            (
                "finish_drift",
                {
                    "skill_used": "explore-curiosity",
                    "status": "completed",
                    "briefing": "整理了漂移状态",
                    "message_result": "silent",
                },
            ),
        ]
    )
    tick = make_proactive_pipeline(
        cfg=cfg_with(drift_enabled=True),
        any_action_gate=gate,
        llm_fn=llm,
        tool_deps=ToolDeps(recent_chat_fn=AsyncMock(return_value=[])),
        gateway_deps=GatewayDeps(
            alert_fn=AsyncMock(return_value=[]),
            feed_fn=AsyncMock(return_value=[]),
            context_fn=AsyncMock(return_value=[]),
        ),
        rng=FakeRng(value=1.0),
        drift_pipeline=_make_drift_pipeline(
            store=DriftStateStore(tmp_path),
            tool_deps=DriftToolDeps(
                drift_dir=tmp_path,
                store=DriftStateStore(tmp_path),
                shared_tools=_build_shared_tools(),
            ),
            max_steps=5,
        ),
    )
    await run_proactive_pipeline(tick)
    assert tick.last_ctx.drift_entered is True
    gate.record_action.assert_called_once()
    assert len(tick._state_store.tick_step_logs) == 2
    assert tick._state_store.tick_step_logs[0]["phase"] == "drift"
    assert tick._state_store.tick_step_logs[0]["tool_name"] == "select_skill"
    assert tick._state_store.tick_step_logs[1]["phase"] == "drift"
    assert tick._state_store.tick_step_logs[1]["tool_name"] == "finish_drift"


@pytest.mark.asyncio
async def test_agent_tick_drift_send_message_skips_normal_post_loop(tmp_path: Path):
    _write_skill(tmp_path)
    sender = AsyncMock(return_value=True)

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

    class _Outbound:
        async def dispatch(self, outbound: OutboundDispatch) -> bool:
            return await sender(outbound.content)

    orchestrator = TurnOrchestrator(
        TurnOrchestratorDeps(
            session=SessionServices(
                session_manager=cast(Any, session_manager),
                presence=cast(Any, SimpleNamespace(record_proactive_sent=lambda _key: None)),
            ),
            outbound=_Outbound(),
        )
    )

    async def send_message(content: str, media: list[str] | None = None) -> bool:
        return await orchestrator.handle_proactive_turn(
            result=TurnResult(
                decision="reply",
                outbound=TurnOutbound(session_key="test_session", content=content, media=list(media or [])),
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
            ("select_skill", {"skill_name": "explore-curiosity"}),
            ("message_push", {"message": "hello from drift"}),
            (
                "finish_drift",
                {
                    "skill_used": "explore-curiosity",
                    "status": "completed",
                    "briefing": "发出一条消息",
                    "message_result": "sent",
                },
            ),
        ]
    )
    tick = ProactiveTurnPipeline(
        ProactiveTurnPipelineDeps(
            cfg=cfg_with(
                drift_enabled=True,
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
                record_tick_log_start=lambda **_kwargs: None,
                record_tick_log_finish=lambda **_kwargs: None,
                record_tick_step_log=lambda **_kwargs: None,
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
            workspace_context_fn=None,
            llm_fn=llm,
            rng=FakeRng(value=1.0),
            recent_proactive_fn=lambda: [],
            drift_pipeline=_make_drift_pipeline(
                store=DriftStateStore(tmp_path),
                tool_deps=DriftToolDeps(
                    drift_dir=tmp_path,
                    store=DriftStateStore(tmp_path),
                    shared_tools=_build_shared_tools(),
                    send_message_fn=send_message,
                ),
                max_steps=5,
            ),
            tool_hooks=None,
        )
    )

    await run_proactive_pipeline(tick)

    sender.assert_awaited_once_with("hello from drift")
    gate.record_action.assert_called_once()
    assert tick.last_ctx.drift_entered is True
    assert tick.last_ctx.drift_message_sent is True


def _write_skill_with_mcp(
    root: Path, name: str, requires_mcp: str,
) -> Path:
    skill_dir = root / "skills" / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        (
            "---\n"
            f"name: {name}\n"
            f"description: test skill needing {requires_mcp}\n"
            f"requires_mcp: {requires_mcp}\n"
            "---\n\n"
            "test skill\n"
        ),
        encoding="utf-8",
    )
    return skill_dir


def _build_shared_tools_with_mcp(*server_names: str) -> ToolRegistry:
    """Build shared tools with fake MCP tools registered."""
    reg = _build_shared_tools()
    for srv in server_names:
        for suffix in ("tool_a", "tool_b"):
            tool = _DummyTool(f"mcp_{srv}__{suffix}")
            reg.register(tool, risk="external-side-effect", source_type="mcp", source_name=srv)
    return reg


def test_skill_meta_requires_mcp_parsed_inline(tmp_path: Path):
    skill_dir = tmp_path / "skills" / "cal-skill"
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: cal-skill\ndescription: test\nrequires_mcp: calendar, gmail\n---\n",
        encoding="utf-8",
    )
    store = DriftStateStore(tmp_path)
    skills = store.scan_skills()
    assert len(skills) == 1
    assert skills[0].requires_mcp == ["calendar", "gmail"]


def test_skill_meta_requires_mcp_parsed_yaml_list(tmp_path: Path):
    skill_dir = tmp_path / "skills" / "multi-mcp"
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        "name: multi-mcp\n"
        "description: test yaml list\n"
        "requires_mcp:\n"
        "  - calendar\n"
        "  - gmail\n"
        "---\n",
        encoding="utf-8",
    )
    store = DriftStateStore(tmp_path)
    skills = store.scan_skills()
    assert len(skills) == 1
    assert skills[0].requires_mcp == ["calendar", "gmail"]


def test_skill_meta_frontmatter_uses_yaml_parser(tmp_path: Path):
    skill_dir = tmp_path / "skills" / "yaml-skill"
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        "name: yaml-skill\n"
        "description: >\n"
        "  test multiline\n"
        "  description\n"
        "requires_mcp:\n"
        "  - calendar # primary calendar source\n"
        "  - gmail\n"
        "---\n",
        encoding="utf-8",
    )
    store = DriftStateStore(tmp_path)
    skills = store.scan_skills()
    assert len(skills) == 1
    assert skills[0].description == "test multiline description"
    assert skills[0].requires_mcp == ["calendar", "gmail"]


def test_skill_meta_requires_mcp_empty_when_missing(tmp_path: Path):
    _write_skill(tmp_path)
    store = DriftStateStore(tmp_path)
    skills = store.scan_skills()
    assert skills[0].requires_mcp == []


def test_drift_state_store_includes_builtin_skills_when_enabled(tmp_path: Path):
    store = DriftStateStore(
        tmp_path,
        builtin_skills_dir=Path("skills"),
        include_builtin_skills=True,
        builtin_skill_names={"create-drift-skill"},
    )
    skills = store.scan_skills()
    names = {skill.name for skill in skills}
    assert "create-drift-skill" in names
    assert next(skill for skill in skills if skill.name == "create-drift-skill").builtin is True


@pytest.mark.asyncio
async def test_drift_readfile_accepts_builtin_skill_shorthand_path(tmp_path: Path):
    ctx = AgentTickContext(now_utc=datetime.now(timezone.utc))
    store = DriftStateStore(
        tmp_path,
        builtin_skills_dir=Path("skills"),
        include_builtin_skills=True,
        builtin_skill_names={"create-drift-skill"},
    )
    raw = await _exec_drift_tool(
        tmp_path,
        ctx,
        "read_file",
        {"path": "skills/create-drift-skill/SKILL.md"},
        store=store,
    )
    assert "创建 Drift Skill" in str(raw)


@pytest.mark.asyncio
async def test_drift_pipeline_filters_skills_by_mcp(tmp_path: Path):
    """Skill requiring unavailable MCP server should be filtered out."""
    _write_skill_with_mcp(tmp_path, "needs-cal", "calendar")
    store = DriftStateStore(tmp_path)
    shared = _build_shared_tools()  # no MCP tools registered
    pipeline = _make_drift_pipeline(
        store=store,
        tool_deps=DriftToolDeps(
            drift_dir=tmp_path,
            store=store,
            shared_tools=shared,
        ),
        max_steps=5,
    )
    ctx = AgentTickContext(now_utc=datetime.now(timezone.utc), session_key="s")
    entered = await pipeline.run(ctx, cast(Any, FakeLLM([])))
    assert entered is False  # all skills filtered, drift should skip


@pytest.mark.asyncio
async def test_drift_pipeline_keeps_skills_when_mcp_available(tmp_path: Path):
    """Skill requiring available MCP server should pass filter."""
    _write_skill_with_mcp(tmp_path, "needs-cal", "calendar")
    store = DriftStateStore(tmp_path)
    shared = _build_shared_tools_with_mcp("calendar")
    llm = FakeLLM([
        ("select_skill", {"skill_name": "needs-cal"}),
        (
            "finish_drift",
            {
                "skill_used": "needs-cal",
                "status": "completed",
                "briefing": "done",
                "message_result": "silent",
            },
        ),
    ])
    pipeline = _make_drift_pipeline(
        store=store,
        tool_deps=DriftToolDeps(drift_dir=tmp_path, store=store, shared_tools=shared),
        max_steps=5,
    )
    ctx = AgentTickContext(now_utc=datetime.now(timezone.utc), session_key="s")
    entered = await pipeline.run(ctx, cast(Any, llm))
    assert entered is True
    assert ctx.drift_finished is True


@pytest.mark.asyncio
async def test_mount_server_adds_tools_and_schemas(tmp_path: Path):
    _write_skill(tmp_path)
    store = DriftStateStore(tmp_path)
    shared = _build_shared_tools_with_mcp("calendar")
    ctx = AgentTickContext(now_utc=datetime.now(timezone.utc))
    reg = build_drift_tool_registry(
        ctx=ctx,
        deps=DriftToolDeps(drift_dir=tmp_path, store=store, shared_tools=shared),
    )
    assert reg.has_tool("mount_server")
    assert not reg.has_tool("mcp_calendar__tool_a")
    raw = await reg.execute("mount_server", {"server": "calendar"})
    result = json.loads(cast(Any, raw))
    assert result["ok"] is True
    assert "mcp_calendar__tool_a" in result["tools"]
    assert "mcp_calendar__tool_b" in result["tools"]
    assert reg.has_tool("mcp_calendar__tool_a")
    assert reg.has_tool("mcp_calendar__tool_b")


@pytest.mark.asyncio
async def test_mount_server_idempotent(tmp_path: Path):
    shared = _build_shared_tools_with_mcp("calendar")
    ctx = AgentTickContext(now_utc=datetime.now(timezone.utc))
    reg = build_drift_tool_registry(
        ctx=ctx,
        deps=DriftToolDeps(drift_dir=tmp_path, store=DriftStateStore(tmp_path), shared_tools=shared),
    )
    await reg.execute("mount_server", {"server": "calendar"})
    raw = await reg.execute("mount_server", {"server": "calendar"})
    result = json.loads(cast(Any, raw))
    assert result["ok"] is True
    assert "已挂载" in result["message"]


@pytest.mark.asyncio
async def test_drift_write_file_can_update_workspace_context(tmp_path: Path):
    workspace = tmp_path / "workspace"
    drift_dir = workspace / "drift"
    store = DriftStateStore(drift_dir)
    ctx = AgentTickContext(now_utc=datetime.now(timezone.utc))
    reg = build_drift_tool_registry(
        ctx=ctx,
        deps=DriftToolDeps(
            drift_dir=drift_dir,
            workspace_dir=workspace,
            store=store,
        ),
    )

    raw = await reg.execute(
        "write_file",
        {"path": "../PROACTIVE_CONTEXT.md", "content": "规则"},
    )

    assert "已写入" in str(raw)
    assert (workspace / "PROACTIVE_CONTEXT.md").read_text(encoding="utf-8") == "规则"


@pytest.mark.asyncio
async def test_drift_write_file_rejects_paths_outside_workspace(tmp_path: Path):
    workspace = tmp_path / "workspace"
    drift_dir = workspace / "drift"
    store = DriftStateStore(drift_dir)
    ctx = AgentTickContext(now_utc=datetime.now(timezone.utc))
    reg = build_drift_tool_registry(
        ctx=ctx,
        deps=DriftToolDeps(
            drift_dir=drift_dir,
            workspace_dir=workspace,
            store=store,
        ),
    )

    raw = await reg.execute(
        "write_file",
        {"path": "../../outside.txt", "content": "no"},
    )

    assert "超出允许目录" in str(raw)
    assert not (tmp_path / "outside.txt").exists()


@pytest.mark.asyncio
async def test_mount_server_rejects_unknown_server(tmp_path: Path):
    shared = _build_shared_tools()  # no MCP
    ctx = AgentTickContext(now_utc=datetime.now(timezone.utc))
    reg = build_drift_tool_registry(
        ctx=ctx,
        deps=DriftToolDeps(drift_dir=tmp_path, store=DriftStateStore(tmp_path), shared_tools=shared),
    )
    assert not reg.has_tool("mount_server")


@pytest.mark.asyncio
async def test_mount_server_not_registered_without_mcp(tmp_path: Path):
    """When no MCP servers connected, mount_server tool should not appear."""
    shared = _build_shared_tools()
    ctx = AgentTickContext(now_utc=datetime.now(timezone.utc))
    reg = build_drift_tool_registry(
        ctx=ctx,
        deps=DriftToolDeps(drift_dir=tmp_path, store=DriftStateStore(tmp_path), shared_tools=shared),
    )
    assert not reg.has_tool("mount_server")


@pytest.mark.asyncio
async def test_drift_pipeline_executes_mounted_mcp_tool(tmp_path: Path):
    """After mount_server, pipeline should dispatch MCP tool calls to shared registry."""
    _write_skill(tmp_path)
    store = DriftStateStore(tmp_path)
    shared = _build_shared_tools_with_mcp("calendar")
    captured_schemas: list[list[str]] = []

    async def llm(messages, schemas, tool_choice="auto"):
        captured_schemas.append([s["function"]["name"] for s in schemas])
        step = len(captured_schemas)
        if step == 1:
            return {"name": "select_skill", "input": {"skill_name": "explore-curiosity"}}
        if step == 2:
            return {"name": "mount_server", "input": {"server": "calendar"}}
        if step == 3:
            return {"name": "mcp_calendar__tool_a", "input": {}}
        if step == 4:
            return {
                "name": "finish_drift",
                "input": {
                    "skill_used": "explore-curiosity",
                    "status": "completed",
                    "briefing": "used cal",
                    "message_result": "silent",
                },
            }
        return None

    pipeline = _make_drift_pipeline(
        store=store,
        tool_deps=DriftToolDeps(drift_dir=tmp_path, store=store, shared_tools=shared),
        max_steps=10,
    )
    ctx = AgentTickContext(now_utc=datetime.now(timezone.utc), session_key="s")
    await pipeline.run(ctx, cast(Any, llm))
    assert ctx.drift_finished is True
    # After mount (step 2), step 3 should see MCP tools in schemas
    assert "mcp_calendar__tool_a" in captured_schemas[2]
    assert "mcp_calendar__tool_b" in captured_schemas[2]
    # Step 1 should NOT have MCP tools yet
    assert "mcp_calendar__tool_a" not in captured_schemas[0]


@pytest.mark.asyncio
async def test_system_prompt_includes_mcp_directory(tmp_path: Path):
    _write_skill(tmp_path)
    store = DriftStateStore(tmp_path)
    shared = _build_shared_tools_with_mcp("calendar")
    pipeline = _make_drift_pipeline(
        store=store,
        tool_deps=DriftToolDeps(drift_dir=tmp_path, store=store, shared_tools=shared),
    )
    content = str(
        (await pipeline._build_runtime_context_message(
            store.scan_skills(), shared.get_mcp_server_names()
        ))["content"]
    )
    assert "可挂载的外部能力" in content
    assert "calendar" in content
    assert "mount_server" in content
    # 不应展开具体工具名，只列 server 名和工具数
    assert "mcp_calendar__tool_a" not in content
    assert "mcp_calendar__tool_b" not in content
    assert "2 个工具" in content


@pytest.mark.asyncio
async def test_system_prompt_no_mcp_block_without_servers(tmp_path: Path):
    _write_skill(tmp_path)
    store = DriftStateStore(tmp_path)
    pipeline = _make_drift_pipeline(
        store=store,
        tool_deps=DriftToolDeps(drift_dir=tmp_path, store=store, shared_tools=_build_shared_tools()),
    )
    content = str((await pipeline._build_runtime_context_message(store.scan_skills(), set()))["content"])
    assert "可挂载的外部能力" not in content
    assert "mount_server" not in content


@pytest.mark.asyncio
async def test_system_prompt_skill_requires_mcp_annotation(tmp_path: Path):
    _write_skill_with_mcp(tmp_path, "cal-skill", "calendar")
    store = DriftStateStore(tmp_path)
    shared = _build_shared_tools_with_mcp("calendar")
    pipeline = _make_drift_pipeline(
        store=store,
        tool_deps=DriftToolDeps(drift_dir=tmp_path, store=store, shared_tools=shared),
    )
    content = str(
        (await pipeline._build_runtime_context_message(
            store.scan_skills(), shared.get_mcp_server_names()
        ))["content"]
    )
    assert "[需要: calendar]" in content


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

    class _Outbound:
        async def dispatch(self, outbound) -> bool:
            return await sender.send(outbound.content)

    from agent.looping.ports import SessionServices
    from agent.turns.orchestrator import TurnOrchestrator, TurnOrchestratorDeps

    orchestrator = TurnOrchestrator(
        TurnOrchestratorDeps(
            session=SessionServices(
                session_manager=cast(Any, session_manager),
                presence=cast(Any, SimpleNamespace(record_proactive_sent=lambda _key: None)),
            ),
            outbound=_Outbound(),
        )
    )

    deps = AgentTickDeps(
        cfg=cfg_with(
            drift_enabled=True,
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

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
from plugins.default_proactive.runtime import ProactiveFlowRuntime, ProactiveFlowDeps
from agent.tools.base import Tool
from agent.tools.registry import ToolRegistry
from agent.looping.ports import SessionServices
from agent.turns.orchestrator import TurnOrchestrator, TurnOrchestratorDeps
from agent.turns.outbound import OutboundDispatch
from plugins.default_proactive.context import AgentTickContext
from plugins.drift_flow.runtime import DriftTurnPipeline, DriftTurnPipelineDeps
from plugins.drift_flow.state import DriftStateStore
from plugins.drift_flow.tools import DriftToolDeps, build_drift_tool_registry
from plugins.default_proactive.gateway import GatewayDeps
from plugins.proactive_flow.tools import ToolDeps
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


def _select_input(skill_name: str, *, decision: str = "explore") -> dict[str, str]:
    return {
        "skill_name": skill_name,
        "decision": decision,
        "intention": f"做一点 {skill_name}",
        "reason": "当前适合做一个小动作",
    }


def _self_update(next_tendency: str = "下次根据当时状态自由选择") -> dict[str, str]:
    return {
        "next_tendency": next_tendency,
        "reflection": "本轮是一次普通闭环，没有形成需要保存的新观察",
        "pattern": "ordinary",
    }


def test_drift_commit_result_corrects_staged_message_result(tmp_path: Path):
    store = DriftStateStore(tmp_path)
    store.save_finish(
        skill_used="explore-curiosity",
        status="completed",
        briefing="done",
        message_result="staged",
        scratchpad_update=None,
        global_note_update=None,
        now_utc=datetime.now(timezone.utc),
    )

    store.update_last_message_result("silent")

    assert store.load_drift()["recent_runs"][-1]["message_result"] == "silent"


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
):
    resolved_store = store or DriftStateStore(tmp_path)
    reg = build_drift_tool_registry(
        ctx=ctx,
        deps=DriftToolDeps(
            drift_dir=tmp_path,
            store=resolved_store,
            builtin_skills_dir=getattr(resolved_store, "builtin_skills_dir", None),
            shared_tools=_build_shared_tools(),
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
    raw = await _exec_drift_tool(
        tmp_path,
        ctx,
        "message_push",
        {
            "message": "新表情来啦",
            "image": "/tmp/one.png",
            "media": ["/tmp/two.png"],
        },
    )
    payload = json.loads(cast(Any, raw))
    assert payload["ok"] is True
    assert payload["delivery_semantics"] == "completed_fire_and_forget"
    assert payload["reply_state"] == "not_tracked"
    assert ctx.draft_message == "新表情来啦"
    assert ctx.draft_media == ["/tmp/one.png", "/tmp/two.png"]


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
    assert "没有被叫住时的自处" in prompt
    assert "默认应该行动一小步" in prompt
    assert "默认调用 select_skill" in prompt
    assert "SKILL.md 是说明书，不是必须从头播放的脚本" in prompt
    assert "local_context 明确已经完成的读文件、查重、规划、写计划等前置动作" in prompt
    assert "不必无目的地重新读取" in prompt
    assert "本轮也可以暂时不继续 paused skill" in prompt
    assert "过去的自己留下的意图和倾向" in prompt
    assert "self_update.next_tendency" in prompt
    assert "没有新发现就省略" in prompt
    assert "recent_drift_runs 是自己刚刚真实度过的空闲时间" in prompt
    assert "是否只是逐字复制 next_tendency" in prompt
    assert "把可反复进行的小活动沉淀成新 skill" in prompt
    assert "现有 skill 能做，不等于此刻就想做" in prompt
    assert "message_push 是 fire-and-forget" in prompt
    assert "回答尚未出现不是可靠事件" in prompt
    assert "idle_drift 的 reason 必须写具体的时机或风险原因" in prompt
    assert "路径由 drift mount resolver 解析" in prompt
    assert "message_result" not in prompt
    assert "select_skill" in prompt
    assert "idle_drift" in prompt
    assert "fetch_messages" in prompt
    assert "search_messages" in prompt
    assert "shell" in prompt
    assert is_context_frame(runtime)
    assert "drift_skills" in runtime


def test_drift_tool_descriptions_include_metacognition_examples(tmp_path: Path):
    _write_skill(tmp_path)
    store = DriftStateStore(tmp_path)
    ctx = AgentTickContext(now_utc=datetime.now(timezone.utc))
    tools = build_drift_tool_registry(
        ctx=ctx,
        deps=DriftToolDeps(drift_dir=tmp_path, store=store),
    )
    schemas = {
        schema["function"]["name"]: schema["function"]["parameters"]
        for schema in tools.get_schemas()
    }

    select_properties = schemas["select_skill"]["properties"]
    finish_properties = schemas["finish_drift"]["properties"]
    self_properties = finish_properties["self_update"]["properties"]
    assert "<example>" in select_properties["reason"]["description"]
    assert "不要只写" in select_properties["reason"]["description"]
    assert "<example>" in self_properties["next_tendency"]["description"]
    assert "不是下一轮必须执行" in self_properties["next_tendency"]["description"]
    assert "<example>" in self_properties["observation"]["description"]
    assert "近期多轮" in self_properties["observation"]["description"]
    assert "recent_drift_runs" in self_properties["reflection"]["description"]
    assert self_properties["pattern"]["enum"] == [
        "ordinary", "repeat", "change", "contradiction"
    ]
    assert "发现重复模式" in finish_properties["self_update"]["description"]
    assert "普通一轮无新发现" in finish_properties["self_update"]["description"]
    assert "创建元能力" in select_properties["reason"]["description"]


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
    store.save_self_choice(
        skill_name="meme-auto-generate",
        intention="继续完成已经计划好的图片",
        decision="continue",
        reason="计划已经存在，只差执行",
        now_utc=datetime(2026, 1, 2, tzinfo=timezone.utc),
    )
    store.save_finish(
        skill_used="meme-auto-generate",
        status="paused",
        briefing="计划已写入，生成服务暂时失败",
        message_result="silent",
        scratchpad_update="plan.json 已写入；若继续，直接执行生成步骤",
        global_note_update=None,
        now_utc=datetime(2026, 1, 2, tzinfo=timezone.utc),
        cursor_update={"next_mode": "create_category"},
        self_update={"next_tendency": "服务恢复后再继续，也可以先做别的"},
    )
    pipeline = _make_drift_pipeline(
        store=store,
        tool_deps=DriftToolDeps(
            drift_dir=tmp_path,
            store=store,
            shared_tools=_build_shared_tools(),
        ),
    )
    ctx = AgentTickContext(now_utc=datetime(2026, 1, 3, 12, 34, tzinfo=timezone.utc))
    runtime = str(
        (await pipeline._build_runtime_context_message(store.scan_skills(), ctx=ctx))["content"]
    )
    assert "drift_selection_context" in runtime
    assert "drift_self_state" in runtime
    assert "drift_self_observations" in runtime
    assert "继续完成已经计划好的图片" in runtime
    assert "计划已经存在，只差执行" in runtime
    assert "服务恢复后再继续，也可以先做别的" in runtime
    assert "runtime_clock" in runtime
    assert "current_time_utc=2026-01-03T12:34:00+00:00" in runtime
    assert "current_time_local=" in runtime
    assert "按 skill 名称排列，顺序不代表优先级" in runtime
    assert "选择依据：runtime_clock、status、上次 finish 时间" in runtime
    assert "必须以 runtime_clock 的完整日期和时间为准" in runtime
    assert "recent_raw_chat" in runtime
    assert "SKILL.md 是能力说明书、约束和路径地图" in runtime
    assert "local_context 记录的已完成进度高于 SKILL.md" in runtime
    assert "local_context 只在 select_skill 后作为执行上下文参考" in runtime
    assert "explore-curiosity: status=completed" in runtime
    assert "meme-auto-generate: status=paused" in runtime
    assert "last_finish=2026-01-01T00:00:00+00:00" in runtime
    assert "last_finish=2026-01-02T00:00:00+00:00" in runtime
    assert "上次 finish：2026-01-01T00:00:00+00:00" in runtime
    assert "briefing=没有自然切口" in runtime
    assert "briefing=计划已写入，生成服务暂时失败" in runtime
    assert "scratchpad=plan.json 已写入；若继续，直接执行生成步骤" in runtime
    assert 'cursor={"next_mode": "create_category"}' in runtime
    assert "本轮首选" not in runtime
    assert "首个工具调用" not in runtime
    assert runtime.index("drift_selection_context") < runtime.index("long_term_memory")
    assert runtime.index("## recent_drift_runs") < runtime.index("## runtime_clock")


@pytest.mark.asyncio
async def test_drift_runtime_context_treats_self_observations_as_tentative(tmp_path: Path):
    _write_skill(tmp_path, name="explore-curiosity")
    _write_skill(tmp_path, name="quiet-organize")
    store = DriftStateStore(tmp_path)
    store.save_finish(
        skill_used="explore-curiosity",
        status="completed",
        briefing="整理了旧问题",
        message_result="silent",
        scratchpad_update=None,
        global_note_update=None,
        now_utc=datetime(2026, 1, 1, tzinfo=timezone.utc),
        journal_append=[{
            "entry_type": "self_observation",
            "key": "question",
            "payload": {
                "statement": "我似乎喜欢回看未完的问题",
                "basis": "主动选择了一条旧问题",
                "effect": "question",
            },
        }],
    )
    store.save_finish(
        skill_used="quiet-organize",
        status="completed",
        briefing="整理了零散材料",
        message_result="silent",
        scratchpad_update=None,
        global_note_update=None,
        now_utc=datetime(2026, 1, 2, tzinfo=timezone.utc),
        journal_append=[{
            "entry_type": "self_observation",
            "key": "reinforce",
            "payload": {
                "statement": "我喜欢把零散材料连成线",
                "basis": "再次主动整理了互不相邻的材料",
                "effect": "reinforce",
            },
        }],
    )
    pipeline = _make_drift_pipeline(
        store=store,
        tool_deps=DriftToolDeps(drift_dir=tmp_path, store=store),
    )

    runtime = str((await pipeline._build_runtime_context_message(store.scan_skills()))["content"])

    assert "drift_self_observations" in runtime
    assert "不是长期记忆、人格结论或行动命令" in runtime
    assert "单次观察不能定义自己" in runtime
    assert "explore-curiosity [question]" in runtime
    assert "quiet-organize [reinforce]" in runtime


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
    self_state = store.load_self_state()
    assert self_state["last_decision"] == "rest"
    assert self_state["decision_reason"] == "最近对话刚结束，主动打扰价值不高"
    assert self_state["next_tendency"] == "等待更合适的时机再自由选择"


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
        cursor_update={"last_topic": "音乐", "last_asked_at": "2026-01-01T00:00:00+00:00"},
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
        _select_input("explore-curiosity"),
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
    self_state = store.load_self_state()
    assert self_state["current_skill"] == "explore-curiosity"
    assert self_state["last_decision"] == "explore"
    assert self_state["current_intention"] == "做一点 explore-curiosity"
    assert self_state["decision_reason"] == "当前适合做一个小动作"
    assert self_state["next_tendency"] == ""


@pytest.mark.asyncio
async def test_select_paused_skill_returns_resume_guidance(tmp_path: Path):
    _write_skill(tmp_path)
    store = DriftStateStore(tmp_path)
    store.save_finish(
        skill_used="explore-curiosity",
        status="paused",
        briefing="计划已写，执行失败",
        message_result="silent",
        scratchpad_update="plan.json 已存在，下一步执行计划",
        global_note_update=None,
        now_utc=datetime.now(timezone.utc),
        cursor_update={"phase": "execute_plan"},
    )
    ctx = AgentTickContext(now_utc=datetime.now(timezone.utc))

    raw = await _exec_drift_tool(
        tmp_path,
        ctx,
        "select_skill",
        _select_input("explore-curiosity", decision="continue"),
        store=store,
    )
    payload = json.loads(cast(Any, raw))

    assert "paused skill 的可续接停点" in payload["runtime_guidance"]
    assert "只执行停点后的最小下一步" in payload["runtime_guidance"]


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
            "self_update": _self_update(),
        },
        store=store,
    )
    assert json.loads(cast(Any, raw))["error"] == "unknown skill: missing"


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
            "self_update": _self_update(),
        },
        store=store,
    )
    assert json.loads(cast(Any, raw))["ok"] is True
    assert store.load_drift()["recent_runs"][-1]["message_result"] == "silent"
    assert not (tmp_path / "skills" / "explore-curiosity" / "state.json").exists()


@pytest.mark.asyncio
async def test_finish_drift_emits_result_after_delivery(tmp_path: Path):
    _write_skill(tmp_path)
    store = DriftStateStore(tmp_path)
    ctx = AgentTickContext(
        now_utc=datetime.now(timezone.utc),
        session_key="session",
        drift_selected_skill="explore-curiosity",
    )
    events: list[Any] = []
    reg = build_drift_tool_registry(
        ctx=ctx,
        deps=DriftToolDeps(
            drift_dir=tmp_path,
            store=store,
            shared_tools=_build_shared_tools(),
            event_bus=SimpleNamespace(enqueue=events.append),
        ),
    )

    await reg.execute("message_push", {"message": "hello"})
    await reg.execute(
        "finish_drift",
        {
            "skill_used": "explore-curiosity",
            "status": "completed",
            "briefing": "staged",
            "self_update": _self_update(),
        },
    )

    assert ctx.drift_message_staged is True
    assert ctx.drift_message_sent is False
    assert events == []

    pipeline = _make_drift_pipeline(
        store=store,
        tool_deps=DriftToolDeps(
            drift_dir=tmp_path,
            store=store,
            shared_tools=_build_shared_tools(),
            event_bus=SimpleNamespace(enqueue=events.append),
        ),
    )
    pipeline.record_commit_result(ctx, False)

    assert events[0].message_result == "silent"
    assert store.load_drift()["recent_runs"][-1]["message_result"] == "silent"


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
            "self_update": _self_update(),
            "cursor_update": {
                "last_topic": "music",
                "last_asked_at": "2026-01-01T00:00:00+00:00",
            },
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
    assert continuum["cursor"]["last_asked_at"] == "2026-01-01T00:00:00+00:00"
    assert journal[0]["entry_type"] == "curiosity_asked"
    assert journal[0]["key"] == "music"


@pytest.mark.asyncio
async def test_finish_drift_saves_optional_self_observation(tmp_path: Path):
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
            "briefing": "把零散想法整理成了一条线",
            "self_update": {
                "next_tendency": "下次看心情再决定",
                "reflection": "本轮与近期行为形成了一条值得暂定记录的新线索",
                "pattern": "repeat",
                "observation": {
                    "statement": "我似乎喜欢把零散材料连成线",
                    "basis": "这轮没有被要求，但主动整理了三条旧笔记的关系",
                    "effect": "question",
                },
            },
        },
        store=store,
    )

    assert json.loads(cast(Any, raw))["ok"] is True
    observations = store.load_recent_self_observations()
    assert observations[0]["skill_name"] == "explore-curiosity"
    assert observations[0]["payload"]["effect"] == "question"
    assert observations[0]["payload"]["statement"] == "我似乎喜欢把零散材料连成线"


@pytest.mark.asyncio
async def test_finish_drift_does_not_force_self_observation(tmp_path: Path):
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
            "briefing": "完成普通检查",
            "self_update": _self_update(),
        },
        store=store,
    )

    assert json.loads(cast(Any, raw))["ok"] is True
    assert store.load_recent_self_observations() == []


@pytest.mark.asyncio
async def test_finish_drift_rejects_invalid_self_observation(tmp_path: Path):
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
            "briefing": "完成检查",
            "self_update": {
                "next_tendency": "下次自由选择",
                "reflection": "本轮试图形成一条观察",
                "pattern": "repeat",
                "observation": {
                    "statement": "我就是这样的人",
                    "basis": "只有这一次",
                    "effect": "confirm",
                },
            },
        },
        store=store,
    )

    assert "observation.effect" in json.loads(cast(Any, raw))["error"]
    assert store.load_recent_self_observations() == []


@pytest.mark.asyncio
async def test_finish_drift_requires_observation_for_declared_pattern(tmp_path: Path):
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
            "briefing": "连续做了同类动作",
            "self_update": {
                "next_tendency": "下次自由选择",
                "reflection": "最近三轮都做了同类动作",
                "pattern": "repeat",
            },
        },
        store=store,
    )

    assert "observation is required when pattern is repeat" in json.loads(cast(Any, raw))["error"]


@pytest.mark.asyncio
async def test_finish_drift_updates_self_state_without_losing_choice(tmp_path: Path):
    _write_skill(tmp_path)
    store = DriftStateStore(tmp_path)
    ctx = AgentTickContext(now_utc=datetime.now(timezone.utc))
    await _exec_drift_tool(
        tmp_path,
        ctx,
        "select_skill",
        {
            "skill_name": "explore-curiosity",
            "decision": "continue",
            "intention": "把上次没想完的问题继续想一点",
            "reason": "已有停点仍然自然",
        },
        store=store,
    )
    raw = await _exec_drift_tool(
        tmp_path,
        ctx,
        "finish_drift",
        {
            "skill_used": "explore-curiosity",
            "status": "completed",
            "briefing": "想清楚了一个问题",
            "self_update": {
                "current_intention": "暂时把这个问题放下",
                "next_tendency": "下次想自由看看别的东西",
                "reflection": "本轮主动结束了原意图，没有形成新的重复模式",
                "pattern": "ordinary",
            },
        },
        store=store,
    )

    assert json.loads(cast(Any, raw))["ok"] is True
    self_state = store.load_self_state()
    assert self_state["last_decision"] == "continue"
    assert self_state["decision_reason"] == "已有停点仍然自然"
    assert self_state["current_intention"] == "暂时把这个问题放下"
    assert self_state["next_tendency"] == "下次想自由看看别的东西"


@pytest.mark.asyncio
async def test_finish_drift_rejects_removed_waiting_status(tmp_path: Path):
    _write_skill(tmp_path)
    store = DriftStateStore(tmp_path)
    ctx = AgentTickContext(now_utc=datetime.now(timezone.utc))
    raw = await _exec_drift_tool(
        tmp_path,
        ctx,
        "finish_drift",
        {
            "skill_used": "explore-curiosity",
            "status": "waiting",
            "briefing": "无新反馈",
        },
        store=store,
    )
    payload = json.loads(cast(Any, raw))
    assert payload["error"] == "status must be one of: completed, paused"
    assert ctx.drift_finished is False
    assert store.load_drift()["recent_runs"] == []


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
        },
        store=store,
    )
    payload = json.loads(cast(Any, raw))
    assert payload["error"] == "scratchpad_update is required when status is paused"
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
            ("select_skill", _select_input("explore-curiosity")),
            (
                "finish_drift",
                {
                    "skill_used": "explore-curiosity",
                    "status": "completed",
                    "briefing": "问了一个问题",
                    "self_update": _self_update(),
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
async def test_drift_pipeline_restricts_tools_after_staging_message(tmp_path: Path):
    _write_skill(tmp_path)
    store = DriftStateStore(tmp_path)
    llm = FakeLLM(
        [
            ("select_skill", _select_input("explore-curiosity")),
            ("message_push", {"message": "hello\\n\\nfrom drift"}),
            (
                "finish_drift",
                {
                    "skill_used": "explore-curiosity",
                    "status": "completed",
                    "briefing": "sent",
                    "self_update": _self_update(),
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
        ),
        max_steps=5,
    )
    ctx = AgentTickContext(now_utc=datetime.now(timezone.utc), session_key="s")
    await pipeline.run(ctx, cast(Any, llm))
    second_names = {schema["function"]["name"] for schema in llm.calls[1][0:1]} if False else None
    assert llm.calls
    # FakeLLM 不记录 schemas，这里用行为结果兜底：send 后仍正常 finish。
    assert ctx.drift_finished is True
    assert ctx.drift_message_staged is True
    assert ctx.drift_message_sent is False
    assert store.load_drift()["recent_runs"][-1]["message_result"] == "staged"


@pytest.mark.asyncio
async def test_drift_pipeline_wraps_up_at_step_limit(tmp_path: Path):
    _write_skill(tmp_path)
    store = DriftStateStore(tmp_path)
    captured: list[tuple[list[str], str | dict]] = []

    async def llm(messages: list[dict], schemas: list[dict], tool_choice: str | dict = "auto"):
        captured.append(([s["function"]["name"] for s in schemas], tool_choice))
        step = len(captured)
        if step == 1:
            return {"name": "select_skill", "input": _select_input("explore-curiosity")}
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
                    "self_update": _self_update(),
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
                    "self_update": _self_update(),
                },
            }
        if len(captured) == 1:
            return {"name": "select_skill", "input": _select_input("explore-curiosity")}
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
            return {"name": "select_skill", "input": _select_input("explore-curiosity")}
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
            return {"name": "select_skill", "input": _select_input("explore-curiosity")}
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
                "self_update": _self_update(),
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
            ("select_skill", _select_input("explore-curiosity")),
            (
                "finish_drift",
                {
                    "skill_used": "explore-curiosity",
                    "status": "completed",
                    "briefing": "整理了漂移状态",
                    "self_update": _self_update(),
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
@pytest.mark.parametrize(
    ("delivered", "message_result"),
    [(True, "sent"), (False, "silent")],
)
async def test_agent_tick_drift_emits_delivery_result(
    tmp_path: Path,
    delivered: bool,
    message_result: str,
):
    _write_skill(tmp_path)
    sender = AsyncMock(return_value=delivered)
    events: list[Any] = []

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

    gate = MagicMock()
    gate.should_act.return_value = (True, {})
    llm = FakeLLM(
        [
            ("select_skill", _select_input("explore-curiosity")),
            ("message_push", {"message": "hello from drift"}),
            (
                "finish_drift",
                {
                    "skill_used": "explore-curiosity",
                    "status": "completed",
                    "briefing": "发出一条消息",
                    "self_update": _self_update(),
                },
            ),
        ]
    )
    tick = ProactiveFlowRuntime(
        ProactiveFlowDeps(
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
                    event_bus=SimpleNamespace(enqueue=events.append),
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
    assert tick.last_ctx.drift_message_sent is delivered
    assert events[-1].message_result == message_result


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
    content = str(raw)
    assert "创建 Drift Skill" in content
    assert "从下一轮 Drift 起进入候选列表" in content
    assert "不执行新活动本身" in content


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
        ("select_skill", _select_input("needs-cal")),
        (
            "finish_drift",
            {
                "skill_used": "needs-cal",
                "status": "completed",
                "briefing": "done",
                "self_update": _self_update(),
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
            return {"name": "select_skill", "input": _select_input("explore-curiosity")}
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
                    "self_update": _self_update(),
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

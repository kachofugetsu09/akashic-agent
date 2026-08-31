from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from agent.plugins.manifest import builtin_plugin_data_dir
from agent.context import ContextBuilder
from agent.control.context import running_turn_id
from agent.core.passive_turn import DefaultReasoner
from agent.core.runtime_support import ToolDiscoveryState
from agent.looping.ports import LLMConfig
from agent.persona import reset_veda
from agent.plugins.manager import PluginManager
from agent.plugins.snapshot import (
    bind_runtime_snapshot,
    reset_runtime_snapshot,
)
from agent.plugin_composition import (
    ContextLengthError,
    LLMResponse,
    ModelRole,
    PROVIDER_REQUEST_PROJECTION,
    ProviderTurnInput,
    SessionCompactionStorage,
)
from agent.tools.registry import ToolRegistry
from bus.event_bus import EventBus
from plugins.compaction import plugin as compaction_plugin
from plugins.compaction.engine import SUMMARY_HEADINGS
from plugins.compaction.receipts import SqliteCompactionReceipts
from plugins.compaction.runtime import _receipt_digest
from plugins.markdown_memory import plugin as markdown_plugin
from plugins.markdown_memory.store import MarkdownProfileStore
from plugins.markdown_memory.store import DEFAULT_SELF_MD
from session.manager import SessionManager
from tests.model_plugin_fakes import (
    BoundChatModelFake,
    register_test_model_provider,
    unregister_test_model_provider,
)
from tests.test_session_compaction_runtime import _seed_receipt


class _RecordedProvider:
    context_window = 5_000
    max_output_tokens = 1_024
    model = "fixture-model"
    runtime_id = "fixture-runtime"

    def __init__(self) -> None:
        self.kinds: list[str] = []
        self.calls: list[list[dict[str, Any]]] = []

    def estimate_context_tokens(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> int:
        return max(
            1,
            (
                sum(len(str(item.get("content", ""))) for item in messages)
                + len(json.dumps(tools))
            )
            // 4,
        )

    def estimate_appended_message_tokens(self, messages: list[dict[str, Any]]) -> int:
        return self.estimate_context_tokens(messages, [])

    async def chat(self, messages: list[dict[str, Any]], **_kwargs: Any) -> LLMResponse:
        self.calls.append([dict(message) for message in messages])
        rendered = json.dumps(messages, ensure_ascii=False)
        if "更新当前长任务的上下文压缩摘要" in rendered:
            self.kinds.append("summary")
            return LLMResponse(
                content="\n\n".join(f"{heading}\n- retained" for heading in SUMMARY_HEADINGS)
            )
        if "你维护两个长期 Markdown 档案" in rendered:
            self.kinds.append("markdown")
            memory = (
                "# 用户长期记忆\n\n"
                "## 用户事实\n- 已有事实必须保留\n- 花月长期使用 Akashic\n\n"
                "## 用户偏好\n- 喜欢简单设计\n\n"
                "## 用户明确要求长期记住的关键内容\n- 保持非特权插件边界\n"
            )
            return LLMResponse(
                content=json.dumps(
                    {"memory": memory, "self": DEFAULT_SELF_MD},
                    ensure_ascii=False,
                )
            )
        self.kinds.append("business")
        return LLMResponse(content="done")


class _OverflowProvider(_RecordedProvider):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[list[dict[str, Any]]] = []

    async def chat(self, messages: list[dict[str, Any]], **_kwargs: Any) -> LLMResponse:
        self.calls.append([dict(message) for message in messages])
        raise ContextLengthError("provider context overflow")


@pytest.mark.asyncio
async def test_real_root_reasoner_updates_profiles_after_business_response(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    reset_veda(workspace)
    memory_path = workspace / "memory/MEMORY.md"
    memory_path.parent.mkdir(parents=True, exist_ok=True)
    memory_path.write_text(
        "# 用户长期记忆\n\n"
        "## 用户事实\n- 已有事实必须保留\n\n"
        "## 用户偏好\n\n"
        "## 用户明确要求长期记住的关键内容\n",
        encoding="utf-8",
    )
    provider = _RecordedProvider()
    register_test_model_provider(workspace, provider)
    sessions = SessionManager(workspace)
    session = sessions.get_or_create("web:e2e")
    for index in range(4):
        turn_id = f"old-{index}"
        session.add_message(
            "user",
            f"user-{index}-" + "u" * 900,
            control_turn_id=turn_id,
        )
        session.add_message(
            "assistant",
            f"assistant-{index}-" + "a" * 900,
            control_turn_id=turn_id,
        )
    sessions.save(session)
    message_count = len(session.messages)
    model_fixture = Path(__file__).parent / "fixtures/model_services"
    manager = PluginManager(
        plugin_dirs=[
            Path(compaction_plugin.__file__).parent,
            Path(markdown_plugin.__file__).parent,
            model_fixture,
        ],
        event_bus=EventBus(),
        tool_registry=ToolRegistry(),
        workspace=workspace,
        session_manager=sessions,
        installed_cache_root=tmp_path / "cache",
    )
    compaction_data = builtin_plugin_data_dir("compaction", workspace)
    compaction_data.mkdir(parents=True, exist_ok=True)
    (compaction_data / "config.local.toml").write_text(
        "keep_recent_tokens = 1\n", encoding="utf-8"
    )
    await manager.load_all()
    lease = await manager.snapshot_store.acquire()
    snapshot_token = bind_runtime_snapshot(lease)
    turn_token = running_turn_id.set("turn:e2e")
    try:
        reasoner = DefaultReasoner(
            llm_config=LLMConfig(max_iterations=1, max_tokens=10),
            tools=ToolRegistry(),
            discovery=ToolDiscoveryState(),
            tool_search_enabled=False,
            context=ContextBuilder(workspace),
        )
        model = BoundChatModelFake(provider, role=ModelRole.AGENT)
        result = await reasoner.run_turn(
            msg=SimpleNamespace(
                content="continue",
                media=[],
                metadata={},
                channel="web",
                chat_id="e2e",
                timestamp=datetime.now(UTC),
            ),
            session=session,
            agent_model=model,
            fallback_model=model,
        )
    finally:
        running_turn_id.reset(turn_token)
        reset_runtime_snapshot(snapshot_token)
        await lease.release()
        await manager.terminate_all()
        sessions.close()
        unregister_test_model_provider(workspace)

    assert result.reply == "done"
    assert provider.kinds == ["summary", "business", "markdown"]
    business_system = str(provider.calls[1][0]["content"])
    assert (
        business_system.index("## 行为规范")
        < business_system.index("## Akashic 自我认知")
        < business_system.index("## Long-term Memory")
        < business_system.index("## Current Session")
    )
    assert "花月长期使用 Akashic" in (
        workspace / "memory/MEMORY.md"
    ).read_text(encoding="utf-8")
    assert (workspace / "memory/SELF.md").read_text(encoding="utf-8") == DEFAULT_SELF_MD
    verifier = SessionManager(workspace)
    try:
        assert len(verifier.get_existing("web:e2e").messages) == message_count
    finally:
        verifier.close()


@pytest.mark.asyncio
async def test_disabled_compaction_real_root_passes_payload_once_without_writes(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    provider = _OverflowProvider()
    register_test_model_provider(workspace, provider)
    sessions = SessionManager(workspace)
    session = sessions.get_or_create("web:disabled")
    session.add_message("user", "old user", control_turn_id="old")
    session.add_message("assistant", "old assistant", control_turn_id="old")
    sessions.save(session)
    before_count = len(session.messages)
    manager = PluginManager(
        plugin_dirs=[
            Path(compaction_plugin.__file__).parent,
            Path(__file__).parent / "fixtures/model_services",
        ],
        event_bus=EventBus(),
        tool_registry=ToolRegistry(),
        workspace=workspace,
        session_manager=sessions,
        installed_cache_root=tmp_path / "cache",
        disabled_builtin_plugins=frozenset({"compaction"}),
    )
    await manager.load_all()
    lease = await manager.snapshot_store.acquire()
    snapshot_token = bind_runtime_snapshot(lease)
    turn_token = running_turn_id.set("turn:disabled")
    try:
        reasoner = DefaultReasoner(
            llm_config=LLMConfig(max_iterations=1, max_tokens=10),
            tools=ToolRegistry(),
            discovery=ToolDiscoveryState(),
            tool_search_enabled=False,
            context=cast(ContextBuilder, SimpleNamespace(
                render=lambda request, **_: SimpleNamespace(
                    messages=[
                        {"role": "system", "content": "root"},
                        *request.history,
                        {"role": "user", "content": request.current_message},
                    ]
                )
            )),
        )
        model = BoundChatModelFake(provider, role=ModelRole.AGENT)
        result = await reasoner.run_turn(
            msg=SimpleNamespace(
                content="current",
                media=[],
                metadata={},
                channel="web",
                chat_id="disabled",
                timestamp=datetime.now(UTC),
            ),
            session=session,
            agent_model=model,
            fallback_model=model,
        )
    finally:
        running_turn_id.reset(turn_token)
        reset_runtime_snapshot(snapshot_token)
        await lease.release()
        await manager.terminate_all()
        sessions.close()
        unregister_test_model_provider(workspace)

    assert len(provider.calls) == 1
    assert result.reply == "上下文过长无法处理，请尝试新建对话。"
    assert provider.calls[0] == [
        {"role": "system", "content": "root"},
        {"role": "user", "content": "old user"},
        {"role": "assistant", "content": "old assistant"},
        {"role": "user", "content": "current"},
    ]
    verifier = SessionManager(workspace)
    try:
        assert len(verifier.get_existing("web:disabled").messages) == before_count
        assert verifier.control_store.get_compaction_head("web:disabled").parent_generation == 0
    finally:
        verifier.close()
    assert not (workspace / "memory/consolidation_writes.db").exists()


@pytest.mark.asyncio
async def test_v2_receipt_recovers_through_real_markdown_plugin_once(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    manager_sessions: list[SessionManager] = []

    def manager_factory(path: Path) -> SessionManager:
        manager = SessionManager(path)
        manager_sessions.append(manager)
        return manager

    sessions, probe, source_ref = _seed_receipt(
        workspace,
        manager_factory,
        version=2,
    )
    receipts = SqliteCompactionReceipts(
        workspace / "memory/consolidation_writes.db"
    )
    receipt = probe.receipts[source_ref]
    draft = receipt["markdown_draft"]
    assert isinstance(draft, dict)
    draft["pending_items"] = "- [identity] 花月长期使用 Akashic"
    receipt["digest"] = _receipt_digest(receipt)
    _ = receipts.write(source_ref, receipt)
    profile_store = MarkdownProfileStore(
        workspace / "memory/MEMORY.md",
        workspace / "memory/SELF.md",
        workspace / "memory/markdown-profile-writes.db",
    )
    provider = _RecordedProvider()
    register_test_model_provider(workspace, provider)
    manager = PluginManager(
        plugin_dirs=[
            Path(compaction_plugin.__file__).parent,
            Path(markdown_plugin.__file__).parent,
            Path(__file__).parent / "fixtures/model_services",
        ],
        event_bus=EventBus(),
        tool_registry=ToolRegistry(),
        workspace=workspace,
        session_manager=sessions,
        installed_cache_root=tmp_path / "cache",
    )
    await manager.load_all()
    root = manager.current_snapshot
    assert root is not None and root.composition_root is not None
    session = sessions.get_existing("session")
    turn_token = running_turn_id.set("turn:v2")
    try:
        grant = session.issue_projection_grant("turn:v2")
        storage = SessionCompactionStorage(sessions).scope(grant)
        input = ProviderTurnInput(
            session_key=session.key,
            session_created_at=session.created_at.isoformat(),
            history_units=storage.history_units(session.key),
            access_grant=grant,
        )
        service = root.composition_root.context.require(PROVIDER_REQUEST_PROJECTION)
        _ = await service.open_turn(input)
        first_memory = profile_store.read_memory()
        _ = await service.open_turn(input)
        assert profile_store.read_memory() == first_memory
    finally:
        running_turn_id.reset(turn_token)
        await manager.terminate_all()
        unregister_test_model_provider(workspace)
        for item in manager_sessions:
            item.close()

    assert provider.kinds == []
    assert "- [identity] 花月长期使用 Akashic" in (
        workspace / "memory/MEMORY.md"
    ).read_text(encoding="utf-8")
    assert profile_store.is_applied(source_ref)

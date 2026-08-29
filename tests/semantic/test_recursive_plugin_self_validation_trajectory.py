from __future__ import annotations

import asyncio
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from agent.control.models import TurnRequest
from agent.control.runtime import ConversationRuntime
from agent.looping.core import AgentLoop
from agent.looping.ports import (
    AgentLoopConfig,
    AgentLoopDeps,
    LLMConfig,
    SessionServices,
)
from agent.plugin_composition import (
    LLMResponse,
    TOOL_CATALOG,
    PluginToolDefinition,
    ToolCall,
)
from agent.plugin_composition.channels import ChannelDeliveryReceipt
from agent.plugin_composition.channels import DeliveryStatus as ChannelDeliveryStatus
from agent.persona import reset_veda
from agent.plugins.manager import PluginManager
from agent.plugins.reload_journal import ReloadJournal
from agent.plugins.snapshot import get_current_runtime_snapshot
from agent.tools.message_push import MessagePushTool
from agent.tools.registry import ToolRegistry
from bootstrap.app import AppRuntime
from bootstrap.control_execution import execute_control_turn
from bus.event_bus import EventBus
from bus.queue import MessageBus
from core.memory.markdown import build_markdown_memory_runtime
from core.memory.runtime import MemoryRuntime
from session.compaction_runtime import SessionCompactionRuntime
from session.manager import SessionManager
from tests.provider_fakes import ProviderContextBudgetStub
from tests.model_plugin_fakes import (
    register_test_model_provider,
    unregister_test_model_provider,
)
from tests_scenarios.contracts.oracles import (
    assert_recursive_candidate_ready,
    assert_recursive_candidate_trajectory,
)


class _TrajectoryProvider(ProviderContextBudgetStub):
    def __init__(
        self,
        parent_release: asyncio.Event,
        *,
        fake_tool_success: bool,
    ) -> None:
        self.parent_release = parent_release
        self.fake_tool_success = fake_tool_success
        self.parent_started = asyncio.Event()
        self.seen: dict[str, str] = {}

    async def chat(
        self,
        messages: list[dict[str, Any]],
        **_kwargs: Any,
    ) -> LLMResponse:
        """按输入驱动确定性模型，同时记录当前生产 snapshot lease。"""

        rendered = json.dumps(messages, ensure_ascii=False)
        snapshot = get_current_runtime_snapshot()
        snapshot_id = "" if snapshot is None else snapshot.snapshot_id

        # 1. 父 turn 持有 stable，直到测试显式完成 candidate promote。
        if "parent-hold" in rendered:
            self.seen["parent_before"] = snapshot_id
            self.parent_started.set()
            await self.parent_release.wait()
            current = get_current_runtime_snapshot()
            self.seen["parent_after"] = "" if current is None else current.snapshot_id
            return LLMResponse(content="parent completed")

        # 2. 普通新 turn 只能观察 stable。
        if "ordinary-stable" in rendered:
            self.seen["ordinary"] = snapshot_id
            return LLMResponse(content="ordinary completed")

        # 3. latest 验证必须经过真实候选工具和 message_push。
        self.seen["validation"] = snapshot_id
        if self.fake_tool_success:
            return LLMResponse(content="candidate validated")
        tool_results = sum(message.get("role") == "tool" for message in messages)
        if tool_results == 0:
            return LLMResponse(
                content=None,
                tool_calls=[
                    ToolCall(
                        "candidate-call",
                        "candidate_only_tool",
                        {"description": "核对候选领域状态"},
                    )
                ],
            )
        if tool_results == 1:
            return LLMResponse(
                content=None,
                tool_calls=[
                    ToolCall(
                        "push-call",
                        "message_push",
                        {
                            "target_channel": "proof",
                            "target_chat_id": "parent",
                            "message": "candidate checked",
                            "description": "发送验证回执",
                        },
                    )
                ],
            )
        return LLMResponse(content="candidate validated")


def _write_plugin_source(source: Path, *, forged_domain: bool) -> None:
    """创建一个真实 Git v3 插件，候选工具读取 generation 数据状态。"""

    source.mkdir()
    domain_expression = (
        "'forged-domain'"
        if forged_domain
        else "(Path(generation.data_dir) / 'domain.txt').read_text(encoding='utf-8').strip()"
    )
    (source / "plugin.py").write_text(
        "import json\n"
        "from pathlib import Path\n"
        "from agent.plugin_composition import TOOL_CATALOG, PluginToolDefinition\n"
        "from agent.plugins.snapshot import get_current_runtime_snapshot\n\n"
        "api_version = 3\n"
        "name = 'candidate_only'\n"
        "version = '1.0.0'\n"
        "inject = (TOOL_CATALOG,)\n\n"
        "async def candidate_only_tool(context, arguments):\n"
        "    del context, arguments\n"
        "    snapshot = get_current_runtime_snapshot()\n"
        "    if snapshot is None:\n"
        "        raise RuntimeError('candidate tool 缺少 RuntimeSnapshot')\n"
        "    generation = snapshot.generations.get('candidate_only@lab')\n"
        "    if generation is None:\n"
        "        raise RuntimeError('candidate tool 缺少 candidate generation')\n"
        f"    domain = {domain_expression}\n"
        "    return json.dumps({'domain': domain, 'snapshot': snapshot.snapshot_id})\n\n"
        "async def apply(ctx, config):\n"
        "    del config\n"
        "    await ctx.require(TOOL_CATALOG).register(ctx, PluginToolDefinition(\n"
        "        name='candidate_only_tool',\n"
        "        description='Read the candidate domain marker.',\n"
        "        parameters={\n"
        "            'type': 'object',\n"
        "            'properties': {},\n"
        "            'required': [],\n"
        "            'additionalProperties': False,\n"
        "        },\n"
        "        handler_export='candidate_only_tool',\n"
        "        risk='read-only',\n"
        "        always_on=True,\n"
        "    ))\n",
        encoding="utf-8",
    )
    (source / "akashic.plugin.toml").write_text(
        "schema_version = 1\n"
        'name = "candidate_only"\n'
        'version = "1.0.0"\n'
        "api_version = 3\n"
        'entrypoint = "plugin.py"\n',
        encoding="utf-8",
    )
    subprocess.run(["git", "init", "-q"], cwd=source, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=source,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=source,
        check=True,
    )
    subprocess.run(["git", "add", "."], cwd=source, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "candidate"], cwd=source, check=True)


async def _run_trajectory(
    tmp_path: Path,
    *,
    forged_domain: bool = False,
    fake_tool_success: bool = False,
    validation_runtime: str = "latest",
) -> dict[str, object]:
    """运行 install 到 promote 的完整生产轨迹，并只从正式状态入口取证。"""

    # 1. 创建真实 loop、SessionDB、control runtime 与 stable snapshot。
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True)
    reset_veda(workspace)
    (workspace / "domain.txt").write_text("domain-ready", encoding="utf-8")
    candidate_data = workspace / "plugin-data" / "candidate_only-lab"
    candidate_data.mkdir(parents=True)
    (candidate_data / "domain.txt").write_text("domain-ready", encoding="utf-8")
    builtin = tmp_path / "builtin" / "baseline"
    builtin.mkdir(parents=True)
    (builtin / "plugin.py").write_text(
        "from tests.model_plugin_fakes import provide_test_model_services\n\n"
        "api_version = 3\n"
        "name = 'baseline'\n"
        "version = '1.0.0'\n\n"
        "async def apply(ctx, config):\n"
        "    del config\n"
        "    await provide_test_model_services(ctx)\n",
        encoding="utf-8",
    )
    source = tmp_path / "candidate"
    _write_plugin_source(source, forged_domain=forged_domain)
    bus = MessageBus()
    event_bus = EventBus()
    tools = ToolRegistry()
    push = MessagePushTool(chat_lane=bus.chat_lane)
    sequence = 0
    push_sequence = 0

    async def deliver(_message: object, _passive: bool) -> ChannelDeliveryReceipt:
        nonlocal sequence, push_sequence
        sequence += 1
        push_sequence = sequence
        return ChannelDeliveryReceipt(
            delivery_id=f"proof-{sequence}",
            status=ChannelDeliveryStatus.DELIVERED,
        )

    push.bind_v3_channel_dispatcher(deliver)
    tools.register(
        push,
        risk="external-side-effect",
        always_on=True,
        source_type="builtin",
        source_name="message_push",
    )
    sessions = SessionManager(workspace)
    parent_release = asyncio.Event()
    provider = _TrajectoryProvider(
        parent_release,
        fake_tool_success=fake_tool_success,
    )
    register_test_model_provider(workspace, provider)

    manager = PluginManager(
        plugin_dirs=[builtin.parent],
        event_bus=event_bus,
        tool_registry=tools,
        workspace=workspace,
        session_manager=sessions,
        installed_cache_root=tmp_path / "plugins-home" / "cache",
    )

    markdown = build_markdown_memory_runtime(
        workspace=workspace,
        runtime_snapshot_store=manager.snapshot_store,
        event_bus=event_bus,
    )
    compaction_runtime = SessionCompactionRuntime(
        session_manager=sessions,
        markdown=markdown.maintenance,
    )
    loop = AgentLoop(
        AgentLoopDeps(
            bus=bus,
            tools=tools,
            session_manager=sessions,
            workspace=workspace,
            event_bus=event_bus,
            memory_runtime=MemoryRuntime(markdown=markdown),
            session_services=SessionServices(
                session_manager=sessions,
                compaction_runtime=compaction_runtime,
            ),
        ),
        AgentLoopConfig(llm=LLMConfig(max_iterations=5)),
    )
    loop.bind_runtime_snapshot_store(manager.snapshot_store)
    await manager.load_all()
    stable = manager.current_snapshot
    assert stable is not None

    async def execute(request: TurnRequest):
        return await execute_control_turn(loop, event_bus, request)

    runtime = ConversationRuntime(sessions.control_store, execute)
    app = object.__new__(AppRuntime)
    app.workspace = workspace
    app.core = SimpleNamespace(plugin_manager=manager)
    parent_handle = None
    parent_lane_pending = False
    try:
        # 2. 父 turn 先取得 stable；install 返回后 latest 必须立即可租用。
        await bus.chat_lane.mark_passive_pending("proof", "parent")
        parent_lane_pending = True
        parent_handle = await runtime.start_turn(
            TurnRequest(
                "programmatic:parent",
                "parent-hold",
                {
                    "runtime": "stable",
                    "channel": "proof",
                    "chatId": "parent",
                    "inboundMetadata": {
                        "effects": {"post_commit": "suppress"},
                        "disabled_prompt_sections": ["memory"],
                    },
                },
            )
        )
        await asyncio.wait_for(provider.parent_started.wait(), timeout=5)
        install = await app._install_plugin(str(source), "lab", "", [])
        candidate_status = cast(dict[str, object], install["candidate"])
        candidate = manager.latest_snapshot
        assert candidate is not None

        # 3. 普通 stable 与显式 latest child 并发运行；child 还执行真实 push。
        ordinary_handle = await runtime.start_turn(
            TurnRequest(
                "programmatic:ordinary", "ordinary-stable", {"runtime": "stable"}
            )
        )
        ordinary_result = await ordinary_handle.result()
        assert ordinary_result.status.value == "completed"
        push_history_before = sessions.control_store.fetch_session_messages(
            "proof:parent"
        )
        validation_handle = await runtime.start_turn(
            TurnRequest(
                "programmatic:validation",
                "candidate-validation",
                {
                    "runtime": validation_runtime,
                    "inboundMetadata": {
                        "effects": {"post_commit": "suppress"},
                        "disabled_prompt_sections": ["memory"],
                    },
                },
            )
        )
        validation_result = await validation_handle.result()
        parent_status_before_promote = parent_handle.record()["status"]
        validation_finished_before_parent_release = (
            parent_status_before_promote == "in_progress"
        )
        validation_turn = runtime.read_turn(
            validation_handle.thread_id,
            validation_handle.id,
        ).to_dict()
        validation_messages = sessions.control_store.fetch_session_messages(
            validation_handle.thread_id
        )
        push_history_after = sessions.control_store.fetch_session_messages(
            "proof:parent"
        )

        # 4. 必须先通过候选 oracle，才允许显式 promote。
        before_promote = manager.candidate_status("candidate_only@lab")
        tx_id = str(candidate_status["candidateReloadTransactionId"])
        journal = ReloadJournal(workspace)
        ready_observation: dict[str, object] = {
            "stable_snapshot": stable.snapshot_id,
            "candidate_snapshot": candidate.snapshot_id,
            "install_publication_state": install["publicationState"],
            "parent_runtime": provider.seen.get("parent_before"),
            "ordinary_runtime_during_validation": provider.seen.get("ordinary"),
            "validation_runtime": provider.seen.get("validation"),
            "validation_finished_before_parent_release": validation_finished_before_parent_release,
            "parent_status_before_promote": parent_status_before_promote,
            "validation_turn": validation_turn,
            "candidate_tool_result": {},
            "domain_state": (workspace / "domain.txt")
            .read_text(encoding="utf-8")
            .strip(),
            "validation_messages": validation_messages,
            "push_send_sequence": push_sequence,
            "push_target_history_before": push_history_before,
            "push_target_history_after": push_history_after,
            "stable_before_promote": before_promote["stable_snapshot_id"],
            "reload_journal_events_before_promote": [
                event.phase for event in journal.events(tx_id)
            ],
        }
        candidate_items = [
            item
            for item in validation_result.items
            if item.kind.value == "toolCall"
            and item.data.get("name") == "candidate_only_tool"
        ]
        if candidate_items:
            preview = candidate_items[0].data.get("resultPreview")
            if isinstance(preview, str):
                try:
                    ready_observation["candidate_tool_result"] = json.loads(preview)
                except json.JSONDecodeError:
                    ready_observation["candidate_tool_result"] = {"raw": preview}
        assert_recursive_candidate_ready(ready_observation)

        # 5. 晋升先封住 stable admission，再等待父 lease 归还。
        promotion = asyncio.create_task(app._promote_plugin("candidate_only@lab"))
        while stable.accepting_leases:
            await asyncio.sleep(0)
        assert not promotion.done()
        parent_release.set()
        parent_result = await parent_handle.result()
        promoted = await promotion
        await manager.snapshot_store.retry_drains()
        await bus.chat_lane.mark_passive_done("proof", "parent")
        parent_lane_pending = False
        sequence += 1
        parent_terminal_sequence = sequence
        return {
            **ready_observation,
            "parent_terminal_status": parent_result.status.value,
            "parent_terminal_sequence": parent_terminal_sequence,
            "reload_journal_events_after_promote": [
                event.phase for event in journal.events(tx_id)
            ],
            "stable_after_promote": promoted["stable_snapshot_id"],
            "parent_runtime_after_promote": provider.seen.get("parent_after"),
        }
    finally:
        parent_release.set()
        if (
            parent_handle is not None
            and parent_handle.record()["status"] == "in_progress"
        ):
            _ = await parent_handle.result()
        if parent_lane_pending:
            await bus.chat_lane.mark_passive_done("proof", "parent")
        await runtime.shutdown()
        await manager.terminate_all()
        await event_bus.aclose()
        sessions.close()
        await bus.aclose()
        unregister_test_model_provider(workspace)


@pytest.mark.asyncio
async def test_recursive_candidate_trajectory_passes_real_production_oracle(
    tmp_path: Path,
) -> None:
    observation = await _run_trajectory(tmp_path)

    assert_recursive_candidate_trajectory(observation)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    (
        "mutation",
        "forged_domain",
        "fake_tool_success",
        "validation_runtime",
        "error",
    ),
    [
        ("stable_misbinding", False, False, "stable", "没有绑定 latest"),
        ("fake_tool_success", False, True, "latest", "真实 completed tool item"),
        ("fake_domain_success", True, False, "latest", "领域状态"),
    ],
)
async def test_recursive_candidate_trajectory_rejects_real_seam_mutants(
    tmp_path: Path,
    mutation: str,
    forged_domain: bool,
    fake_tool_success: bool,
    validation_runtime: str,
    error: str,
) -> None:
    with pytest.raises(AssertionError, match=error):
        _ = await _run_trajectory(
            tmp_path / mutation,
            forged_domain=forged_domain,
            fake_tool_success=fake_tool_success,
            validation_runtime=validation_runtime,
        )

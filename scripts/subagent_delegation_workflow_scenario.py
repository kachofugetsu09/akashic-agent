"""真实 Agent 全链路验收：发现模型渠道 → 按渠道/模型与继承目录委派子任务 → 外部 Runner 委派与读取结果。"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
import tempfile
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import cast

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.plugin_composition import ServiceKey
from agent.plugin_composition.config_input import save_config
from agent.plugins.bundles import set_plugin_choice
from agent.plugins.manager import PluginManager
from plugins.channels.contract import CHANNEL_INPUT_V2, ChannelInboundMessage
from plugins.ledger.contract import MESSAGE_CATALOG, ContentPart, Input, Output, ToolCall as LedgerToolCall, ToolResult
from plugins.models.contract import (
    CHAT_MODELS, MODEL_CATALOG, BoundModelDescriptor, CapabilitySources,
    ChatModelSelection, ConnectionDescriptor, LLMResponse, ModelCapabilities,
    ModelCatalogSnapshot, ModelDescriptor, ToolCall,
)
from plugins.models.content import MODEL_CONTENT, ContentOwner
from plugins.models.projection import (
    MODEL_CALLS, MODEL_MESSAGE_CHECKS, MODEL_PROJECTION, MessageChecksOwner, ProjectionOwner,
)
from plugins.models.selection import MODEL_SELECTION, SelectionOwner
from plugins.models.settings import MODEL_SETTINGS
from plugins.models.state import _BoundChat, ModelsState
from plugins.models.store import ModelsStore
from plugins.working_directory.contract import WORKING_DIRECTORY
from tests.fixtures.plugin_workspace import initialize_plugin_workspace
from tests.support.delivery_sources import sources as delivery_sources


async def run_scenario(tmp_path: Path) -> dict[str, object]:
    os.environ.update(AKASHIC_PLUGIN_DISTRIBUTION="", AKASHIC_EXECUTION_MODE="local")
    sources = tmp_path / "plugins"
    workspace = tmp_path / "workspace"
    repo_dir = tmp_path / "project-repo"
    repo_dir.mkdir(parents=True)
    (repo_dir / "module.py").write_text("VALUE = 1\n", encoding="utf-8")

    initialize_plugin_workspace(workspace)
    save_config(workspace / "plugin-data/reply-builtin", {"max_output_tokens": 4096})

    launcher = tmp_path / "peer"
    launcher.write_text(
        "#!/bin/sh\nexport PEER_CODEX_PERMISSION=approve-for-me\nexec "
        + sys.executable + " " + str(ROOT / "scripts/agent_protocol_peer.py") + "\n",
        encoding="utf-8",
    )
    launcher.chmod(0o700)
    save_config(workspace / "plugin-data/subagent_codex-builtin", {
        "executable": str(launcher),
        "permission_mode": "approve-for-me",
    })

    for name in (
        "ledger", "commands", "sources", "content", "context", "tools",
        "conversation", "react", "turn_projection", "reply_program",
        "reply", "working_directory", "shell", "files", "tool_search",
        "jobs", "subagent", "host_execution", "assets",
        "subagent_codex", "subagent_claude_code",
    ):
        shutil.copytree(ROOT / "plugins" / name, sources / name, ignore=shutil.ignore_patterns("__pycache__"))
    delivery_sources(sources)

    provider = sources / "test_provider"
    provider.mkdir()
    (provider / "plugin.py").write_text(f'''
from contextlib import asynccontextmanager
from types import SimpleNamespace
from agent.plugin_composition import ServiceKey
from plugins.models.contract import (
    CHAT_MODELS, MODEL_CATALOG, BoundModelDescriptor, CapabilitySources,
    ChatModelSelection, ConnectionDescriptor, LLMResponse, ModelCapabilities,
    ModelCatalogSnapshot, ModelDescriptor, ToolCall,
)
from plugins.models.projection import MODEL_CALLS, MODEL_PROJECTION, ProjectionOwner, MODEL_MESSAGE_CHECKS, MessageChecksOwner
from plugins.models.content import MODEL_CONTENT, ContentOwner
from plugins.models.selection import MODEL_SELECTION, SelectionOwner
from plugins.models.state import _BoundChat, ModelsState
from plugins.models.settings import MODEL_SETTINGS
from plugins.models.store import ModelsStore
from agent.plugin_composition.tasks import TASKS
from plugins.ledger.contract import BINDINGS
from plugins.tools.plugin import TOOLS

api_version = 3
name = "test_provider"
version = "1.0.0"
inject = (TOOLS, TASKS, BINDINGS)

async def apply(ctx):
    executions = []
    store = ModelsStore(ctx.data_root / "models.db", ctx.data_root / "backups")
    store.initialize()
    await ctx.effect(lambda: store.close, label="model-store")
    settings = ModelsState(store, context=ctx)
    await ctx.provide(MODEL_SETTINGS, settings.settings)

    snapshot = ModelCatalogSnapshot(
        revision=1,
        connections=(
            ConnectionDescriptor("conn-magpie", "Magpie 内网", "openai-compatible", "api:magpie", "available", "192.168.0.107", "all", "http://192.168.0.107:8000/v1"),
            ConnectionDescriptor("conn-official", "DeepSeek 官方", "openai-compatible", "api:ds", "available", "api.deepseek.com", "picked", "https://api.deepseek.com/v1"),
        ),
        models=(
            ModelDescriptor("conn-magpie__ds", "conn-magpie", "chat", "deepseek-chat", None,
                            ModelCapabilities(context_window=64000, supported_reasoning_efforts=("low", "high")), CapabilitySources(), "available"),
            ModelDescriptor("conn-magpie__qwen", "conn-magpie", "chat", "qwen3-coder", "medium",
                            ModelCapabilities(context_window=64000, supported_reasoning_efforts=("low", "medium", "high")), CapabilitySources(), "available"),
            ModelDescriptor("conn-official__ds", "conn-official", "chat", "deepseek-chat", None,
                            ModelCapabilities(context_window=64000), CapabilitySources(), "available"),
        ),
        role_bindings={{"default": "conn-magpie__ds"}},
        default_embedding_model_id=None,
    )

    class Catalog:
        def snapshot(self):
            return snapshot
        def validate_chat_selection(self, sel):
            return sel

    await ctx.provide(MODEL_CATALOG, Catalog())

    parent_step = 0
    child_steps = {{}}

    class Driver:
        max_tool_schemas = None
        def __init__(self, model_id, effort):
            self.model_id = model_id
            self.effort = effort
        def estimate_context_tokens(self, messages, tools):
            return 10
        async def complete(self, request):
            nonlocal parent_step
            is_child = "被委派的独立子 agent" in request.system_prompt or any("被委派的独立子 agent" in str(m) for m in request.messages)
            executions.append({{"model_id": self.model_id, "effort": self.effort, "is_child": is_child}})
            if is_child:
                step = child_steps.get(self.model_id, 0) + 1
                child_steps[self.model_id] = step
                if step == 1:
                    return LLMResponse(None, [
                        ToolCall("c-read", "read_file", {{"path": "module.py"}}),
                    ])
                if step == 2:
                    return LLMResponse(None, [
                        ToolCall("c-edit", "edit_file", {{"path": "module.py", "old_text": "VALUE = 1", "new_text": "VALUE = 42"}}),
                        ToolCall("c-shell", "shell", {{"command": "pwd", "description": "check inherited cwd"}}),
                    ])
                return LLMResponse(f"子任务已在继承目录完成修改与验证（model={{self.model_id}}, effort={{self.effort}}）")

            parent_step += 1
            if parent_step == 1:
                # 独占调用 set_working_directory 切换到外部项目目录 repo_dir
                return LLMResponse(None, [
                    ToolCall("set-cwd", "set_working_directory", {{"path": {repr(str(repo_dir))}}}),
                ])
            if parent_step == 2:
                return LLMResponse(None, [
                    ToolCall("load-sub", "load_tools", {{"plugin": "subagent"}}),
                    ToolCall("load-codex", "load_tools", {{"plugin": "subagent_codex@local"}}),
                ])
            if parent_step == 3:
                # 1. Agent 先通过 spawn_manage(action="models") 查询有哪些连接渠道与模型可用
                return LLMResponse(None, [
                    ToolCall("discover-models", "tool_call", {{
                        "name": "spawn_manage",
                        "arguments": {{"action": "models"}},
                    }}),
                ])
            if parent_step == 4:
                # 2. Agent 根据查到的目录，指定 Magpie 内网的 deepseek-chat (effort=high)，不传 profile 和 cwd（使用默认 general + 继承父会话目录）
                return LLMResponse(None, [
                    ToolCall("spawn-child", "tool_call", {{
                        "name": "spawn",
                        "arguments": {{
                            "task": "读取并修改当前项目目录下的 module.py 将 VALUE 改为 42，然后运行 pwd 验证目录",
                            "label": "修改项目文件",
                            "connection": "Magpie 内网",
                            "model": "deepseek-chat",
                            "reasoning_effort": "high",
                        }},
                    }}),
                    ToolCall("delegate-ext", "tool_call", {{
                        "name": "delegate_codex",
                        "arguments": {{
                            "task": "审查当前目录改动",
                            "label": "Codex 审查",
                        }},
                    }}),
                ])
            return LLMResponse("主会话已完成模型发现、跨渠道子任务委派与外部 Codex 委派。")

    def make_bound(model_id, effort):
        mid = model_id or "conn-magpie__ds"
        descriptor = BoundModelDescriptor(
            binding_id=f"bind:{{mid}}:{{effort}}", plugin_snapshot_id="fixture", model_revision=1,
            model_id=mid, connection_id=mid.split("__")[0], driver_id="openai-compatible",
            driver_contract_version="1", auth_identity="fixture", model=mid, role="chat",
            reasoning_effort=effort, capabilities=ModelCapabilities(context_window=64000),
            capability_sources=CapabilitySources(), capability_digest="fixture",
        )
        return _BoundChat(descriptor, Driver(mid, effort), store)

    class Models:
        async def resolve(self, use, choice, session=ChatModelSelection()):
            return session
        @asynccontextmanager
        async def execution(self, *, model_id=None, reasoning_effort=None):
            bound = make_bound(model_id, reasoning_effort)
            async def use(u, c):
                return bound
            async with ctx.runtime_scope():
                yield SimpleNamespace(chat=lambda: bound, use=use)

    await ctx.provide(CHAT_MODELS, Models())
    await ctx.provide(MODEL_CALLS, store.read_call)
    await ctx.provide(MODEL_PROJECTION, ProjectionOwner())
    await ctx.provide(MODEL_MESSAGE_CHECKS, MessageChecksOwner())
    await ctx.provide(MODEL_CONTENT, ContentOwner())
    await ctx.provide(MODEL_SELECTION, SelectionOwner())
    await ctx.provide(ServiceKey("fixture.executions"), executions)
''', encoding="utf-8")

    set_plugin_choice(workspace, "models", enabled=False)
    host = PluginManager([sources], workspace=workspace, installed_cache_root=tmp_path / "home/cache")
    await host.load_all()
    try:
        root = host.live_root
        assert root is not None
        session_id = "akashic:workflow-e2e"

        await root.context.require(CHANNEL_INPUT_V2)(
            session_id, "in-1",
            ChannelInboundMessage("test", "user", "room", "请先查可用模型，再用 Magpie 内网的 deepseek-chat 派子任务把 module.py 改成 42", datetime.now(UTC), {}),
        )
        reader = root.context.require(MESSAGE_CATALOG).reader(session_id)
        async with asyncio.timeout(20):
            while True:
                msgs = reader.snapshot()
                if msgs and isinstance(msgs[-1].body, Output) and msgs[-1].body.finish == "complete":
                    break
                await asyncio.sleep(0.05)

        from plugins.subagent.contract import AGENT_RUNNERS
        runners = root.context.require(AGENT_RUNNERS)
        async with asyncio.timeout(20):
            while True:
                async with runners.ctx.runtime_scope():
                    ext_rows = runners.records(session_id)
                if ext_rows and ext_rows[0]["status"] not in {"accepted", "running"}:
                    break
                await asyncio.sleep(0.05)

        tool_results = [
            str(m.body.parts[0].value)
            for m in reader.snapshot()
            if isinstance(m.body, ToolResult)
        ]
        executions = root.context.require(ServiceKey("fixture.executions"))

        # 验证点 1：Agent 通过 spawn_manage(action="models") 成功拿到连接渠道与模型列表
        models_json = next(json.loads(r) for r in tool_results if '"default_model"' in r and '"connections"' in r)
        assert [c["connection"] for c in models_json["connections"]] == ["Magpie 内网", "DeepSeek 官方"]

        # 验证点 2：子任务在指定的 conn-magpie__ds (effort=high) 上运行，且默认 general profile 继承了父会话的 repo_dir（在 workspace 之外）并成功读写文件与执行命令
        if (repo_dir / "module.py").read_text(encoding="utf-8") != "VALUE = 42\n":
            raise AssertionError(f"tool_results={tool_results}, executions={executions}")
        child_execs = [e for e in executions if e["is_child"]]
        assert child_execs and all(e["model_id"] == "conn-magpie__ds" and e["effort"] == "high" for e in child_execs)

        # 验证点 3：delegate_codex 省略 cwd 时同样自动继承父会话目录 repo_dir，并以配置的 approve-for-me (isolated-write) 模式完成
        assert ext_rows[0]["status"] == "completed"
        assert ext_rows[0]["cwd"] == str(repo_dir)
        assert any("permission_mode=approve-for-me" in a["text"] for a in ext_rows[0]["activities"])

        return {
            "discovered_connections": [c["connection"] for c in models_json["connections"]],
            "child_executions": child_execs,
            "modified_file_content": (repo_dir / "module.py").read_text(encoding="utf-8").strip(),
            "external_runner": {
                "runner": ext_rows[0]["runner"],
                "status": ext_rows[0]["status"],
                "inherited_cwd": ext_rows[0]["cwd"],
                "activities": ext_rows[0]["activities"],
            },
        }
    finally:
        if host.live_root is not None:
            await host.live_root.dispose()


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="akashic-subagent-workflow-") as tmp:
        summary = asyncio.run(run_scenario(Path(tmp)))
        print(json.dumps(summary, ensure_ascii=False, indent=2))

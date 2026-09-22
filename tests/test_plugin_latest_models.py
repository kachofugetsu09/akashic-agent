"""真实 models、driver 与普通 latest 程序接线；仅 HTTP 传输使用受控响应。"""
import asyncio
import json
import shutil
import sqlite3
from pathlib import Path

import httpx
import pytest

from agent.plugin_composition.bindings import BINDINGS
from agent.plugins.install import install_git_plugin
from agent.plugins.model_control import RuntimeModelControl
from agent.plugins.snapshot import lease_runtime_snapshot
from plugins.content.plugin import check_text
from plugins.models.store import ModelsStore
from plugins.models.settings import MODEL_SETTINGS, ModelSettingsSource, SetDefaultModel
from plugins.tools.api import MessageReply
from plugins.tools.plugin import ALL_TOOLS, TOOLS
from session.message import CallRef, ContentPart, Input, Output, ToolCall, ToolResult
from tests.test_default_reply import application
from tests.test_model_execution import _model_command
from tests.test_plugin_install import _commit
from tests.test_plugin_latest_control import update_tools


@pytest.mark.asyncio
@pytest.mark.parametrize("target,configured_default", [("models", True), ("openai_compatible", True), ("models", False)])
async def test_latest_uses_selected_model_code_existing_settings_and_local_call_log(
    tmp_path, monkeypatch, target, configured_default,
):
    """新 models/driver 读原设置，刷新原凭据，结果与调用账保存在候选。"""
    source = tmp_path / "updated-plugin"
    plugin_sources = Path(__file__).parents[1] / "plugins"

    def sources(path):
        for name in ("ui", "models", "openai_compatible"):
            destination = source if name == target else path / name
            shutil.copytree(plugin_sources / name, destination, ignore=shutil.ignore_patterns("__pycache__"))
        _commit(source)
        install_git_plugin(workspace=tmp_path / "workspace", source=str(source), marketplace="lab",
                           plugins_home=tmp_path / "home")
        # 不提供假 CHAT_MODELS；普通程序只能使用真实 models 插件。
        (path / "test_provider/plugin.py").write_text('''
from agent.plugin_composition import ServiceKey
from plugins.standard_tools.shell import shell_cleanup
api_version = 3
name = "test_provider"
version = "1.0.0"
async def apply(ctx):
    await ctx.provide(ServiceKey("tools.cleanup.v1"), shell_cleanup)
''')

    requests = []

    async def handle(request):
        assert request.url.host == "existing-provider.test"
        if request.method == "GET":
            assert request.headers["authorization"] == "Bearer existing-key"
            return httpx.Response(200, json={"data": [{"id": "default-wire"}, {"id": "agent-wire"}]})
        token = "refreshed-key" if target == "openai_compatible" else "existing-key"
        assert request.headers["authorization"] == "Bearer " + token
        assert request.extensions["timeout"]["read"] == 7
        body = json.loads(request.content)
        requests.append((request, body))
        assert request.url.path == "/v1/chat/completions"
        assert body["model"] == "agent-wire" and body["reasoning_effort"] == "high"
        if target == "models":
            assert any(message.get("content") == "selected models" for message in body["messages"])
        else:
            assert request.headers["x-selected-driver"] == "latest"
        if body.get("stream"):
            chunks = [{"choices": [{"index": 0, "delta": {"content": "real latest result"}, "finish_reason": None}]},
                      {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                       "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5}}]
            return httpx.Response(200, text="".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks)
                                  + "data: [DONE]\n\n", headers={"content-type": "text/event-stream"})
        return httpx.Response(200, json={"choices": [{"message": {"content": "real latest result"},
                                                     "finish_reason": "stop"}],
                                         "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5}})

    client = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda *args, **kwargs: client(
        *args, **kwargs, transport=httpx.MockTransport(handle)))
    async with asyncio.timeout(30), application(tmp_path, replying=False, updates=True, extra_sources=sources) as (log, host):
        control = RuntimeModelControl(host.snapshot_store)
        await _model_command(control, {"type": "add_connection", "expected_revision": 0, "connection_id": "existing",
            "name": "Existing", "driver_id": "openai-compatible", "endpoint": "http://existing-provider.test/v1",
            "auth_identity": "existing-account", "credential": {"api_key": "existing-key"},
            "driver_config": {"max_retries": 0, "read_timeout": 7}})
        for revision, model_id in ((1, "default-wire"), (2, "agent-wire")):
            await _model_command(control, {"type": "add_model", "expected_revision": revision,
                "model_id": model_id, "connection_id": "existing", "kind": "chat", "model": model_id,
                "capabilities": {"context_window": 32000, "max_output_tokens": 4096, "supports_tool_calls": True,
                                 "supported_reasoning_efforts": ["high"]}, "capability_sources": {},
                "default_reasoning_effort": "high", "driver_config": {"max_tool_schemas": 4}})
        roles = [("default", "default-wire"), ("agent", "agent-wire")] if configured_default else [("agent", "agent-wire")]
        for revision, (role, model_id) in enumerate(roles, 3):
            await _model_command(control, {"type": "set_default", "expected_revision": revision,
                                           "role": role, "model_id": model_id})
        formal_store = ModelsStore(tmp_path / "workspace/model-registry.sqlite3", tmp_path / "backups")
        before = formal_store.read_snapshot()
        if target == "models":
            module = source / "state.py"
            module.write_text(module.read_text().replace(
                '            call_id = self._store.resume_call(',
                '            request = replace(request, messages=(*request.messages, {"role": "user", "content": "selected models"}))\n'
                '            call_id = self._store.resume_call('))
        else:
            module = source / "driver.py"
            module.write_text(module.read_text().replace(
                '{"Authorization": f"Bearer {token}"}',
                '{"Authorization": f"Bearer {token}", "X-Selected-Driver": "latest"}').replace(
                '        body = _chat_body(self._descriptor, connection, self._config, request)',
                '        assert self.max_tool_schemas == 4\n'
                '        async with self._credential.exclusive():\n'
                '            assert (await self._credential.read())["api_key"] == "existing-key"\n'
                '            await self._credential.refresh({"api_key": "refreshed-key"})\n'
                '        body = _chat_body(self._descriptor, connection, self._config, request)'))
        _commit(source)

        async with lease_runtime_snapshot(host.snapshot_store) as stable:
            root = stable.composition_root.context
            tools = root.require(TOOLS)
            reader = log.reader("caller")
            inputs = log.writer("caller", author="user", source="conversation", body_types=(Input,),
                                content={"text": check_text})
            inputs.append("input", Input((ContentPart("text", "update and use my existing model"),)))

            async def call(identity, name, arguments):
                binding = tools.bind(root.require(ALL_TOOLS)().select(name), root.require(BINDINGS))
                output = log.writer("caller", author="assistant", source="conversation", body_types=(Output,),
                                    content={}, check_call=lambda call: None)
                output.append(identity, Output((ToolCall(binding, arguments),), "continue"))
                writer = log.writer("caller", author="tool", source="conversation", body_types=(ToolResult,),
                                    content={"text": check_text}, call_ref=CallRef(identity, 0))
                async def authorize(binding, arguments):
                    return {"approved": True}
                return await tools.execution(authorize).execute_call(
                    MessageReply(identity + ":result", CallRef(identity, 0), reader, writer, lambda: None))

            installed = await call("install", "plugin_install", {"source": str(source), "marketplace": "lab",
                "validation_prompt": "Report the actual result.", "validation_tools": []})
            assert installed.outcome == "success"
            identity = json.loads(installed.parts[0].value)["update_id"]
            started = await call("run", "plugin_latest", {"update_id": identity, "action": "run"})
            assert started.outcome == "success"
            async for _ in host.watch_updates():
                status = host.read_update(identity)
                if not configured_default and status.error:
                    break
                assert not status.error, status.error
                if status.publishing:
                    break
            observed = await call("status", "plugin_latest", {"update_id": identity, "action": "status"})
            if not configured_default:
                assert "default" in status.error
                assert "real latest result" not in observed.parts[0].value
                assert not requests and host._update_publication is None
                with pytest.raises(RuntimeError, match="不得重跑"):
                    await call("retry", "plugin_latest", {"update_id": identity, "action": "run"})
                reverted = await call("revert", "plugin_latest", {"update_id": identity, "action": "revert"})
                assert reverted.outcome == "success"
                assert host.current_snapshot is stable
                assert formal_store.read_snapshot() == before
                assert formal_store.read_calls("", 100) == ()
                return
            assert "real latest result" in observed.parts[0].value
        await host._update_publication[1]
        assert host.read_update(identity).phase == "committed"
        assert len(requests) == 1
        assert formal_store.read_snapshot() == before
        assert formal_store.read_calls("", 100) == ()
        credential = await formal_store.credential_handle("existing", "existing-account").read()
        assert credential["api_key"] == ("refreshed-key" if target == "openai_compatible" else "existing-key")
        evidence = Path(host.read_update(identity).evidence)
        calls = ModelsStore(evidence / "model-registry.sqlite3", evidence / "backups").read_calls("", 100)
        assert len(calls) == 1 and calls[0]["state"] == "success"
        assert calls[0]["binding"]["plugin_snapshot_id"] != stable.snapshot_id
        assert calls[0]["binding"]["model_id"] == "agent-wire"


@pytest.mark.asyncio
async def test_settings_source_keeps_scope_owner_and_rejects_missing_or_corrupt_data(tmp_path):
    """真实双 Root 核对来源、一次接续和失败；缺库不能被初始化为成功。"""
    async with asyncio.timeout(30), update_tools(tmp_path) as (host, stable, identity, call):
        original = stable.composition_root.context.require(MODEL_SETTINGS)
        source = original.read_source()
        missing = tmp_path / "missing-models.sqlite3"
        corrupt = tmp_path / "corrupt-models.sqlite3"
        corrupt.write_bytes(b"not a sqlite database")
        async with host.open_validation(identity) as scope:
            candidate = scope.require(MODEL_SETTINGS)
            with pytest.raises(RuntimeError, match="不属于当前"):
                original.read_source()
            with pytest.raises(RuntimeError, match="不存在"):
                candidate.use_source(ModelSettingsSource(missing, tmp_path / "backups"))
            assert not missing.exists()
            with pytest.raises(sqlite3.DatabaseError):
                candidate.use_source(ModelSettingsSource(corrupt, tmp_path / "backups"))
            assert corrupt.read_bytes() == b"not a sqlite database"
            candidate.use_source(source)
            assert candidate.read_source() == source
            with pytest.raises(RuntimeError, match="接续一次"):
                candidate.use_source(source)
            with pytest.raises(RuntimeError, match="只用于执行"):
                await candidate.apply(SetDefaultModel(0, "default", "unconfigured"))
        with pytest.raises(RuntimeError, match="不属于当前"):
            candidate.read_source()
        reverted = await call("plugin_latest", {"update_id": identity, "action": "revert"})
        assert reverted.outcome == "success"

from __future__ import annotations

import asyncio
import json
import runpy
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent.mcp.registry import McpServerRegistry, _mcp_search_keywords
from agent.provider import ContextLengthError, ContentSafetyError, LLMProvider
from infra.channels.cli import CLIClient, _print_banner
from infra.channels.group_filter import DefaultGroupFilter, strip_at_segments
from memory2.models import MemoryItem
from proactive_v2.anyaction import AnyActionGate, QuotaStore
from proactive_v2.memory_sampler import sample_memory_chunks, split_memory_chunks


class _Response:
    def __init__(self, content: str = "ok", tool_calls: list | None = None) -> None:
        message = SimpleNamespace(content=content, tool_calls=tool_calls or [])
        self.choices = [SimpleNamespace(message=message)]


class _ToolCall:
    def __init__(self, id: str, name: str, arguments: dict) -> None:
        self.id = id
        self.function = SimpleNamespace(
            name=name, arguments=json.dumps(arguments, ensure_ascii=False)
        )


class _FakeClient:
    def __init__(self, responses: list[object]) -> None:
        self._responses = responses
        self.calls: list[dict] = []
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self.create),
        )

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


@pytest.mark.asyncio
async def test_provider_chat_and_retry_paths(monkeypatch: pytest.MonkeyPatch):
    fake = _FakeClient(
        [
            RuntimeError("timeout"),
            _Response(
                content="done",
                tool_calls=[_ToolCall("1", "search", {"q": "x"})],
            ),
        ]
    )
    monkeypatch.setattr("agent.provider.AsyncOpenAI", lambda **_: fake)
    slept = []

    async def _sleep(sec: float) -> None:
        slept.append(sec)

    monkeypatch.setattr("agent.provider.asyncio.sleep", _sleep)
    provider = LLMProvider(
        api_key="k",
        base_url="https://example.com",
        system_prompt="system",
        extra_body={"x": 1},
        request_timeout_s=3,
        max_retries=1,
    )
    result = await provider.chat(
        messages=[{"role": "user", "content": "hi"}],
        tools=[{"type": "function"}],
        model="m",
        max_tokens=10,
    )
    assert result.content == "done"
    assert result.tool_calls[0].arguments == {"q": "x"}
    assert fake.calls[-1]["messages"][0]["role"] == "system"
    assert fake.calls[-1]["extra_body"] == {"x": 1}
    assert slept == [1.0]

    fake = _FakeClient([RuntimeError("content_policy_violation")])
    monkeypatch.setattr("agent.provider.AsyncOpenAI", lambda **_: fake)
    with pytest.raises(ContentSafetyError):
        await LLMProvider(api_key="k").chat([], [], "m", 1)

    fake = _FakeClient([RuntimeError("maximum context length exceeded")])
    monkeypatch.setattr("agent.provider.AsyncOpenAI", lambda **_: fake)
    with pytest.raises(ContextLengthError):
        await LLMProvider(api_key="k").chat([], [], "m", 1)

    fake = _FakeClient([RuntimeError("bad request")])
    monkeypatch.setattr("agent.provider.AsyncOpenAI", lambda **_: fake)
    with pytest.raises(RuntimeError):
        await LLMProvider(api_key="k", max_retries=0).chat([], [], "m", 1)


@pytest.mark.asyncio
async def test_mcp_registry_anyaction_and_sampler_cover_core_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    info = SimpleNamespace(name="readDocsNow", description="Read docs quickly now")
    keywords = _mcp_search_keywords(info, "DocsServer")
    assert "docsserver" in keywords
    assert "read" in keywords

    class _Client:
        def __init__(self, name: str, command: list[str], env=None, cwd=None):
            self.name = name
            self.command = command
            self.env = env
            self.cwd = cwd

        async def connect(self):
            return [SimpleNamespace(name="mcp_tool", description="Read remote docs")]

        async def disconnect(self):
            return None

    class _Wrapper:
        def __init__(self, client, info):
            self.client = client
            self.info = info
            self.name = f"{client.name}:{info.name}"
            self.description = info.description
            self.parameters = {"type": "object", "properties": {}, "required": []}

        def to_schema(self):
            return {"type": "function", "function": {"name": self.name}}

    monkeypatch.setattr("agent.mcp.registry.McpClient", _Client)
    monkeypatch.setattr("agent.mcp.registry.McpToolWrapper", _Wrapper)
    tools = MagicMock()
    registry = McpServerRegistry(tmp_path / "mcp.json", tools)
    added = await registry.add("docs", ["python", "srv.py"], env={"K": "V"}, cwd="/tmp")
    assert "已连接 MCP server 'docs'" in added
    tools.register.assert_called_once()
    assert "docs（1 个工具）" in registry.list_servers()
    assert "不存在" in await registry.remove("missing")
    assert "已注销" in await registry.remove("docs")

    (tmp_path / "mcp.json").write_text("{bad", encoding="utf-8")
    assert registry._load_raw_configs() == {}
    (tmp_path / "mcp.json").write_text(
        json.dumps({"servers": {"docs": {"command": ["x"]}}}), encoding="utf-8"
    )
    await registry.load_and_connect_all()

    quota = QuotaStore(tmp_path / "quota.json")
    now = datetime(2025, 6, 1, 12, tzinfo=timezone.utc)
    snap = quota.snapshot(now_utc=now, reset_hour=8, timezone_name="UTC")
    assert snap.used == 0
    quota.record_action(now_utc=now, reset_hour=8, timezone_name="UTC")
    snap = quota.snapshot(now_utc=now, reset_hour=8, timezone_name="UTC")
    assert snap.used == 1

    cfg = SimpleNamespace(
        anyaction_reset_hour_local=8,
        anyaction_timezone="UTC",
        anyaction_daily_max_actions=1,
        anyaction_min_interval_seconds=300,
        anyaction_idle_scale_minutes=60.0,
        anyaction_probability_min=0.1,
        anyaction_probability_max=0.9,
    )
    gate = AnyActionGate(cfg=cfg, quota_store=quota, rng=SimpleNamespace(random=lambda: 0.0))
    act, meta = gate.should_act(now_utc=now, last_user_at=now - timedelta(hours=2))
    assert act is False
    assert meta["reason"] == "quota_exhausted"

    cfg.anyaction_daily_max_actions = 3
    act, meta = gate.should_act(now_utc=now + timedelta(seconds=10), last_user_at=now)
    assert act is False
    assert meta["reason"] == "min_interval"

    quota = QuotaStore(tmp_path / "quota2.json")
    gate = AnyActionGate(cfg=cfg, quota_store=quota, rng=SimpleNamespace(random=lambda: 0.0))
    act, meta = gate.should_act(now_utc=now, last_user_at=now - timedelta(hours=2))
    assert act is True
    assert meta["reason"] == "probability"
    gate.record_action(now_utc=now)

    text = "## A\n\n第一段\n\n- 一\n- 二\n\n## B\n\n很长内容 " + ("句子。" * 80)
    chunks = split_memory_chunks(text, max_chunk_chars=30)
    assert chunks
    sampled = sample_memory_chunks(text, 2, rng=__import__("random").Random(1))
    assert len(sampled) == 2
    assert sample_memory_chunks("", 2) == []


@pytest.mark.asyncio
async def test_group_filter_and_cli_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    group = SimpleNamespace(group_id="1", allow_from=["42"], require_at=True)
    event = SimpleNamespace(user_id="42", raw_message="[CQ:at,qq=10001] hi")
    assert await DefaultGroupFilter("10001").should_process(event, group) is True
    assert strip_at_segments("x [CQ:at,qq=10001] y") == "x  y".strip()
    bad_user = SimpleNamespace(user_id="9", raw_message="hi")
    assert await DefaultGroupFilter("10001").should_process(bad_user, group) is False

    reader = MagicMock()
    reader.readline = AsyncMock(side_effect=[b'{"content":"hi"}\n', b""])
    writer = MagicMock()
    writer.write = MagicMock()
    writer.drain = AsyncMock()
    writer.close = MagicMock()
    monkeypatch.setattr(
        "infra.channels.cli.asyncio.open_unix_connection",
        AsyncMock(return_value=(reader, writer)),
    )
    lines = iter(["hello\n", "exit\n"])

    async def _fake_read_line() -> str:
        return next(lines)

    monkeypatch.setattr("infra.channels.cli._read_line", _fake_read_line)
    await CLIClient("/tmp/sock").run()
    writer.write.assert_called()
    assert "再见" in capsys.readouterr().out

    monkeypatch.setattr(
        "infra.channels.cli.asyncio.open_unix_connection",
        AsyncMock(side_effect=FileNotFoundError()),
    )
    await CLIClient("/tmp/missing").run()
    _print_banner()
    assert "Akasic Agent CLI" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_bootstrap_trigger_and_entrypoints_cover_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    item = MemoryItem(
        id="1",
        memory_type="procedure",
        summary="s",
        content_hash="h",
        embedding=[0.1],
        reinforcement=1,
        extra_json={},
        source_ref=None,
        happened_at=None,
        created_at="2025-01-01T00:00:00+00:00",
        updated_at="2025-01-01T00:00:00+00:00",
    )
    assert item.id == "1"

    monkeypatch.setattr("pathlib.Path.exists", lambda self: False)
    monkeypatch.setattr(sys, "argv", ["main.py", "--config", "missing.json"])
    with pytest.raises(SystemExit) as exc:
        runpy.run_module("main", run_name="__main__")
    assert exc.value.code == 1

    def _fake_asyncio_run(coro):
        coro.close()
        return None

    monkeypatch.setattr("pathlib.Path.exists", lambda self: True)
    monkeypatch.setattr("asyncio.run", _fake_asyncio_run)
    monkeypatch.setattr(
        "agent.config.Config.load",
        classmethod(
            lambda cls, path="config.json": SimpleNamespace(
                channels=SimpleNamespace(socket="/tmp/sock")
            )
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "infra.channels.cli_tui",
        SimpleNamespace(run_tui=MagicMock()),
    )
    monkeypatch.setattr(
        "bootstrap.app.build_app_runtime",
        lambda *args, **kwargs: SimpleNamespace(run=AsyncMock()),
    )
    monkeypatch.setattr(sys, "argv", ["main.py", "cli"])
    runpy.run_module("main", run_name="__main__")

    monkeypatch.setattr(sys, "argv", ["main.py"])
    runpy.run_module("main", run_name="__main__")


def test_bootstrap_proactive_builders_cover_enabled_and_disabled_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    from bootstrap.proactive import build_memory_optimizer_task, build_proactive_runtime

    cfg = SimpleNamespace(
        proactive=SimpleNamespace(
            enabled=False,
            fitbit_enabled=False,
            fitbit_monitor_path="",
        ),
        memory_optimizer_enabled=False,
        memory_optimizer_interval_seconds=3600,
        model="m",
        max_tokens=128,
        light_model="lm",
    )
    tasks, loop = build_proactive_runtime(
        cfg,
        tmp_path,
        session_manager=MagicMock(),
        provider=MagicMock(),
        light_provider=None,
        push_tool=MagicMock(),
        memory_store=None,
        presence=MagicMock(),
        agent_loop=SimpleNamespace(processing_state=None),
    )
    assert tasks == []
    assert loop is None
    assert build_memory_optimizer_task(cfg, provider=MagicMock(), memory_store=MagicMock()) == []

    proactive_loop = SimpleNamespace(
        run=lambda: "loop-task",
    )
    monkeypatch.setattr("bootstrap.proactive.ProactiveLoop", lambda **kwargs: proactive_loop)
    monkeypatch.setattr("bootstrap.proactive.ProactiveStateStore", lambda path: path)
    monkeypatch.setattr(
        "bootstrap.proactive.MemoryOptimizer",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )
    monkeypatch.setattr(
        "bootstrap.proactive.MemoryOptimizerLoop",
        lambda opt, interval_seconds: SimpleNamespace(run=lambda: ("mem-task", interval_seconds)),
    )
    monkeypatch.setattr(
        "proactive_v2.fitbit_sleep.run_fitbit_monitor",
        lambda path, url: ("fitbit-task", path, url),
    )

    cfg = SimpleNamespace(
        proactive=SimpleNamespace(
            enabled=True,
            fitbit_enabled=True,
            fitbit_monitor_path="/tmp/fitbit.json",
            fitbit_url="http://fitbit",
        ),
        memory_optimizer_enabled=True,
        memory_optimizer_interval_seconds=7200,
        model="m",
        max_tokens=128,
        light_model="lm",
    )
    tasks, loop = build_proactive_runtime(
        cfg,
        tmp_path,
        session_manager=MagicMock(),
        provider=MagicMock(),
        light_provider=MagicMock(),
        push_tool=MagicMock(),
        memory_store=MagicMock(),
        presence=MagicMock(),
        agent_loop=SimpleNamespace(processing_state=SimpleNamespace(is_busy=lambda: False)),
    )
    assert tasks == ["loop-task", ("fitbit-task", "/tmp/fitbit.json", "http://fitbit")]
    assert loop is proactive_loop
    mem_tasks = build_memory_optimizer_task(
        cfg,
        provider=MagicMock(),
        memory_store=MagicMock(),
    )
    assert mem_tasks == [("mem-task", 7200)]

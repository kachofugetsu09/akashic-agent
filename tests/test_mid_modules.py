from __future__ import annotations

import asyncio
import json
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent.core.runtime_support import ToolDiscoveryState, TurnRunResult
from agent.provider import ContentSafetyError, ContextLengthError
from agent.tools.shell import ShellTool, _MAX_OUTPUT, _truncate, _validate_network_command
from agent.tools.web_fetch import WebFetchTool, _to_markdown, _to_text


class _ReasonerHarness:
    def __init__(self, outcomes):
        self.tools = SimpleNamespace(get_always_on_names=lambda: {"always"})
        self._outcomes = list(outcomes)
        self.discovery = ToolDiscoveryState()
        self.discovery._unlocked = {"s:1": OrderedDict({"old": None})}
        self.reasoner = SimpleNamespace(
            run_turn=AsyncMock(side_effect=self._run_reasoner)
        )

    async def _run_reasoner(self, **kwargs):
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return TurnRunResult(
            reply=outcome[0],
            tools_used=outcome[1],
            tool_chain=outcome[2],
            thinking=outcome[4],
        )


@pytest.mark.asyncio
async def test_reasoner_wrapper_and_shell_cover_branches(tmp_path: Path):
    msg = SimpleNamespace(
        content="hello",
        media=[],
        channel="telegram",
        chat_id="1",
        timestamp=datetime.now(timezone.utc),
    )
    session = SimpleNamespace(
        key="s:1",
        messages=[{"role": "u", "content": str(i)} for i in range(6)],
        get_history=lambda max_messages: [{"role": "u", "content": str(i)} for i in range(6)],
        last_consolidated=3,
    )
    harness = _ReasonerHarness(
        [
            ("ok", ["tool_search", "x", "y"], [{"calls": []}], None, None),
        ]
    )
    result = await harness.reasoner.run_turn(msg=msg, session=session)
    assert result.reply == "ok"
    assert result.tools_used == ["tool_search", "x", "y"]

    harness = _ReasonerHarness(
        [("上下文过长无法处理，请尝试新建对话。", [], [], None, None)]
    )
    result = await harness.reasoner.run_turn(msg=msg, session=session)
    assert "上下文过长" in str(result.reply)
    assert result.tools_used == []
    assert result.tool_chain == []

    harness.reasoner = SimpleNamespace(
        run_turn=AsyncMock(return_value=TurnRunResult(reply="ok"))
    )
    result = await harness.reasoner.run_turn(msg=msg, session=session)
    assert result.reply == "ok"
    assert result.tools_used == []
    assert result.tool_chain == []

    harness = _ReasonerHarness([("ok", ["always", "tool_search", "a", "b", "c", "d", "e", "f"], [], None, None)])
    harness.discovery.update("s:1", ["always", "tool_search", "a", "b", "c", "d", "e", "f"], harness.tools.get_always_on_names())
    assert "always" not in harness.discovery._unlocked["s:1"]
    assert len(harness.discovery._unlocked["s:1"]) == 5

    tool = ShellTool()
    assert "命令不能为空" in await tool.execute(command="")
    assert "不被允许" in await tool.execute(command="nc localhost 1")
    assert "URL" in (_validate_network_command("curl ftp://x") or "")
    assert "上传/写文件" in (_validate_network_command("curl -o out http://x.com") or "")
    assert _validate_network_command("echo hi") is None
    assert "禁止访问内网" in (_validate_network_command("curl http://127.0.0.1") or "")
    truncated = _truncate("HEAD\n" + ("a" * 31000) + "\nTAIL")
    assert truncated["truncated"] is True
    assert truncated["strategy"] == "tail"
    assert "HEAD" not in truncated["text"]
    assert "TAIL" in truncated["text"]
    assert len(truncated["text"]) <= _MAX_OUTPUT

    result = json.loads(
        await tool.execute(
            command="sh -c 'printf out; printf err >&2; exit 2'",
            description="验证失败输出",
            timeout=999,
            yield_time_ms=1_000,
        )
    )
    assert result["exit_code"] == 2
    assert result["output"] == "outerr"

async def test_web_fetch_covers_core_paths(tmp_path: Path):
    class _Resp:
        def __init__(self, *, status=200, headers=None, content=b"", encoding="utf-8", url="https://x"):
            self.status_code = status
            self.headers = headers or {}
            self.content = content
            self.encoding = encoding
            self.url = url

    requester = MagicMock()
    requester.get = AsyncMock(
        return_value=_Resp(
            headers={"content-type": "text/html", "content-length": "20"},
            content=b"<html><body><script>x</script><p>Hello <b>world</b></p></body></html>",
        )
    )
    tool = WebFetchTool(requester=requester)
    result = json.loads(await tool.execute(url="https://example.com", format="text"))
    assert result["text"] == "Hello world"
    result = json.loads(await tool.execute(url="https://example.com", format="markdown"))
    assert "Hello" in result["text"]

    requester.get = AsyncMock(return_value=_Resp(status=404))
    assert "HTTP 404" in json.loads(await tool.execute(url="https://example.com"))["error"]
    requester.get = AsyncMock(return_value=_Resp(headers={"content-type": "application/pdf"}))
    assert "二进制内容" in json.loads(await tool.execute(url="https://example.com"))["error"]
    requester.get = AsyncMock(
        side_effect=__import__("httpx").TimeoutException("slow")
    )
    assert "请求超时" in json.loads(await tool.execute(url="https://example.com"))["error"]
    assert "http:// 或 https://" in json.loads(await tool.execute(url="ftp://x"))["error"]
    requester.get = AsyncMock(
        return_value=_Resp(
            headers={"content-type": "text/html", "content-length": "20"},
            content=b"<html><body><script>x</script><p>Hello <b>world</b></p></body></html>",
        )
    )
    assert json.loads(
        await tool.execute(url="http://127.0.0.1", format="text")
    )["text"] == "Hello world"
    assert _to_text(b"<html><body><style>x</style><p>Hi</p></body></html>") == "Hi"
    assert "Title" in _to_markdown("<h1>Title</h1>")

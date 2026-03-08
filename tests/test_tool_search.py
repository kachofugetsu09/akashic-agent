"""
tool_search 搜索质量回归测试。

覆盖场景：
- 中文自然语言查询能命中目标工具
- 同义词扩展（phrase 级 + token 级）
- risk 过滤
- search_keywords 权重与 name 相同
- MCP 工具能被搜索到
"""

import asyncio
from typing import Any
from unittest.mock import MagicMock

import pytest

from agent.mcp.client import McpToolInfo
from agent.mcp.tool import McpToolWrapper
from agent.tools.base import Tool
from agent.tools.registry import ToolRegistry, _expand_query
from agent.tools.tool_search import ToolSearchTool


# ── 辅助工具桩 ────────────────────────────────────────────────────────────────

class _StubTool(Tool):
    def __init__(self, name: str, description: str, params: dict | None = None) -> None:
        self._name = name
        self._description = description
        self._params = params or {"type": "object", "properties": {}}

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def parameters(self) -> dict[str, Any]:
        return self._params

    async def execute(self, **kwargs: Any) -> str:
        return "ok"


def _make_registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(
        ToolSearchTool(reg),
        always_on=True,
        tags=["meta"],
        risk="read-only",
    )
    reg.register(
        _StubTool("write_file", "将内容写入指定文件路径"),
        tags=["filesystem", "memory"],
        risk="write",
        search_keywords=["写文件", "保存文件", "创建文件", "写入文件", "新建文件"],
    )
    reg.register(
        _StubTool("edit_file", "编辑已有文件的指定行"),
        tags=["filesystem", "memory"],
        risk="write",
        search_keywords=["编辑文件", "修改文件", "更新文件", "patch文件"],
    )
    reg.register(
        _StubTool("list_dir", "列出目录下的文件和子目录"),
        tags=["filesystem"],
        risk="read-only",
        search_keywords=["查看目录", "列出文件", "ls", "目录内容", "浏览目录", "dir"],
    )
    reg.register(
        _StubTool("read_file", "读取文件内容"),
        tags=["filesystem"],
        risk="read-only",
        always_on=True,
        search_keywords=["读文件", "查看文件", "文件内容"],
    )
    reg.register(
        _StubTool(
            "schedule",
            "创建定时任务，在指定时间执行动作",
            params={"type": "object", "properties": {"cron": {}, "action": {}}},
        ),
        tags=["scheduling"],
        risk="write",
        search_keywords=["定时任务", "设置提醒", "计划任务", "cron", "timer"],
    )
    reg.register(
        _StubTool("feed_manage", "管理 RSS 订阅源，支持添加、删除、列出"),
        tags=["feed"],
        risk="write",
        search_keywords=["RSS订阅", "订阅管理", "添加订阅", "删除订阅"],
    )
    reg.register(
        _StubTool("fitbit_health_snapshot", "[Fitbit] 获取健康快照：步数、心率等"),
        tags=["health", "fitbit"],
        risk="read-only",
        search_keywords=["健康数据", "运动数据", "fitbit", "心率", "步数"],
    )
    reg.register(
        _StubTool("message_push", "向用户推送一条消息"),
        tags=["message"],
        risk="external-side-effect",
        search_keywords=["推送消息", "发送消息", "通知用户", "给用户发消息"],
    )
    reg.register(
        _StubTool("memorize", "将信息存入长期记忆"),
        tags=["memory"],
        risk="write",
        search_keywords=["记忆", "存储知识", "记录信息", "备忘"],
    )
    reg.register(
        _StubTool("web_search", "在互联网上搜索信息"),
        tags=["web"],
        risk="read-only",
        always_on=True,
        search_keywords=["搜索", "网络搜索"],
    )
    return reg


# ── _expand_query 单元测试 ────────────────────────────────────────────────────

class TestExpandQuery:
    def test_phrase_expansion(self):
        result = _expand_query("查看目录")
        assert "list_dir" in result
        assert "ls" in result

    def test_token_expansion(self):
        result = _expand_query("目录")
        assert "list_dir" in result
        assert "dir" in result

    def test_mixed_phrase_and_token(self):
        result = _expand_query("文件写入")
        assert "write_file" in result
        assert "write" in result

    def test_original_token_preserved(self):
        result = _expand_query("定时任务")
        assert "定时任务" in result

    def test_rss_token(self):
        result = _expand_query("rss")
        assert "feed" in result

    def test_unspaced_chinese_substring(self):
        # "目录" 作为更长中文字符串的子串也应能扩展
        result = _expand_query("列出目录下的文件")
        assert "list_dir" in result or "ls" in result


# ── ToolRegistry.search 集成测试 ──────────────────────────────────────────────

class TestRegistrySearch:
    @pytest.fixture
    def reg(self) -> ToolRegistry:
        return _make_registry()

    def _names(self, results: list[dict]) -> list[str]:
        return [r["name"] for r in results]

    # H1 核心场景：之前搜不到的查询
    def test_文件写入(self, reg):
        assert "write_file" in self._names(reg.search("文件写入"))

    def test_编辑文件(self, reg):
        assert "edit_file" in self._names(reg.search("编辑文件"))

    def test_查看目录(self, reg):
        assert "list_dir" in self._names(reg.search("查看目录"))

    def test_rss订阅(self, reg):
        assert "feed_manage" in self._names(reg.search("RSS订阅"))

    def test_健康数据(self, reg):
        assert "fitbit_health_snapshot" in self._names(reg.search("健康数据"))

    def test_推送消息(self, reg):
        assert "message_push" in self._names(reg.search("推送消息给用户"))

    def test_定时任务(self, reg):
        assert "schedule" in self._names(reg.search("定时任务"))

    def test_记忆(self, reg):
        assert "memorize" in self._names(reg.search("记忆存储"))

    # token 级同义词
    def test_token_目录(self, reg):
        assert "list_dir" in self._names(reg.search("目录"))

    def test_token_推送(self, reg):
        assert "message_push" in self._names(reg.search("推送"))

    def test_token_订阅(self, reg):
        assert "feed_manage" in self._names(reg.search("订阅"))

    # tool_search 自身不出现在结果中
    def test_tool_search_excluded(self, reg):
        results = reg.search("搜索工具")
        assert all(r["name"] != "tool_search" for r in results)

    # top_k 限制
    def test_top_k(self, reg):
        results = reg.search("文件", top_k=2)
        assert len(results) <= 2

    # risk 过滤
    def test_risk_filter_read_only(self, reg):
        results = reg.search("文件", allowed_risk=["read-only"])
        for r in results:
            assert r["risk"] == "read-only"

    def test_risk_filter_excludes_write(self, reg):
        results = reg.search("文件写入", allowed_risk=["read-only"])
        names = self._names(results)
        assert "write_file" not in names

    # search_keywords 权重 = name（应在描述匹配前排序）
    def test_search_keywords_score_equal_to_name(self, reg):
        # "写文件" 是 write_file 的 search_keyword，不在 name/description 里
        results = reg.search("写文件")
        assert len(results) > 0
        assert results[0]["name"] == "write_file"

    # why_matched 字段存在
    def test_why_matched_populated(self, reg):
        results = reg.search("定时任务")
        assert results
        assert results[0]["why_matched"]


# ── MCP 工具可被搜索 ──────────────────────────────────────────────────────────

class TestMcpToolSearch:
    def test_mcp_tool_discoverable_by_capability(self):
        reg = ToolRegistry()
        client = MagicMock()
        client.name = "calendar"
        info = McpToolInfo(
            name="create_event",
            description="Create a calendar event with title and time",
            input_schema={"type": "object", "properties": {"title": {}, "time": {}}},
        )
        wrapper = McpToolWrapper(client, info)

        from agent.mcp.registry import _mcp_search_keywords
        kws = _mcp_search_keywords(info, "calendar")

        reg.register(
            wrapper,
            tags=["mcp", "calendar"],
            risk="external-side-effect",
            search_keywords=kws,
        )

        results = reg.search("calendar")
        assert any(r["name"] == "mcp_calendar__create_event" for r in results)

        results2 = reg.search("create event")
        assert any(r["name"] == "mcp_calendar__create_event" for r in results2)


# ── ToolSearchTool 执行测试 ───────────────────────────────────────────────────

class TestToolSearchTool:
    def test_returns_json_with_matched(self):
        reg = _make_registry()
        tool = ToolSearchTool(reg)
        result = asyncio.run(tool.execute(query="定时任务"))
        import json
        data = json.loads(result)
        assert "matched" in data
        assert any(r["name"] == "schedule" for r in data["matched"])

    def test_no_match_returns_tip(self):
        reg = ToolRegistry()
        reg.register(ToolSearchTool(reg), always_on=True, tags=["meta"], risk="read-only")
        tool = ToolSearchTool(reg)
        result = asyncio.run(tool.execute(query="xxxxxxxxxxxxxxx"))
        import json
        data = json.loads(result)
        assert data["matched"] == []
        assert "tip" in data

    def test_top_k_respected(self):
        reg = _make_registry()
        tool = ToolSearchTool(reg)
        result = asyncio.run(tool.execute(query="文件", top_k=2))
        import json
        data = json.loads(result)
        assert len(data["matched"]) <= 2

    def test_top_k_clamped_to_10(self):
        reg = _make_registry()
        tool = ToolSearchTool(reg)
        result = asyncio.run(tool.execute(query="文件", top_k=999))
        import json
        data = json.loads(result)
        assert len(data["matched"]) <= 10

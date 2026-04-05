"""
tool_search 搜索质量回归测试。

覆盖场景：
- 工具 name / description 自动索引，无需手写 search_hint
- CJK bigram 归一化：中文查询无需分词库
- risk 过滤
- MCP 工具能被搜索到
- baseline 回归（从 tests/fixtures/tool_search_baseline.json 加载）
"""

import asyncio
import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock


import pytest

from agent.mcp.client import McpToolInfo
from agent.mcp.tool import McpToolWrapper
from agent.tools.base import Tool
from agent.tools.registry import ToolRegistry
from agent.tools.search_backend import _default_normalize
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
    """构建测试用 registry。

    描述比原来更丰富，以覆盖各种中文查询 ——
    不依赖手写 search_hint 或同义词表。
    """
    reg = ToolRegistry()
    reg.register(
        ToolSearchTool(reg),
        always_on=True,
        risk="read-only",
    )
    reg.register(
        _StubTool("write_file", "将内容写入指定文件路径，可用于保存、创建文件"),
        risk="write",
    )
    reg.register(
        _StubTool("edit_file", "编辑或修改已有文件的指定行"),
        risk="write",
    )
    reg.register(
        _StubTool("list_dir", "列出目录下的文件和子目录，即 ls 命令"),
        risk="read-only",
    )
    reg.register(
        _StubTool("read_file", "读取或查看文件内容"),
        risk="read-only",
        always_on=True,
    )
    reg.register(
        _StubTool(
            "schedule",
            "创建定时任务或提醒，在指定时间自动执行动作（cron）",
            params={
                "type": "object",
                "properties": {
                    "cron": {"description": "cron 表达式，例如 * * * * *"},
                    "action": {"description": "到时间后执行的动作描述"},
                },
            },
        ),
        risk="write",
    )
    reg.register(
        _StubTool("feed_manage", "管理 RSS 订阅源，支持添加、删除、查询订阅"),
        risk="write",
    )
    reg.register(
        _StubTool(
            "fitbit_health_snapshot",
            "[Fitbit] 获取健康快照：步数、心率、运动数据等",
        ),
        risk="read-only",
    )
    reg.register(
        _StubTool("message_push", "向用户推送或发送一条消息通知"),
        risk="external-side-effect",
    )
    reg.register(
        _StubTool("memorize", "将信息存入长期记忆或备忘录"),
        risk="write",
    )
    reg.register(
        _StubTool("web_search", "在互联网上搜索信息"),
        risk="read-only",
        always_on=True,
    )
    return reg


# ── _default_normalize 单元测试 ───────────────────────────────────────────────


class TestDefaultNormalize:
    """验证 CJK bigram 归一化行为，无需同义词表即可中文召回。"""

    def test_chinese_bigrams(self):
        result = _default_normalize("定时提醒")
        assert "定时" in result  # bigram
        assert "提醒" in result  # bigram
        assert "定时提醒" in result  # 原始串

    def test_chinese_unigrams(self):
        result = _default_normalize("目录")
        assert "目" in result
        assert "录" in result
        assert "目录" in result

    def test_english_space_split(self):
        result = _default_normalize("web search")
        assert "web" in result
        assert "search" in result

    def test_mixed(self):
        result = _default_normalize("RSS订阅")
        assert "rss" in result  # lowercase
        assert "订阅" in result  # CJK bigram
        assert "订" in result  # unigram

    def test_original_preserved(self):
        result = _default_normalize("schedule")
        assert "schedule" in result


# ── ToolRegistry.search 集成测试 ──────────────────────────────────────────────


class TestRegistrySearch:
    @pytest.fixture
    def reg(self) -> ToolRegistry:
        return _make_registry()

    def _names(self, results: list[dict]) -> list[str]:
        return [r["name"] for r in results]

    # 核心场景：基于描述自动召回，无 search_hint
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

    def test_设置提醒(self, reg):
        # 依赖描述中有"提醒"，不依赖同义词表
        assert "schedule" in self._names(reg.search("设置提醒"))

    def test_记忆(self, reg):
        assert "memorize" in self._names(reg.search("记忆存储"))

    # 中文单字也能通过 bigram 召回
    def test_单字_目录(self, reg):
        assert "list_dir" in self._names(reg.search("目录"))

    def test_单字_推送(self, reg):
        assert "message_push" in self._names(reg.search("推送"))

    def test_单字_订阅(self, reg):
        assert "feed_manage" in self._names(reg.search("订阅"))

    # cron 在工具描述中，可被召回
    def test_cron_in_description(self, reg):
        results = reg.search("cron")
        assert "schedule" in self._names(results)

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

    # 描述质量高的工具应排在前面
    def test_best_match_ranks_first(self, reg):
        # "写文件" 最匹配 write_file（描述含"写入"+"文件"）
        results = reg.search("写文件")
        assert len(results) > 0
        assert results[0]["name"] == "write_file"

    # why_matched 字段存在
    def test_why_matched_populated(self, reg):
        results = reg.search("定时任务")
        assert results
        assert results[0]["why_matched"]

    # always_on 字段存在于结果中
    def test_always_on_field_present(self, reg):
        results = reg.search("文件")
        assert all("always_on" in r for r in results)


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

        reg.register(
            wrapper,
            risk="external-side-effect",
            source_type="mcp",
            source_name="calendar",
        )

        results = reg.search("calendar")
        assert any(r["name"] == "mcp_calendar__create_event" for r in results)

        results2 = reg.search("create event")
        assert any(r["name"] == "mcp_calendar__create_event" for r in results2)

    def _make_feed_registry(self) -> ToolRegistry:
        """模拟真实 feed MCP 工具注册（含中文 docstring）。"""
        reg = ToolRegistry()
        client = MagicMock()
        client.name = "feed"

        # feed_manage：与 mcp_bridge.py 真实 docstring 对齐
        info_manage = McpToolInfo(
            name="feed_manage",
            description="管理 RSS 订阅源：添加、删除、列出订阅。支持 rss add / 添加订阅 / 订阅管理 / 取消订阅。",
            input_schema={
                "type": "object",
                "properties": {
                    "action": {"description": "list / add / remove"},
                    "name": {},
                    "url": {},
                },
            },
        )
        wrapper_manage = McpToolWrapper(client, info_manage)
        reg.register(
            wrapper_manage,
            risk="external-side-effect",
            source_type="mcp",
            source_name="feed",
        )

        # feed_query：与 mcp_bridge.py 真实 docstring 对齐
        info_query = McpToolInfo(
            name="feed_query",
            description="查询 RSS 订阅内容，获取最近新闻、最新文章、最新资讯、rss查询。",
            input_schema={
                "type": "object",
                "properties": {
                    "action": {"description": "latest / search / sources"},
                    "keyword": {},
                },
            },
        )
        wrapper_query = McpToolWrapper(client, info_query)
        reg.register(
            wrapper_query,
            risk="external-side-effect",
            source_type="mcp",
            source_name="feed",
        )
        return reg

    def test_feed_manage_chinese_discovery(self):
        """S4 场景：中文 RSS 订阅管理发现路径（无手写同义词表）。"""
        reg = self._make_feed_registry()
        for query in ["RSS订阅", "添加订阅", "订阅管理"]:
            names = [r["name"] for r in reg.search(query)]
            assert "mcp_feed__feed_manage" in names, f"query={query!r} 未找到 feed_manage"

    def test_feed_query_chinese_discovery(self):
        """中文新闻/最新资讯查询发现路径。"""
        reg = self._make_feed_registry()
        for query in ["最近新闻", "最新资讯"]:
            names = [r["name"] for r in reg.search(query)]
            assert "mcp_feed__feed_query" in names, f"query={query!r} 未找到 feed_query"

    def test_feed_manage_rss_add_selfheal(self):
        """S5 场景：rss_add 废弃工具 → query hint 'rss add' → 自愈到 feed_manage。"""
        reg = self._make_feed_registry()
        names = [r["name"] for r in reg.search("rss add")]
        assert "mcp_feed__feed_manage" in names, "query='rss add' 未找到 feed_manage"


# ── ToolSearchTool 执行测试 ───────────────────────────────────────────────────


class TestToolSearchTool:
    def test_returns_json_with_matched(self):
        reg = _make_registry()
        tool = ToolSearchTool(reg)
        result = asyncio.run(tool.execute(query="定时任务"))
        data = json.loads(result)
        assert "matched" in data
        assert any(r["name"] == "schedule" for r in data["matched"])

    def test_no_match_returns_tip(self):
        reg = ToolRegistry()
        reg.register(
            ToolSearchTool(reg), always_on=True, risk="read-only"
        )
        tool = ToolSearchTool(reg)
        result = asyncio.run(tool.execute(query="xxxxxxxxxxxxxxx"))
        data = json.loads(result)
        assert data["matched"] == []
        assert "tip" in data

    def test_empty_query_returns_empty_not_all_tools(self):
        """空/纯空白 query 不能返回全量工具目录（安全防护）。"""
        reg = _make_registry()
        tool = ToolSearchTool(reg)
        for bad_query in ["", "   ", "\t\n"]:
            result = asyncio.run(tool.execute(query=bad_query))
            data = json.loads(result)
            assert data["matched"] == [], f"query={bad_query!r} 不应返回任何工具"
            assert "tip" in data

    def test_empty_query_registry_search_returns_empty(self):
        """registry.search 层面的空 query 保护。"""
        reg = _make_registry()
        assert reg.search("") == []
        assert reg.search("   ") == []

    def test_top_k_respected(self):
        reg = _make_registry()
        tool = ToolSearchTool(reg)
        result = asyncio.run(tool.execute(query="文件", top_k=2))
        data = json.loads(result)
        assert len(data["matched"]) <= 2

    def test_top_k_clamped_to_10(self):
        reg = _make_registry()
        tool = ToolSearchTool(reg)
        result = asyncio.run(tool.execute(query="文件", top_k=999))
        data = json.loads(result)
        assert len(data["matched"]) <= 10

    # ── select: 精确加载路径 ─────────────────────────────────────────────

    def test_select_single_found(self):
        """select:单个工具名 → 精确命中，返回完整结果。"""
        reg = _make_registry()
        tool = ToolSearchTool(reg)
        result = asyncio.run(tool.execute(query="select:schedule"))
        data = json.loads(result)
        assert len(data["matched"]) == 1
        assert data["matched"][0]["name"] == "schedule"
        assert data["matched"][0]["why_matched"] == ["名称:精确匹配"]
        assert "tip" not in data

    def test_select_multi_found(self):
        """select:A,B,C → 多个精确命中。"""
        reg = _make_registry()
        tool = ToolSearchTool(reg)
        result = asyncio.run(tool.execute(query="select:schedule,write_file,memorize"))
        data = json.loads(result)
        names = [r["name"] for r in data["matched"]]
        assert "schedule" in names
        assert "write_file" in names
        assert "memorize" in names
        assert "tip" not in data

    def test_select_partial_match(self):
        """select: 部分命中 → 返回 found 列表 + tip 说明 missing。"""
        reg = _make_registry()
        tool = ToolSearchTool(reg)
        result = asyncio.run(tool.execute(query="select:schedule,nonexistent_tool"))
        data = json.loads(result)
        names = [r["name"] for r in data["matched"]]
        assert "schedule" in names
        assert "tip" in data
        assert "nonexistent_tool" in data["tip"]

    def test_select_all_missing(self):
        """select: 全部不存在 → matched 为空，tip 说明。"""
        reg = _make_registry()
        tool = ToolSearchTool(reg)
        result = asyncio.run(tool.execute(query="select:ghost_tool,phantom_tool"))
        data = json.loads(result)
        assert data["matched"] == []
        assert "tip" in data

    def test_select_case_insensitive_prefix(self):
        """SELECT: 大写前缀也能正常处理。"""
        reg = _make_registry()
        tool = ToolSearchTool(reg)
        result = asyncio.run(tool.execute(query="SELECT:schedule"))
        data = json.loads(result)
        assert any(r["name"] == "schedule" for r in data["matched"])

    def test_select_with_spaces(self):
        """select: 工具名两侧有空格应被 strip。"""
        reg = _make_registry()
        tool = ToolSearchTool(reg)
        result = asyncio.run(tool.execute(query="select: schedule , write_file "))
        data = json.loads(result)
        names = [r["name"] for r in data["matched"]]
        assert "schedule" in names
        assert "write_file" in names

    def test_select_result_has_expected_fields(self):
        """select: 结果包含 summary / risk / always_on 字段。"""
        reg = _make_registry()
        tool = ToolSearchTool(reg)
        result = asyncio.run(tool.execute(query="select:schedule"))
        data = json.loads(result)
        r = data["matched"][0]
        for field in ("name", "summary", "why_matched", "risk", "always_on"):
            assert field in r, f"缺少字段: {field}"

    # ── excluded_names 排除：已可见工具不出现在搜索结果 ─────────────────

    def test_visible_tools_excluded_from_keyword_search(self):
        """excluded_names 传入 schedule 时，keyword 搜索不应返回它。"""
        reg = _make_registry()
        results = reg.search("定时任务", excluded_names={"schedule"})
        assert all(r["name"] != "schedule" for r in results)

    def test_visible_tools_excluded_from_exact_fast_path(self):
        """精确名称 fast path：工具名在 excluded_names 中时不返回。"""
        reg = _make_registry()
        results = reg.search("schedule", excluded_names={"schedule"})
        assert all(r["name"] != "schedule" for r in results)

    def test_no_excluded_names_searches_all(self):
        """excluded_names 未传（None）时搜索全量工具，仅排除 meta 工具。"""
        reg = _make_registry()
        results = reg.search("定时任务")
        assert any(r["name"] == "schedule" for r in results)

    def test_excluded_names_are_per_call_not_shared_state(self):
        """excluded_names 是调用级参数，两次调用互不干扰（无共享 registry 状态）。"""
        reg = _make_registry()
        # Turn A：schedule 已可见
        results_a = reg.search("定时任务", excluded_names={"schedule"})
        # Turn B：schedule 未可见（另一 session 或下一轮）
        results_b = reg.search("定时任务", excluded_names=set())
        assert all(r["name"] != "schedule" for r in results_a)
        assert any(r["name"] == "schedule" for r in results_b)

    def test_get_deferred_names_excludes_visible(self):
        """get_deferred_names(visible=...) 不包含已可见（preloaded）工具。"""
        reg = _make_registry()
        deferred = reg.get_deferred_names(visible={"schedule"})
        assert "schedule" not in deferred.get("builtin", [])
        # write_file 未在 visible 中，应出现在 deferred 里
        assert "write_file" in deferred.get("builtin", [])

    def test_select_respects_allowed_risk(self):
        """select: 加载时尊重 allowed_risk，write 工具在 read-only 过滤下不返回。"""
        reg = _make_registry()
        tool = ToolSearchTool(reg)
        # schedule 是 write 风险，只允许 read-only 时不应返回
        result = asyncio.run(
            tool.execute(query="select:schedule", allowed_risk=["read-only"])
        )
        data = json.loads(result)
        assert all(r["name"] != "schedule" for r in data.get("matched", []))
        assert "tip" in data
        assert "风险等级不符" in data["tip"]

    def test_select_excludes_already_visible(self):
        """select: 不返回已可见工具，tip 中说明可直接调用。"""
        reg = _make_registry()
        tool = ToolSearchTool(reg)

        async def _run():
            tool.set_excluded_names({"schedule"})
            return await tool.execute(query="select:schedule")

        data = json.loads(asyncio.run(_run()))
        assert all(r["name"] != "schedule" for r in data.get("matched", []))
        assert "tip" in data
        assert "schedule" in data["tip"]

    def test_select_meta_tools_are_excluded(self):
        """select:tool_search 与 search() 语义一致 → matched 为空。"""
        reg = _make_registry()
        tool = ToolSearchTool(reg)
        result = asyncio.run(tool.execute(query="select:tool_search"))
        data = json.loads(result)
        assert data["matched"] == [], "select:tool_search 不应返回 meta tool"
        assert "tip" in data

    # ── 精确名称 fast path（独立验证）───────────────────────────────────

    def test_exact_name_fast_path(self):
        """精确工具名查询命中 fast path，why_matched 为精确匹配。"""
        reg = _make_registry()
        results = reg.search("schedule")
        assert results[0]["name"] == "schedule"
        assert results[0]["why_matched"] == ["名称:精确匹配"]

    def test_exact_name_fast_path_respects_risk_filter(self):
        """精确名称 fast path 仍然遵守 risk 过滤。"""
        reg = _make_registry()
        # schedule 是 write 风险，只允许 read-only 时不应返回
        results = reg.search("schedule", allowed_risk=["read-only"])
        assert all(r["name"] != "schedule" for r in results)


# ── Baseline 回归测试 ─────────────────────────────────────────────────────────

_BASELINE_PATH = Path(__file__).parent / "fixtures" / "tool_search_baseline.json"


class TestBaseline:
    """从 tool_search_baseline.json 加载固定 case，验证搜索质量不退化。

    每个 case 字段：
      query          搜索词（必填）
      expected_top1  top1 必须是该工具名（可选）
      expected_top3  这些工具名必须全部出现在 top3 结果中（可选）
      expected_excluded  这些工具名不能出现在结果中（可选）
      allowed_risk   传给 search() 的 risk 过滤（可选）
    """

    @pytest.fixture(scope="class")
    def reg(self):
        return _make_registry()

    @pytest.fixture(scope="class")
    def cases(self):
        return json.loads(_BASELINE_PATH.read_text(encoding="utf-8"))

    def test_baseline_cases(self, reg, cases):
        failures = []
        for case in cases:
            query = case["query"]
            allowed_risk = case.get("allowed_risk")
            results = reg.search(query, top_k=5, allowed_risk=allowed_risk)
            names = [r["name"] for r in results]

            expected_top1 = case.get("expected_top1")
            if expected_top1 and (not names or names[0] != expected_top1):
                failures.append(
                    f"query={query!r}: expected top1={expected_top1!r}, got {names[:3]}"
                )

            for want in case.get("expected_top3", []):
                if want not in names[:3]:
                    failures.append(
                        f"query={query!r}: {want!r} not in top3, got {names[:3]}"
                    )

            for excluded in case.get("expected_excluded", []):
                if excluded in names:
                    failures.append(
                        f"query={query!r}: {excluded!r} should be excluded, got {names}"
                    )

        if failures:
            pytest.fail("\n".join(failures))

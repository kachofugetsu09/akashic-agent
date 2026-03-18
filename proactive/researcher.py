"""
Evidence-First Research 模块

在发送主动消息前，通过 SubAgent 检索正文级证据，确保消息基于真实内容而非标题。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal

logger = logging.getLogger(__name__)


@dataclass
class Evidence:
    """单条证据记录"""
    id: str
    source_url_or_path: str
    title: str
    snippet: str  # 正文片段
    fetched_at: str  # ISO 时间戳


@dataclass
class FactClaim:
    """事实陈述与证据的映射"""
    claim_text: str
    evidence_id: str


@dataclass
class ResearchResult:
    """Research 阶段的输出契约"""
    status: Literal["success", "insufficient", "error"]
    rounds_used: int
    tools_called: list[str] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)
    fact_claims: list[FactClaim] = field(default_factory=list)
    reason: str = ""  # insufficient/error 时必填


class Researcher:
    """封装 SubAgent research 逻辑"""

    def __init__(
        self,
        *,
        max_iterations: int = 10,
        allowed_tools: list[str] | None = None,
        min_body_chars: int = 200,
        timeout_seconds: int = 30,
        provider: object | None = None,
        model: str = "",
        tool_registry: dict | None = None,
        include_all_mcp_tools: bool = False,
    ):
        self.max_iterations = max_iterations
        self.allowed_tools = allowed_tools or ["web_search", "web_fetch", "read_file"]
        self.min_body_chars = min_body_chars
        self.timeout_seconds = timeout_seconds
        self.provider = provider
        self.model = model
        self.tool_registry = tool_registry
        self.include_all_mcp_tools = include_all_mcp_tools

    async def research(
        self,
        *,
        items: list,  # ContentEvent or AlertEvent
        primary_source: str,  # alert | content | context
    ) -> ResearchResult:
        """
        对候选主源进行事实检索。

        Args:
            items: 候选事件列表
            primary_source: 主源类型

        Returns:
            ResearchResult: 包含证据、事实映射和状态
        """
        if not items:
            return ResearchResult(
                status="insufficient",
                rounds_used=0,
                reason="no_items_to_research",
            )

        if not self.provider:
            logger.warning("[researcher] provider 未设置，返回 insufficient")
            return ResearchResult(
                status="insufficient",
                rounds_used=0,
                reason="provider_not_configured",
            )

        # 构造 research prompt
        research_prompt = self._build_research_prompt(items, primary_source)

        # 调用 SubAgent 进行 research
        try:
            from agent.subagent import SubAgent
            from agent.tools.web_search import WebSearchTool
            from agent.tools.web_fetch import WebFetchTool
            from agent.tools.filesystem import ReadFileTool

            # 构造工具集（只读工具）
            tools = []
            if "web_search" in self.allowed_tools:
                tools.append(WebSearchTool())
            if "web_fetch" in self.allowed_tools:
                tools.append(WebFetchTool())
            if "read_file" in self.allowed_tools:
                tools.append(ReadFileTool())

            # 添加全量 MCP 工具
            mcp_tools_added = 0
            if self.include_all_mcp_tools and self.tool_registry:
                # ToolRegistry 对象，访问 _tools 字典
                tool_dict = getattr(self.tool_registry, "_tools", {})
                for tool_name, tool_instance in tool_dict.items():
                    # 只添加 MCP 工具，排除管理类工具
                    if tool_name.startswith("mcp_"):
                        # 排除管理工具
                        if tool_name in ("mcp_add", "mcp_remove", "mcp_list", "mcp_reload"):
                            continue
                        # 去重：避免重复添加
                        if not any(t.name == tool_name for t in tools):
                            tools.append(tool_instance)
                            mcp_tools_added += 1
                            logger.debug("[researcher] 添加 MCP 工具: %s", tool_name)

                if mcp_tools_added > 0:
                    logger.info("[researcher] 已添加 %d 个 MCP 工具", mcp_tools_added)

            if not tools:
                logger.warning("[researcher] 无可用工具，返回 insufficient")
                return ResearchResult(
                    status="insufficient",
                    rounds_used=0,
                    reason="no_available_tools",
                )

            # 创建 SubAgent
            subagent = SubAgent(
                provider=self.provider,
                model=self.model,
                tools=tools,
                system_prompt=self._build_system_prompt(),
                max_iterations=self.max_iterations,
            )

            logger.info("[researcher] 开始 SubAgent research max_iterations=%d", self.max_iterations)
            result_text = await subagent.run(research_prompt)
            rounds_used = subagent.iterations_used
            tools_called = subagent.tools_called  # 实际调用的工具

            logger.info(
                "[researcher] SubAgent 完成 rounds_used=%d exit_reason=%s tools=%s",
                rounds_used,
                subagent.last_exit_reason,
                tools_called,
            )

            # 解析结果
            return self._parse_result(result_text, rounds_used, tools_called)

        except Exception as e:
            logger.warning("[researcher] SubAgent 调用失败: %s", e, exc_info=True)
            return ResearchResult(
                status="error",
                rounds_used=0,
                reason=f"subagent_exception: {e}",
            )

    def _build_system_prompt(self) -> str:
        """构造 SubAgent 的 system prompt"""
        return (
            "你是一个事实检索助手。你的任务是为主动消息候选内容检索正文级证据。\n"
            "\n"
            "工作流程：\n"
            "1. 对每个候选内容，尝试获取完整正文（优先使用 URL）\n"
            "2. 提取至少 1 个可回溯的事实点（人物身份/职位/时间/结果/数字等）\n"
            "3. 将事实点与证据 ID 绑定\n"
            "4. 如果无法获取足够证据，说明失败原因\n"
            "\n"
            f"最低证据标准：至少 {self.min_body_chars} 字符的正文片段\n"
            "\n"
            "关键要求：\n"
            "- snippet 必须包含足够上下文，能明确人物身份（如：选手/教练/经理）\n"
            "- 不要只截取标题或摘要，要包含正文关键段落\n"
            "- 如果正文很长，优先截取包含核心事实的段落\n"
            "\n"
            "输出格式（JSON）：\n"
            "{\n"
            '  "status": "success" | "insufficient" | "error",\n'
            '  "evidence": [\n'
            '    {\n'
            '      "id": "ev1",\n'
            '      "source_url_or_path": "https://...",\n'
            '      "title": "...",\n'
            f'      "snippet": "正文片段（至少{self.min_body_chars}字符，包含关键身份信息）",\n'
            '      "fetched_at": "2026-03-19T00:00:00Z"\n'
            '    }\n'
            '  ],\n'
            '  "fact_claims": [\n'
            '    {\n'
            '      "claim_text": "具体事实陈述（含人物身份）",\n'
            '      "evidence_id": "ev1"\n'
            '    }\n'
            '  ],\n'
            '  "reason": "失败原因（status!=success 时必填）"\n'
            "}\n"
            "\n"
            "注意：\n"
            "- 禁止基于标题做事实推断\n"
            "- snippet 必须是正文原文，不是摘要\n"
            "- 每个 fact_claim 必须能在对应 evidence 的 snippet 中找到依据\n"
            "- 如果步骤预算用完仍未获取足够证据，返回 insufficient 状态"
        )

    def _build_research_prompt(self, items: list, primary_source: str) -> str:
        """构造 research prompt"""
        lines = [
            f"主源类型: {primary_source}",
            f"最低证据标准: 至少 {self.min_body_chars} 字符的正文片段",
            "",
            "候选内容:",
        ]

        for i, item in enumerate(items[:3], 1):
            title = getattr(item, "title", "") or ""
            url = getattr(item, "url", "") or ""
            content = getattr(item, "content", "") or ""
            lines.append(f"{i}. 标题: {title}")
            if url:
                lines.append(f"   URL: {url}")
            if content:
                lines.append(f"   摘要: {content[:100]}")
            lines.append("")

        lines.extend([
            "任务：对每个候选内容获取正文证据，并提取可回溯的事实点。",
            "输出 JSON 格式的结果（见 system prompt）。",
        ])

        return "\n".join(lines)

    def _parse_result(self, result_text: str, rounds_used: int, tools_called: list[str]) -> ResearchResult:
        """解析 SubAgent 返回的结果"""
        # 尝试提取 JSON
        try:
            # 查找 JSON 块
            start = result_text.find("{")
            end = result_text.rfind("}") + 1
            if start >= 0 and end > start:
                json_text = result_text[start:end]
                data = json.loads(json_text)

                status = data.get("status", "error")
                if status not in ("success", "insufficient", "error"):
                    status = "error"

                evidence = []
                for ev in data.get("evidence", []):
                    evidence.append(Evidence(
                        id=str(ev.get("id", "")),
                        source_url_or_path=str(ev.get("source_url_or_path", "")),
                        title=str(ev.get("title", "")),
                        snippet=str(ev.get("snippet", "")),
                        fetched_at=str(ev.get("fetched_at", datetime.now(timezone.utc).isoformat())),
                    ))

                fact_claims = []
                for fc in data.get("fact_claims", []):
                    fact_claims.append(FactClaim(
                        claim_text=str(fc.get("claim_text", "")),
                        evidence_id=str(fc.get("evidence_id", "")),
                    ))

                # 验证证据是否满足最低标准
                if status == "success":
                    valid_evidence = [ev for ev in evidence if len(ev.snippet) >= self.min_body_chars]
                    if not valid_evidence:
                        status = "insufficient"
                        reason = "no_evidence_meets_min_body_chars"
                    else:
                        evidence = valid_evidence
                        reason = ""
                else:
                    reason = str(data.get("reason", "unknown"))

                return ResearchResult(
                    status=status,
                    rounds_used=rounds_used,
                    tools_called=tools_called,
                    evidence=evidence,
                    fact_claims=fact_claims,
                    reason=reason,
                )

        except Exception as e:
            logger.warning("[researcher] 解析结果失败: %s", e)

        # 解析失败，返回 error
        return ResearchResult(
            status="error",
            rounds_used=rounds_used,
            tools_called=tools_called,
            reason=f"parse_failed: {result_text[:200]}",
        )

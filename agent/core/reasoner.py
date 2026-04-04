from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import TYPE_CHECKING

from agent.core.runtime_support import ToolDiscoveryState
from agent.core.types import ReasonerResult, ToolCall
from agent.looping.constants import (
    _INCOMPLETE_SUMMARY_PROMPT,
    _PRE_FLIGHT_PROMPT,
    _REFLECT_PROMPT,
    _SUMMARY_MAX_TOKENS,
    _TOOL_LOOP_REPEAT_LIMIT,
    _tool_call_signature,
)
from agent.prompting import build_runtime_guard_message
from agent.procedure_hint import (
    _match_procedure_items,
    build_intercept_hint,
    build_procedure_hint,
)
from agent.tool_runtime import append_assistant_tool_calls, append_tool_result
from agent.tools.base import normalize_tool_result
from agent.tools.tool_search import _excluded_names_ctx

if TYPE_CHECKING:
    from agent.looping.ports import LLMConfig, LLMServices
    from agent.tools.registry import ToolRegistry
    from core.memory.port import MemoryPort

logger = logging.getLogger("agent.core.reasoner")


def _unlock_from_tool_search(result: str, visible_names: set[str]) -> None:
    try:
        data = json.loads(result)
        for item in data.get("matched", []):
            name = item.get("name")
            if isinstance(name, str) and name:
                visible_names.add(name)
    except Exception:
        pass


def _build_reflect_content(
    pending_hints: list[str],
    visible_names: set[str] | None = None,
    always_on_names: set[str] | None = None,
) -> str:
    # 1. 先根据当前 tool visibility 生成动态提示。
    tool_state_hint = ""
    if visible_names is not None and always_on_names is not None:
        unlocked_extra = visible_names - always_on_names - {"tool_search"}
        if unlocked_extra:
            tool_state_hint = (
                f"【当前会话已额外解锁工具: {', '.join(sorted(unlocked_extra))}】\n"
            )
        else:
            tool_state_hint = (
                "【当前仅 always-on 工具可见】\n"
                "若需其他工具：已知工具名 → tool_search(query=\"select:工具名\") 加载；"
                "不知道工具名 → tool_search(query=\"关键词\") 搜索。\n"
            )

    # 2. 本轮没有 procedure hint 时，直接返回默认 reflect prompt。
    if not pending_hints:
        return tool_state_hint + _REFLECT_PROMPT

    # 3. 有 hint 时，把 hint 和默认 reflect prompt 拼在一起。
    combined = "\n".join(h for h in pending_hints if h.strip())
    if not combined.strip():
        return tool_state_hint + _REFLECT_PROMPT
    return (
        "【⚠️ 操作规范提醒 | 适用于本轮工具调用】\n"
        f"{combined}\n\n---\n\n"
        + tool_state_hint
        + _REFLECT_PROMPT
    )


def _build_deferred_tools_hint(
    tools: "ToolRegistry",
    visible: set[str] | None = None,
) -> str:
    # 1. 先从 registry 里取 deferred 工具目录。
    get_deferred_names = getattr(tools, "get_deferred_names", None)
    if not callable(get_deferred_names):
        return ""
    deferred = get_deferred_names(visible=visible)
    builtin: list[str] = deferred.get("builtin", [])
    mcp: dict[str, list[str]] = deferred.get("mcp", {})

    # 2. 没有 deferred 工具时，不补任何目录提示。
    if not builtin and not mcp:
        return ""

    # 3. 有 deferred 工具时，按 builtin / mcp 分组输出。
    lines: list[str] = ["【未加载工具目录（知道名字但 schema 未暴露）】"]
    if builtin:
        lines.append(f"内置: {', '.join(builtin)}")
    for server, names in mcp.items():
        lines.append(f"MCP ({server}): {', '.join(names)}")

    total = len(builtin) + sum(len(v) for v in mcp.values())
    lines.append(
        f"\n共 {total} 个。加载方式：\n"
        "- 已知工具名 → tool_search(query=\"select:工具名\")，支持逗号分隔多个\n"
        "- 描述功能   → tool_search(query=\"关键词\") 搜索匹配"
    )
    return "\n".join(lines) + "\n\n"


def build_preflight_prompt(
    *,
    request_time: datetime | None,
    tools: "ToolRegistry",
    tool_search_enabled: bool,
    visible_names: set[str] | None,
) -> str:
    # 1. 先生成时间锚点。
    anchor = DefaultReasoner.format_request_time_anchor(request_time)

    # 2. 再按需拼接 deferred 工具目录提示。
    deferred_hint = (
        _build_deferred_tools_hint(tools, visible=visible_names)
        if tool_search_enabled
        else ""
    )

    # 3. 最后输出本轮 preflight runtime guard。
    return (
        f"【本轮时间锚点】{anchor}\n"
        "所有时间相关判断必须与该锚点一致；无法验证时必须明确不确定。\n\n"
        + deferred_hint
        + _PRE_FLIGHT_PROMPT
    )


class Reasoner(ABC):
    """
    ┌──────────────────────────────────────┐
    │ Reasoner                             │
    ├──────────────────────────────────────┤
    │ 1. append preflight                  │
    │ 2. call llm                          │
    │ 3. execute tool calls                │
    │ 4. append tool results               │
    │ 5. return final reply                │
    └──────────────────────────────────────┘
    """

    @abstractmethod
    async def run(
        self,
        initial_messages: list[dict],
        *,
        request_time: datetime | None = None,
        preloaded_tools: set[str] | None = None,
        preflight_injected: bool = False,
    ) -> ReasonerResult:
        """执行多轮 tool loop，并返回本轮结果。"""


class DefaultReasoner(Reasoner):
    def __init__(
        self,
        llm: "LLMServices",
        llm_config: "LLMConfig",
        tools: "ToolRegistry",
        discovery: ToolDiscoveryState,
        memory_port: "MemoryPort",
        *,
        tool_search_enabled: bool,
    ) -> None:
        self._llm = llm
        self._llm_config = llm_config
        self._tools = tools
        self._discovery = discovery
        self._memory_port = memory_port
        self._tool_search_enabled = tool_search_enabled

    async def run(
        self,
        initial_messages: list[dict],
        *,
        request_time: datetime | None = None,
        preloaded_tools: set[str] | None = None,
        preflight_injected: bool = False,
    ) -> ReasonerResult:
        # 1. 初始化消息上下文、本轮工具轨迹、循环检测状态。
        messages = initial_messages
        tools_used: list[str] = []
        tool_chain: list[dict] = []
        last_tool_signature = ""
        repeat_count = 0
        injected_proc_ids: set[str] = set()

        # 2. 初始化本轮可见工具集合。
        visible_names: set[str] | None = None
        if self._tool_search_enabled:
            always_on = self._tools.get_always_on_names()
            visible_names = always_on | (preloaded_tools or set())
            logger.info(
                "[tool_search] visible=%d 个工具 always_on=%d preloaded=%d need_search=%s",
                len(visible_names),
                len(always_on),
                len(preloaded_tools or set()),
                "yes" if len(visible_names) == len(always_on) else "maybe",
            )

        # 3. 如果 assembled input 里还没塞 preflight，就在这里补进去。
        if not preflight_injected:
            messages = messages + [
                build_runtime_guard_message(
                    build_preflight_prompt(
                        request_time=request_time,
                        tools=self._tools,
                        tool_search_enabled=self._tool_search_enabled,
                        visible_names=visible_names,
                    )
                )
            ]

        for iteration in range(self._llm_config.max_iterations):
            # 4. 调用 LLM，带上当前可见工具 schema。
            response = await self._llm.provider.chat(
                messages=messages,
                tools=self._tools.get_schemas(names=visible_names),
                model=self._llm_config.model,
                max_tokens=self._llm_config.max_tokens,
                tool_choice="auto",
            )

            # 5. 模型返回 tool_calls 时，进入工具执行分支。
            if response.tool_calls:
                signature = _tool_call_signature(response.tool_calls)
                if signature and signature == last_tool_signature:
                    repeat_count += 1
                else:
                    repeat_count = 1
                    last_tool_signature = signature

                if repeat_count >= _TOOL_LOOP_REPEAT_LIMIT:
                    summary = await self._summarize_incomplete_progress(
                        messages,
                        reason="tool_call_loop",
                        iteration=iteration + 1,
                        tools_used=tools_used,
                    )
                    return self._build_result(
                        reply=summary,
                        tools_used=tools_used,
                        tool_chain=tool_chain,
                        visible_names=visible_names,
                        thinking=None,
                    )

                append_assistant_tool_calls(
                    messages,
                    content=response.content,
                    tool_calls=response.tool_calls,
                )

                # 6. 逐个执行本轮工具调用。
                iter_calls: list[dict] = []
                pending_hints: list[str] = []
                for tool_call in response.tool_calls:
                    # 6.1 deferred 工具未解锁时，先回填 select: 引导错误。
                    if visible_names is not None and tool_call.name not in visible_names:
                        result = (
                            f"工具 '{tool_call.name}' 当前未加载（schema 不可见）。"
                            f"请先调用 tool_search(query=\"select:{tool_call.name}\") 加载，"
                            "然后再调用该工具。不要放弃当前任务。"
                        )
                        append_tool_result(
                            messages,
                            tool_call_id=tool_call.id,
                            content=result,
                        )
                        iter_calls.append(
                            {
                                "call_id": tool_call.id,
                                "name": tool_call.name,
                                "arguments": tool_call.arguments,
                                "result": result,
                            }
                        )
                        continue

                    # 6.2 真实执行前，先做 procedure intercept 判断。
                    all_items = _match_procedure_items(
                        memory=self._memory_port,
                        tool_name=tool_call.name,
                        tool_arguments=tool_call.arguments,
                        logger=logger,
                    )
                    intercept_items = [
                        item
                        for item in all_items
                        if bool(item.get("intercept", False))
                        and str(item.get("id", "")) not in injected_proc_ids
                    ]
                    if intercept_items:
                        result = build_intercept_hint(intercept_items, tool_call.name)
                        injected_proc_ids.update(
                            str(item.get("id", "")) for item in intercept_items
                        )
                        append_tool_result(
                            messages,
                            tool_call_id=tool_call.id,
                            content=result,
                        )
                        iter_calls.append(
                            {
                                "call_id": tool_call.id,
                                "name": tool_call.name,
                                "arguments": tool_call.arguments,
                                "result": result,
                            }
                        )
                        continue

                    # 6.3 工具未被拦截时，真正执行工具并回填结果。
                    if tool_call.name == "tool_search" and visible_names is not None:
                        token = _excluded_names_ctx.set(visible_names)
                    else:
                        token = None
                    tools_used.append(tool_call.name)
                    try:
                        result = await self._tools.execute(
                            tool_call.name,
                            tool_call.arguments,
                        )
                    finally:
                        if token is not None:
                            _excluded_names_ctx.reset(token)
                    normalized = normalize_tool_result(result)
                    append_tool_result(
                        messages,
                        tool_call_id=tool_call.id,
                        content=result,
                        tool_name=tool_call.name,
                    )

                    # 6.4 工具执行完后，再补充 procedure hint。
                    hint_items = [
                        item
                        for item in all_items
                        if not bool(item.get("intercept", False))
                    ]
                    raw_hint, new_ids = build_procedure_hint(
                        hint_items,
                        injected_proc_ids,
                    )
                    if new_ids:
                        injected_proc_ids.update(new_ids)
                        if raw_hint:
                            pending_hints.append(raw_hint.split("\n", 1)[1])

                    # 6.5 tool_search 的结果会扩展下一轮可见工具。
                    if tool_call.name == "tool_search" and visible_names is not None:
                        _unlock_from_tool_search(normalized.text, visible_names)
                    iter_calls.append(
                        {
                            "call_id": tool_call.id,
                            "name": tool_call.name,
                            "arguments": tool_call.arguments,
                            "result": normalized.preview(),
                        }
                    )

                # 7. 本轮工具执行完后，追加 reflect prompt，继续下一轮。
                tool_chain.append({"text": response.content, "calls": iter_calls})
                messages.append(
                    build_runtime_guard_message(
                        _build_reflect_content(
                            pending_hints,
                            visible_names=visible_names,
                            always_on_names=self._tools.get_always_on_names()
                            if self._tool_search_enabled
                            else None,
                        )
                    )
                )
                continue

            # 8. 没有 tool_calls 时，说明本轮得到最终回复。
            messages.append({"role": "assistant", "content": response.content})
            return self._build_result(
                reply=response.content or "（无响应）",
                tools_used=tools_used,
                tool_chain=tool_chain,
                visible_names=visible_names,
                thinking=response.thinking,
            )

        # 9. 达到最大迭代次数后，生成不完整进展总结。
        summary = await self._summarize_incomplete_progress(
            messages,
            reason="max_iterations",
            iteration=self._llm_config.max_iterations,
            tools_used=tools_used,
        )
        return self._build_result(
            reply=summary,
            tools_used=tools_used,
            tool_chain=tool_chain,
            visible_names=visible_names,
            thinking=None,
        )

    async def _summarize_incomplete_progress(
        self,
        messages: list[dict],
        *,
        reason: str,
        iteration: int,
        tools_used: list[str],
    ) -> str:
        # 1. 先构造收尾总结 prompt。
        summary_prompt = (
            f"[收尾原因] {reason}\n"
            f"[已执行轮次] {iteration}\n"
            f"[已调用工具] {', '.join(tools_used[-8:]) if tools_used else '无'}\n\n"
            + _INCOMPLETE_SUMMARY_PROMPT
        )

        # 2. 先尝试让模型给一段中文收尾总结。
        try:
            response = await self._llm.provider.chat(
                messages=messages + [build_runtime_guard_message(summary_prompt)],
                tools=[],
                model=self._llm_config.model,
                max_tokens=min(_SUMMARY_MAX_TOKENS, self._llm_config.max_tokens),
            )
            text = (response.content or "").strip()
            if text:
                return text
        except Exception as exc:
            logger.warning("生成预算收尾总结失败: %s", exc)

        # 3. 模型收尾失败时，返回固定兜底文案。
        done = f"已尝试 {iteration} 轮，调用工具 {len(tools_used)} 次。"
        return (
            f"这次任务还没完全收束。{done}"
            "我先停在当前进度，后续会继续补齐缺失信息并给你最终结论。"
        )

    def _build_result(
        self,
        *,
        reply: str,
        tools_used: list[str],
        tool_chain: list[dict],
        visible_names: set[str] | None,
        thinking: str | None,
    ) -> ReasonerResult:
        # 1. 先把 tool_chain 扁平化成 invocations。
        invocations: list[ToolCall] = []
        for group in tool_chain:
            for call in group.get("calls") or []:
                args = call.get("arguments")
                invocations.append(
                    ToolCall(
                        id=str(call.get("call_id", "") or ""),
                        name=str(call.get("name", "") or ""),
                        arguments=args if isinstance(args, dict) else {},
                    )
                )

        # 2. 再把运行时元数据统一塞进 metadata。
        metadata = {
            "tools_used": list(tools_used),
            "tool_chain": list(tool_chain),
            "visible_names": set(visible_names) if visible_names is not None else None,
        }

        # 3. 最后返回标准 ReasonerResult。
        return ReasonerResult(
            reply=reply,
            invocations=invocations,
            thinking=thinking,
            metadata=metadata,
        )

    @staticmethod
    def format_request_time_anchor(ts: datetime | None) -> str:
        # 1. 空时间戳时，使用当前本地时间。
        if ts is None:
            ts = datetime.now().astimezone()
        elif ts.tzinfo is None:
            ts = ts.astimezone()

        # 2. 输出稳定的 request_time 锚点字符串。
        return f"request_time={ts.isoformat()} ({ts.strftime('%Y-%m-%d %H:%M:%S %Z')})"

import json
import logging
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

from agent.looping.constants import (
    _INCOMPLETE_SUMMARY_PROMPT,
    _PRE_FLIGHT_PROMPT,
    _REFLECT_PROMPT,
    _SUMMARY_MAX_TOKENS,
    _TOOL_LOOP_REPEAT_LIMIT,
    _tool_call_signature,
)
from agent.procedure_hint import (
    _match_procedure_items,
    build_intercept_hint,
    build_procedure_hint,
)
from agent.tool_runtime import append_assistant_tool_calls, append_tool_result
from agent.tools.base import normalize_tool_result
from agent.tools.tool_search import _excluded_names_ctx

logger = logging.getLogger("agent.loop")

if TYPE_CHECKING:
    from agent.looping.ports import LLMConfig, LLMServices
    from agent.tools.registry import ToolRegistry
    from core.memory.port import MemoryPort

_LRU_CAPACITY = 5


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
    # 动态工具状态提示
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

    if not pending_hints:
        return tool_state_hint + _REFLECT_PROMPT
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
    tools: "ToolRegistry", visible: set[str] | None = None
) -> str:
    """构建 deferred 工具目录提示，注入到每个 turn 的 preflight 里。

    visible: 当前 turn 已可见工具名（always_on + preloaded），从目录中排除，
    避免把已加载的工具误报为"未加载"。
    """
    deferred = tools.get_deferred_names(visible=visible)
    builtin: list[str] = deferred.get("builtin", [])
    mcp: dict[str, list[str]] = deferred.get("mcp", {})

    if not builtin and not mcp:
        return ""

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


@dataclass
class ToolDiscoveryState:
    _unlocked: dict[str, OrderedDict[str, None]] = field(default_factory=dict)
    capacity: int = _LRU_CAPACITY

    def get_preloaded(self, session_key: str) -> set[str]:
        return set(self._unlocked.get(session_key, {}).keys())

    def update(self, session_key: str, tools_used: list[str], always_on: set[str]) -> None:
        skip = always_on | {"tool_search"}
        lru: OrderedDict[str, None] = self._unlocked.setdefault(
            session_key, OrderedDict()
        )
        for name in tools_used:
            if name in skip:
                continue
            if name in lru:
                lru.move_to_end(name)
            else:
                lru[name] = None
            while len(lru) > self.capacity:
                lru.popitem(last=False)


class TurnExecutor:
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

    async def execute(
        self,
        initial_messages: list[dict],
        request_time: datetime | None = None,
        preloaded_tools: set[str] | None = None,
    ) -> tuple[str, list[str], list[dict], set[str] | None, str | None]:
        # 1. 先初始化本轮对话状态：消息上下文、工具使用轨迹、循环检测状态。
        messages = initial_messages
        tools_used: list[str] = []
        tool_chain: list[dict] = []
        last_tool_signature = ""
        repeat_count = 0
        injected_proc_ids: set[str] = set()

        # 2. tool_search 模式下，只向 LLM 暴露 always_on + LRU 预加载工具的完整 schema。
        #    其余工具的名字通过 preflight 的 deferred 工具目录告知模型，
        #    模型用 tool_search 解锁后下一 iteration 才拿到完整 schema。
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

        preflight_prompt = (
            f"【本轮时间锚点】{self._format_request_time_anchor(request_time)}\n"
            "所有时间相关判断必须与该锚点一致；无法验证时必须明确不确定。\n\n"
            + (
                _build_deferred_tools_hint(self._tools, visible=visible_names)
                if self._tool_search_enabled
                else ""
            )
            + _PRE_FLIGHT_PROMPT
        )
        # 3. 每轮开始前都补一条 preflight 提示，约束时间判断和工具使用方式。
        messages = messages + [{"role": "user", "content": preflight_prompt}]

        for iteration in range(self._llm_config.max_iterations):
            # 4. 用当前 messages + visible tools 调一次 LLM。
            logger.debug("LLM 调用  iteration=%d", iteration + 1)
            response = await self._llm.provider.chat(
                messages=messages,
                tools=self._tools.get_schemas(names=visible_names),
                model=self._llm_config.model,
                max_tokens=self._llm_config.max_tokens,
                tool_choice="auto",
            )

            # 5. 模型请求了工具时，先检测是否陷入重复工具调用循环。
            if response.tool_calls:
                signature = _tool_call_signature(response.tool_calls)
                if signature and signature == last_tool_signature:
                    repeat_count += 1
                else:
                    repeat_count = 1
                    last_tool_signature = signature

                if repeat_count >= _TOOL_LOOP_REPEAT_LIMIT:
                    logger.warning(
                        "检测到工具调用循环 signature=%s repeat=%d，提前收尾",
                        signature[:160],
                        repeat_count,
                    )
                    summary = await self._summarize_incomplete_progress(
                        messages,
                        reason="tool_call_loop",
                        iteration=iteration + 1,
                        tools_used=tools_used,
                    )
                    return summary, tools_used, tool_chain, visible_names, None

                logger.info(
                    "LLM 请求调用 %d 个工具: %s",
                    len(response.tool_calls),
                    [tc.name for tc in response.tool_calls],
                )
                append_assistant_tool_calls(
                    messages,
                    content=response.content,
                    tool_calls=response.tool_calls,
                )

                # 6. 逐个执行本轮 tool calls，并把结果回写进消息历史。
                iter_calls: list[dict] = []
                pending_hints: list[str] = []
                for tc in response.tool_calls:
                    # 6.1 工具未在本 turn 可见集合内：必须先通过 tool_search 加载，不自动解锁。
                    if visible_names is not None and tc.name not in visible_names:
                        logger.warning(
                            "  x 工具 %s 未加载（不在 visible_names），引导使用 select:",
                            tc.name,
                        )
                        result = (
                            f"工具 '{tc.name}' 当前未加载（schema 不可见）。"
                            f"请先调用 tool_search(query=\"select:{tc.name}\") 加载，"
                            f"然后再调用该工具。不要放弃当前任务。"
                        )
                        append_tool_result(messages, tool_call_id=tc.id, content=result)
                        iter_calls.append(
                            {
                                "call_id": tc.id,
                                "name": tc.name,
                                "arguments": tc.arguments,
                                "result": result,
                            }
                        )
                        continue

                    args_str = json.dumps(tc.arguments, ensure_ascii=False)
                    logger.info("  → 工具 %s  参数: %s", tc.name, args_str[:120])

                    # 6.2 在真正执行前，先按关键词匹配 procedure 规范。
                    all_items = _match_procedure_items(
                        memory=self._memory_port,
                        tool_name=tc.name,
                        tool_arguments=tc.arguments,
                        logger=logger,
                    )
                    intercept_items = [
                        item
                        for item in all_items
                        if bool(item.get("intercept", False))
                        and str(item.get("id", "")) not in injected_proc_ids
                    ]
                    if intercept_items:
                        result = build_intercept_hint(intercept_items, tc.name)
                        injected_proc_ids.update(
                            str(item.get("id", "")) for item in intercept_items
                        )
                        append_tool_result(messages, tool_call_id=tc.id, content=result)
                        iter_calls.append(
                            {
                                "call_id": tc.id,
                                "name": tc.name,
                                "arguments": tc.arguments,
                                "result": result,
                            }
                        )
                        logger.info("  ⛔ 工具 %s 被规范拦截，未执行", tc.name)
                        continue

                    # 6.3 未被拦截时才真正执行工具，并把结果追加为 tool message。
                    # tool_search 需要知道当前 visible_names 以排除已可见工具；
                    # 通过 ContextVar 注入，不污染 execute() 的 arguments dict，
                    # 避免 set 进入 RecordingToolRegistry 的 artifact 序列化。
                    if tc.name == "tool_search" and visible_names is not None:
                        _ctx_token = _excluded_names_ctx.set(visible_names)
                    else:
                        _ctx_token = None
                    tools_used.append(tc.name)
                    try:
                        result = await self._tools.execute(tc.name, tc.arguments)
                    finally:
                        if _ctx_token is not None:
                            _excluded_names_ctx.reset(_ctx_token)
                    normalized = normalize_tool_result(result)
                    result_preview = normalized.preview()
                    if len(result_preview) > 80:
                        result_preview = result_preview[:80] + "..."
                    logger.info("  ← 工具 %s  结果: %r", tc.name, result_preview)
                    append_tool_result(
                        messages,
                        tool_call_id=tc.id,
                        content=result,
                        tool_name=tc.name,
                    )

                    hint_items = [
                        item for item in all_items if not bool(item.get("intercept", False))
                    ]
                    raw_hint, new_ids = build_procedure_hint(
                        hint_items,
                        injected_proc_ids,
                    )
                    if new_ids:
                        injected_proc_ids.update(new_ids)
                        if raw_hint:
                            pending_hints.append(raw_hint.split("\n", 1)[1])

                    # 6.4 tool_search 的返回结果会继续解锁一批工具，供下一轮 LLM 可见。
                    if tc.name == "tool_search" and visible_names is not None:
                        _unlock_from_tool_search(normalized.text, visible_names)
                        logger.debug("tool_search 解锁后 visible=%d 个工具", len(visible_names))
                    iter_calls.append(
                        {
                            "call_id": tc.id,
                            "name": tc.name,
                            "arguments": tc.arguments,
                            "result": normalized.preview(),
                        }
                    )

                # 7. 本轮工具执行完后，补一条 reflect 提示，让模型基于结果继续下一轮决策。
                tool_chain.append({"text": response.content, "calls": iter_calls})
                messages.append(
                    {
                        "role": "user",
                        "content": _build_reflect_content(
                            pending_hints,
                            visible_names=visible_names,
                            always_on_names=self._tools.get_always_on_names() if self._tool_search_enabled else None,
                        ),
                    }
                )
                continue

            # 8. 没有 tool_calls 时，说明模型已经给出最终回答，本轮结束。
            logger.info("LLM 返回最终回复  iteration=%d", iteration + 1)
            messages.append({"role": "assistant", "content": response.content})
            return (
                response.content or "（无响应）",
                tools_used,
                tool_chain,
                visible_names,
                response.thinking,
            )

        # 9. 如果超过最大迭代次数还没收尾，就生成一个不完整进展总结返回。
        logger.warning("已达到最大迭代次数 %d", self._llm_config.max_iterations)
        summary = await self._summarize_incomplete_progress(
            messages,
            reason="max_iterations",
            iteration=self._llm_config.max_iterations,
            tools_used=tools_used,
        )
        return summary, tools_used, tool_chain, visible_names, None

    async def _summarize_incomplete_progress(
        self,
        messages: list[dict],
        *,
        reason: str,
        iteration: int,
        tools_used: list[str],
    ) -> str:
        summary_prompt = (
            f"[收尾原因] {reason}\n"
            f"[已执行轮次] {iteration}\n"
            f"[已调用工具] {', '.join(tools_used[-8:]) if tools_used else '无'}\n\n"
            + _INCOMPLETE_SUMMARY_PROMPT
        )
        try:
            resp = await self._llm.provider.chat(
                messages=messages + [{"role": "user", "content": summary_prompt}],
                tools=[],
                model=self._llm_config.model,
                max_tokens=min(_SUMMARY_MAX_TOKENS, self._llm_config.max_tokens),
            )
            text = (resp.content or "").strip()
            if text:
                return text
        except Exception as e:
            logger.warning("生成预算收尾总结失败: %s", e)

        done = f"已尝试 {iteration} 轮，调用工具 {len(tools_used)} 次。"
        return (
            f"这次任务还没完全收束。{done}"
            "我先停在当前进度，后续会继续补齐缺失信息并给你最终结论。"
        )

    @staticmethod
    def _format_request_time_anchor(ts: datetime | None) -> str:
        if ts is None:
            ts = datetime.now().astimezone()
        elif ts.tzinfo is None:
            ts = ts.astimezone()
        return f"request_time={ts.isoformat()} ({ts.strftime('%Y-%m-%d %H:%M:%S %Z')})"

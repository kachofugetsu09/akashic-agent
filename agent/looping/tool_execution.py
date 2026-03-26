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
        unlocked_extra = visible_names - always_on_names - {"tool_search", "list_tools"}
        if unlocked_extra:
            tool_state_hint = (
                f"【当前会话已额外解锁工具: {', '.join(sorted(unlocked_extra))}】\n"
            )
        else:
            tool_state_hint = (
                "【当前仅 always-on 工具可见】\n"
                "若需其他工具：已知工具名可直接调用（系统自动解锁）；"
                "不知道工具名时调用 tool_search 搜索。\n"
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
        messages = initial_messages
        tools_used: list[str] = []
        tool_chain: list[dict] = []
        last_tool_signature = ""
        repeat_count = 0
        injected_proc_ids: set[str] = set()

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
            + _PRE_FLIGHT_PROMPT
        )
        messages = messages + [{"role": "user", "content": preflight_prompt}]

        for iteration in range(self._llm_config.max_iterations):
            logger.debug("LLM 调用  iteration=%d", iteration + 1)
            response = await self._llm.provider.chat(
                messages=messages,
                tools=self._tools.get_schemas(names=visible_names),
                model=self._llm_config.model,
                max_tokens=self._llm_config.max_tokens,
                tool_choice="auto",
            )

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

                iter_calls: list[dict] = []
                pending_hints: list[str] = []
                for tc in response.tool_calls:
                    if visible_names is not None and tc.name not in visible_names:
                        if self._tools.has_tool(tc.name):
                            visible_names.add(tc.name)
                            logger.info("  ↑ 工具 %s 从历史记忆自动解锁", tc.name)
                        else:
                            logger.warning("  ✗ 工具 %s 不存在，已注入 query hint", tc.name)
                            suggested_query = tc.name.replace("_", " ").replace("-", " ")
                            result = (
                                f"工具 '{tc.name}' 当前不可见或不存在。"
                                f"请立即调用 tool_search(query=\"{suggested_query}\") 搜索等价工具，"
                                f"然后从结果中选择正确工具继续执行。不要放弃当前任务。"
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

                    tools_used.append(tc.name)
                    result = await self._tools.execute(tc.name, tc.arguments)
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

            logger.info("LLM 返回最终回复  iteration=%d", iteration + 1)
            messages.append({"role": "assistant", "content": response.content})
            return (
                response.content or "（无响应）",
                tools_used,
                tool_chain,
                visible_names,
                response.thinking,
            )

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

import json
import logging
from datetime import datetime

from agent.procedure_hint import (
    _match_procedure_items,
    build_intercept_hint,
    build_procedure_hint,
)
from agent.looping.constants import (
    _INCOMPLETE_SUMMARY_PROMPT,
    _PRE_FLIGHT_PROMPT,
    _REFLECT_PROMPT,
    _SUMMARY_MAX_TOKENS,
    _TOOL_LOOP_REPEAT_LIMIT,
    _tool_call_signature,
)
from agent.tool_runtime import append_assistant_tool_calls, append_tool_result

logger = logging.getLogger("agent.loop")


def _unlock_from_tool_search(result: str, visible_names: set[str]) -> None:
    """解析 tool_search 结果，将匹配到的工具名加入 visible_names。"""
    try:
        data = json.loads(result)
        for item in data.get("matched", []):
            name = item.get("name")
            if isinstance(name, str) and name:
                visible_names.add(name)
    except Exception:
        pass


def _build_reflect_content(pending_hints: list[str]) -> str:
    if not pending_hints:
        return _REFLECT_PROMPT
    combined = "\n".join(h for h in pending_hints if h.strip())
    if not combined.strip():
        return _REFLECT_PROMPT
    return (
        "【⚠️ 操作规范提醒 | 适用于本轮工具调用】\n"
        f"{combined}\n\n---\n\n"
        + _REFLECT_PROMPT
    )


class AgentLoopToolExecutionMixin:
    async def _run_agent_loop(
        self,
        initial_messages: list[dict],
        request_time: datetime | None = None,
        preloaded_tools: set[str] | None = None,
    ) -> tuple[str, list[str], list[dict], set[str] | None]:
        """运行 agent 主循环。

        返回 (final_content, tools_used, tool_chain, visible_names)。
        visible_names 仅在 tool_search_enabled=True 时返回集合，否则返回 None。
        """
        messages = initial_messages
        tools_used: list[str] = []
        tool_chain: list[dict] = []
        last_tool_signature = ""
        repeat_count = 0
        injected_proc_ids: set[str] = set()  # 本轮已注入的 procedure item id，用于去重

        # 按需工具可见性：仅 tool_search_enabled 时生效
        # preloaded_tools 来自上一轮请求的缓存，实现跨请求持久化
        visible_names: set[str] | None = None
        if getattr(self, "_tool_search_enabled", False):
            always_on = self.tools.get_always_on_names()
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

        for iteration in range(self.max_iterations):
            logger.debug(f"LLM 调用  iteration={iteration + 1}")
            response = await self.provider.chat(
                messages=messages,
                tools=self.tools.get_schemas(names=visible_names),
                model=self.model,
                max_tokens=self.max_tokens,
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
                    f"LLM 请求调用 {len(response.tool_calls)} 个工具: "
                    f"{[tc.name for tc in response.tool_calls]}"
                )
                append_assistant_tool_calls(
                    messages,
                    content=response.content,
                    tool_calls=response.tool_calls,
                )

                iter_calls: list[dict] = []
                pending_hints: list[str] = []
                for tc in response.tool_calls:
                    # 硬约束：visible_names 启用时，检查工具可见性
                    if visible_names is not None and tc.name not in visible_names:
                        # 工具在 registry 里存在（模型从历史记忆到）→ auto-unlock，直接执行
                        # 工具根本不存在（幻觉）→ 拦截，返回错误
                        if tc.name in self.tools._tools:
                            visible_names.add(tc.name)
                            logger.info("  ↑ 工具 %s 从历史记忆自动解锁", tc.name)
                        else:
                            logger.warning("  ✗ 工具 %s 不存在，拒绝执行", tc.name)
                            result = f"工具 '{tc.name}' 不存在，请调用 tool_search 查找可用工具。"
                            append_tool_result(
                                messages,
                                tool_call_id=tc.id,
                                content=result,
                            )
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
                    logger.info(f"  → 工具 {tc.name}  参数: {args_str[:120]}")

                    # 1. 单次匹配规范，先分流执行前拦截
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
                        append_tool_result(
                            messages,
                            tool_call_id=tc.id,
                            content=result,
                        )
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

                    # 2. 执行工具，并保留干净的真实输出
                    tools_used.append(tc.name)
                    result = await self.tools.execute(tc.name, tc.arguments)
                    result_preview = result[:80] + "..." if len(result) > 80 else result
                    logger.info(f"  ← 工具 {tc.name}  结果: {result_preview!r}")

                    append_tool_result(
                        messages,
                        tool_call_id=tc.id,
                        content=result,
                    )

                    # 3. 从同一批匹配结果里收集执行后提醒，统一并入 reflect prompt
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

                    # tool_search 返回后解锁匹配工具
                    if tc.name == "tool_search" and visible_names is not None:
                        _unlock_from_tool_search(result, visible_names)
                        logger.debug(
                            "tool_search 解锁后 visible=%d 个工具",
                            len(visible_names),
                        )
                    iter_calls.append(
                        {
                            "call_id": tc.id,
                            "name": tc.name,
                            "arguments": tc.arguments,
                            "result": result,
                        }
                    )
                tool_chain.append({"text": response.content, "calls": iter_calls})
                messages.append(
                    {"role": "user", "content": _build_reflect_content(pending_hints)}
                )
            else:
                logger.info(f"LLM 返回最终回复  iteration={iteration + 1}")
                messages.append({"role": "assistant", "content": response.content})
                return (
                    response.content or "（无响应）",
                    tools_used,
                    tool_chain,
                    visible_names,
                    response.thinking,
                )

        logger.warning(f"已达到最大迭代次数 {self.max_iterations}")
        summary = await self._summarize_incomplete_progress(
            messages,
            reason="max_iterations",
            iteration=self.max_iterations,
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
            resp = await self.provider.chat(
                messages=messages + [{"role": "user", "content": summary_prompt}],
                tools=[],
                model=self.model,
                max_tokens=min(_SUMMARY_MAX_TOKENS, self.max_tokens),
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

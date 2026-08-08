from __future__ import annotations

import inspect
import logging
from contextvars import ContextVar
from dataclasses import replace
from typing import Any, Sequence

from agent.control.ids import new_operation_id, new_turn_id
from agent.model_runtime.context_compaction import (
    ContextCompactionError,
    ContextCompactor,
    ContextPayloadSegments,
)
from agent.model_runtime.types import ModelUsage
from agent.provider import LLMProvider
from agent.tool_hooks import ToolExecutionRequest, ToolExecutor
from agent.tool_hooks.base import ToolHook
from agent.tool_runtime import (
    append_assistant_tool_calls,
    append_tool_result,
    build_tool_map,
    build_tool_schemas,
    tool_call_batch_snapshot,
)
from agent.tool_hooks.types import ToolExecutionResult
from agent.tools.base import (
    Tool,
    ToolExecutionContext,
    get_current_tool_context,
    normalize_tool_result,
    tool_execution_context_scope,
)
from prompts.completion import VERIFIABLE_COMPLETION_RULES

logger = logging.getLogger(__name__)

_SUBAGENT_TURN_ID: ContextVar[str | None] = ContextVar(
    "akashic_subagent_turn_id",
    default=None,
)
_SUBAGENT_BASE_CONTEXT: ContextVar[ToolExecutionContext | None] = ContextVar(
    "akashic_subagent_base_context",
    default=None,
)

_REFLECT_PROMPT = (
    "根据上述工具结果，决定下一步操作。\n"
    "若任务已完成，直接输出最终结果；若需要继续，继续调用工具。\n"
    "禁止把工具调用失败的原因写进最终回复，遇到失败时换个方式或跳过该步骤。"
)
_REFLECT_PROMPT_WARN = (
    "根据上述工具结果，决定下一步操作。\n"
    "⚠️ 步骤预算剩余 {remaining} 步，请优先完成核心目标，跳过非必要步骤。\n"
    "若任务已完成，直接输出最终结果；若需要继续，继续调用工具。\n"
    "禁止把工具调用失败的原因写进最终回复，遇到失败时换个方式或跳过该步骤。"
)
_REFLECT_PROMPT_LAST = (
    "⚠️ 步骤预算将在下一步耗尽。请立即优先完成核心目标，"
    "下一步将进入强制收尾。"
)
_CLEANUP_PROMPT = (
    "步骤预算已耗尽，进入强制收尾阶段。\n"
    "你必须调用 {tool_name}，如实汇报当前进度（已完成的步骤、产出路径、未完成的原因）。"
)
_WARN_THRESHOLD = 5
_MAX_TOOL_RESULT_CHARS = 100_000
_SUMMARY_MAX_TOKENS = 512
_INCOMPLETE_SUMMARY_PROMPT = (
    "当前任务未在步骤预算内完成，请直接输出中文进度总结，不要 JSON。\n"
    "必须覆盖：1) 已完成内容；2) 当前未完成点；3) 下一步计划。\n"
    "禁止输出模板句“已达到最大迭代次数”。"
)
_FORCED_FINAL_SUMMARY_PROMPT = (
    "你已用完任务执行预算，禁止再调用工具。\n"
    "现在必须直接输出中文最终总结，供主 agent 回传给用户。\n"
    "必须覆盖：1) 已完成内容；2) 当前未完成内容；3) 产出文件路径（如果有）；4) 下一步建议。\n"
    "禁止：继续规划工具调用；说“需要继续调用工具”；输出“已达到最大迭代次数”等模板句。"
)
_FORCED_FINAL_SUMMARY_FALLBACK = (
    "这次后台任务已先停在当前进度。我已经完成了一部分关键步骤，"
    "但还有剩余工作未收束；下一次可从当前检查点继续推进。"
)


def _is_tool_loop_guard_denial(exec_result: object) -> bool:
    traces = getattr(exec_result, "pre_hook_trace", ()) or ()
    return any(
        getattr(item, "decision", "") == "deny"
        and str(getattr(item, "reason", "")).startswith("tool_loop_guard:")
        for item in traces
    )


class _SubagentContextGate:
    """持有一次 subagent 运行的内存态 Pi 风格上下文投影。"""

    def __init__(
        self,
        *,
        provider: LLMProvider,
        model: str,
        messages: list[dict[str, Any]],
        max_tokens: int,
        scope_id: str,
    ) -> None:
        self._scope_id = scope_id
        self._compactor = ContextCompactor(
            provider=provider,
            model=model,
            scope_id=scope_id,
            payload_segments=ContextPayloadSegments(
                prefix=tuple(dict(message) for message in messages),
                committed_units=(),
                current_anchor=(),
            ),
            max_output_tokens=max_tokens,
            ledger_parent_generation=0,
            next_generation=1,
        )

    async def prepare(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]],
        max_tokens: int,
    ) -> None:
        """检查下一次 provider payload，并且只替换本次内存视图。"""

        self._compactor.set_pending(messages)
        prepared = await self._compactor.prepare(
            messages,
            pending_start=self._compactor.pending_start,
            tools=tools,
            max_output_tokens=max_tokens,
        )
        if prepared.compacted:
            logger.info(
                "[subagent] context gate compacted scope=%s estimated=%d "
                "trigger=%s",
                self._scope_id,
                prepared.estimated_tokens,
                prepared.checkpoint.trigger if prepared.checkpoint else "unknown",
            )

    def record_response(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        usage: ModelUsage | None,
    ) -> None:
        self._compactor.record_response(
            message_count=len(messages),
            tools=tools,
            usage=usage,
        )

    def record_completed_batch(
        self,
        messages: list[dict[str, Any]],
    ) -> None:
        self._compactor.record_completed_batch(
            messages,
            batch_start=self._compactor.pending_start,
        )

    def sync_pending(self, messages: list[dict[str, Any]]) -> None:
        self._compactor.set_pending(messages)


class SubAgent:
    """使用固定工具集执行有界单任务，不持有会话或记忆状态。"""

    def __init__(
        self,
        provider: LLMProvider,
        model: str,
        tools: list[Tool],
        *,
        system_prompt: str = "",
        max_iterations: int = 30,
        max_tokens: int = 0,
        mandatory_exit_tools: Sequence[str] = (),
    ) -> None:
        self._provider = provider
        self._model = model
        self._system_prompt = system_prompt
        self._max_iterations = max_iterations
        self._max_tokens = max_tokens
        self._mandatory_exit_tools = list(mandatory_exit_tools)
        self.last_exit_reason: str = "idle"
        self.iterations_used: int = 0
        self.tools_called: list[str] = []
        self._run_seq = 0
        tool_schemas = build_tool_schemas(tools)
        tool_map = build_tool_map(tools)
        self._tool_map: dict[str, Tool] = tool_map
        self._tool_schemas: list[dict[str, Any]] = tool_schemas
        self._tool_executor = ToolExecutor([])

    def add_tool_hooks(self, hooks: list[ToolHook]) -> None:
        self._tool_executor.add_hooks(hooks)

    async def run(self, task: str) -> str:
        """执行单次任务，并在 owner 结束时回收 shell execution。"""

        run_turn_id = new_turn_id()
        inherited_context = get_current_tool_context()
        base_context = replace(
            inherited_context or ToolExecutionContext(),
            turn_id=run_turn_id,
            execution_id="",
        )
        turn_token = _SUBAGENT_TURN_ID.set(run_turn_id)
        context_token = _SUBAGENT_BASE_CONTEXT.set(base_context)
        primary_error: BaseException | None = None
        try:
            return await self._run(task)
        except BaseException as exc:
            primary_error = exc
            raise
        finally:
            try:
                await self._shutdown_shell()
            except BaseException:
                if primary_error is None:
                    raise
                logger.exception("[subagent] shell cleanup 失败，保留原始异常")
            finally:
                _SUBAGENT_BASE_CONTEXT.reset(context_token)
                _SUBAGENT_TURN_ID.reset(turn_token)

    async def _run(self, task: str) -> str:
        """执行单次任务，并返回完成结果或预算收尾总结。"""
        messages: list[dict[str, Any]] = []
        self.last_exit_reason = "running"
        self.iterations_used = 0
        self.tools_called = []
        self._run_seq += 1
        tool_session_key = f"subagent:{id(self)}:{self._run_seq}"
        system_prompt = "\n\n".join(
            part for part in (self._system_prompt, VERIFIABLE_COMPLETION_RULES) if part
        )
        messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": task})
        gate = _SubagentContextGate(
            provider=self._provider,
            model=self._model,
            messages=messages,
            max_tokens=self._max_tokens,
            scope_id=f"subagent:{id(self)}:{self._run_seq}",
        )
        for iteration in range(self._max_iterations):
            self.iterations_used = iteration + 1
            try:
                response = await self._provider_chat(
                    gate,
                    messages=messages,
                    tools=self._tool_schemas,
                    max_tokens=self._max_tokens,
                    tool_choice="auto",
                )
            except Exception:
                self.last_exit_reason = "error"
                raise

            if not response.tool_calls:
                result = (response.content or "").strip()
                if not result:
                    self.last_exit_reason = "error"
                    raise RuntimeError("SubAgent 模型未返回结果")
                logger.info("[subagent] 任务完成 iterations=%d", iteration + 1)
                self.last_exit_reason = "completed"
                return result

            # 保持 assistant 调用与后续 tool 结果的消息闭链
            append_assistant_tool_calls(
                messages,
                content=response.content,
                tool_calls=response.tool_calls,
                provider_fields=response.provider_fields,
            )
            tool_batch = tool_call_batch_snapshot(response.tool_calls)

            for tool_batch_index, tc in enumerate(response.tool_calls):
                logger.info(
                    "[subagent] 调用工具 %s args=%s",
                    tc.name,
                    str(tc.arguments)[:120],
                )
                exec_result = await self._execute_tool_call(
                    tc.id,
                    tc.name,
                    tc.arguments,
                    session_key=tool_session_key,
                    tool_batch=tool_batch,
                    tool_batch_index=tool_batch_index,
                )
                if (
                    exec_result.status == "success"
                    and tc.name not in self.tools_called
                ):
                    self.tools_called.append(tc.name)
                normalized = normalize_tool_result(exec_result.output)
                logger.info(
                    "[subagent] 工具结果 %s: %s",
                    tc.name,
                    normalized.preview()[:120],
                )
                # 限制单次工具结果，避免挤占后续推理上下文
                if len(normalized.text) > _MAX_TOOL_RESULT_CHARS:
                    original_len = len(normalized.text)
                    normalized.text = (
                        normalized.text[:_MAX_TOOL_RESULT_CHARS]
                        + f"\n...[结果已截断，原始长度 {original_len} 字符，超出上限 {_MAX_TOOL_RESULT_CHARS}]"
                    )
                    logger.warning(
                        "[subagent] 工具结果 %s 过长已截断 original=%d",
                        tc.name,
                        original_len,
                    )
                append_tool_result(
                    messages,
                    tool_call_id=tc.id,
                    content=normalized,
                    tool_name=tc.name,
                    execution_status=exec_result.status,
                )
                if _is_tool_loop_guard_denial(exec_result):
                    logger.warning(
                        "[subagent] 插件截断重复工具调用 tool=%s，提前收尾",
                        tc.name,
                    )
                    self.last_exit_reason = "tool_loop"
                    for skipped in response.tool_calls[tool_batch_index + 1:]:
                        append_tool_result(
                            messages,
                            tool_call_id=skipped.id,
                            content="工具调用已因重复循环检测跳过。",
                            tool_name=skipped.name,
                            execution_status="skipped",
                        )
                    gate.record_completed_batch(messages)
                    if self._mandatory_exit_tools:
                        await self._run_mandatory_exit(
                            messages,
                            tool_session_key,
                            gate,
                        )
                    return await self._summarize_incomplete_progress(
                        messages,
                        gate,
                        reason="tool_call_loop",
                        iteration=iteration + 1,
                    )

            gate.record_completed_batch(messages)
            remaining = self._max_iterations - iteration - 1
            if remaining == 0:
                reflect = _REFLECT_PROMPT_LAST
            elif remaining <= _WARN_THRESHOLD:
                reflect = _REFLECT_PROMPT_WARN.format(remaining=remaining)
            else:
                reflect = _REFLECT_PROMPT
            messages.append({"role": "user", "content": reflect})

        logger.warning("[subagent] 已达到最大迭代次数 %d", self._max_iterations)
        if self._mandatory_exit_tools:
            await self._run_mandatory_exit(messages, tool_session_key, gate)
        return await self._force_final_summary(
            messages,
            gate,
            reason="max_iterations",
            iteration=self._max_iterations,
        )

    async def _provider_chat(
        self,
        gate: _SubagentContextGate,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        max_tokens: int,
        tool_choice: object = None,
    ):
        """统一检查并执行 subagent 的四类 provider 入口。"""

        await gate.prepare(messages, tools=tools, max_tokens=max_tokens)
        request: dict[str, Any] = {
            "messages": messages,
            "tools": tools,
            "model": self._model,
            "max_tokens": max_tokens,
        }
        if tool_choice is not None:
            request["tool_choice"] = tool_choice
        response = await self._provider.chat(**request)
        gate.record_response(
            messages=messages,
            tools=tools,
            usage=response.usage,
        )
        return response

    async def _shutdown_shell(self) -> None:
        shell = self._tool_map.get("shell")
        shutdown = getattr(shell, "shutdown", None)
        if callable(shutdown):
            result = shutdown()
            if not inspect.isawaitable(result):
                raise TypeError("shell.shutdown 必须返回 awaitable")
            await result

    async def _summarize_incomplete_progress(
        self,
        messages: list[dict[str, Any]],
        gate: _SubagentContextGate,
        *,
        reason: str,
        iteration: int,
    ) -> str:
        prompt = (
            f"[收尾原因] {reason}\n"
            f"[已执行轮次] {iteration}\n\n" + _INCOMPLETE_SUMMARY_PROMPT
        )
        prompt_message = {"role": "user", "content": prompt}
        messages.append(prompt_message)
        try:
            resp = await self._provider_chat(
                gate,
                messages=messages,
                tools=[],
                max_tokens=(
                    min(_SUMMARY_MAX_TOKENS, self._max_tokens)
                    if self._max_tokens > 0
                    else _SUMMARY_MAX_TOKENS
                ),
            )
            text = (resp.content or "").strip()
            if text:
                return text
        except ContextCompactionError:
            raise
        except Exception as e:
            logger.warning("[subagent] 生成收尾总结失败: %s", e)
        finally:
            if not messages or messages[-1] != prompt_message:
                raise RuntimeError("subagent summary prompt projection 不一致")
            messages.pop()
            gate.sync_pending(messages)
        return "本轮步骤预算已用完：已完成部分关键步骤，但仍有未完成项，下一轮将从当前检查点继续推进。"

    async def _force_final_summary(
        self,
        messages: list[dict[str, Any]],
        gate: _SubagentContextGate,
        *,
        reason: str,
        iteration: int,
    ) -> str:
        prompt = (
            f"[结束原因] {reason}\n"
            f"[已执行任务轮次] {iteration}\n\n" + _FORCED_FINAL_SUMMARY_PROMPT
        )
        prompt_message = {"role": "user", "content": prompt}
        messages.append(prompt_message)
        try:
            resp = await self._provider_chat(
                gate,
                messages=messages,
                tools=[],
                max_tokens=(
                    min(_SUMMARY_MAX_TOKENS, self._max_tokens)
                    if self._max_tokens > 0
                    else _SUMMARY_MAX_TOKENS
                ),
            )
            text = (resp.content or "").strip()
            if text:
                self.last_exit_reason = "forced_summary"
                return text
        except ContextCompactionError:
            raise
        except Exception as e:
            logger.warning("[subagent] 强制最终总结失败: %s", e)
        finally:
            if not messages or messages[-1] != prompt_message:
                raise RuntimeError("subagent final summary prompt projection 不一致")
            messages.pop()
            gate.sync_pending(messages)
        self.last_exit_reason = "forced_summary_fallback"
        return _FORCED_FINAL_SUMMARY_FALLBACK

    async def _run_mandatory_exit(
        self,
        messages: list[dict[str, Any]],
        session_key: str,
        gate: _SubagentContextGate,
    ) -> None:
        """强制收尾：逐个调用 mandatory_exit_tools 中的工具。"""
        for tool_name in self._mandatory_exit_tools:
            if tool_name not in self._tool_map:
                continue
            prompt = _CLEANUP_PROMPT.format(tool_name=tool_name)
            messages.append({"role": "user", "content": prompt})
            try:
                response = await self._provider_chat(
                    gate,
                    messages=messages,
                    tools=self._tool_schemas,
                    max_tokens=self._max_tokens,
                    tool_choice={"type": "function", "function": {"name": tool_name}},
                )
            except ContextCompactionError:
                raise
            except Exception as e:
                logger.error("[subagent] mandatory_exit %s 调用失败: %s", tool_name, e)
                continue

            if not response.tool_calls:
                self.last_exit_reason = "error"
                raise RuntimeError(
                    "mandatory_exit 未调用指定工具: "
                    f"expected={tool_name} actual=none"
                )

            tc = response.tool_calls[0]
            if tc.name != tool_name:
                self.last_exit_reason = "error"
                raise RuntimeError(
                    "mandatory_exit 调用了错误工具: "
                    f"expected={tool_name} actual={tc.name}"
                )
            append_assistant_tool_calls(
                messages,
                content=response.content,
                tool_calls=[tc],
                provider_fields=response.provider_fields,
            )
            exec_result = await self._execute_tool_call(
                tc.id,
                tc.name,
                tc.arguments,
                session_key=session_key,
            )
            normalized = normalize_tool_result(exec_result.output)
            logger.info(
                "[subagent] mandatory_exit %s 结果: %s",
                tc.name,
                normalized.preview()[:120],
            )
            append_tool_result(
                messages,
                tool_call_id=tc.id,
                content=normalized,
                tool_name=tc.name,
                execution_status=exec_result.status,
            )
            gate.record_completed_batch(messages)

    async def _execute_tool_call(
        self,
        call_id: str,
        tool_name: str,
        arguments: dict[str, Any],
        *,
        session_key: str = "",
        tool_batch: tuple[dict[str, Any], ...] = (),
        tool_batch_index: int = 0,
    ):
        tool = self._tool_map.get(tool_name)
        if tool is None:
            return ToolExecutionResult(
                status="error",
                output=f"未知工具: {tool_name}",
                final_arguments=dict(arguments),
            )

        async def _invoke(name: str, kwargs: dict[str, Any]):
            inherited_context = _SUBAGENT_BASE_CONTEXT.get()
            run_turn_id = _SUBAGENT_TURN_ID.get()
            if inherited_context is None or run_turn_id is None:
                raise RuntimeError("subagent tool execution owner 未建立")
            execution_context = replace(
                inherited_context,
                turn_id=run_turn_id,
                execution_id=new_operation_id(),
            )
            with tool_execution_context_scope(execution_context):
                return await tool.execute(**kwargs)

        return await self._tool_executor.execute(
            ToolExecutionRequest(
                call_id=call_id,
                tool_name=tool_name,
                arguments=arguments,
                source="subagent",
                session_key=session_key,
                tool_batch=tool_batch,
                tool_batch_index=tool_batch_index,
            ),
            _invoke,
        )

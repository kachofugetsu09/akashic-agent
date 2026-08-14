from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Awaitable, Callable

from agent.plugin_composition.events import SerialEventKey
from agent.plugin_composition.model import CompositionError
from agent.tool_hooks.base import ToolHook
from agent.tool_hooks.types import (
    HookContext,
    HookTraceItem,
    ToolExecStatus,
    ToolExecutionRequest,
    ToolExecutionResult,
    ToolSource,
)

ToolInvoker = Callable[[str, dict[str, Any]], Awaitable[Any]]


@dataclass(frozen=True, slots=True)
class ToolExecutionBefore:
    call_id: str
    tool_name: str
    arguments: Mapping[str, Any]
    source: ToolSource
    session_key: str
    channel: str
    chat_id: str


@dataclass(frozen=True, slots=True)
class ToolExecutionAfter:
    call_id: str
    tool_name: str
    final_arguments: Mapping[str, Any]
    source: ToolSource
    session_key: str
    channel: str
    chat_id: str
    status: ToolExecStatus
    output: Any


TOOL_EXECUTION_BEFORE = SerialEventKey[ToolExecutionBefore, str](
    "tool.execute.before_invoker"
)
TOOL_EXECUTION_AFTER = SerialEventKey[ToolExecutionAfter, None](
    "tool.execute.after_pipeline"
)


class HookExecutionError(RuntimeError):
    def __init__(self, hook_name: str, event: str, cause: Exception) -> None:
        self.hook_name = hook_name
        self.event = event
        self.cause = cause
        super().__init__(f"hook {hook_name} ({event}) failed: {cause}")


class ToolExecutor:
    def __init__(self, hooks: Sequence[ToolHook] | None = None) -> None:
        self._hooks = list(hooks or [])

    def add_hooks(self, hooks: Sequence[ToolHook]) -> None:
        self._hooks.extend(hooks)

    def _runtime_hooks(self) -> list[ToolHook]:
        from agent.plugins.snapshot import get_current_runtime_snapshot

        snapshot = get_current_runtime_snapshot()
        if snapshot is not None:
            fixed = [
                hook
                for hook in self._hooks
                if not getattr(hook, "snapshot_managed", False)
            ]
            return [*fixed, *snapshot.tool_hooks]
        return self._hooks

    async def execute(
        self,
        request: ToolExecutionRequest,
        invoker: ToolInvoker,
    ) -> ToolExecutionResult:
        """执行单次工具调用。

        request 描述“这次想调用什么工具、带什么参数”；
        invoker 是真实执行入口（通常是 ToolRegistry.execute）。

        固定流程：
        1. legacy pre hooks：匹配、改参、必要时拒绝
        2. composition before：只允许观察或 Bail，不允许改参
        3. invoker：用最终参数执行真实工具
        4. legacy post hooks 后发出 composition after
        """
        current_arguments = dict(request.arguments)
        extra_messages: list[str] = []
        pre_trace: list[HookTraceItem] = []
        post_trace: list[HookTraceItem] = []

        try:
            # pre_hook 是唯一允许改输入/直接 deny 的阶段。
            denied_reason, current_arguments = await self._run_pre_hooks(
                request=request,
                current_arguments=current_arguments,
                extra_messages=extra_messages,
                traces=pre_trace,
            )
        except HookExecutionError as exc:
            return await self._finish(
                request,
                ToolExecutionResult(
                    status="error",
                    output=f"工具执行出错: {exc}",
                    final_arguments=dict(current_arguments),
                    extra_messages=extra_messages,
                    pre_hook_trace=pre_trace,
                    post_hook_trace=post_trace,
                ),
            )
        final_arguments = dict(current_arguments)
        if denied_reason:
            return await self._finish(
                request,
                ToolExecutionResult(
                    status="denied",
                    output=denied_reason,
                    final_arguments=final_arguments,
                    extra_messages=extra_messages,
                    pre_hook_trace=pre_trace,
                    post_hook_trace=post_trace,
                ),
            )

        try:
            denied_reason = await _run_composition_before(
                request,
                final_arguments,
            )
        except Exception as exc:
            return await self._finish(
                request,
                ToolExecutionResult(
                    status="error",
                    output=f"工具执行出错: tool event failed: {exc}",
                    final_arguments=final_arguments,
                    extra_messages=extra_messages,
                    pre_hook_trace=pre_trace,
                    post_hook_trace=post_trace,
                ),
            )
        if denied_reason:
            return await self._finish(
                request,
                ToolExecutionResult(
                    status="denied",
                    output=denied_reason,
                    final_arguments=final_arguments,
                    extra_messages=extra_messages,
                    pre_hook_trace=pre_trace,
                    post_hook_trace=post_trace,
                ),
            )

        try:
            # 这里才进入真实工具执行；hook 本身不直接替代工具实现。
            output = await invoker(request.tool_name, final_arguments)
        except Exception as exc:
            error_text = str(exc)
            try:
                # 工具自身报错后，允许 post_tool_error 做记录型处理。
                await self._run_post_hooks(
                    HookContext(
                        event="post_tool_error",
                        request=request,
                        current_arguments=final_arguments,
                        error=error_text,
                    ),
                    extra_messages=extra_messages,
                    traces=post_trace,
                )
            except HookExecutionError as hook_exc:
                return await self._finish(
                    request,
                    ToolExecutionResult(
                        status="error",
                        output=f"工具执行出错: {hook_exc}",
                        final_arguments=final_arguments,
                        extra_messages=extra_messages,
                        pre_hook_trace=pre_trace,
                        post_hook_trace=post_trace,
                    ),
                )
            return await self._finish(
                request,
                ToolExecutionResult(
                    status="error",
                    output=f"工具执行出错: {error_text}",
                    final_arguments=final_arguments,
                    extra_messages=extra_messages,
                    pre_hook_trace=pre_trace,
                    post_hook_trace=post_trace,
                ),
            )

        try:
            # post_tool_use 只做观察和补充信息，不回写执行参数。
            await self._run_post_hooks(
                HookContext(
                    event="post_tool_use",
                    request=request,
                    current_arguments=final_arguments,
                    result=output,
                ),
                extra_messages=extra_messages,
                traces=post_trace,
                fail_open=True,
            )
        except HookExecutionError as exc:
            return await self._finish(
                request,
                ToolExecutionResult(
                    status="error",
                    output=f"工具执行出错: {exc}",
                    final_arguments=final_arguments,
                    extra_messages=extra_messages,
                    pre_hook_trace=pre_trace,
                    post_hook_trace=post_trace,
                ),
            )
        return await self._finish(
            request,
            ToolExecutionResult(
                status="success",
                output=output,
                final_arguments=final_arguments,
                extra_messages=extra_messages,
                pre_hook_trace=pre_trace,
                post_hook_trace=post_trace,
            ),
        )

    async def _finish(
        self,
        request: ToolExecutionRequest,
        result: ToolExecutionResult,
    ) -> ToolExecutionResult:
        """Emit the settled pipeline result and expose listener failures."""

        try:
            await _run_composition_after(request, result)
        except Exception as exc:
            return ToolExecutionResult(
                status="error",
                output=f"工具执行出错: tool event failed: {exc}",
                final_arguments=dict(result.final_arguments),
                extra_messages=result.extra_messages,
                pre_hook_trace=result.pre_hook_trace,
                post_hook_trace=result.post_hook_trace,
            )
        return result

    async def preflight(
        self,
        request: ToolExecutionRequest,
    ) -> ToolExecutionResult:
        current_arguments = dict(request.arguments)
        extra_messages: list[str] = []
        pre_trace: list[HookTraceItem] = []
        try:
            denied_reason, current_arguments = await self._run_pre_hooks(
                request=request,
                current_arguments=current_arguments,
                extra_messages=extra_messages,
                traces=pre_trace,
            )
        except HookExecutionError as exc:
            return ToolExecutionResult(
                status="error",
                output=f"工具执行出错: {exc}",
                final_arguments=dict(current_arguments),
                extra_messages=extra_messages,
                pre_hook_trace=pre_trace,
            )
        if denied_reason:
            return ToolExecutionResult(
                status="denied",
                output=denied_reason,
                final_arguments=dict(current_arguments),
                extra_messages=extra_messages,
                pre_hook_trace=pre_trace,
            )
        return ToolExecutionResult(
            status="success",
            output="",
            final_arguments=dict(current_arguments),
            extra_messages=extra_messages,
            pre_hook_trace=pre_trace,
        )

    async def _run_pre_hooks(
        self,
        *,
        request: ToolExecutionRequest,
        current_arguments: dict[str, Any],
        extra_messages: list[str],
        traces: list[HookTraceItem],
    ) -> tuple[str, dict[str, Any]]:
        for hook in self._runtime_hooks():
            if hook.event != "pre_tool_use":
                continue
            ctx = HookContext(
                event="pre_tool_use",
                request=request,
                current_arguments=dict(current_arguments),
            )
            try:
                matched = hook.matches(ctx)
            except Exception as exc:
                raise HookExecutionError(hook.name, hook.event, exc) from exc
            if not matched:
                traces.append(
                    HookTraceItem(
                        hook_name=hook.name,
                        event=hook.event,
                        matched=False,
                    )
                )
                continue
            try:
                outcome = await hook.run(ctx)
            except Exception as exc:
                raise HookExecutionError(hook.name, hook.event, exc) from exc
            if outcome.updated_input is not None:
                current_arguments = dict(outcome.updated_input)
            if outcome.extra_message:
                extra_messages.append(outcome.extra_message)
            traces.append(
                HookTraceItem(
                    hook_name=hook.name,
                    event=hook.event,
                    matched=True,
                    decision=outcome.decision,
                    reason=outcome.reason,
                    extra_message=outcome.extra_message,
                )
            )
            if outcome.decision == "deny":
                reason = outcome.reason.strip() or "工具调用被拦截"
                return reason, current_arguments
        return "", current_arguments

    async def _run_post_hooks(
        self,
        ctx: HookContext,
        *,
        extra_messages: list[str],
        traces: list[HookTraceItem],
        fail_open: bool = False,
    ) -> None:
        for hook in self._runtime_hooks():
            if hook.event != ctx.event:
                continue
            try:
                matched = hook.matches(ctx)
            except Exception as exc:
                if fail_open:
                    traces.append(
                        HookTraceItem(
                            hook_name=hook.name,
                            event=hook.event,
                            matched=False,
                            reason=f"hook failed: {exc}",
                        )
                    )
                    continue
                raise HookExecutionError(hook.name, hook.event, exc) from exc
            if not matched:
                traces.append(
                    HookTraceItem(
                        hook_name=hook.name,
                        event=hook.event,
                        matched=False,
                    )
                )
                continue
            try:
                outcome = await hook.run(ctx)
            except Exception as exc:
                if fail_open:
                    traces.append(
                        HookTraceItem(
                            hook_name=hook.name,
                            event=hook.event,
                            matched=True,
                            reason=f"hook failed: {exc}",
                        )
                    )
                    continue
                raise HookExecutionError(hook.name, hook.event, exc) from exc
            if outcome.extra_message:
                extra_messages.append(outcome.extra_message)
            traces.append(
                HookTraceItem(
                    hook_name=hook.name,
                    event=hook.event,
                    matched=True,
                    decision=outcome.decision,
                    reason=outcome.reason,
                    extra_message=outcome.extra_message,
                )
            )


def _composition_context() -> Any | None:
    from agent.plugins.snapshot import get_current_runtime_snapshot

    snapshot = get_current_runtime_snapshot()
    if snapshot is None or snapshot.composition_root is None:
        return None
    return snapshot.composition_root.context


async def _run_composition_before(
    request: ToolExecutionRequest,
    final_arguments: Mapping[str, Any],
) -> str:
    """Dispatch the Core-owned pre-invoker event without exposing mutation."""

    context = _composition_context()
    if context is None:
        return ""
    outcome = await context.serial(
        TOOL_EXECUTION_BEFORE,
        ToolExecutionBefore(
            call_id=request.call_id,
            tool_name=request.tool_name,
            arguments=MappingProxyType(dict(final_arguments)),
            source=request.source,
            session_key=request.session_key,
            channel=request.channel,
            chat_id=request.chat_id,
        ),
    )
    if outcome is None:
        return ""
    reason = outcome.value
    if not isinstance(reason, str) or not reason.strip():
        raise CompositionError(
            "INVALID_TOOL_DENIAL",
            "工具执行事件的 Bail 必须携带非空字符串原因",
        )
    return reason.strip()


async def _run_composition_after(
    request: ToolExecutionRequest,
    result: ToolExecutionResult,
) -> None:
    """Dispatch the Core-owned settled event and reject meaningless Bail."""

    context = _composition_context()
    if context is None:
        return
    outcome = await context.serial(
        TOOL_EXECUTION_AFTER,
        ToolExecutionAfter(
            call_id=request.call_id,
            tool_name=request.tool_name,
            final_arguments=MappingProxyType(dict(result.final_arguments)),
            source=request.source,
            session_key=request.session_key,
            channel=request.channel,
            chat_id=request.chat_id,
            status=result.status,
            output=result.output,
        ),
    )
    if outcome is not None:
        raise CompositionError(
            "TOOL_AFTER_BAIL_NOT_ALLOWED",
            "工具执行完成事件不接受 Bail",
        )

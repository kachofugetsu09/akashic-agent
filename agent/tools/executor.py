from __future__ import annotations

from typing import TYPE_CHECKING, Any, Awaitable, Callable

from agent.plugin_composition import CompositionError
from agent.tools.events import (
    TOOL_EXECUTION_AUTHORIZE,
    TOOL_INPUT_PREPARE,
    TOOL_RESULT,
    ToolExecutionRequest,
    ToolExecutionResult,
    ToolInput,
    ToolResult,
)

if TYPE_CHECKING:
    from agent.plugin_composition import CompositionSnapshotRoot

ToolInvoker = Callable[[str, dict[str, Any]], Awaitable[Any]]


class ToolExecutor:
    """Run typed tool admission, invocation, and result observation in order."""

    async def execute(
        self,
        request: ToolExecutionRequest,
        invoker: ToolInvoker,
    ) -> ToolExecutionResult:
        """Prepare, authorize, invoke, and settle one tool call."""

        root = self._runtime_composition_root()
        admission = await self._admit(root, request)
        if admission.status != "success":
            return await self._settle(root, request, admission)

        # Admission 成功后只由 invoker 执行真实工具。
        try:
            output = await invoker(request.tool_name, admission.final_arguments)
        except Exception as exc:
            result = self._error_result(admission.final_arguments, exc)
        else:
            result = ToolExecutionResult(
                status="success",
                output=output,
                final_arguments=admission.final_arguments,
            )
        return await self._settle(root, request, result)

    async def preflight(
        self,
        request: ToolExecutionRequest,
    ) -> ToolExecutionResult:
        """Run typed admission without invoking a tool or publishing a result."""

        return await self._admit(self._runtime_composition_root(), request)

    async def _admit(
        self,
        root: CompositionSnapshotRoot | None,
        request: ToolExecutionRequest,
    ) -> ToolExecutionResult:
        """Own the single grant, prepare, and authorize state machine."""

        current_arguments = dict(request.arguments)
        if not request.grant.allows(request.tool_name):
            return ToolExecutionResult(
                status="denied",
                output=f"工具未被当前 Turn 授权: {request.tool_name}",
                final_arguments=current_arguments,
            )

        try:
            current_arguments = await self._run_input_prepare(
                root,
                request,
                current_arguments,
            )
        except Exception as exc:
            return self._error_result(current_arguments, exc)

        final_arguments = dict(current_arguments)
        try:
            denied_reason = await self._run_execution_authorize(
                root,
                request,
                final_arguments,
            )
        except Exception as exc:
            return self._error_result(final_arguments, exc)
        if denied_reason:
            return ToolExecutionResult(
                status="denied",
                output=denied_reason,
                final_arguments=final_arguments,
            )
        return ToolExecutionResult(
            status="success",
            output="",
            final_arguments=final_arguments,
        )

    @staticmethod
    def _error_result(
        final_arguments: dict[str, Any],
        cause: Exception,
    ) -> ToolExecutionResult:
        return ToolExecutionResult(
            status="error",
            output=f"工具执行出错: {cause}",
            final_arguments=dict(final_arguments),
        )

    @staticmethod
    def _runtime_composition_root() -> CompositionSnapshotRoot | None:
        from agent.plugins.snapshot import get_current_runtime_snapshot

        snapshot = get_current_runtime_snapshot()
        return snapshot.composition_root if snapshot is not None else None

    async def _run_input_prepare(
        self,
        root: CompositionSnapshotRoot | None,
        request: ToolExecutionRequest,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        if root is None:
            return arguments
        original = ToolInput.from_request(request, arguments)
        prepared = await root.context.transform(TOOL_INPUT_PREPARE, original)
        if not prepared.same_call(original):
            raise CompositionError(
                "TOOL_INPUT_IDENTITY_CHANGED",
                "tool.input.prepare 只能通过 with_arguments() 修改参数",
            )
        return prepared.mutable_arguments()

    async def _run_execution_authorize(
        self,
        root: CompositionSnapshotRoot | None,
        request: ToolExecutionRequest,
        arguments: dict[str, Any],
    ) -> str:
        if root is None:
            return ""
        tool_input = ToolInput.from_request(request, arguments)
        decision = await root.context.serial(
            TOOL_EXECUTION_AUTHORIZE,
            tool_input,
        )
        if decision is None:
            return ""
        return decision.value.strip() or "工具调用被拦截"

    async def _settle(
        self,
        root: CompositionSnapshotRoot | None,
        request: ToolExecutionRequest,
        result: ToolExecutionResult,
    ) -> ToolExecutionResult:
        if root is not None:
            await root.context.observe(
                TOOL_RESULT,
                ToolResult.from_execution(request, result),
            )
        return result

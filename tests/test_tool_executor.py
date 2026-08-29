from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import replace
from typing import Any, AsyncIterator

import pytest

from agent.control.turn_scope import ToolGrant
from agent.plugin_composition import Bail, CompositionRoot
from agent.plugins.snapshot import (
    RuntimeSnapshotCompiler,
    RuntimeSnapshotStore,
    bind_runtime_snapshot,
    reset_runtime_snapshot,
)
from agent.tools.events import (
    TOOL_EXECUTION_AUTHORIZE,
    TOOL_INPUT_PREPARE,
    TOOL_RESULT,
    ToolExecutionRequest,
    ToolInput,
    ToolResult,
)
from agent.tools.executor import ToolExecutor


async def _invoke(tool_name: str, arguments: dict[str, Any]) -> Any:
    return {"tool": tool_name, "arguments": dict(arguments)}


@asynccontextmanager
async def _bound_root(root: CompositionRoot) -> AsyncIterator[None]:
    store = RuntimeSnapshotStore()
    store.install(RuntimeSnapshotCompiler().compile({}, composition_root=root))
    lease = store.lease()
    token = bind_runtime_snapshot(lease)
    try:
        yield
    finally:
        reset_runtime_snapshot(token)
        await lease.release()
        await store.close()


@pytest.mark.parametrize("source", ["passive", "subagent"])
@pytest.mark.asyncio
async def test_tool_executor_runs_typed_events_in_order(source: str) -> None:
    order: list[str] = []
    observed: list[ToolResult] = []
    root = CompositionRoot("typed-tool-events")

    async def composition(ctx) -> None:
        def prepare(tool_input: ToolInput) -> ToolInput:
            order.append("prepare")
            arguments = tool_input.mutable_arguments()
            arguments["command"] = str(arguments["command"]).replace(
                "rm ", "mv ", 1
            )
            return tool_input.with_arguments(arguments)

        def authorize(tool_input: ToolInput) -> None:
            order.append("authorize")
            assert tool_input.arguments["command"] == "mv file.txt"
            return None

        _ = await ctx.on(TOOL_INPUT_PREPARE, prepare)
        _ = await ctx.on(TOOL_EXECUTION_AUTHORIZE, authorize)

        def observe(result: ToolResult) -> None:
            order.append("result")
            observed.append(result)

        _ = await ctx.on(TOOL_RESULT, observe)

    _ = await root.mount(composition, name="composition")

    async def invoke(tool_name: str, arguments: dict[str, Any]) -> str:
        order.append("invoke")
        return f"{tool_name}:{arguments['command']}"

    async with _bound_root(root):
        result = await ToolExecutor().execute(
            ToolExecutionRequest(
                call_id="call-1",
                tool_name="shell",
                arguments={"command": "rm file.txt"},
                source=source,
                session_key=f"{source}:session",
            ),
            invoke,
        )

    assert result.status == "success"
    assert result.final_arguments == {"command": "mv file.txt"}
    assert order == ["prepare", "authorize", "invoke", "result"]
    assert len(observed) == 1
    assert observed[0].status == "success"
    assert observed[0].arguments == {"command": "mv file.txt"}


@pytest.mark.asyncio
async def test_turn_tool_grant_denies_before_plugin_hooks_and_invocation() -> None:
    order: list[str] = []
    root = CompositionRoot("tool-grant")

    async def composition(ctx) -> None:
        _ = await ctx.on(TOOL_INPUT_PREPARE, lambda item: order.append("prepare") or item)
        _ = await ctx.on(TOOL_EXECUTION_AUTHORIZE, lambda _: order.append("authorize"))
        _ = await ctx.on(TOOL_RESULT, lambda _: order.append("result"))

    _ = await root.mount(composition, name="listeners")

    async def invoke(_: str, __: dict[str, Any]) -> str:
        order.append("invoke")
        return "unreachable"

    request = ToolExecutionRequest(
        call_id="grant-denied",
        tool_name="shell",
        arguments={"command": "pwd"},
        source="passive",
        grant=ToolGrant.only(("read_file",)),
    )
    async with _bound_root(root):
        result = await ToolExecutor().execute(request, invoke)
        preflight = await ToolExecutor().preflight(request)

    assert result.status == "denied"
    assert preflight.status == "denied"
    assert order == ["result"]


@pytest.mark.asyncio
async def test_v3_authorize_denial_is_observed_without_invoking() -> None:
    observed: list[ToolResult] = []
    invoked = False
    root = CompositionRoot("tool-authorize-deny")

    async def composition(ctx) -> None:
        _ = await ctx.on(
            TOOL_EXECUTION_AUTHORIZE,
            lambda _: Bail("blocked by v3"),
        )
        _ = await ctx.on(TOOL_RESULT, observed.append)

    _ = await root.mount(composition, name="authorizer")

    async def invoke(_: str, __: dict[str, Any]) -> str:
        nonlocal invoked
        invoked = True
        return "unreachable"

    async with _bound_root(root):
        result = await ToolExecutor().execute(
            ToolExecutionRequest(
                call_id="call-2",
                tool_name="shell",
                arguments={"command": "sudo pacman -S pkg"},
                source="passive",
            ),
            invoke,
        )

    assert result.status == "denied"
    assert result.output == "blocked by v3"
    assert invoked is False
    assert [item.status for item in observed] == ["denied"]


@pytest.mark.asyncio
async def test_typed_prepare_failure_records_incident_and_settles_error() -> None:
    observed: list[ToolResult] = []
    root = CompositionRoot("tool-prepare-failure")

    def fail(_: ToolInput) -> ToolInput:
        raise RuntimeError("prepare failed")

    async def composition(ctx) -> None:
        _ = await ctx.on(TOOL_INPUT_PREPARE, fail)
        _ = await ctx.on(TOOL_RESULT, observed.append)

    _ = await root.mount(composition, name="preparer")

    async with _bound_root(root):
        result = await ToolExecutor().execute(
            ToolExecutionRequest(
                call_id="call-3",
                tool_name="dummy",
                arguments={"x": 1},
                source="passive",
            ),
            _invoke,
        )
        incidents = root.receipt().incidents

    assert result.status == "error"
    assert "prepare failed" in str(result.output)
    assert [item.status for item in observed] == ["error"]
    assert (incidents[-1].owner, incidents[-1].kind) == (
        "preparer",
        "transform_failure",
    )


@pytest.mark.asyncio
async def test_typed_authorize_failure_records_incident_and_settles_error() -> None:
    observed: list[ToolResult] = []
    root = CompositionRoot("tool-authorize-failure")

    def fail(_: ToolInput) -> None:
        raise RuntimeError("authorize failed")

    async def composition(ctx) -> None:
        _ = await ctx.on(TOOL_EXECUTION_AUTHORIZE, fail)
        _ = await ctx.on(TOOL_RESULT, observed.append)

    _ = await root.mount(composition, name="authorizer")

    async with _bound_root(root):
        result = await ToolExecutor().execute(
            ToolExecutionRequest(
                call_id="call-authorize-failure",
                tool_name="dummy",
                arguments={"x": 1},
                source="passive",
            ),
            _invoke,
        )
        incidents = root.receipt().incidents

    assert result.status == "error"
    assert "authorize failed" in str(result.output)
    assert [item.status for item in observed] == ["error"]
    assert (incidents[-1].owner, incidents[-1].kind) == (
        "authorizer",
        "serial_failure",
    )


@pytest.mark.asyncio
async def test_typed_authorize_rejects_invalid_bail_with_owner_incident() -> None:
    observed: list[ToolResult] = []
    invoked = False
    root = CompositionRoot("tool-authorize-invalid-bail")

    async def composition(ctx) -> None:
        _ = await ctx.on(TOOL_EXECUTION_AUTHORIZE, lambda _: Bail(7))
        _ = await ctx.on(TOOL_RESULT, observed.append)

    _ = await root.mount(composition, name="bad-authorizer")
    assert root.topology_view().listeners == (
        "serial:tool.execution.authorize"
        "[bail=akashic.tool-deny-reason.v1]:bad-authorizer",
        "observe:tool.result:bad-authorizer",
    )

    async def invoke(_: str, __: dict[str, Any]) -> str:
        nonlocal invoked
        invoked = True
        return "unreachable"

    async with _bound_root(root):
        result = await ToolExecutor().execute(
            ToolExecutionRequest(
                call_id="call-invalid-bail",
                tool_name="dummy",
                arguments={"x": 1},
                source="passive",
            ),
            invoke,
        )
        incidents = root.receipt().incidents

    assert result.status == "error"
    assert "akashic.tool-deny-reason.v1" in str(result.output)
    assert invoked is False
    assert [item.status for item in observed] == ["error"]
    assert (incidents[-1].owner, incidents[-1].kind) == (
        "bad-authorizer",
        "serial_failure",
    )


def test_tool_input_replace_cannot_bypass_recursive_freeze() -> None:
    request = ToolExecutionRequest(
        call_id="call-replace",
        tool_name="dummy",
        arguments={"nested": [1]},
        source="passive",
    )
    original = ToolInput.from_request(request, request.arguments)
    raw = {"nested": [2]}

    replaced = replace(original, arguments=raw)
    raw["nested"].append(3)

    assert replaced.same_call(original)
    assert replaced.arguments["nested"] == (2,)
    assert replaced.mutable_arguments() == {"nested": [2]}
    with pytest.raises(TypeError, match="JSON"):
        _ = replace(original, arguments={"bad": object()})


@pytest.mark.asyncio
async def test_invoker_error_is_observed_as_one_settled_error() -> None:
    observed: list[ToolResult] = []
    root = CompositionRoot("tool-invoker-error")

    async def composition(ctx) -> None:
        _ = await ctx.on(TOOL_RESULT, observed.append)

    _ = await root.mount(composition, name="observer")

    async def broken(_: str, __: dict[str, Any]) -> str:
        raise RuntimeError("invoker failed")

    async with _bound_root(root):
        result = await ToolExecutor().execute(
            ToolExecutionRequest(
                call_id="call-invoker-error",
                tool_name="dummy",
                arguments={"x": 1},
                source="passive",
            ),
            broken,
        )

    assert result.status == "error"
    assert "invoker failed" in str(result.output)
    assert [item.status for item in observed] == ["error"]


@pytest.mark.asyncio
async def test_v3_result_observer_failure_does_not_change_settled_result() -> None:
    root = CompositionRoot("tool-result-failure")

    def fail(result: ToolResult) -> None:
        result.arguments["nested"]["items"].append(2)

    async def composition(ctx) -> None:
        _ = await ctx.on(TOOL_RESULT, fail)

    _ = await root.mount(composition, name="observer")

    async with _bound_root(root):
        result = await ToolExecutor().execute(
            ToolExecutionRequest(
                call_id="call-4",
                tool_name="dummy",
                arguments={"nested": {"items": [1]}},
                source="passive",
            ),
            _invoke,
        )
        incidents = root.receipt().incidents

    assert result.status == "success"
    assert result.final_arguments == {"nested": {"items": [1]}}
    assert (incidents[-1].owner, incidents[-1].kind) == (
        "observer",
        "observer_failure",
    )


@pytest.mark.asyncio
async def test_preflight_runs_typed_admission_without_publishing_result() -> None:
    observed: list[ToolResult] = []
    root = CompositionRoot("tool-preflight")

    async def composition(ctx) -> None:
        _ = await ctx.on(
            TOOL_INPUT_PREPARE,
            lambda item: item.with_arguments({"x": 2}),
        )
        _ = await ctx.on(
            TOOL_EXECUTION_AUTHORIZE,
            lambda item: Bail("x denied") if item.arguments["x"] == 2 else None,
        )
        _ = await ctx.on(TOOL_RESULT, observed.append)

    _ = await root.mount(composition, name="preflight")

    async with _bound_root(root):
        result = await ToolExecutor().preflight(
            ToolExecutionRequest(
                call_id="call-5",
                tool_name="dummy",
                arguments={"x": 1},
                source="passive",
            )
        )

    assert result.status == "denied"
    assert result.final_arguments == {"x": 2}
    assert observed == []

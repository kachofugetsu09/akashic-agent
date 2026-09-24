from __future__ import annotations

import inspect
from pathlib import Path
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import TypeVar, cast

Snapshot = TypeVar("Snapshot")
Result = TypeVar("Result")


def assert_rows_unchanged(
    before: Sequence[tuple[object, ...]],
    after: Sequence[tuple[object, ...]],
    *,
    state_name: str,
) -> None:
    """断言权威状态中的既有行没有被删改。"""
    if list(after) != list(before):
        raise AssertionError(f"{state_name} 的既有行发生删改")


def assert_no_forbidden_writes(
    statements: Sequence[str],
    *,
    tables: Sequence[str],
) -> None:
    """断言执行轨迹没有删改受保护表。"""
    protected = {table.casefold() for table in tables}
    violations: list[str] = []
    for statement in statements:
        normalized = " ".join(statement.casefold().split())
        if not normalized.startswith(("delete ", "update ", "replace ")):
            continue
        if any(table in normalized for table in protected):
            violations.append(statement)
    if violations:
        raise AssertionError(f"检测到受保护状态删改: {violations}")


async def assert_call_finality(
    invoke: Callable[[], Awaitable[Result]],
    observe: Callable[[], Snapshot | Awaitable[Snapshot]],
    *,
    expected: Snapshot,
) -> Result:
    """断言普通调用成功返回时，承诺状态已经可由正式读取入口观察。"""

    # 1. 等待正式调用完成，不能把排队或后台启动当成完成。
    result = await invoke()

    # 2. 立即从独立读取入口核对终态，不提供额外等待窗口。
    snapshot = observe()
    if inspect.isawaitable(snapshot):
        snapshot = await cast(Awaitable[Snapshot], snapshot)
    else:
        snapshot = cast(Snapshot, snapshot)
    if snapshot != expected:
        raise AssertionError(
            f"调用已返回，但承诺状态不可见: expected={expected!r}, actual={snapshot!r}"
        )
    return result


def assert_process_resources_released(
    *,
    live_descendant_pids: Sequence[int],
    listening_ports: Sequence[int],
) -> None:
    """断言 owned process leader 退出后没有后代或监听端口残留。"""
    if live_descendant_pids or listening_ports:
        raise AssertionError(
            "owned process tree 仍持有运行资源: "
            f"pids={list(live_descendant_pids)}, ports={list(listening_ports)}"
        )


def assert_committed_turn_finality(
    *,
    status: str,
    final_response: str | None,
    dispatch_count: int,
) -> None:
    """断言后置 cleanup 不能反向改变已经提交的 turn。"""
    if status != "completed" or final_response is None or dispatch_count != 1:
        raise AssertionError(
            "cleanup 反向破坏已提交 turn: "
            f"status={status!r}, final_response={final_response!r}, "
            f"dispatch_count={dispatch_count}"
        )


def assert_unconfirmed_cleanup_retains_ownership(
    *,
    cleanup_confirmed: bool,
    tracked_execution_ids: Sequence[int],
) -> None:
    """断言未确认清理时 execution 仍由 supervisor 跟踪。"""
    if not cleanup_confirmed and not tracked_execution_ids:
        raise AssertionError("cleanup 未确认却提前丢失 execution ownership")


def assert_snapshot_fields(
    snapshot: Mapping[str, object],
    expected: Mapping[str, object],
) -> None:
    """核对场景声明的完整观察字段。"""
    actual = {key: snapshot.get(key) for key in expected}
    if actual != dict(expected):
        raise AssertionError(
            f"状态快照不匹配: expected={expected!r}, actual={actual!r}"
        )


def assert_recursive_plugin_self_validation(observation: Mapping[str, object]) -> None:
    """Check one real selection, message, tool, and delivery observation."""

    def section(name: str) -> Mapping[str, object]:
        value = observation[name]
        if not isinstance(value, Mapping):
            raise AssertionError(f"{name} 缺少真实 owner 观察")
        return cast(Mapping[str, object], value)

    # 1. Selection owns committed input; the live Root and Fibers own execution.
    selection = section("selection")
    before = selection["before_ref"]
    committed = selection["committed_ref"]
    if not isinstance(before, str) or not before:
        raise AssertionError("缺少初始持久 selection")
    if selection["after_compile_ref"] != before or selection["old_fiber_after_compile"] != "active":
        raise AssertionError("编译失败改变了 selection 或旧 Fiber")
    if not isinstance(committed, str) or committed == before:
        raise AssertionError("有效更新未提交新的 selection")
    if selection["root_before"] != selection["root_after"]:
        raise AssertionError("局部更新替换了 live Root")
    if selection["new_fiber"] != "active" or selection["old_scope_closed"] is not True:
        raise AssertionError("新 Fiber 未 ACTIVE 或旧 Scope 未关闭")

    # 2. A selected failure remains failed until a real retry closes its owner.
    failure = section("failure")
    if failure["selected_ref"] == failure["previous_ref"]:
        raise AssertionError("失败 B 没有成为持久 selected input")
    if failure["failed_state"] != "failed" or failure["failed_fiber"] == "active":
        raise AssertionError("selected B 未 ACTIVE 却被报告为 active")
    if failure["scope_retained"] is not True or failure["old_a_closed"] is not True:
        raise AssertionError("失败 B 或旧 A 的 cleanup owner 不准确")
    if failure["recovered_before_retry"] is True:
        raise AssertionError("失败 B 被误报为 A 已恢复")
    if failure["retry_state"] != "active" or failure["failed_scope_closed_after_retry"] is not True:
        raise AssertionError("显式 retry 未建立活动 Fiber 并清理失败 owner")
    if failure["root_before"] != failure["root_after"]:
        raise AssertionError("恢复替换了 live Root")

    # 3. Admission fixes learning policy while both Sessions keep their messages.
    programmatic = section("programmatic")
    if (programmatic["excluded_learning"] != "excluded"
            or programmatic["false_learning"] != "excluded"
            or programmatic["eligible_learning"] != "eligible"):
        raise AssertionError("Programmatic admission 学习资格错误")
    if programmatic["default_retry_equal"] is not True or programmatic["conflict_rejected"] is not True:
        raise AssertionError("Programmatic admission 重试或冲突未按固定资格结算")
    if programmatic["parent_open_during_validation"] is not True:
        raise AssertionError("验证被跨 Session 工作阻塞")
    if programmatic["validation_terminal"] != "complete" or programmatic["parent_terminal"] != "complete":
        raise AssertionError("Programmatic terminal 未完整保存")
    if programmatic["validation_bodies"] != ("Input", "Output"):
        raise AssertionError("验证 Session 消息未完整保存")

    # 4. Child work and the caller receipt both have durable independent owners.
    child = section("child")
    if child["parent_open_during_child"] is not True or child["call_outcome"] != "success":
        raise AssertionError("child 与 parent 未在真实并发窗口进展")
    if child["child_bodies"] != ("Input", "Output", "ToolResult", "Output"):
        raise AssertionError("child history 未完整保存")
    if child["domain_file"] != "once" or child["caller_tool_results"] != 1:
        raise AssertionError("工具完成没有领域效果与 caller receipt")
    if child["parent_terminal"] != "complete":
        raise AssertionError("父 Session terminal 未保存")

    # 5. Push has one caller ToolExecution receipt and one separate target Output.
    push = section("push")
    if push["tool_outcome"] != "success" or push["tool_phase"] != "done":
        raise AssertionError("MessagePush caller 工具未完成")
    if push["delivery_phase"] != "delivered":
        raise AssertionError("MessagePush Delivery 未确认")
    if push["target_turn_open_across_push"] is not True or push["target_input_unchanged"] is not True:
        raise AssertionError("MessagePush 改写了目标未完成 Turn")
    if push["new_target_bodies"] != (("Output", "message_push"),):
        raise AssertionError("目标 Session 未恰好增加独立 MessagePush Output")
    if push["push_message_id_match"] is not True:
        raise AssertionError("MessagePush Delivery 与目标 Output 身份不一致")
    if push["repeat_same_receipt"] is not True or push["send_count"] != 1:
        raise AssertionError("MessagePush 重试重复发送或更换工具回执")


def assert_isolated_gate_paths(
    *,
    sandbox: Path,
    workspace: Path,
    plugin_home: Path,
    config: Path,
) -> None:
    """断言 Gate 的所有可写输入都位于本次一次性 sandbox。"""
    root = sandbox.resolve()
    resolved = {
        "workspace": workspace.resolve(),
        "plugin_home": plugin_home.resolve(),
        "config": config.resolve(),
    }
    for name, path in resolved.items():
        if not path.is_relative_to(root):
            raise AssertionError(f"{name} 逃逸 Gate sandbox: {path}")
    if len(set(resolved.values())) != len(resolved):
        raise AssertionError("workspace、plugin home 和 config 必须彼此隔离")


def assert_paths_retained(paths: Sequence[Path], *, operation: str) -> None:
    """断言普通生命周期操作没有物理删除用户数据。"""
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise AssertionError(f"{operation} 越权删除持久数据: {missing}")


def assert_companion_contract(observation: Mapping[str, object]) -> None:
    """断言 Companion 合同快照没有静默丢失、越权删除或错误升级。"""

    # 1. 失败必须属于公开分类，且可恢复项不能宣称 runtime 已退出。
    allowed = {
        "operation_rejected",
        "item_quarantined",
        "degraded_continuation",
        "unit_failed",
        "cleanup_degraded",
        "runtime_fatal",
    }
    failure = observation.get("failure_semantics")
    if not isinstance(failure, str) or failure not in allowed:
        raise AssertionError(f"未知失败分类: {failure!r}")
    if failure != "runtime_fatal" and observation.get("runtime_alive") is False:
        raise AssertionError("可恢复失败错误结束 Companion runtime")

    # 2. 已提交状态和 live subscriber 必须保持可观察。
    if observation.get("committed_result") is False:
        raise AssertionError("cleanup 或容量处理反向破坏已提交结果")
    if observation.get("live_event_dropped") is True:
        raise AssertionError("replay eviction 丢弃 live subscriber 事件")

    # 3. 物理减少必须具备 owner、恢复证据和明确授权。
    if observation.get("physical_reduction") is True and not all(
        observation.get(key)
        for key in ("physical_reduction_owner", "recovery_evidence")
    ):
        raise AssertionError("物理减少缺少 owner 或恢复证据")


def assert_companion_capacity(observation: Mapping[str, object]) -> None:
    """断言容量拒绝保持既有状态并只影响当前操作。"""
    if observation.get("capacity_rejected") is True:
        if observation.get("existing_state_changed") is True:
            raise AssertionError("容量拒绝改变了既有状态")
        if observation.get("runtime_alive") is False:
            raise AssertionError("容量拒绝错误结束 Companion runtime")


def assert_tool_context_contract(observation: Mapping[str, object]) -> None:
    """断言 runtime provenance 不被模型参数覆盖，显式 target 仍可不同。"""
    if observation.get("origin_overridden") is True:
        raise AssertionError("模型参数覆盖 runtime provenance")
    if observation.get("target_required") is True and not observation.get("target"):
        raise AssertionError("显式 target 缺失")


def assert_external_io_contract(observation: Mapping[str, object]) -> None:
    """断言 spill 结果仍绑定 execution owner。"""
    if observation.get("spill_owner") in (None, ""):
        raise AssertionError("spill 结果缺少 execution owner")
    if observation.get("redirect_validated") is False:
        raise AssertionError("redirect hop 未执行地址校验")


def assert_peer_removed(observation: Mapping[str, object]) -> None:
    """断言 Peer 生产表面已经消失。"""
    if observation.get("peer_route_registered") is True:
        raise AssertionError("Peer route 仍然注册")
    if observation.get("legacy_peer_config") == "enabled":
        raise AssertionError("遗留 Peer 配置被静默启用")


def assert_content_wake_delivery_contract(observation: Mapping[str, object]) -> None:
    """断言 Content 只在真实送达后结算并向原 source 确认。"""
    if observation.get("source_ack_before_delivery") is True:
        raise AssertionError("Content source ack 早于真实送达")
    if observation.get("settled_without_delivery_receipt") is True:
        raise AssertionError("Content 在缺少 durable delivery receipt 时结算")


def assert_schedule_capacity_contract(observation: Mapping[str, object]) -> None:
    """断言第 11 个 Schedule add 不改变已有任务。"""
    active_jobs = observation.get("active_jobs", 0)
    if not isinstance(active_jobs, int):
        raise AssertionError("Schedule active_jobs 不是整数")
    if active_jobs > 10 and observation.get("operation_accepted") is True:
        raise AssertionError("Schedule 超过默认 10 个仍被接受")
    assert_companion_capacity(observation)


def assert_receipt_contract(observation: Mapping[str, object]) -> None:
    """断言高水位清理不会删除仍在有效窗口内的 receipt。"""
    if observation.get("valid_receipt_deleted") is True:
        raise AssertionError("有效 receipt 被提前删除")
    if observation.get("stale_processing_replayed_blindly") is True:
        raise AssertionError("processing receipt 被盲目重放")


def assert_shell_contract(observation: Mapping[str, object]) -> None:
    """断言 cleanup 失败不能改写已提交 turn。"""
    assert_committed_turn_finality(
        status=cast(str, observation.get("status")),
        final_response=cast(str | None, observation.get("final_response")),
        dispatch_count=cast(int, observation.get("dispatch_count", 0)),
    )


def assert_control_replay_contract(observation: Mapping[str, object]) -> None:
    """断言 replay eviction 不丢 live subscriber 的新事件。"""
    if observation.get("live_event_dropped") is True:
        raise AssertionError("replay eviction 丢失 live subscriber 事件")
    if observation.get("expired_without_snapshot") is True:
        raise AssertionError("replay 过期后静默返回空流")


def assert_dashboard_contract(observation: Mapping[str, object]) -> None:
    """断言外部字段不经 innerHTML 进入展示层。"""
    if observation.get("html_sink") is True:
        raise AssertionError("外部 efficiency 值进入 HTML sink")
    if observation.get("invalid_efficiency_display") != "--":
        raise AssertionError("非法 efficiency 未显示 --")

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


def assert_atomic_generation_switch(
    observations: Sequence[tuple[str, str]],
    *,
    previous_generation: str,
    next_generation: str,
) -> None:
    """断言候选准备期间旧 generation 始终可见，提交后才原子切换。"""
    expected = [
        ("before", previous_generation),
        ("candidate_ready", previous_generation),
        ("committed", next_generation),
    ]
    if list(observations) != expected:
        raise AssertionError(
            f"plugin generation 未原子发布: expected={expected!r}, actual={list(observations)!r}"
        )


def assert_recursive_plugin_self_validation(
    observation: Mapping[str, object],
) -> None:
    """断言父 turn 能隔离执行候选、回读证据并安全提交。"""

    assert_recursive_candidate_trajectory(observation)

    stable = observation.get("stable_snapshot")
    candidate = observation.get("candidate_snapshot")

    # crash 恢复不能误晋升，显式 promote 才改变默认 pointer。
    if observation.get("recovered_stable_snapshot") != stable:
        raise AssertionError("candidate crash recovery 改变了 stable")
    if observation.get("recovered_latest_snapshot") not in {stable, candidate}:
        raise AssertionError("crash recovery 恢复出未知 latest identity")
    if observation.get("recovery_promoted_candidate") is not False:
        raise AssertionError("crash recovery 未经 oracle 自动晋升候选")


def assert_recursive_candidate_trajectory(
    observation: Mapping[str, object],
) -> None:
    """断言候选通过预提交 oracle，且显式晋升形成完整终态。"""

    assert_recursive_candidate_ready(observation)

    stable = observation.get("stable_snapshot")
    candidate = observation.get("candidate_snapshot")

    # 1. promote 之后，旧父 lease 仍保持 stable 且 terminal 完整送达。
    if observation.get("parent_terminal_status") != "completed":
        raise AssertionError("父 programmatic terminal 未完整送达调用方")
    push_seq = observation.get("push_send_sequence")
    parent_terminal_seq = observation.get("parent_terminal_sequence")
    if not isinstance(push_seq, int) or not isinstance(parent_terminal_seq, int):
        raise AssertionError("message_push 缺少可比较的投递时序")
    if push_seq >= parent_terminal_seq:
        raise AssertionError("child message_push 等待了父 session 终态")
    if observation.get("parent_runtime_after_promote") != stable:
        raise AssertionError("promote 改写了父 turn 已持有的 stable lease")

    # 2. 显式 promote 才改变 stable，旧 lease 排空后 journal 必须 complete。
    journal_events = observation.get("reload_journal_events_after_promote")
    if not isinstance(journal_events, Sequence) or not journal_events:
        raise AssertionError("reload journal 缺少候选发布轨迹")
    if journal_events[-1] != "complete":
        raise AssertionError("reload journal 未到达 complete")
    if observation.get("stable_after_promote") != candidate:
        raise AssertionError("显式 promote 后 stable 未指向已验证候选")


def assert_recursive_candidate_ready(
    observation: Mapping[str, object],
) -> None:
    """从真实 control、SessionDB 与插件状态轨迹判定候选能否晋升。"""

    stable = observation.get("stable_snapshot")
    candidate = observation.get("candidate_snapshot")
    if not isinstance(stable, str) or not stable:
        raise AssertionError("stable snapshot identity 缺失")
    if not isinstance(candidate, str) or not candidate or candidate == stable:
        raise AssertionError("candidate snapshot identity 无效")

    # 1. install 返回即代表 latest 可租用，父与普通 turn 仍绑定 stable。
    if observation.get("install_publication_state") != "latest_ready":
        raise AssertionError("plugin-install 返回时 latest 尚不可租用")
    if observation.get("parent_runtime") != stable:
        raise AssertionError("父 turn 的 stable lease 被候选反向替换")
    if observation.get("ordinary_runtime_during_validation") != stable:
        raise AssertionError("普通 turn 看见了未晋升候选")
    if observation.get("validation_runtime") != candidate:
        raise AssertionError("programmatic 验证没有绑定 latest candidate")
    if observation.get("validation_finished_before_parent_release") is not True:
        raise AssertionError("验证仍被跨 session 全局锁阻塞")
    if observation.get("parent_status_before_promote") != "in_progress":
        raise AssertionError("验证完成前父 turn 已释放，无法证明跨 session 并发")

    # 2. 行为成功必须由真实 tool item、领域状态和持久 trace 共同证明。
    raw_turn = observation.get("validation_turn")
    if not isinstance(raw_turn, Mapping):
        raise AssertionError("验证 turn 未从 SessionDB 读取到 completed 终态")
    turn = cast(Mapping[str, object], raw_turn)
    if turn.get("status") != "completed":
        raise AssertionError("验证 turn 未从 SessionDB 读取到 completed 终态")
    raw_metadata = turn.get("metadata")
    if not isinstance(raw_metadata, Mapping):
        raise AssertionError("验证 turn metadata 不是 SessionDB 原始证据")
    metadata = cast(Mapping[str, object], raw_metadata)
    raw_inbound = metadata.get("inboundMetadata")
    if metadata.get("runtime") != "latest" or not isinstance(raw_inbound, Mapping):
        raise AssertionError("验证 turn metadata 未声明 latest 与只读记忆策略")
    inbound = cast(Mapping[str, object], raw_inbound)
    if inbound.get("skip_post_memory") is not True or inbound.get(
        "disable_memory_writes"
    ) is not True:
        raise AssertionError("验证 turn metadata 未声明 latest 与只读记忆策略")
    raw_items: object = turn.get("items")
    if not isinstance(raw_items, list) or not all(
        isinstance(item, Mapping) for item in raw_items
    ):
        raise AssertionError("验证 turn items 不是 SessionDB JSON array")
    items = cast(list[Mapping[str, object]], raw_items)
    candidate_items = [
        item
        for item in items
        if item.get("type") == "toolCall"
        and isinstance(item.get("data"), Mapping)
        and cast(Mapping[str, object], item["data"]).get("name")
        == "candidate_only_tool"
    ]
    if len(candidate_items) != 1:
        raise AssertionError("候选工具缺少真实 completed tool item")
    candidate_data = cast(Mapping[str, object], candidate_items[0]["data"])
    if candidate_data.get("status") != "success":
        raise AssertionError("候选工具缺少真实 completed tool item")
    result = observation.get("candidate_tool_result")
    if not isinstance(result, Mapping):
        raise AssertionError("候选工具结果不是结构化领域证据")
    result = cast(Mapping[str, object], result)
    if result.get("snapshot") != candidate:
        raise AssertionError("候选工具结果不是来自 latest snapshot")
    if result.get("domain") != observation.get("domain_state"):
        raise AssertionError("工具 success 没有对应领域状态证据")
    raw_messages = observation.get("validation_messages")
    if not isinstance(raw_messages, list) or not all(
        isinstance(message, Mapping) for message in raw_messages
    ):
        raise AssertionError("验证 session 的消息或工具 trace 未持久化")
    messages = cast(list[Mapping[str, object]], raw_messages)
    if [message.get("role") for message in messages] != ["user", "assistant"]:
        raise AssertionError("验证 session 的消息或工具 trace 未持久化")
    assistant = cast(Mapping[str, object], messages[-1])
    tool_chain = assistant.get("tool_chain")
    if "candidate_only_tool" not in str(tool_chain):
        raise AssertionError("验证 session 的消息或工具 trace 未持久化")

    # 3. 默认 child 可读历史但不得写语义记忆。
    validation_thread = turn.get("threadId")
    recall_sessions = observation.get("recall_session_keys")
    if not isinstance(recall_sessions, Sequence) or validation_thread not in recall_sessions:
        raise AssertionError("默认验证 session 无法检索既有记忆")
    if observation.get("semantic_memory_write_set") != []:
        raise AssertionError("默认验证 session 写入了语义记忆")

    # 4. message_push 只提交短出站效果，receipt 属于 child trace。
    push_seq = observation.get("push_send_sequence")
    if not isinstance(push_seq, int) or push_seq <= 0:
        raise AssertionError("child message_push 等待了父 session 终态")
    push_items = [
        item
        for item in items
        if item.get("type") == "toolCall"
        and isinstance(item.get("data"), Mapping)
        and cast(Mapping[str, object], item["data"]).get("name") == "message_push"
    ]
    if len(push_items) != 1:
        raise AssertionError("DeliveryReceipt 未写入 child 工具 trace")
    push_data = cast(Mapping[str, object], push_items[0]["data"])
    if push_data.get("status") != "success" or push_data.get("resultPreview") != "消息已发送":
        raise AssertionError("DeliveryReceipt 未写入 child 工具 trace")
    if observation.get("push_target_history_before") != observation.get(
        "push_target_history_after"
    ):
        raise AssertionError("message_push 正文污染了目标 session history")

    # 5. 预提交阶段必须仍是 stable + latest_ready，oracle 不能事后补票。
    if observation.get("stable_before_promote") != stable:
        raise AssertionError("候选验证期间 stable 已提前改变")
    journal_events = observation.get("reload_journal_events_before_promote")
    if not isinstance(journal_events, Sequence) or not journal_events:
        raise AssertionError("reload journal 缺少候选发布轨迹")
    if journal_events[-1] != "latest_ready":
        raise AssertionError("候选在 oracle 前不是 latest_ready")


def assert_plugin_drain_finality(
    *,
    status: str,
    old_generation_lease_count: int,
    old_scope_closed: bool,
    cache_exists: bool,
) -> None:
    """断言 uninstall completed 只表示旧代和代码均已真实排空。"""
    if status == "completed" and (
        old_generation_lease_count != 0 or not old_scope_closed or cache_exists
    ):
        raise AssertionError(
            "插件卸载假报完成: "
            f"leases={old_generation_lease_count}, "
            f"scope_closed={old_scope_closed}, cache_exists={cache_exists}"
        )


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
        observation.get(key) for key in ("physical_reduction_owner", "recovery_evidence")
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


def assert_mcp_reservoir_contract(observation: Mapping[str, object]) -> None:
    """断言单条 MCP quarantine 不会中止合法批次。"""
    if observation.get("quarantine_aborted_batch") is True:
        raise AssertionError("MCP quarantine 错误中止合法批次")
    if observation.get("deleted_before_ack") is True:
        raise AssertionError("MCP item 在 ack 前被删除")


def assert_schedule_capacity_contract(observation: Mapping[str, object]) -> None:
    """断言第 11 个 Schedule add 不改变已有任务。"""
    if observation.get("active_jobs", 0) > 10 and observation.get("operation_accepted") is True:
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

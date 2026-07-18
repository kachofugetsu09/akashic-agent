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

#!/usr/bin/env python3
"""Run the new plugin composition path in one explicit isolated workspace."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import uuid
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, cast

SOURCE_ROOT = Path(__file__).resolve().parents[1]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from agent.plugin_composition import (  # noqa: E402
    CompositionAudit,
    CompositionRoot,
    PluginRuntime,
)
from agent.plugins.snapshot import (  # noqa: E402
    RuntimeSnapshot,
    RuntimeSnapshotCompiler,
    RuntimeSnapshotStore,
)
from examples.plugin_composition.probe import (  # noqa: E402
    PROBE_SIGNAL,
    ProbeConsumer,
    ProbeFormatterProvider,
    ProbeProvider,
    ProbeTrace,
)
from infra.persistence.json_store import atomic_write_text  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="在全新隔离 workspace 中运行插件组合、验证和晋升实验"
    )
    _ = parser.add_argument(
        "--workspace",
        type=Path,
        required=True,
        help="必须尚不存在；脚本创建后保留全部实验证据",
    )
    return parser.parse_args()


def _create_workspace(requested: Path) -> Path:
    """Create only the explicitly named, previously absent workspace."""

    workspace = requested.expanduser().resolve(strict=False)
    source_root = SOURCE_ROOT.resolve(strict=True)
    if workspace == source_root or source_root in workspace.parents:
        raise ValueError("实验 workspace 不能位于源码 worktree 内")
    for protected in _protected_roots():
        if workspace == protected or protected in workspace.parents:
            raise ValueError(f"实验 workspace 不能位于正式状态根内: {protected}")
    if workspace.exists():
        raise FileExistsError(f"实验 workspace 必须尚不存在: {workspace}")
    if not workspace.parent.is_dir():
        raise FileNotFoundError(f"实验 workspace 父目录不存在: {workspace.parent}")
    workspace.mkdir()
    return workspace


def _protected_roots() -> tuple[Path, ...]:
    """Resolve formal workspace and plugin-home roots without importing runtime."""

    configured_workspace = os.environ.get(
        "AKASHIC_WORKSPACE",
        "~/.akashic/workspace",
    )
    configured_plugin_home = os.environ.get(
        "AKASHIC_PLUGIN_HOME",
        "~/.akashic-plugin",
    )
    roots = {
        Path(configured_workspace).expanduser().resolve(strict=False),
        Path(configured_plugin_home).expanduser().resolve(strict=False),
    }
    return tuple(sorted(roots, key=str))


async def _drain_snapshot(snapshot: RuntimeSnapshot) -> None:
    root = snapshot.composition_root
    if root is not None:
        await root.dispose()


async def _run(workspace: Path) -> dict[str, object]:
    """Build, validate, promote, lease, and drain one candidate topology."""

    # 1. Core establishes the run identity and generation-scoped data root.
    run_id = str(uuid.uuid4())
    runtime_dir = workspace / "runtime"
    runtime_dir.mkdir()
    marker = {
        "kind": "plugin-composition-experiment",
        "run_id": run_id,
        "workspace": str(workspace),
    }
    atomic_write_text(
        runtime_dir / "plugin-composition-experiment.json",
        json.dumps(marker, ensure_ascii=False, indent=2) + "\n",
        domain="plugin_composition_experiment",
    )
    audit = CompositionAudit()
    root = CompositionRoot(f"experiment:{run_id}", audit=audit)
    provider_data_root = workspace / "plugin-data" / "probe-provider"
    provider_data_root.mkdir(parents=True)
    provider_runtime = PluginRuntime(
        plugin_id="probe-provider",
        generation_id="composition-experiment",
        plugin_dir=SOURCE_ROOT / "examples" / "plugin_composition",
        data_dir=provider_data_root,
        workspace=workspace,
        config={},
    )

    # 2. New plugins prove required waiting and optional nested injection.
    trace = ProbeTrace()
    consumer_plugin = ProbeConsumer(trace)
    consumer = await root.mount(
        consumer_plugin.apply,
        name=consumer_plugin.name,
        inject=consumer_plugin.inject,
    )
    pending_receipt = root.receipt()
    first_provider_plugin = ProbeProvider("first", trace)
    provider = await root.mount(
        first_provider_plugin.apply,
        name=first_provider_plugin.name,
        inject=first_provider_plugin.inject,
        runtime=provider_runtime,
    )
    optional_receipt = root.receipt()
    formatter_plugin = ProbeFormatterProvider()
    _ = await root.mount(
        formatter_plugin.apply,
        name=formatter_plugin.name,
        inject=formatter_plugin.inject,
    )
    ready_receipt = root.receipt()
    initial_signal = root.context.require(PROBE_SIGNAL)

    # 3. 只在编译冻结前观察依赖退出和重新装配。
    compiler = RuntimeSnapshotCompiler()
    await provider.dispose()
    removed_receipt = root.receipt()
    second_provider_plugin = ProbeProvider("second", trace)
    _ = await root.mount(
        second_provider_plugin.apply,
        name=second_provider_plugin.name,
        inject=second_provider_plugin.inject,
        runtime=provider_runtime,
    )
    restored_receipt = root.receipt()
    candidate = compiler.compile(
        {},
        snapshot_revision=f"candidate:{run_id}",
        composition_root=root,
    )
    _validate_behavior(
        consumer_state=consumer.state.value,
        pending_receipt=pending_receipt,
        optional_receipt=optional_receipt,
        ready_receipt=ready_receipt,
        removed_receipt=removed_receipt,
        restored_receipt=restored_receipt,
        external_effect_count=len(audit.external_effects),
    )

    # 4. 候选只供显式验证，关闭后重新构造正式 Root。
    stable = compiler.compile({}, snapshot_revision=f"stable:{run_id}")
    store = RuntimeSnapshotStore(_drain_snapshot)
    store.install(stable)
    transaction = store.begin_publish(candidate)
    await store.commit_latest(transaction)
    lease = store.lease(selector="latest")
    leased_root = lease.snapshot.composition_root
    if leased_root is None:
        raise RuntimeError("latest snapshot 缺少 composition root")
    validated_signal = leased_root.context.require(PROBE_SIGNAL)
    await lease.release()
    _ = store.pause_candidate_admission(candidate)
    await store.wait_for_no_leases(candidate)
    store.seal_candidate_validation(candidate)
    await store.discard_latest(candidate)
    formal_root = CompositionRoot(f"formal:{run_id}")
    formal_trace = ProbeTrace()
    formal_consumer = ProbeConsumer(formal_trace)
    formal_provider = ProbeProvider("second", formal_trace)
    formal_formatter = ProbeFormatterProvider()
    formal_data_root = workspace / "formal-plugin-data" / "probe-provider"
    formal_data_root.mkdir(parents=True)
    formal_runtime = PluginRuntime(
        plugin_id="probe-provider", generation_id="composition-experiment-formal",
        plugin_dir=provider_runtime.plugin_dir, data_dir=formal_data_root,
        workspace=workspace, config={},
    )
    try:
        await formal_root.mount(formal_consumer.apply, name=formal_consumer.name,
                                inject=formal_consumer.inject)
        await formal_root.mount(formal_provider.apply, name=formal_provider.name,
                                inject=formal_provider.inject, runtime=formal_runtime)
        await formal_root.mount(formal_formatter.apply, name=formal_formatter.name,
                                inject=formal_formatter.inject)
        formal = compiler.compile({}, snapshot_revision=f"formal:{run_id}",
                                  composition_root=formal_root)
        transaction = store.begin_publish(formal)
    except BaseException:
        await formal_root.dispose()
        raise
    await store.commit_provisional(transaction)
    await store.finalize_provisional(transaction)
    await store.retry_drains()
    stable_lease = store.lease(selector="stable")
    promoted_snapshot_id = stable_lease.snapshot.snapshot_id
    await stable_lease.release()
    promoted_signal = formal_root.context.require(PROBE_SIGNAL)
    if promoted_signal is validated_signal or promoted_signal.value != validated_signal.value:
        raise RuntimeError("正式 Root 必须从相同输入创建独立服务实例")
    promoted_receipt = formal_root.receipt()
    await store.close()
    disposed_receipt = formal_root.receipt()

    return {
        "run_id": run_id,
        "workspace": str(workspace),
        "stable_snapshot_id": promoted_snapshot_id,
        "observed_signal": initial_signal.value,
        "promoted_signal": promoted_signal.value,
        "trace": trace.events,
        "formal_trace": formal_trace.events,
        "receipts": {
            "pending": pending_receipt,
            "optional": optional_receipt,
            "ready": ready_receipt,
            "removed": removed_receipt,
            "restored": restored_receipt,
            "promoted": promoted_receipt,
            "disposed": disposed_receipt,
        },
        "workspace_files_before_result": _workspace_files(workspace),
    }


def _validate_behavior(**evidence: object) -> None:
    """Apply Core-owned oracles; plugins do not submit their own verdict."""

    expected: dict[str, object] = {
        "consumer_state": "active",
        "external_effect_count": 0,
    }
    for field, value in expected.items():
        if evidence[field] != value:
            raise RuntimeError(f"实验行为不符合 oracle: {field}={evidence[field]!r}")
    for field in ("pending_receipt", "removed_receipt"):
        receipt = evidence[field]
        if getattr(receipt, "ready") is not False:
            raise RuntimeError(f"实验未观察到 required pending: {field}")
    for field in ("ready_receipt", "restored_receipt"):
        receipt = evidence[field]
        if getattr(receipt, "ready") is not True:
            raise RuntimeError(f"实验拓扑未恢复 ready: {field}")
    optional = evidence["optional_receipt"]
    if getattr(optional, "optional_pending") != ("probe-formatter-consumer",):
        raise RuntimeError("实验未观察到 optional child pending")


def _workspace_files(workspace: Path) -> list[dict[str, str]]:
    files: list[dict[str, str]] = []
    for path in sorted(item for item in workspace.rglob("*") if item.is_file()):
        content = path.read_bytes()
        files.append(
            {
                "path": str(path.relative_to(workspace)),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        )
    return files


def _json_default(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(cast(Any, value))
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"无法序列化实验字段: {type(value).__name__}")


async def _main() -> int:
    args = _parse_args()
    workspace = _create_workspace(args.workspace)
    result = await _run(workspace)
    result_path = workspace / "runtime" / "plugin-composition-result.json"
    atomic_write_text(
        result_path,
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
            default=_json_default,
        )
        + "\n",
        domain="plugin_composition_experiment",
    )
    print(result_path)
    return 0


if __name__ == "__main__":
    import asyncio

    raise SystemExit(asyncio.run(_main()))

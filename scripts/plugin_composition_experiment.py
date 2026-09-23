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
    CompositionReceipt,
    CompositionRoot,
    PluginRuntime,
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
        description="在全新隔离 workspace 中运行单 Root 插件组合实验"
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


async def _run(workspace: Path) -> dict[str, object]:
    """Observe one Root through dependency wait, replacement, and real close."""

    # 1. Fix the isolated run identity and marker before acquiring a Root.
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
    trace = ProbeTrace()
    receipts: dict[str, CompositionReceipt] = {}
    current_signal = None
    body_error: BaseException | None = None
    try:
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

        # 2. The required consumer waits; the first provider leaves one optional child pending.
        consumer_plugin = ProbeConsumer(trace)
        consumer = await root.mount(
            consumer_plugin.apply,
            name=consumer_plugin.name,
            inject=consumer_plugin.inject,
        )
        receipts["pending"] = root.receipt()
        first_provider_plugin = ProbeProvider("first", trace)
        provider = await root.mount(
            first_provider_plugin.apply,
            name=first_provider_plugin.name,
            inject=first_provider_plugin.inject,
            runtime=provider_runtime,
        )
        receipts["optional"] = root.receipt()
        formatter_plugin = ProbeFormatterProvider()
        _ = await root.mount(
            formatter_plugin.apply,
            name=formatter_plugin.name,
            inject=formatter_plugin.inject,
        )
        receipts["ready"] = root.receipt()
        initial_signal = root.context.require(PROBE_SIGNAL)
        if initial_signal.value != "first":
            raise RuntimeError("实验初始 provider 未提供预期信号")

        # 3. The same Root closes the dependent scope before mounting its replacement.
        await provider.dispose()
        receipts["removed"] = root.receipt()
        second_provider_plugin = ProbeProvider("second", trace)
        _ = await root.mount(
            second_provider_plugin.apply,
            name=second_provider_plugin.name,
            inject=second_provider_plugin.inject,
            runtime=provider_runtime,
        )
        receipts["replaced"] = root.receipt()
        current_signal = root.context.require(PROBE_SIGNAL)
        if current_signal is initial_signal or current_signal.value != "second":
            raise RuntimeError("同一 Root 未切换到替换 provider 的信号")
        _validate_behavior(
            consumer_state=consumer.state.value,
            pending_receipt=receipts["pending"],
            optional_receipt=receipts["optional"],
            ready_receipt=receipts["ready"],
            removed_receipt=receipts["removed"],
            replaced_receipt=receipts["replaced"],
        )
    except BaseException as error:
        body_error = error
    finally:
        # Root/Fiber own every acquired Effect; a failed close cannot produce a disposed receipt.
        try:
            await root.dispose()
        except BaseException as cleanup_error:
            if body_error is not None:
                raise BaseExceptionGroup("实验运行与 Root 清理均失败", [body_error, cleanup_error]) from None
            raise
    if body_error is not None:
        raise body_error
    if current_signal is None:
        raise RuntimeError("实验没有取得替换后的信号")
    disposed_receipt = root.receipt()
    if disposed_receipt.fibers or disposed_receipt.effects:
        raise RuntimeError("实验 Root 关闭后仍有 Fiber 或 Effect")
    receipts["disposed"] = disposed_receipt
    external_effect_count = len(audit.external_effects)
    if external_effect_count:
        raise RuntimeError("实验发生外部效果")

    return {
        "run_id": run_id,
        "workspace": str(workspace),
        "current_signal_after_replacement": current_signal.value,
        "trace": trace.events,
        "receipts": receipts,
        "external_effect_count": external_effect_count,
        "workspace_files_before_result": _workspace_files(workspace),
    }


def _validate_behavior(
    *,
    consumer_state: str,
    pending_receipt: CompositionReceipt,
    optional_receipt: CompositionReceipt,
    ready_receipt: CompositionReceipt,
    removed_receipt: CompositionReceipt,
    replaced_receipt: CompositionReceipt,
) -> None:
    """Check the real Root observations before releasing its resources."""

    if consumer_state != "active":
        raise RuntimeError(f"实验消费者未恢复 active: {consumer_state}")
    for name, receipt in (("pending", pending_receipt), ("removed", removed_receipt)):
        if receipt.ready:
            raise RuntimeError(f"实验未观察到 required pending: {name}")
    for name, receipt in (("ready", ready_receipt), ("replaced", replaced_receipt)):
        if not receipt.ready:
            raise RuntimeError(f"实验拓扑未恢复 ready: {name}")
    if not optional_receipt.ready or optional_receipt.optional_pending != ("probe-formatter-consumer",):
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

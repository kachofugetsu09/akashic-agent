"""集中式、一次性 workspace 的 pure-v3 E1 Gate。"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sqlite3
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.config_models import (
    Config,
    MemoryConfig,
    MemoryEmbeddingConfig,
)  # noqa: E402
from agent.plugins.interaction_undo import InteractionUndoCoordinator  # noqa: E402
from agent.plugins.generation_activity_host import ActivityHost  # noqa: E402
from agent.plugins.generation_job_host import BackgroundJobActivityAdapter  # noqa: E402
from agent.plugins.manager import PluginManager  # noqa: E402
from agent.plugins.mobile_ui import PluginMobileUiProvider  # noqa: E402
from agent.provider import LLMProvider  # noqa: E402
from agent.tools.registry import ToolRegistry  # noqa: E402
from bootstrap.memory import (
    build_memory_runtime,
    ensure_memory_plugin_storage,
)  # noqa: E402
from bus.event_bus import EventBus  # noqa: E402
from core.net.http import SharedHttpResources  # noqa: E402
from memory2.store import MemoryStore2  # noqa: E402
from session.manager import SessionManager  # noqa: E402

try:
    from docker.debug import plugin_v3_fleet_gate as fleet_gate  # noqa: E402
except ModuleNotFoundError:  # pragma: no cover
    import plugin_v3_fleet_gate as fleet_gate  # type: ignore[no-redef] # noqa: E402


DEFAULT_LOCK = ROOT / "docker" / "debug" / "plugin-v3-fleet.lock.json"
DEFAULT_REPORT = ROOT / "docker" / "debug" / "reports" / "plugin-v3-e1" / "gate.json"
DEFAULT_PASSIVE_WEBUI_REPORT = (
    ROOT / "docker" / "debug" / "reports" / "plugin-passive-webui-v3" / "gate.json"
)
E1_PLUGIN_IDS = (
    "akasha",
    "default_memory",
    "citation",
    "meme",
    "emotion",
    "observe",
    "proactive_feedback",
    "plugin_undo",
)
E1_EXTERNAL_PLUGIN_IDS = E1_PLUGIN_IDS[2:]
PASSIVE_WEBUI_SCENARIO_PROFILE = "citation-meme-webui-v3-v1"
PASSIVE_WEBUI_PLUGIN_IDS: tuple[str, ...] = ("citation", "meme")
BUILTIN_PLUGIN_ROOTS = {
    "akasha": ROOT / "plugins" / "akasha",
    "default_memory": ROOT / "plugins" / "default_memory",
}


class E1GateError(RuntimeError):
    """报告可复现的 E1 输入或证据失败。"""


@dataclass(frozen=True, slots=True)
class RuntimeBundle:
    """保存一个 disposable Core runtime 的 owner。"""

    workspace: Path
    engine_name: str
    sessions: SessionManager
    memory: Any
    manager: PluginManager
    http: SharedHttpResources


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行 pure-v3 集中式 E1 Gate")
    _ = parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    _ = parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    _ = parser.add_argument(
        "--passive-webui-report", type=Path, default=DEFAULT_PASSIVE_WEBUI_REPORT
    )
    _ = parser.add_argument("--tmp-root", type=Path)
    _ = parser.add_argument(
        "--plugin-root", action="append", default=[], metavar="PLUGIN_ID=PATH"
    )
    _ = parser.add_argument("--offline", action="store_true")
    return parser.parse_args()


def _parse_plugin_roots(raw: list[str]) -> dict[str, Path]:
    """在 CLI 边界解析 explicit exact checkout 路径。"""

    roots: dict[str, Path] = {}
    for value in raw:
        plugin_id, separator, path = value.partition("=")
        if not separator or not plugin_id.strip() or not path.strip():
            raise E1GateError(f"--plugin-root 必须是 PLUGIN_ID=PATH: {value!r}")
        if plugin_id in roots:
            raise E1GateError(f"--plugin-root 重复: {plugin_id}")
        roots[plugin_id] = Path(path).expanduser().resolve(strict=False)
    unknown = sorted(set(roots) - set(E1_EXTERNAL_PLUGIN_IDS))
    if unknown:
        raise E1GateError(f"--plugin-root 只允许 E1 external plugin: {unknown}")
    return roots


def _select_e1_locks(path: Path) -> dict[str, Any]:
    """从完整 immutable fleet lock 选择 E1 external revisions。"""

    locks = fleet_gate._load_lock(path)  # pyright: ignore[reportPrivateUsage]
    by_id = {item.id: item for item in locks}
    missing = sorted(set(E1_EXTERNAL_PLUGIN_IDS).difference(by_id))
    if missing:
        raise E1GateError(f"E1 lock 缺少 external plugin: {missing}")
    return {plugin_id: by_id[plugin_id] for plugin_id in E1_EXTERNAL_PLUGIN_IDS}


def _local_checkout_evidence(lock: Any, root: Path) -> dict[str, object]:
    """验证 local checkout 的 commit、tree、clean 状态。"""

    if not root.is_dir() or root.is_symlink():
        raise E1GateError(f"checkout 不是实体目录: {root}")
    actual = fleet_gate._git_output(
        root, "rev-parse", "HEAD"
    )  # pyright: ignore[reportPrivateUsage]
    if actual != lock.resolved_sha:
        raise E1GateError(
            f"checkout SHA 与锁不一致: {lock.id} expected={lock.resolved_sha} actual={actual}"
        )
    dirty = tuple(
        fleet_gate._git_output(root, "status", "--porcelain").splitlines()
    )  # pyright: ignore[reportPrivateUsage]
    if dirty:
        raise E1GateError(f"checkout 工作树不干净: {lock.id}: {dirty}")
    return {
        "id": lock.id,
        "repository": lock.repository,
        "resolved_sha": lock.resolved_sha,
        "tree": fleet_gate._git_output(
            root, "rev-parse", "HEAD^{tree}"
        ),  # pyright: ignore[reportPrivateUsage]
        "clean": True,
        "history": fleet_gate._git_output(
            root, "rev-parse", "--is-shallow-repository"
        ),  # pyright: ignore[reportPrivateUsage]
        "path": str(root),
        "mode": "provided-checkout",
    }


def _resolve_external_roots(
    locks: dict[str, Any], staging: Path, provided: dict[str, Path], *, offline: bool
) -> tuple[dict[str, Path], list[dict[str, object]], list[str]]:
    """解析 exact lock checkout；绝不回退到旧 v2 checkout。"""

    roots: dict[str, Path] = {}
    evidence: list[dict[str, object]] = []
    blockers: list[str] = []
    for plugin_id in E1_EXTERNAL_PLUGIN_IDS:
        lock = locks[plugin_id]
        try:
            if plugin_id in provided:
                checkout = provided[plugin_id]
                item = _local_checkout_evidence(lock, checkout)
            elif offline:
                raise E1GateError(
                    f"offline 模式没有精确 checkout: {plugin_id}@{lock.resolved_sha}"
                )
            else:
                checkout = staging / plugin_id
                result = fleet_gate._checkout_locked_plugin(
                    lock, checkout
                )  # pyright: ignore[reportPrivateUsage]
                item: dict[str, object] = {
                    **cast(dict[str, object], asdict(result)),
                    "path": str(checkout),
                    "mode": "shallow-lock-checkout",
                }
            root = Path(str(item["path"]))
            static = fleet_gate._inspect_static_plugin(
                root, plugin_id
            )  # pyright: ignore[reportPrivateUsage]
            item["static"] = static
            if static["status"] != "passed":
                raise E1GateError(f"external static v3 inspection failed: {plugin_id}")
            roots[plugin_id] = root
            evidence.append(cast(dict[str, object], item))
        except Exception as error:
            blockers.append(
                f"{plugin_id}: exact locked checkout unavailable: {type(error).__name__}: {error}"
            )
            evidence.append(
                {
                    "id": plugin_id,
                    "resolved_sha": lock.resolved_sha,
                    "status": "blocked",
                    "error": f"{type(error).__name__}: {error}",
                    "mode": "not-run",
                }
            )
    return roots, evidence, blockers


def _report_object(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise E1GateError(f"{label} 必须是 object")
    return cast(dict[str, object], value)


def _report_index(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise E1GateError(f"{label} 必须是非负整数")
    return value


def _validate_passive_sources(
    report: dict[str, object], locks: dict[str, Any]
) -> dict[str, str]:
    """验证 WebUI 报告只引用 E1 fleet 锁定的 Citation/Meme source。"""

    missing_locks = sorted(set(PASSIVE_WEBUI_PLUGIN_IDS) - set(locks))
    if missing_locks:
        raise E1GateError(f"E1 fleet lock 缺少 passive WebUI source: {missing_locks}")
    raw_sources = report.get("sources")
    if not isinstance(raw_sources, list):
        raise E1GateError("passive WebUI report.sources 必须是列表")
    source_shas: dict[str, str] = {}
    for raw_source in cast(list[object], raw_sources):
        source = _report_object(raw_source, "passive WebUI report.sources item")
        if source.get("kind") != "plugin":
            continue
        plugin_id = source.get("id")
        if not isinstance(plugin_id, str) or plugin_id not in PASSIVE_WEBUI_PLUGIN_IDS:
            raise E1GateError(
                f"passive WebUI report 包含非 E1 Citation/Meme source: {plugin_id!r}"
            )
        if plugin_id in source_shas:
            raise E1GateError(f"passive WebUI report source 重复: {plugin_id}")
        resolved_sha = source.get("resolved_sha")
        if (
            not isinstance(resolved_sha, str)
            or resolved_sha != locks[plugin_id].resolved_sha
        ):
            raise E1GateError(
                f"passive WebUI {plugin_id} source SHA 与 E1 fleet lock 不一致: "
                f"expected={locks[plugin_id].resolved_sha} actual={resolved_sha!r}"
            )
        source_shas[plugin_id] = resolved_sha
    if set(source_shas) != set(PASSIVE_WEBUI_PLUGIN_IDS):
        raise E1GateError(
            "passive WebUI report 缺少 Citation/Meme source: "
            + ", ".join(sorted(set(PASSIVE_WEBUI_PLUGIN_IDS) - set(source_shas)))
        )
    return source_shas


def _validate_passive_assistant(report: dict[str, object]) -> dict[str, object]:
    """验证 WebUI 持久 assistant 同时保留 citation metadata 与 Meme media。"""

    runtime = _report_object(report.get("runtime"), "passive WebUI report.runtime")
    if runtime.get("status") != "passed":
        raise E1GateError(
            f"passive WebUI runtime.status 不是 passed: {runtime.get('status')!r}"
        )
    messages = runtime.get("messages")
    if not isinstance(messages, list) or len(cast(list[object], messages)) != 2:
        raise E1GateError("passive WebUI runtime.messages 必须包含 user + assistant")
    message_items = cast(list[object], messages)
    _ = _report_object(message_items[0], "passive WebUI user message")
    assistant = _report_object(message_items[1], "passive WebUI assistant message")
    if assistant.get("role") != "assistant":
        raise E1GateError("passive WebUI assistant message role 错误")
    if assistant.get("cited_memory_ids") != ["mem_1"]:
        raise E1GateError("passive WebUI assistant citation metadata 缺失")
    attachment_ids = assistant.get("attachment_ids")
    attachments = assistant.get("attachments")
    if (
        not isinstance(attachment_ids, list)
        or len(attachment_ids) != 1
        or not isinstance(attachment_ids[0], str)
        or not isinstance(attachments, list)
        or len(attachments) != 1
        or not isinstance(attachments[0], dict)
    ):
        raise E1GateError("passive WebUI assistant attachment 不符合 fixture")
    descriptor = cast(dict[str, object], attachments[0])
    if (
        descriptor.get("artifact_id") != attachment_ids[0]
        or descriptor.get("kind") != "image"
        or descriptor.get("filename") != "001.png"
        or descriptor.get("media_type") != "image/png"
        or descriptor.get("url") != f"/api/chat/artifacts/{attachment_ids[0]}"
    ):
        raise E1GateError("passive WebUI assistant artifact descriptor 漂移")
    return {
        "cited_memory_ids": assistant["cited_memory_ids"],
        "attachment_ids": attachment_ids,
        "attachments": attachments,
    }


def _validate_passive_webui_report(
    path: Path,
    locks: dict[str, Any],
    expected_core: dict[str, object],
) -> dict[str, object]:
    """严格验证已完成的 WebUI Gate report，返回可并入 E1 的证据。"""

    if not path.is_file():
        raise E1GateError(f"passive WebUI report 不存在: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise E1GateError(
            f"passive WebUI report 无法读取: {type(error).__name__}: {error}"
        ) from error
    report = _report_object(payload, "passive WebUI report")
    if report.get("status") != "passed":
        raise E1GateError(
            f"passive WebUI report.status 不是 passed: {report.get('status')!r}"
        )
    if report.get("scenario_profile") != PASSIVE_WEBUI_SCENARIO_PROFILE:
        raise E1GateError(
            f"passive WebUI scenario_profile 不匹配: {report.get('scenario_profile')!r}"
        )
    report_core = _report_object(report.get("core"), "passive WebUI report.core")
    if report_core.get("dirty_status") != []:
        raise E1GateError("passive WebUI report 不是干净 Core 证据")
    for field in ("head", "tree"):
        if report_core.get(field) != expected_core.get(field):
            raise E1GateError(
                f"passive WebUI Core {field} 与当前 E1 不一致: "
                f"expected={expected_core.get(field)!r} actual={report_core.get(field)!r}"
            )
    source_shas = _validate_passive_sources(report, locks)
    runtime = _report_object(report.get("runtime"), "passive WebUI report.runtime")
    request = _report_object(
        runtime.get("model_request"), "passive WebUI model_request"
    )
    citation_index = _report_index(
        request.get("citation_index"), "passive WebUI citation_index"
    )
    meme_index = _report_index(request.get("meme_index"), "passive WebUI meme_index")
    if citation_index >= meme_index:
        raise E1GateError(
            f"passive WebUI prompt 顺序错误: citation={citation_index} meme={meme_index}"
        )
    assistant = _validate_passive_assistant(report)
    cleanup = _report_object(report.get("cleanup"), "passive WebUI report.cleanup")
    if cleanup.get("residuals") != []:
        raise E1GateError(
            f"passive WebUI cleanup residuals 非空: {cleanup.get('residuals')!r}"
        )
    if (
        cleanup.get("sandbox_removed") is not True
        or cleanup.get("source_unchanged") is not True
    ):
        raise E1GateError(
            "passive WebUI cleanup 未同时证明 sandbox_removed/source_unchanged"
        )
    return {
        "status": "passed",
        "report": str(path),
        "scenario_profile": report["scenario_profile"],
        "source_shas": source_shas,
        "prompt_order": {"citation_index": citation_index, "meme_index": meme_index},
        "assistant": assistant,
        "cleanup": {
            "residuals": [],
            "sandbox_removed": True,
            "source_unchanged": True,
        },
    }


def _plugin_dirs(external: dict[str, Path]) -> list[Path]:
    """组装真实 PluginManager 的 in-tree 与 exact checkout source roots。"""

    return [BUILTIN_PLUGIN_ROOTS["akasha"], BUILTIN_PLUGIN_ROOTS["default_memory"]] + [
        external[plugin_id]
        for plugin_id in E1_EXTERNAL_PLUGIN_IDS
        if plugin_id in external
    ]


def _config(engine: str) -> Config:
    return Config(
        provider="fixture",
        model="fixture",
        api_key="",
        system_prompt="",
        base_url="http://127.0.0.1:9",
        memory=MemoryConfig(
            enabled=True,
            engine=engine,
            embedding=MemoryEmbeddingConfig(output_dimensionality=32),
        ),
    )


async def _open_runtime(
    workspace: Path, engine: str, plugin_dirs: list[Path]
) -> RuntimeBundle:
    """在 disposable workspace 启动真实 memory runtime 与 PluginManager。"""

    workspace.mkdir(parents=True, exist_ok=True)
    config = _config(engine)
    sessions = SessionManager(workspace)
    http = SharedHttpResources()
    event_bus = EventBus()
    tools = ToolRegistry()
    provider = LLMProvider(
        api_key="", provider_name="fixture", base_url="http://127.0.0.1:9"
    )
    try:
        _ = ensure_memory_plugin_storage(config, workspace)
        memory = build_memory_runtime(
            config, workspace, tools, provider, None, http, event_publisher=event_bus
        )
        manager = PluginManager(
            plugin_dirs,
            event_bus=event_bus,
            workspace=workspace,
            tool_registry=tools,
            session_manager=sessions,
            memory_engine=memory.engine,
            installed_cache_root=workspace / "installed-plugins",
        )
        manager.bind_activity_host(
            ActivityHost(
                (
                    BackgroundJobActivityAdapter(
                        manager.snapshot_store, workspace=str(workspace)
                    ),
                )
            )
        )
        await manager.load_all()
    except BaseException:
        sessions.close()
        await http.aclose()
        raise
    return RuntimeBundle(workspace, engine, sessions, memory, manager, http)


async def _close_runtime(bundle: RuntimeBundle) -> list[str]:
    """关闭所有 owner，并返回 cleanup failure 证据。"""

    errors: list[str] = []
    for label, action in (
        ("PluginManager", bundle.manager.terminate_all),
        ("MemoryRuntime", bundle.memory.aclose),
        ("SessionManager", bundle.sessions.close),
        ("HTTP", bundle.http.aclose),
    ):
        try:
            result = action()
            if asyncio.iscoroutine(result):
                await result
        except Exception as error:
            errors.append(f"{label}: {type(error).__name__}: {error}")
    return errors


def _runtime_identity(bundle: RuntimeBundle) -> dict[str, object]:
    """读取 stable snapshot、generation 与 mobile catalog。"""

    snapshot = bundle.manager.current_snapshot
    if snapshot is None:
        raise E1GateError(f"{bundle.engine_name} runtime 没有 stable snapshot")
    active = sorted(item.plugin_id for item in bundle.manager.active_plugins())
    generations = {
        plugin_id: generation.source_revision
        for plugin_id in active
        if (generation := bundle.manager.generation(plugin_id)) is not None
    }
    return {
        "engine": bundle.engine_name,
        "snapshot_id": snapshot.snapshot_id,
        "active_plugins": active,
        "generations": generations,
        "mobile_catalog": PluginMobileUiProvider(bundle.manager).catalog(),
        "composition_active_plugin_ids": sorted(
            snapshot.composition_active_plugin_ids or frozenset()
        ),
    }


async def _probe_boot(bundle: RuntimeBundle) -> dict[str, object]:
    """执行 stable lease 与 Akasha bounded mobile query。"""

    identity = _runtime_identity(bundle)
    snapshot = bundle.manager.current_snapshot
    if snapshot is None:
        raise E1GateError("stable snapshot 在 mobile probe 前消失")
    before = snapshot.lease_count
    async with bundle.manager.snapshot_store.lease() as leased:
        lease: dict[str, object] = {
            "snapshot_id": leased.snapshot_id,
            "during": leased.lease_count,
        }
    lease["before"] = before
    lease["after"] = snapshot.lease_count
    identity["stable_lease"] = lease
    if bundle.engine_name == "akasha":
        generation = bundle.manager.generation("akasha")
        if generation is None:
            raise E1GateError("Akasha generation 缺失")
        provider = PluginMobileUiProvider(bundle.manager)
        try:
            result = await provider.query(
                "akasha",
                generation.source_revision,
                "inspector.recent",
                {},
                session_id="e1:mobile",
                turn_id="turn:e1:mobile",
            )
        except Exception as error:
            identity["mobile_query"] = {
                "plugin_id": "akasha",
                "method": "inspector.recent",
                "status": "blocked",
                "error": f"{type(error).__name__}: {error}",
            }
        else:
            identity["mobile_query"] = {
                "plugin_id": "akasha",
                "method": "inspector.recent",
                "status": "passed",
                "result": result,
            }
    else:
        identity["mobile_query"] = {
            "status": "not_applicable",
            "reason": "Default Memory 无 mobile UI",
        }
    if snapshot.lease_count != before:
        raise E1GateError("stable snapshot lease 未归还")
    return identity


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _seed_interaction(
    sessions: SessionManager, *, key: str, turn: str, label: str
) -> tuple[str, ...]:
    """通过 SessionStore append 一个完整 U+A interaction。"""

    timestamp = _now()
    rows = sessions.control_store.persist_session(
        key,
        created_at=timestamp,
        updated_at=timestamp,
        metadata={"gate": label},
        messages=[
            {
                "role": "user",
                "content": f"E1 {label} question",
                "timestamp": timestamp,
                "extra": {"control_turn_id": turn, "turn_input_ordinal": 0},
            },
            {
                "role": "assistant",
                "content": f"E1 {label} answer",
                "timestamp": timestamp,
                "extra": {
                    "control_turn_id": turn,
                    "turn_terminal": True,
                    "turn_input_count": 1,
                },
            },
        ],
    )
    return tuple(str(row["id"]) for row in rows)


def _freeze(value: object) -> object:
    if isinstance(value, bytes):
        return {"sha256": hashlib.sha256(value).hexdigest(), "length": len(value)}
    if value is None or isinstance(value, str | int | float):
        return value
    return repr(value)


def _sqlite_state(path: Path) -> dict[str, object]:
    """读取 SQLite integrity、schema 与所有表的行 hash，不写入数据库。"""

    connection = sqlite3.connect(str(path))
    try:
        integrity = str(connection.execute("PRAGMA integrity_check").fetchone()[0])
        if integrity != "ok":
            raise E1GateError(f"SQLite integrity_check 失败: {path}: {integrity}")
        names = [
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
            )
        ]
        tables: dict[str, object] = {}
        for name in names:
            quote = '"' + name.replace('"', '""') + '"'
            schema = connection.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (name,)
            ).fetchone()[0]
            info = list(connection.execute(f"PRAGMA table_info({quote})"))
            columns = [str(row[1]) for row in info]
            primary = [
                name
                for order, name in sorted(
                    (int(row[5]), str(row[1])) for row in info if int(row[5]) > 0
                )
            ]
            without_rowid = "WITHOUT ROWID" in str(schema or "").upper()
            rows = (
                connection.execute(f"SELECT * FROM {quote}").fetchall()
                if without_rowid
                else connection.execute(f"SELECT rowid, * FROM {quote}").fetchall()
            )
            values: dict[str, str] = {}
            for row in rows:
                frozen = [_freeze(item) for item in row]
                if without_rowid:
                    positions = {column: index for index, column in enumerate(columns)}
                    key_values = (
                        [frozen[positions[item]] for item in primary]
                        if primary
                        else frozen
                    )
                    key = json.dumps(key_values, ensure_ascii=False, sort_keys=True)
                    payload = frozen
                else:
                    key = str(frozen[0])
                    payload = frozen[1:]
                encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True)
                values[key] = hashlib.sha256(encoded.encode()).hexdigest()
            tables[name] = {"schema": schema, "rows": values}
        return {"path": str(path), "integrity": integrity, "tables": tables}
    finally:
        connection.close()


def _sqlite_diff(
    before: dict[str, object], after: dict[str, object]
) -> dict[str, object]:
    """从两个 SQLite snapshot 生成紧凑完整 write-set。"""

    left = cast(dict[str, object], before["tables"])
    right = cast(dict[str, object], after["tables"])
    inserted: list[str] = []
    deleted: list[str] = []
    updated: list[str] = []
    for table in sorted(set(left) | set(right)):
        old = cast(
            dict[str, str],
            cast(dict[str, object], left.get(table, {"rows": {}}))["rows"],
        )
        new = cast(
            dict[str, str],
            cast(dict[str, object], right.get(table, {"rows": {}}))["rows"],
        )
        inserted.extend(f"{table}:{key}" for key in sorted(set(new) - set(old)))
        deleted.extend(f"{table}:{key}" for key in sorted(set(old) - set(new)))
        updated.extend(
            f"{table}:{key}"
            for key in sorted(set(old) & set(new))
            if old[key] != new[key]
        )
    return {
        "inserted": inserted,
        "deleted": deleted,
        "updated": updated,
        "inserted_count": len(inserted),
        "deleted_count": len(deleted),
        "updated_count": len(updated),
    }


def _messages(path: Path) -> dict[str, tuple[object, ...]]:
    connection = sqlite3.connect(str(path))
    try:
        return {
            str(row[0]): tuple(row[1:])
            for row in connection.execute(
                "SELECT id, session_key, seq, role, content, tool_chain, extra, ts FROM messages ORDER BY id"
            )
        }
    finally:
        connection.close()


def _memory_store(bundle: RuntimeBundle) -> MemoryStore2:
    stores = [
        item for item in bundle.memory.closeables if isinstance(item, MemoryStore2)
    ]
    if len(stores) != 1:
        raise E1GateError(f"MemoryStore2 owner 数量异常: {len(stores)}")
    return stores[0]


def _item_id(receipt: str) -> str:
    prefix, separator, value = receipt.partition(":")
    if prefix not in {"new", "reinforced"} or not separator or not value:
        raise E1GateError(f"MemoryStore2 receipt 无效: {receipt}")
    return value


async def _append_only(bundle: RuntimeBundle) -> dict[str, object]:
    """证明既有 SessionDB messages 不变且新 seq 只追加。"""

    key = "e1:append"
    _ = _seed_interaction(
        bundle.sessions, key=key, turn="turn:e1:append:1", label="append-1"
    )
    path = Path(bundle.sessions.db_path)
    before = _sqlite_state(path)
    old_messages = _messages(path)
    old_seq = bundle.sessions.control_store.next_seq(key)
    new_ids = _seed_interaction(
        bundle.sessions, key=key, turn="turn:e1:append:2", label="append-2"
    )
    after = _sqlite_state(path)
    new_messages = _messages(path)
    if any(new_messages[item] != row for item, row in old_messages.items()):
        raise E1GateError("append 修改既有 canonical message")
    if set(old_messages) - set(new_messages):
        raise E1GateError("append 删除 canonical message")
    if set(new_messages) - set(old_messages) != set(new_ids):
        raise E1GateError("append 新增 message 集合不匹配")
    new_seq = bundle.sessions.control_store.next_seq(key)
    if new_seq != old_seq + len(new_ids):
        raise E1GateError(f"seq 高水位异常: {old_seq}->{new_seq}")
    return {
        "status": "passed",
        "database": str(path),
        "before": before,
        "after": after,
        "write_set": _sqlite_diff(before, after),
        "existing_messages_unchanged": True,
        "new_message_ids": list(new_ids),
        "seq_highwater": {"before": old_seq, "after": new_seq},
    }


async def _undo(bundle: RuntimeBundle) -> dict[str, object]:
    """在 disposable workspace 执行 exact latest interaction undo 与 Memory2 rollback。"""

    key = "e1:undo"
    guard_ids = _seed_interaction(
        bundle.sessions, key=key, turn="turn:e1:undo:guard", label="undo-guard"
    )
    target_ids = _seed_interaction(
        bundle.sessions, key=key, turn="turn:e1:undo:target", label="undo-target"
    )
    store = _memory_store(bundle)
    target_source = json.dumps([target_ids[0]], ensure_ascii=False)
    guard_source = json.dumps([guard_ids[0]], ensure_ascii=False)
    old_id = _item_id(
        store.upsert_item("fact", "E1 old memory", [0.0] * 32, target_source)
    )
    new_id = _item_id(
        store.upsert_item("fact", "E1 replacement memory", [0.0] * 32, target_source)
    )
    guard_memory_id = _item_id(
        store.upsert_item("fact", "E1 guard memory", [0.0] * 32, guard_source)
    )
    old_item = bundle.memory.engine.get_item_for_dashboard(old_id)
    new_item = bundle.memory.engine.get_item_for_dashboard(new_id)
    if old_item is None or new_item is None:
        raise E1GateError("Memory2 replacement seed 读取失败")
    bundle.memory.engine.update_item_for_dashboard(old_id, status="superseded")
    if (
        store.record_replacements(
            old_items=[old_item], new_item=new_item, source_ref=target_source
        )
        != 1
    ):
        raise E1GateError("Memory2 replacement seed 未写入")
    path = Path(bundle.sessions.db_path)
    before = _sqlite_state(path)
    old_messages = _messages(path)
    old_seq = bundle.sessions.control_store.next_seq(key)
    result = await InteractionUndoCoordinator(
        cast(Any, bundle.sessions), bundle.memory.engine
    ).undo_latest(key)
    if result is None or result.control_turn_id != "turn:e1:undo:target":
        raise E1GateError(f"未撤销 exact control_turn_id: {result}")
    after = _sqlite_state(path)
    new_messages = _messages(path)
    backup_path = Path(result.backup_path)
    backup = _sqlite_state(backup_path)
    target_missing = set(target_ids).isdisjoint(new_messages)
    guard_unchanged = all(
        new_messages.get(item) == old_messages[item] for item in guard_ids
    )
    if not target_missing or not guard_unchanged:
        raise E1GateError("Undo target/non-target message write-set 不符合合同")
    if bundle.sessions.control_store.pending_interaction_memory_reconciliations(
        "default_memory"
    ):
        raise E1GateError("Undo 成功后仍有 pending receipt")
    old_after = bundle.memory.engine.get_item_for_dashboard(old_id)
    new_after = bundle.memory.engine.get_item_for_dashboard(new_id)
    guard_after = bundle.memory.engine.get_item_for_dashboard(guard_memory_id)
    if old_after is None or new_after is None or guard_after is None:
        raise E1GateError("Memory2 undo item 缺失")
    if (
        old_after["status"] != "active"
        or new_after["status"] != "superseded"
        or guard_after["status"] != "active"
    ):
        raise E1GateError("Memory2 replacement/cursor rollback 不成立")
    new_seq = bundle.sessions.control_store.next_seq(key)
    if new_seq != old_seq:
        raise E1GateError(f"Undo 改变 seq 高水位: {old_seq}->{new_seq}")
    return {
        "status": "passed",
        "workspace": str(bundle.workspace),
        "control_turn_id": result.control_turn_id,
        "message_ids": list(result.message_ids),
        "backup": {"path": str(backup_path), "integrity": backup["integrity"]},
        "before": before,
        "after": after,
        "write_set": _sqlite_diff(before, after),
        "target_missing": target_missing,
        "non_target_messages_unchanged": guard_unchanged,
        "seq_highwater": {"before": old_seq, "after": new_seq},
        "memory2": {
            "old_item_id": old_id,
            "new_item_id": new_id,
            "guard_item_id": guard_memory_id,
            "old_status_after": old_after["status"],
            "new_status_after": new_after["status"],
            "guard_status_after": guard_after["status"],
            "replacement_rows": 1,
        },
    }


async def _crash_recovery(
    workspace: Path, plugin_dirs: list[Path]
) -> dict[str, object]:
    """用真实 closed MemoryStore2 触发内部失败，再模拟 Core 进程重启恢复。"""

    first = await _open_runtime(workspace, "default", plugin_dirs)
    key = "e1:crash"
    ids = _seed_interaction(
        first.sessions, key=key, turn="turn:e1:crash", label="crash"
    )
    deletion = first.sessions.control_store.delete_interaction(
        "turn:e1:crash",
        action_source="plugin_v3_e1_gate.crash",
        expected_latest_session_key=key,
        reconciliation_owner="default_memory",
    )
    if deletion is None:
        raise E1GateError("crash seed delete 未提交")
    _memory_store(first).close()
    failure = ""
    try:
        await InteractionUndoCoordinator(
            cast(Any, first.sessions), first.memory.engine
        ).recover_pending()
    except Exception as error:
        failure = f"{type(error).__name__}: {error}"
    if not failure:
        raise E1GateError("closed MemoryStore2 未触发 process-internal failure")
    pending = first.sessions.control_store.pending_interaction_memory_reconciliations(
        "default_memory"
    )
    if len(pending) != 1 or pending[0].attempts != 1:
        raise E1GateError(f"失败后 receipt 不可重试: {pending}")
    cleanup = await _close_runtime(first)
    if cleanup:
        raise E1GateError(f"重启前 cleanup 失败: {cleanup}")
    restarted = await _open_runtime(workspace, "default", plugin_dirs)
    try:
        await InteractionUndoCoordinator(
            cast(Any, restarted.sessions), restarted.memory.engine
        ).recover_pending()
        if restarted.sessions.control_store.pending_interaction_memory_reconciliations(
            "default_memory"
        ):
            raise E1GateError("Core restart 后 pending receipt 未清空")
        if restarted.sessions.get_existing(key).messages:
            raise E1GateError("Core restart 后 deleted interaction 重现")
        backup = _sqlite_state(Path(deletion.backup_path))
        process_workspace = workspace.parent / "runtime-process-crash"
        child_code = """
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from session.manager import SessionManager

root = Path(sys.argv[1])
sessions = SessionManager(root)
now = datetime.now(UTC).isoformat()
sessions.control_store.persist_session(
    "e1:process-crash", created_at=now, updated_at=now,
    metadata={"gate": "process-crash"},
    messages=[
        {"role": "user", "content": "process crash question", "timestamp": now,
         "extra": {"control_turn_id": "turn:e1:process-crash", "turn_input_ordinal": 0}},
        {"role": "assistant", "content": "process crash answer", "timestamp": now,
         "extra": {"control_turn_id": "turn:e1:process-crash", "turn_terminal": True, "turn_input_count": 1}},
    ],
)
deletion = sessions.control_store.delete_interaction(
    "turn:e1:process-crash", action_source="plugin_v3_e1_gate.process_crash",
    expected_latest_session_key="e1:process-crash", reconciliation_owner="default_memory",
)
if deletion is None:
    raise RuntimeError("process crash seed delete 未提交")
os._exit(17)
"""
        crash = subprocess.run(
            [sys.executable, "-c", child_code, str(process_workspace)],
            cwd=ROOT,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30,
        )
        if crash.returncode != 17:
            raise E1GateError(
                f"Core process crash child exit 不符合预期: {crash.returncode}; stderr={crash.stderr.strip()}"
            )
        crash_bundle = await _open_runtime(process_workspace, "default", plugin_dirs)
        try:
            await InteractionUndoCoordinator(
                cast(Any, crash_bundle.sessions), crash_bundle.memory.engine
            ).recover_pending()
            if crash_bundle.sessions.control_store.pending_interaction_memory_reconciliations(
                "default_memory"
            ):
                raise E1GateError("Core process crash 后 pending receipt 未清空")
            if crash_bundle.sessions.get_existing("e1:process-crash").messages:
                raise E1GateError("Core process crash 后 interaction 重新出现")
            connection = sqlite3.connect(str(crash_bundle.sessions.db_path))
            try:
                backup_row = connection.execute(
                    "SELECT backup_path FROM session_source_mutation_audits ORDER BY completed_at DESC LIMIT 1"
                ).fetchone()
            finally:
                connection.close()
            if backup_row is None or backup_row[0] is None:
                raise E1GateError("Core process crash audit 缺少 backup_path")
            process_backup = _sqlite_state(Path(str(backup_row[0])))
        finally:
            cleanup = await _close_runtime(crash_bundle)
            if cleanup:
                raise E1GateError(f"Core process crash cleanup 失败: {cleanup}")
        return {
            "status": "passed",
            "control_turn_id": deletion.control_turn_id,
            "message_ids": list(ids),
            "process_internal_failure": failure,
            "pending_attempts_before_restart": 1,
            "pending_after_restart": 0,
            "backup": {"path": deletion.backup_path, "integrity": backup["integrity"]},
            "core_process_crash": {
                "child_exit_code": crash.returncode,
                "pending_after_restart": 0,
                "backup_integrity": process_backup["integrity"],
            },
        }
    finally:
        cleanup = await _close_runtime(restarted)
        if cleanup:
            raise E1GateError(f"重启后 cleanup 失败: {cleanup}")


async def _scenarios(
    workspace: Path, plugin_dirs: list[Path], blockers: list[str]
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """运行两个 memory engine、SQLite write-set、Undo 与 restart 场景。"""

    scenarios: list[dict[str, object]] = []
    runtimes: dict[str, RuntimeBundle] = {}
    evidence: dict[str, object] = {}
    for engine in ("default", "akasha"):
        bundle: RuntimeBundle | None = None
        try:
            bundle = await _open_runtime(
                workspace / f"runtime-{engine}", engine, plugin_dirs
            )
            runtimes[engine] = bundle
            boot = await _probe_boot(bundle)
            evidence[engine] = boot
            scenarios.append(
                {"id": f"runtime_boot_{engine}", "status": "passed", "evidence": boot}
            )
            mobile = cast(dict[str, object], boot["mobile_query"])
            if mobile.get("status") == "blocked":
                blockers.append(
                    f"runtime_boot_{engine}.mobile_query: {mobile.get('error', 'unavailable')}"
                )
        except Exception as error:
            scenarios.append(
                {
                    "id": f"runtime_boot_{engine}",
                    "status": "failed",
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            blockers.append(f"runtime_boot_{engine}: {type(error).__name__}: {error}")
            if bundle is not None:
                blockers.extend(
                    f"runtime cleanup: {item}" for item in await _close_runtime(bundle)
                )
                _ = runtimes.pop(engine, None)
    default = runtimes.get("default")
    if default is None:
        scenarios.extend(
            {"id": item, "status": "blocked", "reason": "Default Memory runtime 未启动"}
            for item in ("append_only_sessiondb", "plugin_undo", "core_crash_recovery")
        )
    else:
        for case_id, action in (
            ("append_only_sessiondb", _append_only),
            ("plugin_undo", _undo),
        ):
            try:
                scenarios.append({"id": case_id, **await action(default)})
            except Exception as error:
                scenarios.append(
                    {
                        "id": case_id,
                        "status": "failed",
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
                blockers.append(f"{case_id}: {type(error).__name__}: {error}")
        blockers.extend(
            f"runtime cleanup: {item}" for item in await _close_runtime(default)
        )
        _ = runtimes.pop("default", None)
        try:
            scenarios.append(
                {
                    "id": "core_crash_recovery",
                    **await _crash_recovery(workspace / "runtime-crash", plugin_dirs),
                }
            )
        except Exception as error:
            scenarios.append(
                {
                    "id": "core_crash_recovery",
                    "status": "failed",
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            blockers.append(f"core_crash_recovery: {type(error).__name__}: {error}")
    for bundle in runtimes.values():
        blockers.extend(
            f"runtime cleanup: {item}" for item in await _close_runtime(bundle)
        )
    return scenarios, evidence


async def _run_gate(
    *,
    lock_path: Path,
    report_path: Path,
    tmp_root: Path | None,
    provided_raw: list[str],
    offline: bool,
    passive_webui_report: Path = DEFAULT_PASSIVE_WEBUI_REPORT,
) -> dict[str, object]:
    """执行一次 combined E1 Gate 并持久化 truthful report。"""

    blockers: list[str] = []
    try:
        locks = _select_e1_locks(lock_path)
        provided = _parse_plugin_roots(provided_raw)
    except Exception as error:
        locks, provided = {}, {}
        blockers.append(f"lock/input: {type(error).__name__}: {error}")
    plugin_evidence: list[dict[str, object]] = []
    core = fleet_gate._core_evidence()  # pyright: ignore[reportPrivateUsage]
    lock_evidence: dict[str, object] = {
        "path": str(lock_path),
        "sha256": (
            fleet_gate._sha256(lock_path) if lock_path.is_file() else None
        ),  # pyright: ignore[reportPrivateUsage]
        "selected_external_ids": list(E1_EXTERNAL_PLUGIN_IDS),
    }
    with tempfile.TemporaryDirectory(
        dir=tmp_root, prefix="akashic-plugin-v3-e1-"
    ) as raw:
        workspace = Path(raw) / "workspace"
        staging = Path(raw) / "locked-checkouts"
        staging.mkdir()
        if locks:
            external, external_evidence, external_blockers = _resolve_external_roots(
                locks, staging, provided, offline=offline
            )
            plugin_evidence.extend(external_evidence)
            blockers.extend(external_blockers)
        else:
            external = {}
        for plugin_id, root in BUILTIN_PLUGIN_ROOTS.items():
            status = "available" if root.is_dir() else "blocked"
            plugin_evidence.append(
                {
                    "id": plugin_id,
                    "status": status,
                    "path": str(root),
                    "source": "in-tree",
                }
            )
            if status != "available":
                blockers.append(f"{plugin_id}: in-tree source missing: {root}")
        scenarios, runtime = await _scenarios(
            workspace, _plugin_dirs(external), blockers
        )
        active: set[str] = set()
        for item in runtime.values():
            evidence = cast(dict[str, object], item)
            active.update(cast(list[str], evidence["active_plugins"]))
        missing_runtime = sorted(set(E1_EXTERNAL_PLUGIN_IDS).difference(active))
        if missing_runtime:
            blockers.append(
                "required external plugin runtime coverage absent: "
                + ", ".join(missing_runtime)
            )
        try:
            if not locks:
                raise E1GateError(
                    "E1 fleet lock 不可用，无法绑定 passive WebUI source SHA"
                )
            passive = _validate_passive_webui_report(
                passive_webui_report,
                locks,
                core,
            )
        except E1GateError as error:
            scenarios.append(
                {
                    "id": "passive_prompt_metadata_media",
                    "status": "blocked",
                    "reason": str(error),
                    "report": str(passive_webui_report),
                }
            )
            blockers.append(f"passive_prompt_metadata_media: {error}")
        else:
            scenarios.append({"id": "passive_prompt_metadata_media", **passive})
        report: dict[str, object] = {
            "status": (
                "passed"
                if not blockers
                and all(item["status"] == "passed" for item in scenarios)
                else "blocked"
            ),
            "phase": "e1",
            "gate_version": 1,
            "checked_at": datetime.now(UTC).isoformat(),
            "disposable_workspace": str(workspace),
            "workspace_persisted": False,
            "core": core,
            "lock": lock_evidence,
            "plugins": plugin_evidence,
            "runtime": runtime,
            "scenarios": scenarios,
            "blockers": sorted(set(blockers)),
        }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    _ = report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> int:
    """运行 E1 Gate；blocked/failed 证据返回非零。"""

    args = _parse_args()
    report = asyncio.run(
        _run_gate(
            lock_path=args.lock.resolve(),
            report_path=args.report.resolve(),
            tmp_root=None if args.tmp_root is None else args.tmp_root.resolve(),
            provided_raw=cast(list[str], args.plugin_root),
            offline=bool(args.offline),
            passive_webui_report=args.passive_webui_report.resolve(),
        )
    )
    print(f"plugin v3 E1 Gate {report['status']}: {args.report.resolve()}")
    for blocker in cast(list[str], report["blockers"]):
        print(f"- {blocker}", file=sys.stderr)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

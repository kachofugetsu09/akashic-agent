#!/usr/bin/env python3
"""Run the final pure-v3 rehearsal against a disposable workspace copy."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import shutil
import signal
import sqlite3
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.plugins.interaction_undo import InteractionUndoCoordinator  # noqa: E402
from docker.debug import plugin_v3_e1_gate as e1_gate  # noqa: E402
from scripts.container_rehearsal.prepare import prepare_rehearsal  # noqa: E402
from scripts.container_rehearsal.policy import excluded_reason  # noqa: E402

DEFAULT_LOCK = ROOT / "docker/debug/plugin-v3-fleet.lock.json"
DEFAULT_REPORT = ROOT / "docker/debug/reports/plugin-v3-e4" / "gate.json"
DEFAULT_E1_REPORT = ROOT / "docker/debug/reports/plugin-v3-e1" / "gate.json"
DEFAULT_E2_REPORT = ROOT / "docker/debug/reports/plugin-v3-e2" / "gate.json"
DEFAULT_E3_REPORT = ROOT / "docker/debug/reports/plugin-v3-e3" / "gate.json"
DEFAULT_PASSIVE_REPORT = (
    ROOT / "docker/debug/reports/plugin-passive-webui-v3" / "gate.json"
)

GATE_VERSION = 1
SCENARIO_PROFILE = "plugin-v3-e4-copied-workspace-rehearsal-v1"
SQLITE_HEADER = b"SQLite format 3\x00"
E2_PROFILE = "plugin-v3-e2-shell-v1"
E3_PROFILE = "plugin-v3-e3-fleet-channel-proactive-v3"
PASSIVE_PROFILE = "citation-meme-webui-v3-v1"
E1_SCENARIOS = ("runtime_boot_default", "runtime_boot_akasha")
E1_RUNTIME_ENGINES = ("default", "akasha")
E1_PLUGINS = {
    "akasha", "default_memory", "citation", "meme", "emotion", "observe",
    "proactive_feedback", "plugin_undo",
}
E2_PLUGINS = {
    "shell_restore", "shell_safety", "calendar-mcp",
    "feed-mcp", "fitbit-mcp", "steam-mcp",
}
E3_PLUGINS = {
    "setup_helper",
    "status_commands",
    "daynight_gate",
    "emotion",
    "calendar-mcp",
    "feed-mcp",
    "fitbit-mcp",
    "steam-mcp",
    "huayue-skills",
    "github_watch",
    "feishu",
    "qqbot",
    "citation",
    "meme",
}


class GateBlocked(RuntimeError):
    """Missing prerequisite evidence or unavailable provider input."""


class GateFailure(RuntimeError):
    """Copied-workspace invariant violation."""


def _resolve_tmp_root(value: Path | None) -> Path | None:
    """解析调用方可选的临时目录。"""

    if value is None:
        return None
    root = value.expanduser().resolve()
    if not root.is_dir():
        raise GateFailure(f"E4 tmp root 不是已存在目录: {root}")
    return root


def _sha256_file(path: Path) -> str:
    """Hash a file without loading it all into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _freeze(value: object) -> object:
    """Convert SQLite values to stable, non-secret evidence values."""

    if isinstance(value, bytes):
        return {"sha256": hashlib.sha256(value).hexdigest(), "length": len(value)}
    if value is None or isinstance(value, str | int | float | bool):
        return value
    return repr(value)


def _digest_records(records: list[dict[str, object]]) -> str:
    encoded = json.dumps(
        records, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _is_sqlite_runtime_sidecar(path: Path) -> bool:
    """只在相邻主文件确为 SQLite 时识别它的运行 sidecar。"""

    # 1. 后缀条目必须是实体普通文件。
    if path.is_symlink() or not path.is_file():
        return False
    suffix = next(
        (
            candidate
            for candidate in ("-wal", "-shm", "-journal")
            if path.name.endswith(candidate)
        ),
        None,
    )
    if suffix is None:
        return False

    # 2. 同名主文件必须携带 SQLite 文件头，避免排除插件自有普通文件。
    database = path.with_name(path.name[: -len(suffix)])
    if database.is_symlink() or not database.is_file():
        return False
    with database.open("rb") as stream:
        return stream.read(len(SQLITE_HEADER)) == SQLITE_HEADER


def _tree_summary(
    root: Path,
    *,
    include_entries: bool = False,
    exclude_workspace_runtime: bool = False,
    exclude_sqlite_sidecars: bool = False,
) -> dict[str, object]:
    """Summarize a tree without following symlinks or exposing file contents."""

    records: list[dict[str, object]] = []
    if not root.exists():
        missing: list[dict[str, object]] = [{"kind": "missing", "path": "."}]
        return {
            "path": str(root), "exists": False, "digest": _digest_records(missing),
            "file_count": 0, "directory_count": 0, "symlink_count": 0,
            **({"entries": missing} if include_entries else {}),
        }

    def visit(directory: Path) -> None:
        for child in sorted(directory.iterdir(), key=lambda item: item.name):
            relative = child.relative_to(root).as_posix()
            if exclude_sqlite_sidecars and _is_sqlite_runtime_sidecar(child):
                continue
            if exclude_workspace_runtime and excluded_reason(
                Path(relative), is_symlink=child.is_symlink()
            ) is not None:
                continue
            if child.is_symlink():
                records.append({"kind": "symlink", "path": relative, "target": os.readlink(child)})
            elif child.is_dir():
                records.append({"kind": "directory", "path": relative})
                visit(child)
            elif child.is_file():
                records.append({
                    "kind": "file", "path": relative, "size": child.stat().st_size,
                    "sha256": _sha256_file(child),
                })
            else:
                raise GateFailure(f"不支持的 Workspace 文件系统条目: {child}")

    if root.is_symlink():
        raise GateFailure(f"Workspace root 不能是符号链接: {root}")
    if root.is_file():
        records.append({
            "kind": "file", "path": ".", "size": root.stat().st_size,
            "sha256": _sha256_file(root),
        })
    elif root.is_dir():
        visit(root)
    else:
        raise GateFailure(f"Workspace root 不是实体文件或目录: {root}")
    records.sort(key=lambda item: str(item["path"]))
    return {
        "path": str(root), "exists": True, "digest": _digest_records(records),
        "file_count": sum(item["kind"] == "file" for item in records),
        "directory_count": sum(item["kind"] == "directory" for item in records),
        "symlink_count": sum(item["kind"] == "symlink" for item in records),
        **({"entries": records} if include_entries else {}),
    }


def _artifact_inventory(
    root: Path, *, exclude_sqlite_sidecars: bool = False
) -> dict[str, object]:
    """Inventory artifact and pointer files by digest, never by contents."""

    summary = _tree_summary(
        root,
        include_entries=True,
        exclude_sqlite_sidecars=exclude_sqlite_sidecars,
    )
    entries = cast(list[dict[str, object]], summary.get("entries", []))
    artifacts: list[dict[str, object]] = []
    pointers: list[dict[str, object]] = []
    for entry in entries:
        if entry.get("kind") != "file":
            continue
        relative = str(entry["path"])
        path = Path(relative)
        if ".artifacts" in path.parts:
            artifacts.append(entry)
        if path.name in {"stable.json", "latest.json", "pointers.json"} or "pointer" in path.name.lower():
            pointers.append(entry)
    return {
        "root": str(root), "tree_digest": summary["digest"],
        "artifact_files": artifacts, "pointer_files": pointers,
        "artifact_digest": _digest_records(artifacts),
        "pointer_digest": _digest_records(pointers),
    }


def _sqlite_snapshot(path: Path) -> dict[str, object]:
    """Read-only check SQLite integrity and canonical existing message rows."""

    if not path.is_file():
        raise GateBlocked(f"sessions.db 不存在: {path}")
    uri = f"file:{quote(str(path.resolve()))}?mode=ro"
    try:
        connection = sqlite3.connect(uri, uri=True)
    except sqlite3.Error as error:
        raise GateFailure(f"无法以只读方式打开 sessions.db: {error}") from error
    try:
        integrity = str(connection.execute("PRAGMA integrity_check").fetchone()[0])
        if integrity != "ok":
            raise GateFailure(f"sessions.db integrity_check 失败: {integrity}")
        required = {"id", "session_key", "seq", "role", "content", "tool_chain", "extra", "ts"}
        columns = {str(row[1]) for row in connection.execute("PRAGMA table_info(messages)")}
        if not required.issubset(columns):
            raise GateBlocked("sessions.db 缺少 append-only messages schema: " + ",".join(sorted(required - columns)))
        rows = {
            str(row[0]): json.dumps([_freeze(item) for item in row[1:]], ensure_ascii=False, sort_keys=True)
            for row in connection.execute(
                "SELECT id, session_key, seq, role, content, tool_chain, extra, ts FROM messages ORDER BY id"
            )
        }
        return {
            "path": str(path), "integrity": integrity, "message_count": len(rows),
            "message_rows_digest": _digest_records([{"id": key, "row": rows[key]} for key in sorted(rows)]),
            "_message_rows": rows,
        }
    finally:
        connection.close()


def _append_only_evidence(
    before: Mapping[str, object], after: Mapping[str, object], *, label: str
) -> dict[str, object]:
    """Prove existing canonical messages were neither deleted nor rewritten."""

    old = cast(dict[str, str], before["_message_rows"])
    new = cast(dict[str, str], after["_message_rows"])
    removed = sorted(set(old) - set(new))
    changed = sorted(key for key in set(old) & set(new) if old[key] != new[key])
    if removed or changed:
        raise GateFailure(f"{label} 违反 append-only: removed={removed[:5]} changed={changed[:5]}")
    return {
        "status": "passed", "label": label,
        "integrity_before": before["integrity"], "integrity_after": after["integrity"],
        "existing_message_count": len(old), "new_message_count": len(new) - len(old),
        "removed": [], "changed": [],
    }


def _read_json(path: Path, label: str) -> dict[str, object]:
    if not path.is_file():
        raise GateBlocked(f"{label} report 不存在: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise GateBlocked(f"{label} report 无法读取: {type(error).__name__}: {error}") from error
    if not isinstance(payload, dict):
        raise GateBlocked(f"{label} report 顶层必须是 object")
    return cast(dict[str, object], payload)


def _mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise GateBlocked(f"{label} 必须是 object")
    return cast(dict[str, object], value)


def _report_lock_sha(report: Mapping[str, object], label: str) -> str:
    direct = report.get("lock_sha256")
    if isinstance(direct, str):
        return direct
    lock = report.get("lock")
    if isinstance(lock, dict):
        nested = cast(dict[str, object], lock).get("sha256")
        if isinstance(nested, str):
            return nested
    raise GateBlocked(f"{label} report 缺少 lock_sha256")


def _validate_core_identity(
    report: Mapping[str, object], label: str, current: Mapping[str, str], lock_sha: str
) -> dict[str, object]:
    core = _mapping(report.get("core"), f"{label}.core")
    head, tree = core.get("head"), core.get("tree")
    if head != current["head"] or tree != current["tree"]:
        raise GateBlocked(
            f"{label} core identity 不是当前 HEAD/tree: report={head}/{tree} "
            f"current={current['head']}/{current['tree']}"
        )
    actual_lock_sha = _report_lock_sha(report, label)
    if actual_lock_sha != lock_sha:
        raise GateBlocked(f"{label} lock identity 不匹配: report={actual_lock_sha} current={lock_sha}")
    return {"head": head, "tree": tree, "lock_sha256": actual_lock_sha}


def _mapping_by_id(value: object, label: str) -> dict[str, dict[str, object]]:
    if not isinstance(value, list):
        raise GateBlocked(f"{label} 必须是列表")
    result: dict[str, dict[str, object]] = {}
    for raw in cast(list[object], value):
        item = _mapping(raw, f"{label} item")
        scenario_id = item.get("id")
        if isinstance(scenario_id, str):
            result[scenario_id] = item
    return result


def _validate_report(
    path: Path, label: str, current: Mapping[str, str], lock_sha: str
) -> dict[str, object]:
    """Validate one final report and return compact identity evidence."""

    report = _read_json(path, label)
    if report.get("status") != "passed":
        raise GateBlocked(f"{label} report.status 不是 passed: {report.get('status')!r}")
    identity = _validate_core_identity(report, label, current, lock_sha)
    if label == "E1":
        if report.get("phase") != "e1":
            raise GateBlocked(f"E1 phase 不匹配: {report.get('phase')!r}")
        scenarios = _mapping_by_id(report.get("scenarios"), "E1.scenarios")
        for scenario_id in E1_SCENARIOS:
            if scenarios.get(scenario_id, {}).get("status") != "passed":
                raise GateBlocked(f"E1 缺少 passed 场景: {scenario_id}")
        runtime = _mapping(report.get("runtime"), "E1.runtime")
        for engine in E1_RUNTIME_ENGINES:
            if engine not in runtime:
                raise GateBlocked(f"E1 缺少 {engine} data-read boot evidence")
    elif label == "E2":
        if report.get("scenario_profile") != E2_PROFILE:
            raise GateBlocked(f"E2 scenario_profile 不匹配: {report.get('scenario_profile')!r}")
        crash = _mapping(report.get("core_process_crash"), "E2.core_process_crash")
        if crash.get("status") not in {"passed", None}:
            raise GateBlocked(f"E2 Core crash recovery 未通过: {crash}")
    elif label == "E3":
        if report.get("scenario_profile") != E3_PROFILE:
            raise GateBlocked(f"E3 scenario_profile 不匹配: {report.get('scenario_profile')!r}")
        runtime = _mapping(report.get("runtime"), "E3.runtime")
        for field in ("channel", "message_push", "channel_cleanup"):
            if field not in runtime:
                raise GateBlocked(f"E3 runtime 缺少 {field}")
    elif label == "Passive WebUI":
        if report.get("scenario_profile") != PASSIVE_PROFILE:
            raise GateBlocked(f"Passive WebUI scenario_profile 不匹配: {report.get('scenario_profile')!r}")
        runtime = _mapping(report.get("runtime"), "Passive WebUI.runtime")
        if runtime.get("status") != "passed":
            raise GateBlocked("Passive WebUI runtime.status 不是 passed")
        cleanup = _mapping(report.get("cleanup"), "Passive WebUI.cleanup")
        if cleanup.get("residuals") != [] or cleanup.get("sandbox_removed") is not True:
            raise GateBlocked("Passive WebUI cleanup 证据不完整")
    else:
        raise GateFailure(f"未知 report label: {label}")
    return {
        "label": label, "path": str(path), "status": "passed",
        "scenario_profile": report.get("scenario_profile", "e1"), "core": identity,
    }


def validate_final_reports(
    *, e1_report: Path, e2_report: Path, e3_report: Path,
    passive_webui_report: Path, current_identity: Mapping[str, str], lock_sha256: str,
) -> dict[str, object]:
    """Consume exact final reports without rerunning any upstream E gate."""

    reports = [
        _validate_report(e1_report, "E1", current_identity, lock_sha256),
        _validate_report(e2_report, "E2", current_identity, lock_sha256),
        _validate_report(e3_report, "E3", current_identity, lock_sha256),
        _validate_report(passive_webui_report, "Passive WebUI", current_identity, lock_sha256),
    ]
    return {"status": "passed", "reports": reports}


def _git_output(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args], cwd=ROOT, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, check=False,
    )
    if completed.returncode != 0:
        raise GateFailure(f"git {' '.join(args)} 失败: {completed.stderr.strip()}")
    return completed.stdout.strip()


def _current_identity(lock_path: Path) -> dict[str, str]:
    if not lock_path.is_file():
        raise GateBlocked(f"fleet lock 不存在: {lock_path}")
    return {
        "head": _git_output("rev-parse", "HEAD"),
        "tree": _git_output("rev-parse", "HEAD^{tree}"),
        "lock_sha256": _sha256_file(lock_path),
    }


def _fleet_coverage(lock_path: Path) -> dict[str, object]:
    """Report exact fleet IDs not represented by completed E1-E3 lanes."""

    try:
        payload = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise GateBlocked(f"fleet lock 无法读取: {type(error).__name__}: {error}") from error
    if not isinstance(payload, dict) or not isinstance(payload.get("plugins"), list):
        raise GateBlocked("fleet lock.plugins 必须是列表")
    expected: set[str] = set()
    for raw in cast(list[object], payload["plugins"]):
        plugin = _mapping(raw, "fleet lock plugin")
        plugin_id = plugin.get("id")
        if not isinstance(plugin_id, str) or not plugin_id:
            raise GateBlocked(f"fleet lock plugin.id 无效: {plugin_id!r}")
        expected.add(plugin_id)
    covered = E1_PLUGINS | E2_PLUGINS | E3_PLUGINS
    missing = sorted(expected - covered)
    return {
        "status": "passed" if not missing else "blocked",
        "expected_ids": sorted(expected), "covered_ids": sorted(expected & covered),
        "missing_ids": missing,
        "reason": "full fleet exact provider/runtime coverage unavailable" if missing else None,
    }


def _copy_tree(source: Path, target: Path) -> None:
    """Copy a prepared workspace into another disposable scenario root."""

    shutil.copytree(source, target, symlinks=True)


async def _run_builtin_boot(workspace: Path) -> dict[str, object]:
    """Boot real in-tree Akasha/Default Memory from copied data."""

    plugin_dirs = e1_gate._plugin_dirs({})  # pyright: ignore[reportPrivateUsage]
    evidence: dict[str, object] = {}
    for engine in E1_RUNTIME_ENGINES:
        runtime_workspace = workspace.parent / f"e4-runtime-{engine}"
        _copy_tree(workspace, runtime_workspace)
        bundle: Any | None = None
        try:
            bundle = await e1_gate._open_runtime(  # pyright: ignore[reportPrivateUsage]
                runtime_workspace, engine, plugin_dirs
            )
            boot = await e1_gate._probe_boot(bundle)  # pyright: ignore[reportPrivateUsage]
            if engine == "akasha":
                mobile = _mapping(boot.get("mobile_query"), "Akasha mobile query")
                if mobile.get("status") != "passed":
                    raise GateBlocked(f"Akasha copied data-read boot unavailable: {mobile}")
            evidence[engine] = {"status": "passed", "boot": boot}
        finally:
            if bundle is not None:
                cleanup = await e1_gate._close_runtime(bundle)  # pyright: ignore[reportPrivateUsage]
                if cleanup:
                    raise GateFailure(f"{engine} graceful stop cleanup 失败: {cleanup}")
    return {"status": "passed", "engines": evidence}


async def _run_in_process_failure(workspace: Path) -> dict[str, object]:
    """Exercise a closed memory owner, then recover its pending receipt."""

    plugin_dirs = e1_gate._plugin_dirs({})  # pyright: ignore[reportPrivateUsage]
    runtime_workspace = workspace.parent / "e4-in-process-runtime"
    _copy_tree(workspace, runtime_workspace)
    first: Any | None = None
    failure = ""
    try:
        first = await e1_gate._open_runtime(  # pyright: ignore[reportPrivateUsage]
            runtime_workspace, "default", plugin_dirs
        )
        _ = e1_gate._seed_interaction(  # pyright: ignore[reportPrivateUsage]
            first.sessions,
            key="e4:in-process",
            turn="turn:e4:in-process",
            label="in-process-failure",
        )
        deletion = first.sessions.control_store.delete_interaction(
            "turn:e4:in-process",
            action_source="plugin_v3_e4_gate.in_process_failure",
            expected_latest_session_key="e4:in-process",
            reconciliation_owner="default_memory",
        )
        if deletion is None:
            raise GateFailure("in-process failure seed delete 未提交")
        store = e1_gate._memory_store(first)  # pyright: ignore[reportPrivateUsage]
        store.close()
        try:
            await InteractionUndoCoordinator(
                first.sessions, first.memory.engine
            ).recover_pending()
        except Exception as error:
            failure = f"{type(error).__name__}: {error}"
        if not failure:
            raise GateFailure("closed MemoryStore2 未触发进程内失败")
        pending = first.sessions.control_store.pending_interaction_memory_reconciliations(
            "default_memory"
        )
        if len(pending) != 1 or pending[0].attempts != 1:
            raise GateFailure(f"进程内失败后 pending receipt 异常: {pending}")
    finally:
        if first is not None:
            cleanup = await e1_gate._close_runtime(first)  # pyright: ignore[reportPrivateUsage]
            if cleanup:
                raise GateFailure(f"进程内失败前 cleanup 失败: {cleanup}")

    restarted: Any | None = None
    try:
        restarted = await e1_gate._open_runtime(  # pyright: ignore[reportPrivateUsage]
            runtime_workspace, "default", plugin_dirs
        )
        await InteractionUndoCoordinator(
            restarted.sessions, restarted.memory.engine
        ).recover_pending()
        remaining = restarted.sessions.control_store.pending_interaction_memory_reconciliations(
            "default_memory"
        )
        if remaining or restarted.sessions.get_existing("e4:in-process").messages:
            raise GateFailure(f"进程内失败重启恢复不完整: pending={remaining}")
    finally:
        if restarted is not None:
            cleanup = await e1_gate._close_runtime(restarted)  # pyright: ignore[reportPrivateUsage]
            if cleanup:
                raise GateFailure(f"进程内失败重启 cleanup 失败: {cleanup}")
    return {"status": "passed", "failure_observed": failure, "pending_after_restart": 0}


def _run_sigkill_child(workspace: Path) -> dict[str, object]:
    """Commit a pending reconciliation and SIGKILL the child before recovery."""

    child_lines = [
        "import os, signal, sys",
        "from datetime import UTC, datetime",
        "from pathlib import Path",
        "from session.manager import SessionManager",
        "sessions = SessionManager(Path(sys.argv[1]))",
        "now = datetime.now(UTC).isoformat()",
        "sessions.control_store.persist_session(",
        "    'e4:process-crash', created_at=now, updated_at=now,",
        "    metadata={'gate': 'plugin_v3_e4_gate'},",
        "    messages=[",
        "        {'role': 'user', 'content': 'E4 process crash', 'timestamp': now,",
        "         'extra': {'control_turn_id': 'turn:e4:process-crash', 'turn_input_ordinal': 0}},",
        "        {'role': 'assistant', 'content': 'E4 pending', 'timestamp': now,",
        "         'extra': {'control_turn_id': 'turn:e4:process-crash', 'turn_terminal': True, 'turn_input_count': 1}},",
        "    ],",
        ")",
        "deletion = sessions.control_store.delete_interaction(",
        "    'turn:e4:process-crash', action_source='plugin_v3_e4_gate.process_crash',",
        "    expected_latest_session_key='e4:process-crash', reconciliation_owner='default_memory',",
        ")",
        "if deletion is None: raise RuntimeError('E4 process crash seed delete 未提交')",
        "os.kill(os.getpid(), signal.SIGKILL)",
    ]
    child = subprocess.run(
        [sys.executable, "-c", "\n".join(child_lines), str(workspace)],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=30,
    )
    expected = -signal.SIGKILL
    if child.returncode != expected:
        raise GateFailure(
            f"Core process SIGKILL child exit 异常: {child.returncode}; "
            f"stderr={child.stderr.strip()}"
        )
    return {"status": "passed", "child_exit_code": child.returncode}


async def _run_process_crash(workspace: Path) -> dict[str, object]:
    """Recover a SIGKILL-created pending receipt in a newly booted Core."""

    plugin_dirs = e1_gate._plugin_dirs({})  # pyright: ignore[reportPrivateUsage]
    runtime_workspace = workspace.parent / "e4-process-crash-runtime"
    _copy_tree(workspace, runtime_workspace)
    crash = _run_sigkill_child(runtime_workspace)
    bundle: Any | None = None
    try:
        bundle = await e1_gate._open_runtime(  # pyright: ignore[reportPrivateUsage]
            runtime_workspace, "default", plugin_dirs
        )
        await InteractionUndoCoordinator(
            bundle.sessions, bundle.memory.engine
        ).recover_pending()
        remaining = bundle.sessions.control_store.pending_interaction_memory_reconciliations(
            "default_memory"
        )
        if remaining or bundle.sessions.get_existing("e4:process-crash").messages:
            raise GateFailure(f"SIGKILL 后恢复不完整: pending={remaining}")
    finally:
        if bundle is not None:
            cleanup = await e1_gate._close_runtime(bundle)  # pyright: ignore[reportPrivateUsage]
            if cleanup:
                raise GateFailure(f"SIGKILL 重启 cleanup 失败: {cleanup}")
    return {**crash, "pending_after_restart": 0, "status": "passed"}


def _sanitize_sqlite(snapshot: Mapping[str, object]) -> dict[str, object]:
    return {key: value for key, value in snapshot.items() if key != "_message_rows"}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行 pure-v3 copied-workspace E4 rehearsal")
    parser.add_argument("--source-workspace", type=Path, required=True)
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--plugin-home", type=Path, required=True)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--e1-report", type=Path, default=DEFAULT_E1_REPORT)
    parser.add_argument("--e2-report", type=Path, default=DEFAULT_E2_REPORT)
    parser.add_argument("--e3-report", type=Path, default=DEFAULT_E3_REPORT)
    parser.add_argument("--passive-webui-report", type=Path, default=DEFAULT_PASSIVE_REPORT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--tmp-root", type=Path)
    return parser.parse_args()


async def _run_runtime(args: argparse.Namespace, report: dict[str, object]) -> None:
    lock_path = args.lock.resolve()
    current = _current_identity(lock_path)
    report["core"] = current
    report["reports"] = validate_final_reports(
        e1_report=args.e1_report.resolve(),
        e2_report=args.e2_report.resolve(),
        e3_report=args.e3_report.resolve(),
        passive_webui_report=args.passive_webui_report.resolve(),
        current_identity=current,
        lock_sha256=current["lock_sha256"],
    )
    report["fleet_coverage"] = _fleet_coverage(lock_path)
    source_workspace = args.source_workspace.resolve(strict=True)
    source_config = args.source_config.resolve(strict=True)
    plugin_home = args.plugin_home.resolve(strict=True)
    source_before = {
        "workspace": _tree_summary(source_workspace, exclude_workspace_runtime=True),
        "config": _tree_summary(source_config),
        "plugin_home": _tree_summary(plugin_home),
        "plugin_data": _artifact_inventory(
            source_workspace / "plugin-data", exclude_sqlite_sidecars=True
        ),
        "artifact_pointer": _artifact_inventory(plugin_home),
    }
    source_db_before = _sqlite_snapshot(source_workspace / "sessions.db")
    report["source_before"] = {
        **source_before, "sessions_db": _sanitize_sqlite(source_db_before),
    }
    tmp_parent = _resolve_tmp_root(args.tmp_root)
    with tempfile.TemporaryDirectory(prefix="akashic-plugin-v3-e4-", dir=tmp_parent) as raw:
        target = Path(raw) / "rehearsal"
        manifest = prepare_rehearsal(
            source_workspace=source_workspace,
            source_config=source_config,
            plugin_home=plugin_home,
            target=target,
        )
        copied_workspace = target / "workspace"
        copied_db_before = _sqlite_snapshot(copied_workspace / "sessions.db")
        before_artifact = _artifact_inventory(target / "plugin-home")
        report["rehearsal_copy"] = {
            "status": "passed", "manifest": str(manifest), "target": str(target),
            "workspace": _tree_summary(copied_workspace),
            "plugin_data": _artifact_inventory(
                copied_workspace / "plugin-data", exclude_sqlite_sidecars=True
            ),
            "artifact_pointer": before_artifact,
        }
        report["builtin_e1_data_read_boot"] = await _run_builtin_boot(copied_workspace)
        report["in_process_failure"] = await _run_in_process_failure(copied_workspace)
        report["process_crash_restart"] = await _run_process_crash(copied_workspace)
        copied_db_after = _sqlite_snapshot(copied_workspace / "sessions.db")
        report["copied_sessions_append_only"] = _append_only_evidence(
            copied_db_before, copied_db_after, label="rehearsal copy sessions.db"
        )
        copied_artifact_after = _artifact_inventory(target / "plugin-home")
        if before_artifact != copied_artifact_after:
            raise GateFailure("copied artifact/pointer inventory 在生命周期中发生变化")
        report["artifact_pointer_after"] = copied_artifact_after
        report["plugin_data_after"] = _artifact_inventory(
            copied_workspace / "plugin-data", exclude_sqlite_sidecars=True
        )
        report["graceful_stop_cleanup"] = {
            "status": "passed", "runtime_directories_removed_with_rehearsal": True,
        }
    source_after = {
        "workspace": _tree_summary(source_workspace, exclude_workspace_runtime=True),
        "config": _tree_summary(source_config),
        "plugin_home": _tree_summary(plugin_home),
        "plugin_data": _artifact_inventory(
            source_workspace / "plugin-data", exclude_sqlite_sidecars=True
        ),
        "artifact_pointer": _artifact_inventory(plugin_home),
    }
    source_db_after = _sqlite_snapshot(source_workspace / "sessions.db")
    report["source_after"] = {
        **source_after, "sessions_db": _sanitize_sqlite(source_db_after),
    }
    report["source_sessions_append_only"] = _append_only_evidence(
        source_db_before, source_db_after, label="source sessions.db"
    )
    if source_before != source_after:
        raise GateFailure("source workspace/config/plugin-home 生命周期摘要发生变化")
    report["source_unchanged"] = True


def main() -> int:
    """Run E4 and return blocked/failed status without hiding evidence."""

    args = _parse_args()
    report_path = args.report.resolve()
    report: dict[str, object] = {
        "status": "failed", "gate_version": GATE_VERSION,
        "scenario_profile": SCENARIO_PROFILE,
        "checked_at": datetime.now(UTC).isoformat(), "blockers": [], "failures": [],
    }
    exit_code = 1
    try:
        asyncio.run(_run_runtime(args, report))
        coverage = cast(dict[str, object], report.get("fleet_coverage", {}))
        if coverage.get("status") != "passed":
            report["status"] = "blocked"
            report["blockers"] = [str(coverage.get("reason") or coverage)]
            exit_code = 2
        else:
            report["status"] = "passed"
            exit_code = 0
    except GateBlocked as error:
        report["status"] = "blocked"
        report["blockers"] = [f"{type(error).__name__}: {error}"]
        exit_code = 2
    except (GateFailure, OSError, RuntimeError, sqlite3.Error, subprocess.SubprocessError) as error:
        report["status"] = "failed"
        report["failures"] = [f"{type(error).__name__}: {error}"]
        exit_code = 1
    finally:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(f"plugin v3 E4 {report['status']}: {report_path}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""从运行中的 Workspace 创建只供容器预演使用的一致副本。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sqlite3
import tempfile
import tomllib
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

SQLITE_HEADER = b"SQLite format 3\x00"
EXCLUDED_DIRECTORY_NAMES = frozenset(
    {
        ".git",
        ".tmp",
        ".venv",
        "backups",
        "cache",
        "downloads",
        "mcp-env",
        "rebuilds",
        "runtime",
        "sandboxes",
        "subagent-runs",
        "venv",
    }
)
EXCLUDED_RUNTIME_FILES = frozenset(
    {
        ".app-server-token",
        ".instance.lock",
        ".runtime-ready.json",
        ".supervisor.lock",
        ".supervisor.pid",
        "akashic.sock",
    }
)
WEBUI_ONLY_SETTINGS: tuple[tuple[str, str, Any], ...] = (
    ("channels.chat", "enabled", True),
    ("channels.telegram", "enabled", False),
    ("channels.telegram", "token", ""),
    ("channels.qq", "enabled", False),
    ("channels.qq", "bot_uin", ""),
    ("mobile_realtime", "enabled", False),
    ("mobile_realtime", "public_url", ""),
    ("proactive", "enabled", False),
    ("proactive.target", "channel", ""),
    ("proactive.target", "chat_id", ""),
    ("proactive.drift", "enabled", False),
)
MAX_WORKSPACE_SNAPSHOT_ATTEMPTS = 3


@dataclass(frozen=True)
class CopyRecord:
    path: str
    kind: str
    size: int
    sha256: str | None = None
    link_target: str | None = None


@dataclass(frozen=True)
class SourceEntry:
    relative: Path
    kind: str
    size: int
    mtime_ns: int
    mode: int
    link_target: str | None = None
    sha256: str | None = None


class SnapshotDriftError(RuntimeError):
    """表示在线复制期间 Workspace 的纳入集合发生变化。"""


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        _ = path.relative_to(root)
    except ValueError:
        return False
    return True


def _validate_roots(
    *, source_workspace: Path, source_config: Path, plugin_home: Path, target: Path
) -> tuple[Path, Path, Path, Path]:
    """解析全部边界路径，并拒绝覆盖或递归复制。"""

    # 1. 源必须已经存在，目标必须尚未创建。
    source_workspace = source_workspace.expanduser().resolve(strict=True)
    source_config = source_config.expanduser().resolve(strict=True)
    plugin_home = plugin_home.expanduser().resolve(strict=True)
    target = target.expanduser().absolute()
    if not source_workspace.is_dir():
        raise NotADirectoryError(source_workspace)
    if not source_config.is_file():
        raise FileNotFoundError(source_config)
    if not plugin_home.is_dir():
        raise NotADirectoryError(plugin_home)
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"预演目标必须是尚不存在的新路径: {target}")

    # 2. 目标不能与任何源重叠，避免读写正式状态或递归吞入副本。
    target_resolved = target.resolve(strict=False)
    for label, source in (
        ("Workspace", source_workspace),
        ("配置", source_config),
        ("插件目录", plugin_home),
    ):
        if target_resolved == source or _is_relative_to(target_resolved, source):
            raise ValueError(f"预演目标不能位于源{label}内部: {target}")
        if source.is_dir() and _is_relative_to(source, target_resolved):
            raise ValueError(f"源{label}不能位于预演目标内部: {source}")
    return source_workspace, source_config, plugin_home, target


def _excluded_reason(relative: Path, *, is_symlink: bool) -> str | None:
    """返回排除原因；未命中时允许该路径进入隔离副本。"""

    parts = relative.parts
    name = relative.name
    if any(part in EXCLUDED_DIRECTORY_NAMES for part in parts):
        return "excluded_state_class"
    if any(part.endswith("_rebuild") for part in parts):
        return "rebuild_artifact"
    if any(part.startswith("mobile-webui-build-") for part in parts):
        return "temporary_webui_build"
    if len(parts) >= 2 and parts[-2:] in {
        ("mobile-webui", "staging"),
        ("mobile-webui", "trash"),
    }:
        return "temporary_webui_state"
    if name in EXCLUDED_RUNTIME_FILES:
        return "runtime_control_file"
    if name.endswith(("-wal", "-shm", "-journal")):
        return "sqlite_sidecar"
    if is_symlink and (parts[:1] == ("skills",) or parts[:2] == ("drift", "skills")):
        return "rebuildable_skill_projection"
    return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sqlite(path: Path) -> bool:
    with path.open("rb") as stream:
        return stream.read(len(SQLITE_HEADER)) == SQLITE_HEADER


def _integrity_rows(connection: sqlite3.Connection) -> list[tuple[str]]:
    return connection.execute("PRAGMA integrity_check").fetchall()


def _copy_sqlite(source: Path, destination: Path) -> dict[str, Any]:
    """在线备份一个 SQLite，并验证源与副本完整性。"""

    source_uri = f"{source.resolve().as_uri()}?mode=ro"
    with closing(sqlite3.connect(source_uri, uri=True, timeout=30)) as source_db:
        _ = source_db.execute("PRAGMA query_only = ON")
        source_integrity = _integrity_rows(source_db)
        if source_integrity != [("ok",)]:
            raise sqlite3.DatabaseError(
                f"源 SQLite integrity_check 失败: {source}: {source_integrity[:3]}"
            )
        with closing(sqlite3.connect(destination)) as destination_db:
            source_db.backup(destination_db, pages=256, sleep=0.05)
            destination_db.commit()
            target_integrity = _integrity_rows(destination_db)
            if target_integrity != [("ok",)]:
                raise sqlite3.DatabaseError(
                    f"副本 SQLite integrity_check 失败: {destination}: "
                    f"{target_integrity[:3]}"
                )
            page_count = int(destination_db.execute("PRAGMA page_count").fetchone()[0])
    shutil.copymode(source, destination, follow_symlinks=False)
    return {
        "source_integrity_check": "ok",
        "target_integrity_check": "ok",
        "page_count": page_count,
    }


def _stable_copy(source: Path, destination: Path) -> tuple[os.stat_result, str]:
    """复制普通文件；源在复制期间变化时有限重试后明确失败。"""

    for _attempt in range(3):
        before = source.stat()
        _ = shutil.copy2(source, destination)
        after = source.stat()
        identity_before = (before.st_ino, before.st_size, before.st_mtime_ns)
        identity_after = (after.st_ino, after.st_size, after.st_mtime_ns)
        target_digest = _sha256(destination)
        if identity_before == identity_after and _sha256(source) == target_digest:
            return after, target_digest
        _ = destination.unlink()
    raise SnapshotDriftError(f"普通文件在快照期间持续变化: {source}")


def _copy_symlink(source: Path, destination: Path, source_root: Path) -> CopyRecord:
    """只复制仍指向 Workspace 内部的相对软链。"""

    link_target = os.readlink(source)
    if os.path.isabs(link_target):
        raise ValueError(f"拒绝复制绝对符号链接: {source} -> {link_target}")
    resolved = (source.parent / link_target).resolve(strict=False)
    if not _is_relative_to(resolved, source_root):
        raise ValueError(
            f"拒绝复制逃逸 Workspace 的符号链接: {source} -> {link_target}"
        )
    destination.symlink_to(link_target, target_is_directory=source.is_dir())
    return CopyRecord(
        path=str(destination), kind="symlink", size=0, link_target=link_target
    )


def _copy_workspace(
    source: Path, destination: Path
) -> tuple[
    list[CopyRecord], list[dict[str, str]], list[dict[str, Any]], dict[str, Any]
]:
    """以有界整轮重试复制 Workspace，拒绝跨数据库窗口的文件漂移。"""

    drift_messages: list[str] = []
    for attempt in range(1, MAX_WORKSPACE_SNAPSHOT_ATTEMPTS + 1):
        if destination.exists():
            shutil.rmtree(destination)
        try:
            records, exclusions, databases = _copy_workspace_once(source, destination)
            return (
                records,
                exclusions,
                databases,
                {
                    "attempts": attempt,
                    "max_attempts": MAX_WORKSPACE_SNAPSHOT_ATTEMPTS,
                    "drift_retries": drift_messages,
                },
            )
        except SnapshotDriftError as exc:
            drift_messages.append(str(exc))
            if destination.exists():
                shutil.rmtree(destination)
            if attempt == MAX_WORKSPACE_SNAPSHOT_ATTEMPTS:
                raise RuntimeError(
                    "Workspace 在全部一致性快照尝试中持续变化: "
                    + " | ".join(drift_messages)
                ) from exc
    raise AssertionError("unreachable")


def _scan_workspace(
    source: Path,
) -> tuple[dict[Path, SourceEntry], list[dict[str, str]]]:
    """不跟随软链并在源端剪枝，固定当前纳入集合。"""

    entries: dict[Path, SourceEntry] = {}
    exclusions: list[dict[str, str]] = []

    def visit(directory: Path, parent: Path) -> None:
        try:
            with os.scandir(directory) as iterator:
                children = sorted(iterator, key=lambda item: item.name)
        except FileNotFoundError as exc:
            raise SnapshotDriftError(f"扫描期间目录消失: {directory}") from exc
        for child in children:
            item = Path(child.path)
            relative = parent / child.name
            try:
                is_symlink = child.is_symlink()
                reason = _excluded_reason(relative, is_symlink=is_symlink)
                if reason is not None:
                    exclusions.append({"path": relative.as_posix(), "reason": reason})
                    continue
                metadata = child.stat(follow_symlinks=False)
                if is_symlink:
                    kind = "symlink"
                    link_target = os.readlink(item)
                elif child.is_dir(follow_symlinks=False):
                    kind = "directory"
                    link_target = None
                elif child.is_file(follow_symlinks=False):
                    kind = "sqlite" if _is_sqlite(item) else "file"
                    link_target = None
                else:
                    exclusions.append(
                        {
                            "path": relative.as_posix(),
                            "reason": "non_regular_filesystem_entry",
                        }
                    )
                    continue
            except FileNotFoundError as exc:
                raise SnapshotDriftError(f"扫描期间路径消失: {item}") from exc
            entries[relative] = SourceEntry(
                relative=relative,
                kind=kind,
                size=metadata.st_size,
                mtime_ns=metadata.st_mtime_ns,
                mode=metadata.st_mode,
                link_target=link_target,
            )
            if kind == "directory":
                visit(item, relative)

    visit(source, Path())
    return entries, exclusions


def _copy_workspace_once(
    source: Path, destination: Path
) -> tuple[list[CopyRecord], list[dict[str, str]], list[dict[str, Any]]]:
    """先复制普通文件，再备份数据库，最后核对整个纳入集合。"""

    initial, exclusions = _scan_workspace(source)
    baseline = dict(initial)
    records: list[CopyRecord] = []
    databases: list[dict[str, Any]] = []
    destination.mkdir(mode=0o700)

    # 1. 普通目录、软链和文件先完成稳定复制与摘要确认。
    for relative, entry in initial.items():
        item = source / relative
        target = destination / relative
        if entry.kind == "directory":
            target.mkdir(parents=True, exist_ok=True)
            shutil.copymode(item, target, follow_symlinks=False)
        elif entry.kind == "symlink":
            target.parent.mkdir(parents=True, exist_ok=True)
            record = _copy_symlink(item, target, source)
            records.append(
                CopyRecord(
                    path=relative.as_posix(),
                    kind=record.kind,
                    size=record.size,
                    link_target=record.link_target,
                )
            )
        elif entry.kind == "file":
            target.parent.mkdir(parents=True, exist_ok=True)
            metadata, digest = _stable_copy(item, target)
            baseline[relative] = SourceEntry(
                relative=relative,
                kind="file",
                size=metadata.st_size,
                mtime_ns=metadata.st_mtime_ns,
                mode=metadata.st_mode,
                sha256=digest,
            )
            records.append(
                CopyRecord(
                    path=relative.as_posix(),
                    kind="file",
                    size=target.stat().st_size,
                    sha256=digest,
                )
            )

    # 2. 只有普通状态稳定后，才为每个数据库取得在线一致快照。
    for relative, entry in initial.items():
        if entry.kind != "sqlite":
            continue
        item = source / relative
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        check = _copy_sqlite(item, target)
        databases.append({"path": relative.as_posix(), **check})
        records.append(
            CopyRecord(
                path=relative.as_posix(),
                kind="sqlite_online_backup",
                size=target.stat().st_size,
                sha256=_sha256(target),
            )
        )

    # 3. 数据库窗口关闭后重新扫描；不允许数据库引用一个未纳入的文件世代。
    final, _final_exclusions = _scan_workspace(source)
    _verify_workspace_unchanged(source, destination, baseline, final)
    _verify_session_media_references(source, destination, databases)
    return records, exclusions, databases


def _verify_workspace_unchanged(
    source: Path,
    destination: Path,
    baseline: dict[Path, SourceEntry],
    final: dict[Path, SourceEntry],
) -> None:
    """核对数据库备份窗口两侧的纳入路径与普通文件世代完全相同。"""

    if baseline.keys() != final.keys():
        added = sorted(path.as_posix() for path in final.keys() - baseline.keys())
        removed = sorted(path.as_posix() for path in baseline.keys() - final.keys())
        raise SnapshotDriftError(
            f"纳入路径集合变化: added={added[:8]} removed={removed[:8]}"
        )
    for relative, expected in baseline.items():
        actual = final[relative]
        if expected.kind != actual.kind:
            raise SnapshotDriftError(
                f"路径类型变化: {relative}: {expected.kind} -> {actual.kind}"
            )
        if expected.kind == "sqlite":
            continue
        expected_metadata = (expected.size, expected.mtime_ns, expected.mode)
        actual_metadata = (actual.size, actual.mtime_ns, actual.mode)
        if (
            expected_metadata != actual_metadata
            or expected.link_target != actual.link_target
        ):
            raise SnapshotDriftError(f"普通路径元数据变化: {relative}")
        if expected.kind == "file":
            source_digest = _sha256(source / relative)
            if source_digest != expected.sha256 or source_digest != _sha256(
                destination / relative
            ):
                raise SnapshotDriftError(f"普通文件内容变化: {relative}")


def _verify_session_media_references(
    source: Path, destination: Path, databases: list[dict[str, Any]]
) -> None:
    """验证 SessionDB 中明确属于 Workspace 的媒体路径已进入副本。"""

    session_database = destination / "sessions.db"
    record = next((item for item in databases if item["path"] == "sessions.db"), None)
    if record is None or not session_database.is_file():
        return
    checked = 0
    with closing(sqlite3.connect(session_database)) as connection:
        columns = {
            str(row[1])
            for row in connection.execute("PRAGMA table_info(messages)").fetchall()
        }
        if not {"id", "extra"}.issubset(columns):
            record["workspace_media_references"] = "schema_not_applicable"
            return
        rows = connection.execute(
            "SELECT id, extra FROM messages WHERE extra LIKE ?", ('%"media"%',)
        ).fetchall()
    for message_id, raw_extra in rows:
        try:
            extra = json.loads(raw_extra or "{}")
        except (TypeError, ValueError) as exc:
            raise sqlite3.DatabaseError(
                f"SessionDB message extra JSON 损坏: {message_id}"
            ) from exc
        extra_dict = cast(dict[str, object], extra) if isinstance(extra, dict) else {}
        media: object = extra_dict.get("media")
        if media is None:
            continue
        media_items = cast(list[object], media) if isinstance(media, list) else []
        if not isinstance(media, list) or not all(
            isinstance(item, str) for item in media_items
        ):
            raise sqlite3.DatabaseError(
                f"SessionDB message media 不是字符串数组: {message_id}"
            )
        for item in cast(list[str], media_items):
            media_path = Path(item).expanduser()
            if not media_path.is_absolute():
                continue
            resolved = media_path.resolve(strict=False)
            if not _is_relative_to(resolved, source):
                continue
            relative = resolved.relative_to(source)
            if not (destination / relative).is_file():
                raise SnapshotDriftError(
                    f"SessionDB 引用的 Workspace 媒体未进入副本: {relative}"
                )
            checked += 1
    record["workspace_media_references"] = {"checked": checked, "status": "ok"}


def _toml_literal(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    raise TypeError(f"不支持的 TOML 值: {type(value).__name__}")


def _set_toml_value(content: str, section: str, key: str, value: Any) -> str:
    """在不重写其他字段的情况下设置一个普通 TOML 表字段。"""

    lines = content.splitlines(keepends=True)
    section_pattern = re.compile(rf"^\s*\[{re.escape(section)}\]\s*(?:#.*)?(?:\r?\n)?$")
    heading_pattern = re.compile(r"^\s*\[\[?[^]]+\]\]?\s*(?:#.*)?(?:\r?\n)?$")
    key_pattern = re.compile(rf"^\s*{re.escape(key)}\s*=")
    start = next(
        (index for index, line in enumerate(lines) if section_pattern.match(line)), None
    )
    assignment = f"{key} = {_toml_literal(value)}\n"
    if start is None:
        if lines and not lines[-1].endswith(("\n", "\r")):
            lines[-1] += "\n"
        if lines and lines[-1].strip():
            lines.append("\n")
        lines.extend((f"[{section}]\n", assignment))
        return "".join(lines)
    end = next(
        (
            index
            for index in range(start + 1, len(lines))
            if heading_pattern.match(lines[index])
        ),
        len(lines),
    )
    for index in range(start + 1, end):
        if key_pattern.match(lines[index]):
            lines[index] = assignment
            return "".join(lines)
    lines.insert(end, assignment)
    return "".join(lines)


def _write_candidate_config(source: Path, destination: Path, workspace: Path) -> None:
    """保留模型注册表配置，同时生成没有外部投递面的候选配置。"""

    source_content = source.read_text(encoding="utf-8")
    source_document = tomllib.loads(source_content)
    candidate = _set_toml_value(source_content, "runtime", "workspace", str(workspace))
    for section, key, value in WEBUI_ONLY_SETTINGS:
        candidate = _set_toml_value(candidate, section, key, value)
    candidate_document = tomllib.loads(candidate)

    # 模型选择和 registry 引用必须逐项保持，不能为禁用频道顺手重写。
    if candidate_document.get("llm") != source_document.get("llm"):
        raise RuntimeError("生成候选配置时意外改变了 llm 配置")
    for section, key, expected in WEBUI_ONLY_SETTINGS:
        table: Any = candidate_document
        for part in section.split("."):
            table = table[part]
        if table[key] != expected:
            raise RuntimeError(f"候选配置字段未生效: {section}.{key}")
    _ = destination.write_text(candidate, encoding="utf-8")
    destination.chmod(0o600)


def _copy_plugin_manifest(source_home: Path, destination_home: Path) -> CopyRecord:
    """只迁移插件声明；cache、旧全局 data 与备份均不进入副本。"""

    source = source_home / "manifest.toml"
    if not source.is_file():
        raise FileNotFoundError(f"插件 manifest 不存在: {source}")
    destination_home.mkdir(mode=0o700)
    destination = destination_home / "manifest.toml"
    _ = _stable_copy(source, destination)
    _ = tomllib.loads(destination.read_text(encoding="utf-8"))
    return CopyRecord(
        path="plugin-home/manifest.toml",
        kind="plugin_manifest",
        size=destination.stat().st_size,
        sha256=_sha256(destination),
    )


def prepare_rehearsal(
    *,
    source_workspace: Path,
    source_config: Path,
    plugin_home: Path,
    target: Path,
) -> Path:
    """创建原子发布的隔离预演根，并返回机器清单路径。"""

    source_workspace, source_config, plugin_home, target = _validate_roots(
        source_workspace=source_workspace,
        source_config=source_config,
        plugin_home=plugin_home,
        target=target,
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.preparing-", dir=target.parent)
    )
    stage.chmod(0o700)
    try:
        # 1. 复制 Workspace、配置和唯一必要的全局插件声明。
        records, exclusions, databases, consistency = _copy_workspace(
            source_workspace, stage / "workspace"
        )
        _write_candidate_config(
            source_config, stage / "config.toml", target / "workspace"
        )
        config_record = CopyRecord(
            path="config.toml",
            kind="webui_only_config",
            size=(stage / "config.toml").stat().st_size,
            sha256=_sha256(stage / "config.toml"),
        )
        plugin_record = _copy_plugin_manifest(plugin_home, stage / "plugin-home")

        # 2. 清单只保存摘要和状态边界，不序列化配置或凭据正文。
        manifest: dict[str, Any] = {
            "schema_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source": {
                "workspace": str(source_workspace),
                "config": str(source_config),
                "plugin_home": str(plugin_home),
                "read_only": True,
            },
            "target": str(target),
            "candidate": {
                "workspace": "workspace",
                "config": "config.toml",
                "plugin_manifest": "plugin-home/manifest.toml",
                "config_channels": ["web"],
                "model_registry_preserved": True,
                "plugin_manifest_copied_unmodified": True,
                "plugin_data_source": "workspace/plugin-data",
                "plugin_cache_copied": False,
            },
            "exclusion_policy": {
                "directory_names": sorted(EXCLUDED_DIRECTORY_NAMES),
                "runtime_files": sorted(EXCLUDED_RUNTIME_FILES),
                "additional": [
                    "*_rebuild directories",
                    "mobile-webui-build-* directories",
                    "mobile-webui/staging and mobile-webui/trash",
                    "SQLite -wal/-shm/-journal sidecars",
                    "workspace skills and drift/skills cache symlinks",
                    "non-regular filesystem entries",
                ],
            },
            "excluded": exclusions,
            "databases": databases,
            "consistency": consistency,
            "files": [
                record.__dict__
                for record in sorted(
                    [*records, config_record, plugin_record], key=lambda item: item.path
                )
            ],
            "cleanup": {
                "exact_paths": [str(target)],
                "guard_manifest": str(target / "rehearsal-manifest.json"),
            },
        }
        manifest_path = stage / "rehearsal-manifest.json"
        _ = manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        manifest_path.chmod(0o600)

        # 3. 所有校验完成后一次发布；任何失败都不暴露半成品目标。
        os.replace(stage, target)
        return target / "rehearsal-manifest.json"
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument("--source-workspace", type=Path, required=True)
    _ = parser.add_argument("--source-config", type=Path, required=True)
    _ = parser.add_argument(
        "--plugin-home", type=Path, default=Path("~/.akashic-plugin")
    )
    _ = parser.add_argument("--target", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    manifest = prepare_rehearsal(
        source_workspace=args.source_workspace,
        source_config=args.source_config,
        plugin_home=args.plugin_home,
        target=args.target,
    )
    print(json.dumps({"manifest": str(manifest)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

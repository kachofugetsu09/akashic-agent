"""Capture a stable, isolated copy of runtime workspace state."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

from scripts.container_rehearsal.model import (
    CopyRecord,
    SnapshotDriftError,
    SourceEntry,
    is_relative_to,
    sha256,
)
from scripts.container_rehearsal.policy import excluded_reason
from scripts.container_rehearsal.sqlite_snapshot import (
    copy_sqlite,
    is_sqlite,
    verify_session_media_references,
)

MAX_WORKSPACE_SNAPSHOT_ATTEMPTS = 3


def _stable_copy(source: Path, destination: Path) -> tuple[os.stat_result, str]:
    """Copy one regular file while rejecting a moving source generation."""

    for _attempt in range(3):
        before = source.stat()
        _ = shutil.copy2(source, destination)
        after = source.stat()
        identity_before = (before.st_ino, before.st_size, before.st_mtime_ns)
        identity_after = (after.st_ino, after.st_size, after.st_mtime_ns)
        target_digest = sha256(destination)
        if identity_before == identity_after and sha256(source) == target_digest:
            return after, target_digest
        destination.unlink()
    raise SnapshotDriftError(f"普通文件在快照期间持续变化: {source}")


def _copy_symlink(source: Path, destination: Path, source_root: Path) -> CopyRecord:
    """Copy only relative symlinks that remain inside the workspace."""

    link_target = os.readlink(source)
    if os.path.isabs(link_target):
        raise ValueError(f"拒绝复制绝对符号链接: {source} -> {link_target}")
    resolved = (source.parent / link_target).resolve(strict=False)
    if not is_relative_to(resolved, source_root):
        raise ValueError(f"拒绝复制逃逸 Workspace 的符号链接: {source} -> {link_target}")
    destination.symlink_to(link_target, target_is_directory=source.is_dir())
    return CopyRecord(path=str(destination), kind="symlink", size=0, link_target=link_target)


def _scan_workspace(source: Path) -> tuple[dict[Path, SourceEntry], list[dict[str, str]]]:
    """Freeze the included path set without following symlinks."""

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
                reason = excluded_reason(relative, is_symlink=is_symlink)
                if reason is not None:
                    exclusions.append({"path": relative.as_posix(), "reason": reason})
                    continue
                metadata = child.stat(follow_symlinks=False)
                if is_symlink:
                    kind, link_target = "symlink", os.readlink(item)
                elif child.is_dir(follow_symlinks=False):
                    kind, link_target = "directory", None
                elif child.is_file(follow_symlinks=False):
                    kind, link_target = ("sqlite" if is_sqlite(item) else "file"), None
                else:
                    exclusions.append(
                        {"path": relative.as_posix(), "reason": "non_regular_filesystem_entry"}
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


def _verify_workspace_unchanged(
    source: Path,
    destination: Path,
    baseline: dict[Path, SourceEntry],
    final: dict[Path, SourceEntry],
) -> None:
    """Verify regular paths stayed on one generation around database backup."""

    if baseline.keys() != final.keys():
        added = sorted(path.as_posix() for path in final.keys() - baseline.keys())
        removed = sorted(path.as_posix() for path in baseline.keys() - final.keys())
        raise SnapshotDriftError(f"纳入路径集合变化: added={added[:8]} removed={removed[:8]}")
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
        if expected_metadata != actual_metadata or expected.link_target != actual.link_target:
            raise SnapshotDriftError(f"普通路径元数据变化: {relative}")
        if expected.kind == "file":
            source_digest = sha256(source / relative)
            if source_digest != expected.sha256 or source_digest != sha256(destination / relative):
                raise SnapshotDriftError(f"普通文件内容变化: {relative}")


def _copy_workspace_once(
    source: Path, destination: Path
) -> tuple[list[CopyRecord], list[dict[str, str]], list[dict[str, Any]]]:
    """Copy regular state, back up databases, then verify the source generation."""

    initial, exclusions = _scan_workspace(source)
    baseline = dict(initial)
    records: list[CopyRecord] = []
    databases: list[dict[str, Any]] = []
    destination.mkdir(mode=0o700)

    # 1. Copy stable regular paths before opening the database backup window.
    for relative, entry in initial.items():
        item, target = source / relative, destination / relative
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

    # 2. Back up each SQLite database through its online backup API.
    for relative, entry in initial.items():
        if entry.kind != "sqlite":
            continue
        item, target = source / relative, destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        check = copy_sqlite(item, target)
        databases.append({"path": relative.as_posix(), **check})
        records.append(
            CopyRecord(
                path=relative.as_posix(),
                kind="sqlite_online_backup",
                size=target.stat().st_size,
                sha256=sha256(target),
            )
        )

    # 3. Close the window by proving path identity and media reachability.
    final, _final_exclusions = _scan_workspace(source)
    _verify_workspace_unchanged(source, destination, baseline, final)
    verify_session_media_references(source, destination, databases)
    return records, exclusions, databases


def copy_workspace(
    source: Path, destination: Path
) -> tuple[list[CopyRecord], list[dict[str, str]], list[dict[str, Any]], dict[str, Any]]:
    """Capture a consistent workspace with bounded whole-snapshot retries."""

    drift_messages: list[str] = []
    for attempt in range(1, MAX_WORKSPACE_SNAPSHOT_ATTEMPTS + 1):
        if destination.exists():
            shutil.rmtree(destination)
        try:
            records, exclusions, databases = _copy_workspace_once(source, destination)
            return records, exclusions, databases, {
                "attempts": attempt,
                "max_attempts": MAX_WORKSPACE_SNAPSHOT_ATTEMPTS,
                "drift_retries": drift_messages,
            }
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

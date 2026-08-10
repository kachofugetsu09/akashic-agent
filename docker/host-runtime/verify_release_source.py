from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Any

_MANIFEST_NAME = ".akashic-source-manifest.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_entries(root: Path) -> list[dict[str, Any]]:
    """Return the exact regular-file and symlink identity of a release tree."""

    entries: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if relative == _MANIFEST_NAME:
            continue
        metadata = path.lstat()
        if stat.S_ISDIR(metadata.st_mode):
            continue
        if stat.S_ISLNK(metadata.st_mode):
            entries.append(
                {"path": relative, "kind": "symlink", "target": os.readlink(path)}
            )
            continue
        if not stat.S_ISREG(metadata.st_mode):
            raise RuntimeError(f"release source 含不支持的文件类型: {relative}")
        entries.append(
            {
                "path": relative,
                "kind": "file",
                "executable": bool(metadata.st_mode & stat.S_IXUSR),
                "sha256": _sha256(path),
            }
        )
    return entries


def verify_release_source(
    root: Path,
    manifest_path: Path,
    *,
    expected_commit: str,
    expected_tree: str,
    expected_archive_sha256: str,
) -> dict[str, Any]:
    """Verify that the Docker context is exactly the deployment-owned archive."""

    # 1. Validate manifest identity before trusting its file inventory.
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_identity = {
        "sourceCommit": expected_commit,
        "sourceTree": expected_tree,
        "sourceArchiveSha256": expected_archive_sha256,
    }
    for key, value in expected_identity.items():
        if document.get(key) != value:
            raise RuntimeError(
                f"release source manifest identity 不一致: {key}={document.get(key)!r}"
            )

    # 2. Reject missing, changed, dirty, ignored, or otherwise extra context files.
    expected_entries = document.get("files")
    if not isinstance(expected_entries, list):
        raise RuntimeError("release source manifest files 必须是 array")
    actual_entries = source_entries(root)
    if actual_entries != expected_entries:
        expected_paths = {str(item.get("path")) for item in expected_entries}
        actual_paths = {str(item.get("path")) for item in actual_entries}
        raise RuntimeError(
            "Docker build context 与 commit archive 不一致: "
            f"added={sorted(actual_paths - expected_paths)[:8]} "
            f"removed={sorted(expected_paths - actual_paths)[:8]}"
        )
    return document


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify immutable release source")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--expected-tree", required=True)
    parser.add_argument("--expected-archive-sha256", required=True)
    args = parser.parse_args()
    verify_release_source(
        args.root,
        args.manifest,
        expected_commit=args.expected_commit,
        expected_tree=args.expected_tree,
        expected_archive_sha256=args.expected_archive_sha256,
    )


if __name__ == "__main__":
    main()

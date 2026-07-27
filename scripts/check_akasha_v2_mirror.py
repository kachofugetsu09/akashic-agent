#!/usr/bin/env python3
"""Verify the vendored Akasha package against its pinned upstream checkout."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

_HOST_GENERATED_FILES = {
    "UPSTREAM.json",
    "dashboard_panel_inspector.js",
}


def main() -> None:
    """Validate the upstream revision, Git subtree, and mirrored file bytes."""

    # 1. Resolve the recorded source identity and explicit upstream checkout.
    arguments = _parser().parse_args()
    host_root = Path(__file__).resolve().parents[1]
    target = host_root / "plugins" / "akasha"
    metadata = json.loads(
        (target / "UPSTREAM.json").read_text(encoding="utf-8")
    )
    upstream = arguments.upstream.resolve(strict=True)
    commit = _git(upstream, "rev-parse", "HEAD")
    tree = _git(
        upstream,
        "rev-parse",
        f"{commit}:{metadata['source_subtree']}",
    )
    if commit != metadata["commit"]:
        raise ValueError(
            f"upstream commit mismatch: {commit} != {metadata['commit']}"
        )
    if tree != metadata["source_tree"]:
        raise ValueError(
            f"upstream source tree mismatch: {tree} != {metadata['source_tree']}"
        )

    # 2. Compare the complete source file set and deterministic content hash.
    source = upstream / str(metadata["source_subtree"])
    source_files = _files(source)
    target_files = _files(target, ignored=_HOST_GENERATED_FILES)
    if source_files != target_files:
        missing = sorted(source_files - target_files)
        unexpected = sorted(target_files - source_files)
        raise ValueError(
            f"Akasha mirror file mismatch: missing={missing} "
            f"unexpected={unexpected}"
        )
    for relative in sorted(source_files):
        if (source / relative).read_bytes() != (target / relative).read_bytes():
            raise ValueError(f"Akasha mirror differs: {relative}")
    digest = _tree_sha256(target, ignored=_HOST_GENERATED_FILES)
    if digest != metadata["source_sha256"]:
        raise ValueError(
            f"Akasha mirror digest mismatch: {digest}"
        )
    print(
        json.dumps(
            {
                "commit": commit,
                "source_tree": tree,
                "source_sha256": digest,
                "files": len(source_files),
            },
            sort_keys=True,
        )
    )


def _files(
    root: Path,
    *,
    ignored: set[str] | None = None,
) -> set[str]:
    excluded = ignored or set()
    return {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix != ".pyc"
        and path.name not in excluded
    }


def _tree_sha256(
    root: Path,
    *,
    ignored: set[str],
) -> str:
    digest = hashlib.sha256()
    for relative in sorted(_files(root, ignored=ignored)):
        name = relative.encode("utf-8")
        content = (root / relative).read_bytes()
        digest.update(len(name).to_bytes(8, "big"))
        digest.update(name)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def _git(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    _ = parser.add_argument(
        "--upstream",
        type=Path,
        required=True,
        help="Path to the akasha-v2-engine checkout.",
    )
    return parser


if __name__ == "__main__":
    main()

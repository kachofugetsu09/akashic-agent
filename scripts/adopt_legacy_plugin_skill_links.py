#!/usr/bin/env python3
"""Adopt verified legacy plugin skill symlinks into the ownership registry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from infra.persistence.json_store import atomic_save_json


def _inside(path: Path, roots: tuple[Path, ...]) -> bool:
    return any(path == root or path.is_relative_to(root) for root in roots)


def adopt_legacy_links(*, workspace: Path, plugin_roots: tuple[Path, ...]) -> dict[str, str]:
    """Validate every legacy skill link and atomically create its ownership registry."""

    # 1. Resolve the explicitly approved migration boundary.
    workspace = workspace.resolve(strict=True)
    roots = tuple(root.resolve(strict=True) for root in plugin_roots)
    if not roots:
        raise RuntimeError("至少需要一个 plugin root")
    ownership_path = workspace / "runtime" / "plugin-skill-links.json"
    if ownership_path.exists() or ownership_path.is_symlink():
        raise RuntimeError(f"ownership registry 已存在，拒绝覆盖: {ownership_path}")

    # 2. Adopt only existing symlinks whose targets stay inside an approved root.
    links: dict[str, str] = {}
    for directory in (workspace / "skills", workspace / "drift" / "skills"):
        if not directory.exists():
            continue
        for link in sorted(directory.iterdir()):
            if not link.is_symlink():
                continue
            target = link.resolve(strict=True)
            if not target.is_dir() or not _inside(target, roots):
                raise RuntimeError(f"legacy skill link target 不在 plugin roots 内: {link} -> {target}")
            key = str((link.parent.resolve(strict=True) / link.name).relative_to(workspace))
            links[key] = str(target)

    # 3. Persist one complete registry only after the full set has passed validation.
    ownership_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_save_json(
        ownership_path,
        {"version": 1, "links": links, "pending": {}},
        domain="plugin_skill_links",
    )
    return links


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--plugin-root", type=Path, action="append", required=True)
    args = parser.parse_args()
    links = adopt_legacy_links(
        workspace=args.workspace,
        plugin_roots=tuple(args.plugin_root),
    )
    print(json.dumps({"adopted": len(links), "links": sorted(links)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

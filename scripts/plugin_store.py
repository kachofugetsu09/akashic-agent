#!/usr/bin/env python3
"""镜像外的内置插件制品库。

布局（宿主拥有，容器只读挂载在同一路径）：

    <store>/sources/<name>/<content_commit>/   插件源码，按内容寻址，写入后只读
    <store>/blobs/<sha256>                    发行版其余文件，同内容只存一份
    <store>/distributions/<commit>/           一个发行版：文件硬链接到 blobs，
                                              sources/<name> 是指向内容目录的符号链接

运行时解析 `sources/<name>` 的符号链接，所以内容未变的插件在不同发行版里代码路径相同，
输入描述符与 input_ref 也相同；只有内容变化的插件会被替换。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import tempfile


# 制品对任何运行用户可读、对谁都不可写：发布器可能以 root 运行，容器以普通用户读取。
def _read_only(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        if path.is_symlink():
            continue
        executable = path.is_dir() or path.stat().st_mode & stat.S_IXUSR
        path.chmod(0o555 if executable else 0o444)
    root.chmod(0o555)


# 制品库目录要让运行用户能进入；只改自己拥有的目录，别人创建的保持原样。
def _traversable(directory: Path) -> None:
    if directory.stat().st_uid == os.getuid():
        directory.chmod(0o755)


def _writable(root: Path) -> None:
    for path in [root, *root.rglob("*")]:
        if not path.is_symlink():
            path.chmod(path.stat().st_mode | stat.S_IWUSR)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# 1. 内容目录已存在时核对来源；2. 不存在时先写临时目录再原子改名。
def _store_source(source: Path, target: Path, content_commit: str) -> None:
    if target.exists():
        provenance = json.loads((target / ".akashic-source.json").read_text())
        if provenance.get("commit") != content_commit:
            raise RuntimeError(f"制品库内容目录身份不一致: {target}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    _traversable(target.parent)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    try:
        _ = shutil.copytree(source, staging / "tree", symlinks=True)
        provenance = json.loads((staging / "tree/.akashic-source.json").read_text())
        if provenance.get("commit") != content_commit:
            raise RuntimeError(f"发行版插件来源与 content_commit 不一致: {source}")
        # 目录改名要求它自身可写，先改名再收回写权限。
        (staging / "tree").rename(target)
        _read_only(target)
    finally:
        if staging.exists():
            _writable(staging)
            shutil.rmtree(staging)


def _store_blob(store: Path, source: Path) -> Path:
    blob = store / "blobs" / _sha256(source)
    if not blob.exists():
        blob.parent.mkdir(parents=True, exist_ok=True)
        staging = blob.with_name(f".{blob.name}.{os.getpid()}")
        _ = shutil.copy2(source, staging)
        staging.chmod(0o444)
        staging.rename(blob)
    return blob


# 把一个已构建的发行版放进制品库，返回 distributions/<commit>；已存在时原样返回。
def materialize(distribution: Path, store: Path, commit: str) -> Path:
    report = json.loads((distribution / "distribution.json").read_text())
    if report.get("source_commit") != commit:
        raise RuntimeError("发行版 source_commit 与目标提交不一致")
    target = store / "distributions" / commit
    if target.exists():
        return target
    for directory in (store, store / "distributions", store / "sources", store / "blobs"):
        directory.mkdir(parents=True, exist_ok=True)
        _traversable(directory)
    staging = Path(tempfile.mkdtemp(prefix=f".{commit}.", dir=target.parent))
    try:
        for row in report["plugins"]:
            name, content = row["name"], row["content_commit"]
            content_root = store / "sources" / name / content
            _store_source(distribution / "sources" / name, content_root, content)
            link = staging / "sources" / name
            link.parent.mkdir(parents=True, exist_ok=True)
            link.symlink_to(os.path.relpath(content_root, link.parent))
        for path in sorted(distribution.rglob("*")):
            relative = path.relative_to(distribution)
            if relative.parts[0] == "sources" or path.is_dir():
                continue
            destination = staging / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.link(_store_blob(store, path), destination)
        # mkdtemp 是 0700；发布器可能以 root 运行，容器以普通用户读取。
        for directory in [staging, *(path for path in staging.rglob("*") if path.is_dir() and not path.is_symlink())]:
            directory.chmod(0o755)
        staging.rename(target)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return target


# 只保留仍被引用的发行版，以及它们用到的内容目录与 blob。
def prune(store: Path, keep: set[str]) -> list[str]:
    removed: list[str] = []
    distributions = store / "distributions"
    for entry in sorted(distributions.iterdir()) if distributions.exists() else []:
        if entry.name not in keep and not entry.name.startswith("."):
            shutil.rmtree(entry)
            removed.append(entry.name)
    used_sources: set[Path] = set()
    used_blobs: set[int] = set()
    for entry in distributions.iterdir() if distributions.exists() else []:
        for path in entry.rglob("*"):
            if path.is_symlink():
                used_sources.add(path.resolve())
            elif path.is_file():
                used_blobs.add(path.stat().st_ino)
    for content_root in (store / "sources").glob("*/*"):
        if content_root.resolve() not in used_sources:
            _writable(content_root)
            shutil.rmtree(content_root)
    for blob in (store / "blobs").glob("*"):
        if blob.stat().st_ino not in used_blobs:
            blob.unlink()
    return removed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    put = commands.add_parser("materialize")
    put.add_argument("--distribution", type=Path, required=True)
    put.add_argument("--store", type=Path, required=True)
    put.add_argument("--commit", required=True)
    clean = commands.add_parser("prune")
    clean.add_argument("--store", type=Path, required=True)
    clean.add_argument("--keep", action="append", default=[])
    args = parser.parse_args()
    if args.command == "materialize":
        print(materialize(args.distribution.resolve(strict=True), args.store.resolve(), args.commit))
    else:
        print(json.dumps(prune(args.store.resolve(), set(args.keep))))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build, publish and inspect the server-owned mobile WebUI release store."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from infra.mobile_webui.manifest import manifest_from_directory
from infra.mobile_webui.store import MobileWebUiStore


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    source_repository = args.source_repository.resolve()
    workspace = args.workspace.resolve() if args.workspace is not None else None
    if args.action == "build":
        output = _build(source_repository, args.output, allow_dirty=args.allow_dirty, source_commit=args.source_commit, stable=False)
        print(output)
        return 0
    if args.action == "restore":
        if not args.server_id:
            parser.error("restore 需要 --server-id")
        if args.destination is None or args.restore_root is None:
            parser.error("restore 需要 --destination 和 --restore-root")
        restored = MobileWebUiStore.restore_backup(args.destination, args.restore_root, server_id=args.server_id)
        print(restored)
        return 0
    if not args.server_id:
        parser.error("publish/inspect/promote/clear-preview/gc/backup 需要 --server-id")
    if workspace is None and args.store_root is None:
        parser.error("有状态操作必须显式提供 --workspace 或 --store-root")
    store = MobileWebUiStore(args.store_root or workspace / "mobile-webui", server_id=args.server_id)
    try:
        if args.action == "publish":
            build_dir = args.build_dir
            temporary: Path | None = None
            if build_dir is None:
                temporary = Path(tempfile.mkdtemp(prefix="mobile-webui-build-", dir=workspace if workspace is not None else None))
                build_dir = _build(
                    source_repository,
                    temporary,
                    allow_dirty=args.allow_dirty,
                    source_commit=args.source_commit,
                    stable=args.stable,
                )
            try:
                manifest, contents = _manifest(source_repository, build_dir, allow_dirty=args.allow_dirty)
                release = store.publish(manifest, contents, stable=args.stable, preview=not args.stable, actor=args.actor)
                print(json.dumps(_release_json(release), ensure_ascii=False, sort_keys=True))
            finally:
                if temporary is not None:
                    shutil.rmtree(temporary, ignore_errors=True)
            return 0
        if args.action == "promote-preview":
            print(json.dumps(_release_json(store.promote_preview(actor=args.actor)), ensure_ascii=False, sort_keys=True))
            return 0
        if args.action == "rollback":
            if not args.target_key:
                parser.error("rollback 需要 --target-key")
            print(json.dumps(_release_json(store.rollback(args.target_key, actor=args.actor)), ensure_ascii=False, sort_keys=True))
            return 0
        if args.action == "pin":
            if not args.target_key:
                parser.error("pin 需要 --target-key")
            store.pin_target(args.target_key, reason=args.reason)
            return 0
        if args.action == "unpin":
            if not args.target_key:
                parser.error("unpin 需要 --target-key")
            store.unpin_target(args.target_key)
            return 0
        if args.action == "clear-preview":
            print(json.dumps(_release_json(store.clear_preview(actor=args.actor)), ensure_ascii=False, sort_keys=True))
            return 0
        if args.action == "inspect":
            print(json.dumps(_release_json(store.get_release()), ensure_ascii=False, sort_keys=True))
            return 0
        if args.action == "gc":
            report = store.gc(keep_unreachable=args.keep_unreachable)
            print(json.dumps({"removed_generations": report.removed_generations, "removed_blobs": report.removed_blobs}, ensure_ascii=False, sort_keys=True))
            return 0
        if args.action == "backup":
            if args.destination is None:
                parser.error("backup 需要 --destination")
            destination = store.backup_to(args.destination)
            MobileWebUiStore.verify_backup(destination, server_id=args.server_id)
            print(destination)
            return 0
        if args.action == "release-backup":
            if not args.backup_id:
                parser.error("release-backup 需要 --backup-id")
            store.release_backup(args.backup_id)
            return 0
        raise AssertionError(f"unsupported action: {args.action}")
    finally:
        store.close()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("build", "publish", "inspect", "promote-preview", "clear-preview", "rollback", "pin", "unpin", "gc", "backup", "release-backup", "restore"))
    parser.add_argument("--source-repository", type=Path, default=Path.cwd())
    parser.add_argument("--workspace", type=Path)
    parser.add_argument("--store-root", type=Path)
    parser.add_argument("--server-id")
    parser.add_argument("--build-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--source-commit", help="用于构建/发布的 40 位 commit；Stable 默认使用 HEAD")
    parser.add_argument("--stable", action="store_true")
    parser.add_argument("--keep-unreachable", type=int, default=0)
    parser.add_argument("--target-key")
    parser.add_argument("--reason", default="rollback")
    parser.add_argument("--backup-id")
    parser.add_argument("--restore-root", type=Path)
    parser.add_argument("--actor", default="publish-mobile-webui")
    return parser


def _build(
    workspace: Path,
    output: Path | None,
    *,
    allow_dirty: bool,
    source_commit: str | None,
    stable: bool,
) -> Path:
    workspace = workspace.resolve()
    if stable and allow_dirty:
        raise RuntimeError("Stable 不允许 --allow-dirty")
    commit = _resolve_commit(workspace, source_commit)
    current_head = _git(workspace, "rev-parse", "HEAD")
    if commit == current_head:
        _require_clean_source(workspace, allow_dirty=allow_dirty)
    dirty = commit == current_head and _webui_dirty(workspace)
    if stable and dirty:
        raise RuntimeError("Stable 构建输入存在 dirty provenance")
    lock_available = _commit_has_file(workspace, commit, "package-lock.json")
    if stable and not lock_available:
        raise RuntimeError("Stable 要求指定 commit 自带 package-lock.json")
    if not lock_available and not dirty:
        raise RuntimeError("无 package-lock 的 clean 构建只能拒绝；请使用带 dirty provenance 的 Preview")
    workspace.parent.mkdir(parents=True, exist_ok=True)
    if output is None:
        output = Path(tempfile.mkdtemp(prefix="mobile-webui-build-"))
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"build output 非空: {output}")
    output.mkdir(parents=True, exist_ok=True)
    with _build_source(workspace, commit, dirty=dirty) as build_workspace:
        before = _capture_provenance(build_workspace)
        environment = os.environ.copy()
        environment["AKASHIC_MOBILE_WEB_OUT_DIR"] = str(output)
        if lock_available:
            subprocess.run(
                ["npm", "ci", "--ignore-scripts", "--no-audit", "--no-fund"],
                cwd=build_workspace,
                env=environment,
                check=True,
            )
        else:
            subprocess.run(
                ["npm", "install", "--package-lock=false", "--ignore-scripts", "--no-audit", "--no-fund"],
                cwd=build_workspace,
                env=environment,
                check=True,
            )
        subprocess.run(["npm", "run", "build:mobile-web"], cwd=build_workspace, env=environment, check=True)
        after = _capture_provenance(build_workspace)
        if before != after:
            raise RuntimeError("构建期间 source/input/build_context 发生变化")
    sidecar = output.with_name(output.name + ".provenance.json")
    sidecar.write_text(
        json.dumps(
            {
                **before,
                "artifact_digest": _artifact_digest(output),
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return output


def _manifest(workspace: Path, build_dir: Path, *, allow_dirty: bool):
    sidecar_path = build_dir.with_name(build_dir.name + ".provenance.json")
    if not sidecar_path.is_file():
        raise RuntimeError("外部 build-dir 必须带受控 build provenance sidecar")
    sidecar = _load_sidecar(sidecar_path)
    commit = sidecar["source_commit"]
    tree = sidecar["source_tree"]
    if sidecar["artifact_digest"] != _artifact_digest(build_dir):
        raise RuntimeError("build-dir provenance 与 artifact 不匹配")
    if _git(workspace, "rev-parse", f"{commit}^{{tree}}") != tree:
        raise RuntimeError("build-dir provenance 的 source tree 不存在或不匹配")
    current_head = _git(workspace, "rev-parse", "HEAD")
    dirty = sidecar["dirty_provenance"]
    if dirty is not None:
        if current_head != commit or _capture_provenance(workspace)["input_digest"] != sidecar["input_digest"] or _dirty_provenance(workspace) != dirty:
            raise RuntimeError("dirty build provenance 与当前 source 不匹配")
    elif current_head == commit and _capture_provenance(workspace) != {key: sidecar[key] for key in _build_provenance_keys()}:
        raise RuntimeError("build-dir provenance 与当前 clean source 不匹配")
    if dirty is not None and not allow_dirty:
        raise RuntimeError("publish 默认拒绝 dirty source；Preview 请显式 --allow-dirty")
    if dirty is None and sidecar["builder_identity"]["package_lock_digest"] == _no_lock_digest():
        raise RuntimeError("无 package-lock 的构建不能作为 Stable/reproducible manifest")
    if dirty is None:
        with _build_source(workspace, commit, dirty=False) as snapshot:
            if _capture_provenance(snapshot) != {key: sidecar[key] for key in _build_provenance_keys()}:
                raise RuntimeError("clean build provenance 未通过 commit snapshot 重算")
    return manifest_from_directory(
        build_dir,
        source_repository=sidecar["source_repository"],
        source_commit=commit,
        source_tree=tree,
        input_digest=sidecar["input_digest"],
        build_context_digest=sidecar["build_context_digest"],
        dirty_provenance=dirty,
        reproducible=dirty is None,
        builder_identity=sidecar["builder_identity"],
    )


def _require_clean_source(workspace: Path, *, allow_dirty: bool) -> None:
    if _webui_dirty(workspace) and not allow_dirty:
        raise RuntimeError("WebUI/build inputs require clean source；Preview 可用 --allow-dirty")


def _webui_dirty(workspace: Path) -> bool:
    return bool(_git(workspace, "status", "--porcelain", "--untracked-files=all", "--", *(_webui_paths())))


def _dirty_provenance(workspace: Path) -> dict[str, str]:
    tracked = subprocess.run(["git", "diff", "--binary", "HEAD", "--", *(_webui_paths())], cwd=workspace, check=True, capture_output=True).stdout
    untracked_digest = hashlib.sha256()
    for path in _git(workspace, "ls-files", "--others", "--exclude-standard", "--", *(_webui_paths())).splitlines():
        absolute = workspace / path
        _hash_record(untracked_digest, path.encode("utf-8"), _read_regular_input(absolute))
    return {
        "base_commit": _git(workspace, "rev-parse", "HEAD"),
        "tracked_patch_digest": hashlib.sha256(tracked).hexdigest(),
        "untracked_tree_digest": untracked_digest.hexdigest(),
    }


@contextmanager
def _build_source(workspace: Path, commit: str, *, dirty: bool):
    """Yield an immutable git worktree for clean builds or the explicit dirty preview source."""

    temporary = Path(tempfile.mkdtemp(prefix="mobile-webui-source-", dir=workspace.parent))
    shutil.rmtree(temporary)
    added = False
    try:
        subprocess.run(["git", "worktree", "add", "--detach", str(temporary), commit], cwd=workspace, check=True)
        added = True
        if dirty:
            _overlay_webui_inputs(workspace, temporary)
        yield temporary
    finally:
        if added:
            subprocess.run(["git", "worktree", "remove", "--force", str(temporary)], cwd=workspace, check=False)
        shutil.rmtree(temporary, ignore_errors=True)


def _resolve_commit(workspace: Path, requested: str | None) -> str:
    commit = requested or _git(workspace, "rev-parse", "HEAD")
    if len(commit) != 40 or any(char not in "0123456789abcdef" for char in commit):
        raise RuntimeError("source-commit 必须是 40 位小写 commit SHA-1")
    resolved = _git(workspace, "rev-parse", "--verify", f"{commit}^{{commit}}")
    if resolved != commit:
        raise RuntimeError("source-commit 不是可解析的 commit")
    return commit


def _commit_has_file(workspace: Path, commit: str, path: str) -> bool:
    result = subprocess.run(
        ["git", "cat-file", "-e", f"{commit}:{path}"],
        cwd=workspace,
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def _capture_provenance(workspace: Path) -> dict[str, object]:
    """Capture all source/build identity inputs before and after a build."""

    package_lock = workspace / "package-lock.json"
    package_lock_digest = _sha256(package_lock.read_bytes()) if package_lock.is_file() else _no_lock_digest()
    build_script = workspace / "scripts" / "package-mobile-web.sh"
    script_digest = _sha256(build_script.read_bytes())
    node_version = _run_version("node", "--version")
    npm_version = _run_version("npm", "--version")
    context = {
        "node": node_version,
        "npm": npm_version,
        "package_lock": package_lock_digest,
        "script": script_digest,
    }
    return {
        "source_repository": _repository_url(workspace),
        "source_commit": _git(workspace, "rev-parse", "HEAD"),
        "source_tree": _git(workspace, "rev-parse", "HEAD^{tree}"),
        "input_digest": _input_digest(workspace),
        "build_context_digest": _sha256(json.dumps(context, sort_keys=True, separators=(",", ":")).encode()),
        "dirty_provenance": _dirty_provenance(workspace) if _webui_dirty(workspace) else None,
        "builder_identity": {
            "node_version": node_version,
            "npm_version": npm_version,
            "package_lock_digest": package_lock_digest,
            "build_script_digest": script_digest,
        },
    }


def _input_digest(workspace: Path) -> str:
    paths = _git(workspace, "ls-files", "--cached", "--others", "--exclude-standard", "--", *_webui_paths()).splitlines()
    digest = hashlib.sha256()
    for relative in sorted(set(paths), key=lambda value: value.encode("utf-8")):
        path = workspace / relative
        if path.exists():
            _hash_record(digest, relative.encode("utf-8"), _read_regular_input(path))
        else:
            _hash_record(digest, relative.encode("utf-8"), b"<missing>")
    return digest.hexdigest()


def _repository_url(workspace: Path) -> str:
    remote = _git(workspace, "config", "--get", "remote.origin.url")
    if remote.startswith("git@github.com:"):
        remote = "https://github.com/" + remote.removeprefix("git@github.com:")
    remote = remote.removesuffix(".git")
    if not remote or any(char.isspace() for char in remote):
        raise RuntimeError("source repository remote.origin.url 无效")
    return remote


def _load_sidecar(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "artifact_digest", "build_context_digest", "builder_identity", "dirty_provenance",
        "input_digest", "source_commit", "source_repository", "source_tree",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise RuntimeError("build provenance sidecar 字段集合无效")
    for key in ("artifact_digest", "build_context_digest", "input_digest"):
        if not isinstance(value[key], str) or len(value[key]) != 64 or any(char not in "0123456789abcdef" for char in value[key]):
            raise RuntimeError(f"build provenance {key} 无效")
    for key in ("source_commit", "source_tree"):
        if not isinstance(value[key], str) or len(value[key]) != 40 or any(char not in "0123456789abcdef" for char in value[key]):
            raise RuntimeError(f"build provenance {key} 无效")
    if not isinstance(value["source_repository"], str) or not value["source_repository"] or any(char.isspace() for char in value["source_repository"]):
        raise RuntimeError("build provenance source_repository 无效")
    builder = value["builder_identity"]
    if not isinstance(builder, dict) or set(builder) != {"node_version", "npm_version", "package_lock_digest", "build_script_digest"}:
        raise RuntimeError("build provenance builder_identity 无效")
    dirty = value["dirty_provenance"]
    if dirty is not None and (not isinstance(dirty, dict) or set(dirty) != {"base_commit", "tracked_patch_digest", "untracked_tree_digest"}):
        raise RuntimeError("build provenance dirty_provenance 无效")
    return value


def _no_lock_digest() -> str:
    return _sha256(b"NO_PACKAGE_LOCK\n")


def _read_regular_input(path: Path) -> bytes:
    if path.is_symlink():
        raise RuntimeError(f"WebUI 输入不允许 symlink: {path}")
    try:
        mode = path.lstat().st_mode
    except OSError as error:
        raise RuntimeError(f"WebUI 输入无法读取: {path}") from error
    if not stat.S_ISREG(mode):
        raise RuntimeError(f"WebUI 输入只允许普通文件: {path}")
    return path.read_bytes()


def _overlay_webui_inputs(source: Path, snapshot: Path) -> None:
    """Overlay the exact dirty WebUI input set onto a clean commit worktree."""

    tracked = _git(source, "ls-files", "--cached", "--", *_webui_paths()).splitlines()
    untracked = _git(source, "ls-files", "--others", "--exclude-standard", "--", *_webui_paths()).splitlines()
    for relative in (*tracked, *untracked):
        source_path = source / relative
        target_path = snapshot / relative
        if source_path.exists() or source_path.is_symlink():
            data = _read_regular_input(source_path)
            target_path.parent.mkdir(parents=True, exist_ok=True)
            if target_path.is_dir() and not target_path.is_symlink():
                shutil.rmtree(target_path)
            target_path.write_bytes(data)
        elif relative in tracked:
            if target_path.exists() or target_path.is_symlink():
                target_path.unlink()


def _hash_record(digest: hashlib._Hash, path: bytes, data: bytes) -> None:
    digest.update(len(path).to_bytes(8, "big"))
    digest.update(path)
    digest.update(len(data).to_bytes(8, "big"))
    digest.update(data)


def _build_provenance_keys() -> tuple[str, ...]:
    return (
        "build_context_digest",
        "builder_identity",
        "dirty_provenance",
        "input_digest",
        "source_commit",
        "source_repository",
        "source_tree",
    )


def _webui_paths() -> tuple[str, ...]:
    return (
        "frontend/chat",
        "scripts/package-mobile-web.sh",
        "package.json",
        "package-lock.json",
    )


def _release_json(release):
    return {
        "server_id": release.server_id,
        "release_epoch": release.release_epoch,
        "sequence": release.sequence,
        "selection_digest": release.selection_digest,
        "stable": release.stable.as_json() if release.stable is not None else None,
        "preview": release.preview.as_json() if release.preview is not None else None,
    }


def _git(workspace: Path, *args: str) -> str:
    result = subprocess.run(["git", *args], cwd=workspace, check=True, capture_output=True, text=True)
    return result.stdout.strip()


def _run_version(command: str, flag: str) -> str:
    return subprocess.run([command, flag], check=True, capture_output=True, text=True).stdout.strip()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _artifact_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted((item for item in root.rglob("*") if item.is_file()), key=lambda item: item.relative_to(root).as_posix().encode("utf-8")):
        _hash_record(digest, path.relative_to(root).as_posix().encode("utf-8"), path.read_bytes())
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""从固定提交生成无业务源码的 Core 与可独立安装的插件 Git bundle。"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tarfile
import tempfile

_SOURCE_ROOT = Path(__file__).resolve().parents[1]
if str(_SOURCE_ROOT) not in sys.path:
    # Direct script execution has scripts/ as sys.path[0]. The builder needs
    # the checked out manifest parser, while the produced Core remains
    # independent of this checkout.
    sys.path.insert(0, str(_SOURCE_ROOT))

from agent.plugins.static_manifest import load_static_plugin_manifest


# 只打包宿主运行入口；业务 plugins/ 不进入 Core，也没有开发源码路径。
CORE_PATHS = (
    "agent",
    "bootstrap",
    "bus",
    "core",
    "infra",
    "session",
    "utils",
    "mcp_servers",
    "host_bridge",
    "memory2",
    "prompts",
    "migrations",
    "main.py",
    "config.example.toml",
    "pyproject.toml",
    "requirements.txt",
    "requirements-dev.txt",
    "sdk/python",
    "scripts/install_plugin_distribution.py",
    "scripts/distribution_runtime.py",
    "docker/host-runtime",
)

_DEFAULT_PROFILE_PATH = Path("docker/host-runtime/profiles/default.json")
_RUNTIME_WIRING = (
    ("docker/host-runtime/Dockerfile.distribution", "Dockerfile.distribution"),
    ("docker/host-runtime/distribution-entrypoint.sh", "distribution-entrypoint.sh"),
)


def git(repository: Path, *args: str, env: dict[str, str] | None = None) -> bytes:
    """只执行明确 Git 参数，失败保留命令与 stderr。"""

    return subprocess.run(
        ["git", "-C", str(repository), *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    ).stdout


def _git_file(repository: Path, revision: str, path: str) -> bytes | None:
    try:
        return git(repository, "show", f"{revision}:{path}")
    except subprocess.CalledProcessError:
        return None


def _hash_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _source_file_record(
    repository: Path, commit: str, path: str
) -> dict[str, str] | None:
    content = _git_file(repository, commit, path)
    if content is None:
        return None
    return {"path": path, "sha256": _hash_bytes(content)}


def _subtree_listing_hash(repository: Path, commit: str, path: str) -> str | None:
    try:
        content = git(repository, "ls-tree", "-r", "--full-tree", commit, "--", path)
    except subprocess.CalledProcessError:
        return None
    if not content:
        return None
    return _hash_bytes(content)


def _runtime_dependency_record(
    repository: Path,
    commit: str,
    tree: str,
) -> dict[str, object]:
    """记录 SDK、Python 锁和 Web 输入的固定来源。"""

    python_files = (
        "pyproject.toml",
        "requirements.txt",
        "requirements-dev.txt",
        "docker/host-runtime/requirements.lock",
    )
    records = [
        record
        for path in python_files
        if (record := _source_file_record(repository, commit, path)) is not None
    ]
    package_records = [
        record
        for path in ("package.json", "package-lock.json")
        if (record := _source_file_record(repository, commit, path)) is not None
    ]
    return {
        "schema_version": 1,
        "source_commit": commit,
        "source_tree": tree,
        "python": {
            "files": records,
            "sdk": {
                "path": "sdk/python",
                "tree_listing_sha256": _subtree_listing_hash(
                    repository, commit, "sdk/python"
                ),
            },
        },
        "node": {
            "files": package_records,
            "web_build_commands": [
                "npm ci --ignore-scripts",
                "npm run build:dashboard",
                "npm run build:chat",
            ],
        },
    }


def _run_web_command(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
) -> None:
    result = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if result.returncode:
        output = result.stdout[-4000:]
        raise RuntimeError(
            "固定提交的 Web 资产构建失败: "
            f"command={' '.join(command)} exit={result.returncode}\n{output}"
        )


def _build_web_assets(
    repository: Path,
    commit: str,
    temporary: Path,
) -> tuple[Path | None, dict[str, object]]:
    """在一次性完整源码目录构建 Web，不向调用方 checkout 写入产物。"""

    required = (
        "package.json",
        "package-lock.json",
        "frontend/chat/vite.config.ts",
        "frontend/dashboard/vite.config.ts",
    )
    missing = [path for path in required if _git_file(repository, commit, path) is None]
    if missing:
        return None, {"enabled": False, "missing_source_paths": missing}

    source = temporary / "web-source"
    source.mkdir()
    archive = git(repository, "archive", "--format=tar", commit)
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as stream:
        stream.extractall(source, filter="data")

    env = {
        **os.environ,
        "CI": "1",
        # npm must never populate a shared checkout or shared cache as part of
        # an artifact build.
        "npm_config_cache": str(temporary / "npm-cache"),
    }
    _run_web_command(["npm", "ci", "--ignore-scripts"], cwd=source, env=env)
    _run_web_command(["npm", "run", "build:dashboard"], cwd=source, env=env)
    _run_web_command(["npm", "run", "build:chat"], cwd=source, env=env)
    asset_root = source / "static"
    required_outputs = (asset_root / "dashboard", asset_root / "chat")
    if any(not path.is_dir() for path in required_outputs):
        raise RuntimeError(
            "Web 资产构建没有产生 static/dashboard 与 static/chat: "
            + ", ".join(str(path) for path in required_outputs)
        )
    if any(not any(path.iterdir()) for path in required_outputs):
        raise RuntimeError("Web 资产构建产生了空的静态目录")

    def version(command: list[str]) -> str:
        return subprocess.run(
            command,
            cwd=source,
            env=env,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        ).stdout.strip()

    return asset_root, {
        "enabled": True,
        "source_commit": commit,
        "source_paths": [
            "frontend/chat",
            "frontend/dashboard",
            "frontend/theme",
            "packages",
        ],
        "build_commands": [
            "npm ci --ignore-scripts",
            "npm run build:dashboard",
            "npm run build:chat",
        ],
        "node_version": version(["node", "--version"]),
        "npm_version": version(["npm", "--version"]),
        "file_count": sum(1 for path in asset_root.rglob("*") if path.is_file()),
    }


def _tar_info(
    name: str,
    *,
    mode: int,
    mtime: int,
    size: int = 0,
) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.mode = mode
    info.mtime = mtime
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.size = size
    return info


def _append_bytes(
    archive: bytes,
    name: str,
    content: bytes,
    *,
    mtime: int,
    mode: int = 0o644,
) -> bytes:
    stream = io.BytesIO(archive)
    with tarfile.open(fileobj=stream, mode="a", format=tarfile.PAX_FORMAT) as output:
        output.addfile(
            _tar_info(name, mode=mode, mtime=mtime, size=len(content)),
            io.BytesIO(content),
        )
    return stream.getvalue()


def _append_tree(
    archive: bytes,
    root: Path,
    prefix: str,
    *,
    mtime: int,
) -> bytes:
    result = archive
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        name = f"{prefix.rstrip('/')}/{relative}"
        if path.is_dir():
            result = _append_bytes(result, name + "/", b"", mtime=mtime, mode=0o755)
            continue
        if not path.is_file():
            raise RuntimeError(f"Web 静态产物含不支持的文件类型: {path}")
        result = _append_bytes(
            result,
            name,
            path.read_bytes(),
            mtime=mtime,
            mode=0o755 if os.access(path, os.X_OK) else 0o644,
        )
    return result


def _copy_profile(repository: Path, commit: str, output: Path) -> dict[str, str] | None:
    content = _git_file(repository, commit, _DEFAULT_PROFILE_PATH.as_posix())
    if content is None:
        return None
    profile = output / "profiles" / "default.json"
    profile.parent.mkdir(parents=True, exist_ok=True)
    profile.write_bytes(content)
    return {"path": "profiles/default.json", "sha256": _hash_bytes(content)}


def _copy_runtime_wiring(
    repository: Path, commit: str, output: Path
) -> list[dict[str, str]]:
    """Copy only the distribution Docker wiring from the fixed source tree."""

    copied: list[dict[str, str]] = []
    for source_path, output_name in _RUNTIME_WIRING:
        content = _git_file(repository, commit, source_path)
        if content is None:
            continue
        target = output / output_name
        target.write_bytes(content)
        if target.name.endswith(".sh"):
            target.chmod(0o755)
        copied.append({"path": output_name, "source_path": source_path, "sha256": _hash_bytes(content)})
    return copied


def _bundle_plugin(
    repository: Path,
    commit: str,
    stamp: str,
    root: str,
    output: Path,
    names: set[str],
) -> dict[str, str]:
    with tempfile.TemporaryDirectory(prefix="akashic-plugin-source-") as directory:
        package = Path(directory)
        archive = git(repository, "archive", "--format=tar", commit + ":" + root)
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as stream:
            stream.extractall(package, filter="data")
        manifest = load_static_plugin_manifest(package)
        if manifest.name in names:
            raise ValueError(f"发布插件名称重复: {manifest.name}")
        names.add(manifest.name)
        provenance = package / ".akashic-source.json"
        if provenance.exists():
            raise ValueError(f"插件源码已经占用构建来源文件: {root}")
        provenance.write_text(
            json.dumps({"commit": commit, "path": root}, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        git(package, "init", "--initial-branch=source")
        git(package, "add", "--all")
        env = {
            **os.environ,
            "GIT_AUTHOR_NAME": "Akashic build",
            "GIT_AUTHOR_EMAIL": "build@akashic.invalid",
            "GIT_COMMITTER_NAME": "Akashic build",
            "GIT_COMMITTER_EMAIL": "build@akashic.invalid",
            "GIT_AUTHOR_DATE": stamp,
            "GIT_COMMITTER_DATE": stamp,
        }
        git(
            package,
            "-c",
            "commit.gpgSign=false",
            "commit",
            "-m",
            f"Build {root} from {commit}",
            env=env,
        )
        identity = git(package, "rev-parse", "HEAD").decode().strip()
        bundle = output / f"{manifest.name}.bundle"
        git(package, "bundle", "create", str(bundle.resolve()), "HEAD", "source")
        return {
            "name": manifest.name,
            "source_commit": commit,
            "source_path": root,
            "file": bundle.name,
            "source_revision": identity,
            "sha256": hashlib.sha256(bundle.read_bytes()).hexdigest(),
        }


def build(repository: Path, revision: str, output: Path) -> dict[str, object]:
    """固定源提交生成 Core、插件 bundle、profile 和身份报告。"""

    # 1. 固定源提交；输出必须新建，失败产物保留供检查，不覆盖旧恢复材料。
    commit = git(
        repository,
        "rev-parse",
        "--verify",
        "--end-of-options",
        revision + "^{commit}",
    ).decode().strip()
    tree = git(repository, "rev-parse", f"{commit}^{{tree}}").decode().strip()
    stamp = git(repository, "show", "-s", "--format=%cI", commit).decode().strip()
    commit_epoch = int(git(repository, "show", "-s", "--format=%ct", commit).decode())
    files = git(repository, "ls-tree", "-r", "--name-only", commit).decode().splitlines()
    roots = sorted(
        {
            str(Path(path).parent)
            for path in files
            if path.startswith("plugins/") and path.endswith("/akashic.plugin.toml")
        }
    )
    entry_roots = {
        str(Path(path).parent)
        for path in files
        if path.startswith("plugins/")
        and path.endswith(("/plugin.py", "/message_plugin.py"))
    }
    missing = entry_roots - set(roots)
    if missing:
        raise ValueError(f"插件入口缺少发布 manifest: {sorted(missing)}")
    output.mkdir(parents=True, exist_ok=False)

    runtime_dependencies = _runtime_dependency_record(repository, commit, tree)
    core_paths = [
        path
        for path in CORE_PATHS
        if any(item == path or item.startswith(path + "/") for item in files)
    ]
    core = git(repository, "archive", "--format=tar", commit, "--", *core_paths)

    # 2. Web 构建完全在临时源码副本执行；只把生成的静态目录带入 Core tar。
    with tempfile.TemporaryDirectory(prefix="akashic-distribution-web-") as directory:
        asset_root, web = _build_web_assets(repository, commit, Path(directory))
        runtime_metadata = {**runtime_dependencies, "web": web}
        core = _append_bytes(
            core,
            "runtime-dependencies.json",
            (
                json.dumps(runtime_metadata, ensure_ascii=False, indent=2, sort_keys=True)
                + "\n"
            ).encode(),
            mtime=commit_epoch,
        )
        if asset_root is not None:
            core = _append_tree(
                core, asset_root / "chat", "static/chat", mtime=commit_epoch
            )
            core = _append_tree(
                core,
                asset_root / "dashboard",
                "static/dashboard",
                mtime=commit_epoch,
            )

    (output / "core.tar").write_bytes(core)
    report: dict[str, object] = {
        "schema_version": 2,
        "source_commit": commit,
        "source_tree": tree,
        "core": {
            "file": "core.tar",
            "sha256": hashlib.sha256(core).hexdigest(),
            "source_paths": core_paths,
            "config_example": "config.example.toml" in core_paths,
            "web": web,
        },
        "runtime_dependencies": runtime_dependencies,
        "plugins": [],
    }

    # 3. 独立 Git 源保留来源证明；安装仍走原有 clone、校验及 artifact 发布链。
    names: set[str] = set()
    rows = [
        _bundle_plugin(repository, commit, stamp, root, output, names)
        for root in roots
    ]
    report["plugins"] = rows
    profile = _copy_profile(repository, commit, output)
    if profile is None:
        raise ValueError(
            f"固定提交缺少正式产品 profile: {_DEFAULT_PROFILE_PATH.as_posix()}"
        )
    report["profiles"] = [profile]
    report["runtime_wiring"] = _copy_runtime_wiring(repository, commit, output)
    (output / "distribution.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repository", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--revision", default="HEAD")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            build(args.repository.resolve(), args.revision, args.output.resolve()),
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""从固定提交生成不带业务源码的 Core 与可独立安装的插件 Git bundle。"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import tarfile
import tempfile

from agent.plugins.static_manifest import load_static_plugin_manifest


# 只打包宿主运行入口；业务 plugins/ 不进入 Core，也没有开发源码路径。
CORE_PATHS = (
    "agent", "bootstrap", "bus", "core", "infra", "session", "utils",
    "mcp_servers", "host_bridge", "memory2", "prompts", "migrations",
    "main.py", "pyproject.toml", "requirements.txt", "requirements-dev.txt",
    "sdk/python", "docker/host-runtime",
)


def git(repository: Path, *args: str, env: dict[str, str] | None = None) -> bytes:
    """只执行明确 Git 参数，失败保留命令与 stderr。"""
    return subprocess.run(["git", "-C", str(repository), *args], check=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env).stdout


def build(repository: Path, revision: str, output: Path) -> dict[str, object]:
    """各包只含自身 Git 子树；构建报告记录源提交、产物身份与哈希。"""
    # 1. 固定源提交；输出必须新建，失败产物保留供检查，不覆盖旧恢复材料。
    commit = git(repository, "rev-parse", "--verify", "--end-of-options",
                 revision + "^{commit}").decode().strip()
    stamp = git(repository, "show", "-s", "--format=%cI", commit).decode().strip()
    files = git(repository, "ls-tree", "-r", "--name-only", commit).decode().splitlines()
    roots = sorted({str(Path(path).parent) for path in files
                    if path.startswith("plugins/") and path.endswith("/akashic.plugin.toml")})
    entry_roots = {str(Path(path).parent) for path in files
                   if path.startswith("plugins/") and path.endswith(("/plugin.py", "/message_plugin.py"))}
    missing = entry_roots - set(roots)
    if missing:
        raise ValueError(f"插件入口缺少发布 manifest: {sorted(missing)}")
    output.mkdir(parents=True, exist_ok=False)
    core_paths = [path for path in CORE_PATHS
                  if any(item == path or item.startswith(path + "/") for item in files)]
    core = git(repository, "archive", "--format=tar", commit, "--", *core_paths)
    (output / "core.tar").write_bytes(core)
    report: dict[str, object] = {"source_commit": commit,
        "core": {"file": "core.tar", "sha256": hashlib.sha256(core).hexdigest()},
        "plugins": []}
    rows = []
    names: set[str] = set()
    # 2. 独立 Git 源保留来源证明；安装仍走原有 clone、校验及 artifact 发布链。
    for root in roots:
        with tempfile.TemporaryDirectory(prefix="akashic-plugin-source-") as directory:
            package = Path(directory)
            archive = git(repository, "archive", "--format=tar", commit + ":" + root)
            with tarfile.open(fileobj=io.BytesIO(archive)) as stream:
                stream.extractall(package, filter="data")
            manifest = load_static_plugin_manifest(package)
            if manifest.name in names:
                raise ValueError(f"发布插件名称重复: {manifest.name}")
            names.add(manifest.name)
            provenance = package / ".akashic-source.json"
            if provenance.exists():
                raise ValueError(f"插件源码已经占用构建来源文件: {root}")
            provenance.write_text(json.dumps({"commit": commit, "path": root}, sort_keys=True) + "\n")
            git(package, "init", "--initial-branch=source")
            git(package, "add", "--all")
            env = {**os.environ, "GIT_AUTHOR_NAME": "Akashic build", "GIT_AUTHOR_EMAIL": "build@akashic.invalid",
                   "GIT_COMMITTER_NAME": "Akashic build", "GIT_COMMITTER_EMAIL": "build@akashic.invalid",
                   "GIT_AUTHOR_DATE": stamp, "GIT_COMMITTER_DATE": stamp}
            git(package, "-c", "commit.gpgSign=false", "commit", "-m", f"Build {root} from {commit}", env=env)
            identity = git(package, "rev-parse", "HEAD").decode().strip()
            bundle = output / (manifest.name + ".bundle")
            git(package, "bundle", "create", str(bundle.resolve()), "HEAD", "source")
            rows.append({"name": manifest.name, "source_path": root, "file": bundle.name,
                         "source_revision": identity, "sha256": hashlib.sha256(bundle.read_bytes()).hexdigest()})
    report["plugins"] = rows
    (output / "distribution.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--revision", default="HEAD")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.repository.resolve(), args.revision, args.output.resolve()), ensure_ascii=False))


if __name__ == "__main__":
    main()

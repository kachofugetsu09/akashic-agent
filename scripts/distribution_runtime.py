#!/usr/bin/env python3
"""验证正式 Core/插件分发身份并写入镜像运行身份。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import re

_REVISION = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _report(path: Path) -> dict[str, object]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("distribution report 必须是 object")
    return document


def write_runtime_info(
    *,
    distribution: Path,
    output: Path,
    source_commit: str,
    source_tree: str,
    core_sha256: str,
    base_image: str,
    arch_snapshot: str,
    pypi_index_url: str,
) -> dict[str, object]:
    """核对 build args 与分发报告后写镜像内不可变身份。"""

    if _REVISION.fullmatch(source_commit) is None or _REVISION.fullmatch(source_tree) is None:
        raise ValueError("source commit/tree 必须是完整小写 SHA")
    if _SHA256.fullmatch(core_sha256) is None:
        raise ValueError("Core sha256 必须是完整 SHA256")
    if "@sha256:" not in base_image:
        raise ValueError("base image 必须固定 digest")
    report_path = distribution / "distribution.json"
    report = _report(report_path)
    if report.get("source_commit") != source_commit or report.get("source_tree") != source_tree:
        raise ValueError("distribution source identity 与 Docker build args 不一致")
    core = report.get("core")
    if not isinstance(core, dict) or core.get("sha256") != core_sha256:
        raise ValueError("distribution Core sha256 与 Docker build args 不一致")
    profile_path = distribution / "profiles" / "default.json"
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    if not isinstance(profile, dict) or not isinstance(profile.get("name"), str):
        raise ValueError("default profile 缺少 name")
    lock_path = Path("/opt/akashic/source/docker/host-runtime/requirements.lock")
    runtime: dict[str, object] = {
        "schemaVersion": 3,
        "sourceCommit": source_commit,
        "sourceTree": source_tree,
        "coreSha256": core_sha256,
        "distributionReportSha256": _sha256(report_path),
        "defaultProfile": profile["name"],
        "defaultProfileSha256": _sha256(profile_path),
        "baseImage": base_image,
        "archSnapshot": arch_snapshot,
        "pypiIndexUrl": pypi_index_url,
        "requirementsLockSha256": _sha256(lock_path),
        "pythonVersion": platform.python_version(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(runtime, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return runtime


def check_runtime_info(path: Path, *, source_commit: str, source_tree: str) -> None:
    """在每次容器启动前核对运行身份环境变量。"""

    runtime = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(runtime, dict):
        raise ValueError("runtime-info 必须是 object")
    if runtime.get("schemaVersion") != 3:
        raise ValueError("runtime-info schemaVersion 不受支持")
    if runtime.get("sourceCommit") != source_commit:
        raise ValueError("runtime source commit 与 AKASHIC_RUNTIME_COMMIT 不一致")
    if runtime.get("sourceTree") != source_tree:
        raise ValueError("runtime source tree 与 AKASHIC_RUNTIME_TREE 不一致")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    write = subparsers.add_parser("write")
    write.add_argument("--distribution", type=Path, required=True)
    write.add_argument("--output", type=Path, required=True)
    write.add_argument("--source-commit", required=True)
    write.add_argument("--source-tree", required=True)
    write.add_argument("--core-sha256", required=True)
    write.add_argument("--base-image", required=True)
    write.add_argument("--arch-snapshot", required=True)
    write.add_argument("--pypi-index-url", required=True)
    check = subparsers.add_parser("check")
    check.add_argument("--runtime-info", type=Path, required=True)
    check.add_argument("--expected-commit", required=True)
    check.add_argument("--expected-tree", required=True)
    args = parser.parse_args()
    if args.command == "write":
        write_runtime_info(
            distribution=args.distribution,
            output=args.output,
            source_commit=args.source_commit,
            source_tree=args.source_tree,
            core_sha256=args.core_sha256,
            base_image=args.base_image,
            arch_snapshot=args.arch_snapshot,
            pypi_index_url=args.pypi_index_url,
        )
    else:
        check_runtime_info(
            args.runtime_info,
            source_commit=args.expected_commit,
            source_tree=args.expected_tree,
        )


if __name__ == "__main__":
    main()

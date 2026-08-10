from __future__ import annotations

import argparse
import hashlib
import json
import re
import runpy
import subprocess
import sys
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, cast

_SOURCE_ROOT = Path(__file__).resolve().parents[1]
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

from scripts.host_toolchain_identity import declared_toolchain_identity

_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")
_DEFAULT_BASE_IMAGE = (
    "archlinux@sha256:345a872f6c95e082d4b8c050af637eebb57402c6e2177b411c3acf7df84eb33b"
)
_FORBIDDEN_RELEASE_ROOTS = {".env", "auth.json", "config.toml"}


def _assert_release_paths_safe(repository: Path, commit: str) -> None:
    """Reject tracked deployment credentials before materializing a build context."""

    paths = _run(
        "git", "ls-tree", "-r", "--name-only", commit, cwd=repository
    ).splitlines()
    forbidden = []
    for name in paths:
        path = Path(name)
        lower = path.name.lower()
        if name in _FORBIDDEN_RELEASE_ROOTS:
            forbidden.append(name)
        elif lower.startswith("config.toml.settings-") and lower.endswith(".bak"):
            forbidden.append(name)
        elif lower.endswith((".pem", ".private-key", ".secret")):
            forbidden.append(name)
    if forbidden:
        raise RuntimeError(f"release commit 含禁止发布的凭据路径: {sorted(forbidden)}")


def _run(*arguments: str, cwd: Path | None = None) -> str:
    result = subprocess.run(
        list(arguments), cwd=cwd, check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_commit(repository: Path, requested: str) -> tuple[str, str]:
    if _COMMIT_PATTERN.fullmatch(requested) is None:
        raise RuntimeError("release --commit 必须是用户批准的完整 40 位 SHA")
    commit = _run(
        "git", "rev-parse", "--verify", f"{requested}^{{commit}}", cwd=repository
    )
    if _COMMIT_PATTERN.fullmatch(commit) is None:
        raise RuntimeError("release commit 必须解析为完整 40 位小写 SHA")
    tree = _run("git", "rev-parse", f"{commit}^{{tree}}", cwd=repository)
    if _COMMIT_PATTERN.fullmatch(tree) is None:
        raise RuntimeError("release tree 必须是完整 40 位小写 SHA")
    return commit, tree


def _create_context(
    repository: Path, commit: str, tree: str, target: Path
) -> dict[str, Any]:
    """Materialize exactly one Git commit and add its verifiable inventory."""

    # 1. Archive the immutable Git object, never the caller's working tree.
    archive = target.parent / "source.tar"
    with archive.open("wb") as output:
        subprocess.run(
            ["git", "archive", "--format=tar", commit],
            cwd=repository,
            check=True,
            stdout=output,
        )
    archive_sha256 = _sha256(archive)
    target.mkdir()
    with tarfile.open(archive) as source:
        source.extractall(target, filter="data")

    # 2. Inventory all archived paths; the image verifies this before building assets.
    verifier = target / "docker" / "host-runtime" / "verify_release_source.py"
    namespace = runpy.run_path(str(verifier))
    source_entries = cast(
        Callable[[Path], list[dict[str, Any]]], namespace["source_entries"]
    )
    manifest: dict[str, Any] = {
        "schemaVersion": 1,
        "sourceCommit": commit,
        "sourceTree": tree,
        "sourceArchiveSha256": archive_sha256,
        "files": source_entries(target),
    }
    manifest_path = target / ".akashic-source-manifest.json"
    _ = manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    manifest["sourceManifestSha256"] = _sha256(manifest_path)
    return manifest


def build_release(
    *,
    repository: Path,
    requested_commit: str,
    image_tag: str,
    output_manifest: Path,
    base_image: str,
    arch_snapshot: str,
) -> dict[str, Any]:
    """Build an immutable host-runtime image and record its local content digest."""

    repository = repository.resolve(strict=True)
    commit, tree = _resolve_commit(repository, requested_commit)
    _assert_release_paths_safe(repository, commit)
    with tempfile.TemporaryDirectory(prefix="akashic-host-runtime-") as temporary:
        context = Path(temporary) / "context"
        source = _create_context(repository, commit, tree, context)
        requirements_lock = context / "docker" / "host-runtime" / "requirements.lock"
        package_lock = context / "package-lock.json"
        host_toolchain_identity = declared_toolchain_identity(
            commit, (context / "mise.toml").read_bytes()
        )
        build_arguments = {
            "AKASHIC_BASE_IMAGE": base_image,
            "AKASHIC_ARCH_SNAPSHOT": arch_snapshot,
            "AKASHIC_SOURCE_COMMIT": commit,
            "AKASHIC_SOURCE_TREE": tree,
            "AKASHIC_SOURCE_ARCHIVE_SHA256": str(source["sourceArchiveSha256"]),
            "AKASHIC_SOURCE_MANIFEST_SHA256": str(source["sourceManifestSha256"]),
            "AKASHIC_REQUIREMENTS_LOCK_SHA256": _sha256(requirements_lock),
            "AKASHIC_PACKAGE_LOCK_SHA256": _sha256(package_lock),
        }
        command = ["docker", "build", "--pull=false", "--tag", image_tag]
        for key, value in build_arguments.items():
            command.extend(("--build-arg", f"{key}={value}"))
        command.extend(
            ("--file", str(context / "docker/host-runtime/Dockerfile"), str(context))
        )
        subprocess.run(command, check=True)

    image_id = _run("docker", "image", "inspect", image_tag, "--format", "{{.Id}}")
    if not image_id.startswith("sha256:"):
        raise RuntimeError(f"Docker 未返回 content-addressed image ID: {image_id}")
    runtime_info = json.loads(
        _run(
            "docker",
            "run",
            "--rm",
            "--entrypoint",
            "/bin/cat",
            image_id,
            "/opt/akashic/runtime-info.json",
        )
    )
    if runtime_info.get("sourceCommit") != commit:
        raise RuntimeError("built image runtime-info 与 release commit 不一致")
    result: dict[str, Any] = {
        "schemaVersion": 1,
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "repository": str(repository),
        "imageTag": image_tag,
        "imageId": image_id,
        "runtimeInfo": runtime_info,
        **source,
        "baseImage": base_image,
        "archSnapshot": arch_snapshot,
        "requirementsLockSha256": build_arguments["AKASHIC_REQUIREMENTS_LOCK_SHA256"],
        "packageLockSha256": build_arguments["AKASHIC_PACKAGE_LOCK_SHA256"],
        "hostToolchainIdentity": host_toolchain_identity,
    }
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    _ = output_manifest.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an immutable Akashic runtime")
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--image-tag", required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--base-image", default=_DEFAULT_BASE_IMAGE)
    parser.add_argument("--arch-snapshot", default="2026/08/09")
    args = parser.parse_args()
    result = build_release(
        repository=args.repository,
        requested_commit=args.commit,
        image_tag=args.image_tag,
        output_manifest=args.output_manifest,
        base_image=args.base_image,
        arch_snapshot=args.arch_snapshot,
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()

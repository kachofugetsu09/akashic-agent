from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_ARCH_SNAPSHOT_PATTERN = re.compile(r"\d{4}/\d{2}/\d{2}")


@dataclass(frozen=True)
class RuntimeIdentity:
    source_commit: str
    source_tree: str
    host_checkout: Path
    source_archive_sha256: str
    environment_digest: str
    image_id: str

    @classmethod
    def load(
        cls,
        runtime_info: Path,
        release_manifest: Path,
        *,
        expected_commit: str,
        host_checkout: Path,
    ) -> RuntimeIdentity:
        """Load and verify the immutable image identity contract."""

        # 1. Validate the deployment-owned expected identity.
        if _COMMIT_PATTERN.fullmatch(expected_commit) is None:
            raise RuntimeError("AKASHIC_RUNTIME_COMMIT 必须是完整 40 位小写 commit")
        if not host_checkout.is_absolute():
            raise RuntimeError("AKASHIC_RUNTIME_CHECKOUT 必须是宿主绝对路径")
        if not host_checkout.is_dir():
            raise RuntimeError(f"runtime host checkout 不存在: {host_checkout}")

        # 2. Verify the image-owned manifest matches the requested generation.
        document = json.loads(runtime_info.read_text(encoding="utf-8"))
        release = json.loads(release_manifest.read_text(encoding="utf-8"))
        if release.get("schemaVersion") != 1:
            raise RuntimeError("release manifest schemaVersion 不受支持")
        if release.get("runtimeInfo") != document:
            raise RuntimeError("runtime-info 与部署方 release manifest 不一致")
        image_id = str(release.get("imageId") or "")
        if re.fullmatch(r"sha256:[0-9a-f]{64}", image_id) is None:
            raise RuntimeError("release imageId 必须是 content-addressed SHA256")
        if document.get("schemaVersion") != 2:
            raise RuntimeError("runtime-info schemaVersion 不受支持")
        source_commit = str(document.get("sourceCommit") or "")
        source_tree = str(document.get("sourceTree") or "")
        if source_commit != expected_commit:
            raise RuntimeError(
                f"runtime commit 不一致: image={source_commit} expected={expected_commit}"
            )
        if _COMMIT_PATTERN.fullmatch(source_tree) is None:
            raise RuntimeError("runtime sourceTree 必须是完整 40 位小写 tree")
        source_archive_sha256 = _required_sha256(document, "sourceArchiveSha256")
        environment_fields = (
            "sourceManifestSha256",
            "pacmanDigest",
            "requirementsLockSha256",
            "packageLockSha256",
        )
        environment_values = [
            _required_sha256(document, key) for key in environment_fields
        ]
        base_image = str(document.get("baseImage") or "")
        if "@sha256:" not in base_image:
            raise RuntimeError("runtime baseImage 必须固定 digest")
        arch_snapshot = str(document.get("archSnapshot") or "")
        if _ARCH_SNAPSHOT_PATTERN.fullmatch(arch_snapshot) is None:
            raise RuntimeError("runtime archSnapshot 必须是 YYYY/MM/DD")
        versions = [
            str(document.get(key) or "")
            for key in ("pythonVersion", "nodeVersion", "npmVersion")
        ]
        if not all(versions):
            raise RuntimeError("runtime Python/Node/npm 版本不能为空")
        environment_digest = _identity_digest(
            [base_image, arch_snapshot, *environment_values, *versions]
        )

        # 3. The mounted host checkout must be the same clean source generation.
        head = _git_value(host_checkout, "rev-parse", "HEAD")
        tree = _git_value(host_checkout, "rev-parse", "HEAD^{tree}")
        if head != source_commit or tree != source_tree:
            raise RuntimeError(
                "runtime host checkout 与 image source identity 不一致: "
                f"head={head} tree={tree}"
            )
        status = _git_value(
            host_checkout,
            "status",
            "--porcelain",
            "--untracked-files=all",
        )
        if status:
            raise RuntimeError("runtime host checkout 必须保持 clean")
        return cls(
            source_commit,
            source_tree,
            host_checkout,
            source_archive_sha256,
            environment_digest,
            image_id,
        )


def _required_sha256(document: dict[str, object], key: str) -> str:
    value = str(document.get(key) or "")
    if _SHA256_PATTERN.fullmatch(value) is None:
        raise RuntimeError(f"runtime {key} 必须是 64 位 SHA256")
    return value


def _identity_digest(values: list[str]) -> str:
    return hashlib.sha256("\0".join(values).encode()).hexdigest()


def _git_value(checkout: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(checkout), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify Akashic runtime identity")
    parser.add_argument("--runtime-info", type=Path, required=True)
    parser.add_argument("--release-manifest", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--host-checkout", type=Path, required=True)
    args = parser.parse_args()
    RuntimeIdentity.load(
        args.runtime_info,
        args.release_manifest,
        expected_commit=args.expected_commit,
        host_checkout=args.host_checkout,
    )


if __name__ == "__main__":
    main()

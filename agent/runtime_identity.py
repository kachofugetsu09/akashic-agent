from __future__ import annotations

import argparse
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")


@dataclass(frozen=True)
class RuntimeIdentity:
    source_commit: str
    source_tree: str
    host_checkout: Path

    @classmethod
    def load(
        cls,
        runtime_info: Path,
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
        if document.get("schemaVersion") != 1:
            raise RuntimeError("runtime-info schemaVersion 不受支持")
        source_commit = str(document.get("sourceCommit") or "")
        source_tree = str(document.get("sourceTree") or "")
        if source_commit != expected_commit:
            raise RuntimeError(
                f"runtime commit 不一致: image={source_commit} expected={expected_commit}"
            )
        if _COMMIT_PATTERN.fullmatch(source_tree) is None:
            raise RuntimeError("runtime sourceTree 必须是完整 40 位小写 tree")

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
        return cls(source_commit, source_tree, host_checkout)


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
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--host-checkout", type=Path, required=True)
    args = parser.parse_args()
    RuntimeIdentity.load(
        args.runtime_info,
        expected_commit=args.expected_commit,
        host_checkout=args.host_checkout,
    )


if __name__ == "__main__":
    main()

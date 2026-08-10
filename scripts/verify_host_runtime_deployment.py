from __future__ import annotations

import argparse
import json
import re
import subprocess
import os
import sys
from pathlib import Path

_SOURCE_ROOT = Path(__file__).resolve().parents[1]
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

from scripts.host_toolchain_identity import resolve_toolchain_identity


def verify_deployment_image(release_manifest: Path, image: str) -> str:
    """Verify the exact local Docker image selected by a release manifest."""

    if re.fullmatch(r"sha256:[0-9a-f]{64}", image) is None:
        raise RuntimeError("部署必须使用完整 content-addressed image ID")
    release = json.loads(release_manifest.read_text(encoding="utf-8"))
    if release.get("schemaVersion") != 1 or release.get("imageId") != image:
        raise RuntimeError("部署 image 与 release manifest 不一致")
    actual = subprocess.run(
        ["docker", "image", "inspect", image, "--format", "{{.Id}}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if actual != image:
        raise RuntimeError(f"Docker Engine image identity 不一致: {actual}")
    return actual


def verify_host_toolchain_deployment(
    release_manifest: Path,
    runtime_checkout: Path,
    mise: Path,
    bridge_python: Path,
    expected_digest: str,
) -> dict[str, object]:
    """Prove the Bridge checkout, tools, interpreter, and imported source match release."""

    # 1. Recompute the target host toolchain instead of trusting runtime.env.
    release = json.loads(release_manifest.read_text(encoding="utf-8"))
    expected = release.get("hostToolchainIdentity")
    actual = resolve_toolchain_identity(runtime_checkout, mise)
    if expected != actual or actual.get("toolchainDigest") != expected_digest:
        raise RuntimeError("Host Bridge toolchain 与 release manifest 不一致")

    # 2. Verify the selected Python and agent module are the declared generation.
    bridge_python = bridge_python.absolute()
    if not bridge_python.is_file() or not os.access(bridge_python, os.X_OK):
        raise RuntimeError("Host Bridge Python 不存在或不可执行")
    python_version = subprocess.run(
        [str(bridge_python), "--version"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    declared_python = str(actual["tools"]["python"])
    if declared_python not in python_version:
        raise RuntimeError("Host Bridge Python 与 mise contract 不一致")
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(runtime_checkout.resolve(strict=True))
    module_path = subprocess.run(
        [
            str(bridge_python),
            "-c",
            "from pathlib import Path; import agent.host_bridge.server as m; "
            "print(Path(m.__file__).resolve())",
        ],
        cwd=runtime_checkout,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    expected_module = str(
        (runtime_checkout / "agent" / "host_bridge" / "server.py").resolve(strict=True)
    )
    if module_path != expected_module:
        raise RuntimeError("Host Bridge module 未从 release checkout 加载")
    return actual


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify an Akashic release image")
    parser.add_argument("--release-manifest", type=Path, required=True)
    parser.add_argument("--image")
    parser.add_argument("--host-only", action="store_true")
    parser.add_argument("--runtime-checkout", type=Path, required=True)
    parser.add_argument("--mise", type=Path, required=True)
    parser.add_argument("--bridge-python", type=Path, required=True)
    parser.add_argument("--expected-toolchain-digest", required=True)
    args = parser.parse_args()
    identity = verify_host_toolchain_deployment(
        args.release_manifest,
        args.runtime_checkout,
        args.mise,
        args.bridge_python,
        args.expected_toolchain_digest,
    )
    image: str | None = None
    if not args.host_only:
        if args.image is None:
            parser.error("--image is required unless --host-only is set")
        image = verify_deployment_image(args.release_manifest, args.image)
    print(
        json.dumps(
            {
                "imageId": image,
                "hostToolchainIdentity": identity,
            }
        )
    )


if __name__ == "__main__":
    main()

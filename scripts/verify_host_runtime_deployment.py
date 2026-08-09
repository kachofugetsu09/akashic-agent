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

from scripts.build_host_runtime_release import home_service_images
from scripts.host_toolchain_identity import resolve_toolchain_identity

_REQUIRED_HOME_SERVICE_CONTAINERS = {
    "RSSHUB_IMAGE": "akashic-services-rsshub-1",
    "REDIS_IMAGE": "akashic-services-redis-1",
    "BROWSERLESS_IMAGE": "akashic-services-browserless-1",
    "REAL_BROWSER_IMAGE": "akashic-services-real-browser-1",
}


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


def verify_home_service_images(
    release_manifest: Path, environment_file: Path
) -> dict[str, str]:
    """Reject sidecar image drift from the immutable release manifest."""

    release = json.loads(release_manifest.read_text(encoding="utf-8"))
    actual = home_service_images(environment_file.read_bytes())
    if release.get("homeServiceImages") != actual:
        raise RuntimeError("home-services image 与 release manifest 不一致")
    return actual


def verify_running_home_services(expected_images: dict[str, str]) -> dict[str, str]:
    """Verify required running sidecars use the release-pinned image references."""

    # 1. Inspect the exact Compose-owned containers required by Core.
    actual: dict[str, str] = {}
    for image_key, container_name in _REQUIRED_HOME_SERVICE_CONTAINERS.items():
        inspected = subprocess.run(
            [
                "docker",
                "container",
                "inspect",
                container_name,
                "--format",
                "{{.Config.Image}} {{.State.Running}}",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        image, separator, running = inspected.rpartition(" ")
        if not separator or running != "true":
            raise RuntimeError(f"required home service 未运行: {container_name}")
        actual[image_key] = image

    # 2. Reject a mixed release before Core starts.
    required_expected = {
        key: expected_images[key] for key in _REQUIRED_HOME_SERVICE_CONTAINERS
    }
    if actual != required_expected:
        raise RuntimeError("运行中的 home-services image 与 release manifest 不一致")
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
    parser.add_argument("--home-services-env-file", type=Path)
    parser.add_argument("--verify-running-home-services", action="store_true")
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
    home_images = None
    if args.home_services_env_file is not None:
        home_images = verify_home_service_images(
            args.release_manifest, args.home_services_env_file
        )
    if args.verify_running_home_services:
        if home_images is None:
            parser.error(
                "--verify-running-home-services requires --home-services-env-file"
            )
        verify_running_home_services(home_images)
    print(
        json.dumps(
            {
                "imageId": image,
                "hostToolchainIdentity": identity,
                "homeServiceImages": home_images,
            }
        )
    )


if __name__ == "__main__":
    main()

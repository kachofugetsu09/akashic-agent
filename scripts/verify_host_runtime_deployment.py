from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify an Akashic release image")
    parser.add_argument("--release-manifest", type=Path, required=True)
    parser.add_argument("--image", required=True)
    args = parser.parse_args()
    print(verify_deployment_image(args.release_manifest, args.image))


if __name__ == "__main__":
    main()

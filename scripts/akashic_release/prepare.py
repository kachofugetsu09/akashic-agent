from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Callable

from scripts.akashic_release.bridge import prepare_bridge_venv, prepare_runtime_checkout
from scripts.akashic_release.bridge import verify_bridge
from scripts.akashic_release.image import prepare_core_image
from scripts.akashic_release.manifest import read_json, write_json
from scripts.akashic_release.model import ReleasePaths
from scripts.verify_host_runtime_deployment import verify_deployment_image

Run = Callable[..., subprocess.CompletedProcess[str]]


def verify_host_prerequisites(*, mise: Path, run: Run) -> None:
    """Verify the narrow host commands required by preparation and activation."""

    missing = [
        name
        for name in ("docker", "git", "sudo", "systemctl")
        if shutil.which(name) is None
    ]
    if missing:
        raise RuntimeError(f"宿主缺少安装依赖: {', '.join(missing)}")
    if not mise.is_file() or not os.access(mise, os.X_OK):
        raise RuntimeError(f"mise 不存在或不可执行: {mise}")
    run(["docker", "compose", "version"], check=True, capture_output=True, text=True)
    run(["docker", "info"], check=True, capture_output=True, text=True)


def prepare_generation(
    *,
    paths: ReleasePaths,
    bootstrap_checkout: Path,
    commit: str,
    origin: str,
    mise: Path,
    run: Run,
) -> dict[str, object]:
    """Prepare and publish one immutable Core plus Bridge generation."""

    manifest_path = paths.release(commit)
    source = paths.source(commit)
    bridge_venv = paths.bridge_venv(commit)
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        bridge_python = bridge_venv / "bin/python"
        if (
            manifest.get("sourceCommit") != commit
            or manifest.get("runtimeCheckout") != str(source)
            or manifest.get("bridgePython") != str(bridge_python)
            or manifest.get("releaseRoot") != str(paths.root)
            or not source.is_dir()
        ):
            raise RuntimeError("既有 release manifest identity 漂移")
        if not bridge_python.is_file():
            raise RuntimeError("既有 release 缺少 Bridge Python")
        host_identity = manifest.get("hostToolchainIdentity")
        if not isinstance(host_identity, dict):
            raise RuntimeError("既有 release 缺少 Host Bridge identity")
        verify_deployment_image(manifest_path, str(manifest.get("imageId")))
        verify_bridge(
            manifest=manifest_path,
            checkout=source,
            mise=mise,
            bridge_python=bridge_python,
            toolchain_digest=str(host_identity.get("toolchainDigest")),
        )
        return manifest
    if source.exists() or bridge_venv.exists():
        raise RuntimeError("发现未发布的同 commit generation，需先人工审计")

    temporary_manifest = paths.releases / f".{commit}.{os.getpid()}.preparing.json"
    try:
        prepare_runtime_checkout(bootstrap_checkout, commit, source, origin)
        manifest = prepare_core_image(
            checkout=source,
            commit=commit,
            manifest=temporary_manifest,
            image_tag=f"akashic-core:{commit[:12]}",
        )
        bridge_python = prepare_bridge_venv(
            checkout=source,
            target=bridge_venv,
            mise=mise,
            run=run,
        )
        host_identity = manifest.get("hostToolchainIdentity")
        if not isinstance(host_identity, dict):
            raise RuntimeError("release manifest 缺少 Host Bridge identity")
        verify_bridge(
            manifest=temporary_manifest,
            checkout=source,
            mise=mise,
            bridge_python=bridge_python,
            toolchain_digest=str(host_identity["toolchainDigest"]),
        )
        manifest.update(
            {
                "runtimeCheckout": str(source),
                "bridgePython": str(bridge_python),
                "releaseRoot": str(paths.root),
            }
        )
        write_json(manifest_path, manifest)
        return manifest
    except BaseException:
        if source.exists():
            shutil.rmtree(source)
        if bridge_venv.exists():
            shutil.rmtree(bridge_venv)
        raise
    finally:
        if temporary_manifest.exists():
            temporary_manifest.unlink()

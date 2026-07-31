import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, cast

import pytest

from benchmark.harbor_v4flash.runtime_volume import (
    DEFAULT_BUILDER_IMAGE,
    DEFAULT_PYTHON_VERSION,
    RUNTIME_MOUNT_PATH,
    RUNTIME_TOP_LEVEL,
    RuntimeVolumeError,
    _builder_image_identity,
    _resolver_platform,
    build_runtime_volume,
    create_runtime_manifest,
    inspect_runtime_volume,
    runtime_compose_overlay,
    runtime_volume_labels,
)


def _manifest() -> tuple[dict[str, Any], bytes]:
    lock_bytes = b"example==1.0 --hash=sha256:abc\n"
    manifest = cast(
        dict[str, Any],
        create_runtime_manifest(
            requirements={
                "source": "requirements.txt",
                "source_digest": "sha256:requirements-source",
                "extras": ["tzdata"],
                "digest": "sha256:requirements",
            },
            uv={
                "version": "uv 0.8.23",
                "digest": "sha256:uv",
            },
            python_version=DEFAULT_PYTHON_VERSION,
            platform="linux/amd64",
            resolver_platform="x86_64-manylinux_2_28",
            builder_image={
                "reference": "debian:bullseye-slim",
                "id": "sha256:builder",
                "repo_digests": ["debian@sha256:repo"],
                "platform": "linux/amd64",
                "libc": "glibc 2.31",
            },
            resolved_lock_digest=(f"sha256:{hashlib.sha256(lock_bytes).hexdigest()}"),
        ),
    )
    return manifest, lock_bytes


def test_runtime_manifest_and_compose_freeze_identity() -> None:
    manifest, _ = _manifest()
    volume_name = str(manifest["volume_name"])

    assert volume_name.startswith("akasic-bench-runtime-v1-")
    assert (
        runtime_volume_labels(manifest)["akasic.benchmark.runtime.resolved_lock_digest"]
        == manifest["recipe"]["resolved_lock"]["digest"]
    )
    assert (
        runtime_volume_labels(manifest)["akasic.benchmark.runtime.builder_glibc"]
        == "glibc 2.31"
    )
    assert runtime_compose_overlay(volume_name) == {
        "services": {
            "main": {
                "volumes": [
                    {
                        "type": "volume",
                        "source": "akasic_runtime",
                        "target": RUNTIME_MOUNT_PATH,
                        "read_only": True,
                    }
                ]
            }
        },
        "volumes": {
            "akasic_runtime": {
                "external": True,
                "name": volume_name,
            }
        },
    }
    overlay = cast(
        dict[str, Any],
        runtime_compose_overlay(
            volume_name,
            task_image_id="sha256:task-image",
            git_volume_name="akasic-bench-git-v1-example",
        ),
    )
    assert overlay["services"]["main"] == {
        "image": "sha256:task-image",
        "pull_policy": "never",
        "volumes": [
            {
                "type": "volume",
                "source": "akasic_runtime",
                "target": RUNTIME_MOUNT_PATH,
                "read_only": True,
            },
            {
                "type": "volume",
                "source": "akasic_git",
                "target": "/opt/akashic-git",
                "read_only": True,
            },
        ],
    }


def test_runtime_labels_reject_manifest_without_builder_glibc() -> None:
    manifest, _ = _manifest()
    del manifest["recipe"]["builder_image"]["libc"]

    with pytest.raises(RuntimeVolumeError, match="builder glibc 缺失"):
        runtime_volume_labels(manifest)


def test_inspect_runtime_volume_verifies_manifest_lock_and_inputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manifest, lock_bytes = _manifest()
    volume_name = str(manifest["volume_name"])
    labels = runtime_volume_labels(manifest)
    requirements = manifest["recipe"]["requirements"]
    uv = manifest["recipe"]["uv"]

    monkeypatch.setattr(
        "benchmark.harbor_v4flash.runtime_volume._inspect_volume",
        lambda _: {
            "Driver": "local",
            "Scope": "local",
            "CreatedAt": "fixed",
            "Labels": labels,
        },
    )

    def read_file(**kwargs: str) -> bytes:
        if kwargs["relative_path"] == "manifest.json":
            return json.dumps(manifest).encode()
        return lock_bytes

    monkeypatch.setattr(
        "benchmark.harbor_v4flash.runtime_volume._read_volume_file",
        read_file,
    )
    monkeypatch.setattr(
        "benchmark.harbor_v4flash.runtime_volume._volume_top_level",
        lambda *_: list(RUNTIME_TOP_LEVEL),
    )
    monkeypatch.setattr(
        "benchmark.harbor_v4flash.runtime_volume._docker_platform",
        lambda: "linux/amd64",
    )
    monkeypatch.setattr(
        "benchmark.harbor_v4flash.runtime_volume._requirements_identity",
        lambda _: requirements,
    )
    monkeypatch.setattr(
        "benchmark.harbor_v4flash.runtime_volume._uv_identity",
        lambda _: uv,
    )

    report = inspect_runtime_volume(
        volume_name,
        source_root=tmp_path,
        uv_binary=tmp_path / "uv",
    )

    assert report["name"] == volume_name
    assert report["manifest"] == manifest


def test_inspect_runtime_volume_rejects_label_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manifest, _ = _manifest()
    volume_name = str(manifest["volume_name"])
    labels = runtime_volume_labels(manifest)
    labels["akasic.benchmark.runtime.uv_version"] = "uv changed"
    monkeypatch.setattr(
        "benchmark.harbor_v4flash.runtime_volume._inspect_volume",
        lambda _: {"Labels": labels},
    )
    monkeypatch.setattr(
        "benchmark.harbor_v4flash.runtime_volume._read_volume_file",
        lambda **_: json.dumps(manifest).encode(),
    )

    with pytest.raises(RuntimeVolumeError, match="labels"):
        inspect_runtime_volume(
            volume_name,
            source_root=tmp_path,
            uv_binary=tmp_path / "uv",
        )


def test_builder_rejects_non_frozen_python_before_docker(
    tmp_path: Path,
) -> None:
    with pytest.raises(RuntimeVolumeError, match="已冻结"):
        build_runtime_volume(
            source_root=tmp_path,
            uv_binary=tmp_path / "uv",
            python_version="3.13.8",
            builder_image_reference="debian:bookworm-slim",
        )


def test_runtime_resolver_uses_explicit_manylinux_baseline() -> None:
    assert DEFAULT_BUILDER_IMAGE == "debian:bullseye-slim"
    assert _resolver_platform("linux/amd64") == "x86_64-manylinux_2_28"
    assert _resolver_platform("linux/arm64") == "aarch64-manylinux_2_28"


def test_builder_image_id_reuses_local_immutable_image(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands: list[list[str]] = []

    def run(command: list[str], *, text: bool = True):
        commands.append(command)
        if command[:3] == ["docker", "image", "inspect"]:
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=json.dumps(
                    [
                        {
                            "Id": "sha256:builder",
                            "RepoDigests": ["debian@sha256:repo"],
                            "Os": "linux",
                            "Architecture": "amd64",
                        }
                    ]
                ),
                stderr="",
            )
        return subprocess.CompletedProcess(
            command,
            0,
            stdout="glibc 2.31\n",
            stderr="",
        )

    monkeypatch.setattr(
        "benchmark.harbor_v4flash.runtime_volume._run",
        run,
    )

    identity = _builder_image_identity("sha256:builder")

    assert identity["id"] == "sha256:builder"
    assert not any(command[:2] == ["docker", "pull"] for command in commands)


def test_builder_rejects_glibc_newer_than_official_task_floor(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "benchmark.harbor_v4flash.runtime_volume._docker_platform",
        lambda: "linux/amd64",
    )
    monkeypatch.setattr(
        "benchmark.harbor_v4flash.runtime_volume._requirements_identity",
        lambda _: {"digest": "sha256:requirements"},
    )
    monkeypatch.setattr(
        "benchmark.harbor_v4flash.runtime_volume._uv_identity",
        lambda _: {"version": "uv test", "digest": "sha256:uv"},
    )
    monkeypatch.setattr(
        "benchmark.harbor_v4flash.runtime_volume._builder_image_identity",
        lambda _: {
            "reference": "debian:bookworm-slim",
            "id": "sha256:builder",
            "repo_digests": [],
            "platform": "linux/amd64",
            "libc": "glibc 2.36",
        },
    )

    with pytest.raises(RuntimeVolumeError, match="glibc 高于兼容上限"):
        build_runtime_volume(
            source_root=tmp_path,
            uv_binary=tmp_path / "uv",
            python_version=DEFAULT_PYTHON_VERSION,
            builder_image_reference="debian:bookworm-slim",
        )

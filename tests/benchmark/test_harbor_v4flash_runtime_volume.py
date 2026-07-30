import hashlib
import json
from pathlib import Path

import pytest

from benchmark.harbor_v4flash.runtime_volume import (
    DEFAULT_PYTHON_VERSION,
    RUNTIME_MOUNT_PATH,
    RUNTIME_TOP_LEVEL,
    RuntimeVolumeError,
    build_runtime_volume,
    create_runtime_manifest,
    inspect_runtime_volume,
    runtime_compose_overlay,
    runtime_volume_labels,
)


def _manifest() -> tuple[dict[str, object], bytes]:
    lock_bytes = b"example==1.0 --hash=sha256:abc\n"
    manifest = create_runtime_manifest(
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
        resolver_platform="x86_64-unknown-linux-gnu",
        builder_image={
            "reference": "debian:bookworm-slim",
            "id": "sha256:builder",
            "repo_digests": ["debian@sha256:repo"],
            "platform": "linux/amd64",
        },
        resolved_lock_digest=(
            f"sha256:{hashlib.sha256(lock_bytes).hexdigest()}"
        ),
    )
    return manifest, lock_bytes


def test_runtime_manifest_and_compose_freeze_identity() -> None:
    manifest, _ = _manifest()
    volume_name = str(manifest["volume_name"])

    assert volume_name.startswith("akasic-bench-runtime-v1-")
    assert runtime_volume_labels(manifest)[
        "akasic.benchmark.runtime.resolved_lock_digest"
    ] == manifest["recipe"]["resolved_lock"]["digest"]
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

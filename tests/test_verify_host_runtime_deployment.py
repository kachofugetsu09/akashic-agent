from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from scripts.verify_host_runtime_deployment import (
    verify_deployment_image,
    verify_home_service_images,
    verify_host_toolchain_deployment,
    verify_running_home_services,
)


def test_deployment_requires_exact_engine_image(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    image = "sha256:" + "a" * 64
    manifest = tmp_path / "release.json"
    manifest.write_text(
        json.dumps({"schemaVersion": 1, "imageId": image}), encoding="utf-8"
    )

    def run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess("docker", 0, image + "\n", "")

    monkeypatch.setattr(subprocess, "run", run)
    assert verify_deployment_image(manifest, image) == image


def test_deployment_rejects_mutable_tag(tmp_path: Path) -> None:
    manifest = tmp_path / "release.json"
    manifest.write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="content-addressed"):
        verify_deployment_image(manifest, "akashic:latest")


def test_deployment_rejects_runtime_env_toolchain_digest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    identity = {
        "schemaVersion": 1,
        "releaseCommit": "a" * 40,
        "miseConfigSha256": "b" * 64,
        "tools": {"python": "3.14.6"},
        "toolchainDigest": "c" * 64,
    }
    manifest = tmp_path / "release.json"
    manifest.write_text(
        json.dumps({"hostToolchainIdentity": identity}), encoding="utf-8"
    )
    monkeypatch.setattr(
        "scripts.verify_host_runtime_deployment.resolve_toolchain_identity",
        lambda _checkout, _mise: identity,
    )

    with pytest.raises(RuntimeError, match="toolchain"):
        verify_host_toolchain_deployment(
            manifest,
            checkout,
            tmp_path / "mise",
            tmp_path / "python",
            "d" * 64,
        )


def test_deployment_rejects_sidecar_image_drift(tmp_path: Path) -> None:
    digest = "a" * 64
    values = {
        key: f"example/{key.lower()}@sha256:{digest}"
        for key in (
            "RSSHUB_IMAGE",
            "REDIS_IMAGE",
            "BROWSERLESS_IMAGE",
            "REAL_BROWSER_IMAGE",
            "OPENCLI_BROWSER_IMAGE",
        )
    }
    manifest = tmp_path / "release.json"
    manifest.write_text(json.dumps({"homeServiceImages": values}), encoding="utf-8")
    environment = tmp_path / "home-services.env"
    environment.write_text(
        "\n".join(f"{key}={value}" for key, value in values.items()) + "\n",
        encoding="utf-8",
    )
    assert verify_home_service_images(manifest, environment) == values

    environment.write_text(
        environment.read_text(encoding="utf-8").replace(digest, "b" * 64, 1),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="release manifest"):
        verify_home_service_images(manifest, environment)


def test_toolchain_verification_preserves_venv_python_symlink(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    checkout = tmp_path / "checkout"
    module = checkout / "agent" / "host_bridge" / "server.py"
    module.parent.mkdir(parents=True)
    module.write_text("", encoding="utf-8")
    target = tmp_path / "python-target"
    target.write_text("", encoding="utf-8")
    target.chmod(0o755)
    launcher = tmp_path / "venv" / "bin" / "python"
    launcher.parent.mkdir(parents=True)
    launcher.symlink_to(target)
    identity = {
        "schemaVersion": 1,
        "releaseCommit": "a" * 40,
        "miseConfigSha256": "b" * 64,
        "tools": {"python": "3.14.6"},
        "toolchainDigest": "c" * 64,
    }
    manifest = tmp_path / "release.json"
    manifest.write_text(
        json.dumps({"hostToolchainIdentity": identity}), encoding="utf-8"
    )
    monkeypatch.setattr(
        "scripts.verify_host_runtime_deployment.resolve_toolchain_identity",
        lambda _checkout, _mise: identity,
    )
    calls: list[list[str]] = []

    def run(
        arguments: list[str], **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        calls.append(arguments)
        output = "Python 3.14.6\n" if arguments[1] == "--version" else f"{module}\n"
        return subprocess.CompletedProcess(arguments, 0, output, "")

    monkeypatch.setattr(subprocess, "run", run)
    assert (
        verify_host_toolchain_deployment(
            manifest, checkout, tmp_path / "mise", launcher, "c" * 64
        )
        == identity
    )
    assert calls[0][0] == str(launcher.absolute())
    assert calls[1][0] == str(launcher.absolute())


def test_deployment_verifies_running_required_sidecars(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = {
        "RSSHUB_IMAGE": "example/rsshub@sha256:" + "a" * 64,
        "REDIS_IMAGE": "example/redis@sha256:" + "b" * 64,
        "BROWSERLESS_IMAGE": "example/browserless@sha256:" + "c" * 64,
        "REAL_BROWSER_IMAGE": "example/real-browser@sha256:" + "d" * 64,
        "OPENCLI_BROWSER_IMAGE": "example/opencli@sha256:" + "e" * 64,
    }
    containers = {
        "akashic-services-rsshub-1": expected["RSSHUB_IMAGE"],
        "akashic-services-redis-1": expected["REDIS_IMAGE"],
        "akashic-services-browserless-1": expected["BROWSERLESS_IMAGE"],
        "akashic-services-real-browser-1": expected["REAL_BROWSER_IMAGE"],
    }

    def run(
        arguments: list[str], **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            arguments, 0, containers[arguments[3]] + " true\n", ""
        )

    monkeypatch.setattr(subprocess, "run", run)
    assert verify_running_home_services(expected) == {
        key: expected[key]
        for key in (
            "RSSHUB_IMAGE",
            "REDIS_IMAGE",
            "BROWSERLESS_IMAGE",
            "REAL_BROWSER_IMAGE",
        )
    }


def test_deployment_rejects_running_sidecar_from_previous_release(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = {
        "RSSHUB_IMAGE": "example/rsshub@sha256:" + "a" * 64,
        "REDIS_IMAGE": "example/redis@sha256:" + "b" * 64,
        "BROWSERLESS_IMAGE": "example/browserless@sha256:" + "c" * 64,
        "REAL_BROWSER_IMAGE": "example/real-browser@sha256:" + "d" * 64,
    }

    def run(
        arguments: list[str], **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        image = expected[
            next(key for key in expected if key.lower().split("_")[0] in arguments[3])
        ]
        if arguments[3] == "akashic-services-rsshub-1":
            image = "example/rsshub@sha256:" + "f" * 64
        return subprocess.CompletedProcess(arguments, 0, image + " true\n", "")

    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(RuntimeError, match="运行中的 home-services"):
        verify_running_home_services(expected)

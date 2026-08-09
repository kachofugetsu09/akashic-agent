from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from scripts.verify_host_runtime_deployment import verify_deployment_image


def test_deployment_requires_exact_engine_image(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    image = "sha256:" + "a" * 64
    manifest = tmp_path / "release.json"
    manifest.write_text(
        json.dumps({"schemaVersion": 1, "imageId": image}), encoding="utf-8"
    )

    def run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args[0], 0, image + "\n", "")

    monkeypatch.setattr(subprocess, "run", run)
    assert verify_deployment_image(manifest, image) == image


def test_deployment_rejects_mutable_tag(tmp_path: Path) -> None:
    manifest = tmp_path / "release.json"
    manifest.write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="content-addressed"):
        verify_deployment_image(manifest, "akashic:latest")

import subprocess

import pytest

from benchmark.harbor_v4flash.git_volume import (
    GIT_MOUNT_PATH,
    GIT_TOP_LEVEL,
    GitVolumeError,
    _find_reusable_git_volume,
    create_git_manifest,
    git_volume_labels,
)


def test_git_manifest_freezes_builder_packages_and_content() -> None:
    manifest = create_git_manifest(
        builder_image={
            "reference": "debian:bullseye-slim",
            "id": "sha256:builder",
            "repo_digests": ["debian@sha256:repo"],
            "platform": "linux/amd64",
        },
        metadata={
            "git_version": "git version 2.30.2",
            "git_package": "1:2.30.2-1",
            "ca_certificates_package": "20210119",
        },
        content_digest="sha256:content",
    )

    assert manifest["volume_name"].startswith("akasic-bench-git-v1-")
    assert manifest["contents"] == {
        "mount_path": GIT_MOUNT_PATH,
        "git_path": "bin/git",
        "top_level": list(GIT_TOP_LEVEL),
        "contains_source": False,
        "contains_workspace": False,
        "contains_task_data": False,
        "contains_secrets": False,
    }
    labels = git_volume_labels(manifest)
    assert labels["akasic.benchmark.git.content_digest"] == "sha256:content"
    assert labels["akasic.benchmark.git.git_version"] == "git version 2.30.2"


def test_git_volume_cache_reuses_valid_local_volume(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "benchmark.harbor_v4flash.git_volume._run",
        lambda _: subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="akasic-bench-git-v1-valid\n",
            stderr="",
        ),
    )
    expected = {"name": "akasic-bench-git-v1-valid"}
    monkeypatch.setattr(
        "benchmark.harbor_v4flash.git_volume.inspect_git_volume",
        lambda _: expected,
    )

    assert _find_reusable_git_volume({"id": "sha256:builder"}) == expected


def test_git_volume_cache_exposes_corrupt_local_volume(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "benchmark.harbor_v4flash.git_volume._run",
        lambda _: subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="akasic-bench-git-v1-corrupt\n",
            stderr="",
        ),
    )

    def reject(_: str) -> dict[str, object]:
        raise GitVolumeError("corrupt")

    monkeypatch.setattr(
        "benchmark.harbor_v4flash.git_volume.inspect_git_volume",
        reject,
    )

    with pytest.raises(GitVolumeError, match="损坏"):
        _find_reusable_git_volume({"id": "sha256:builder"})

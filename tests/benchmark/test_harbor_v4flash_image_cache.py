import asyncio
import threading
from pathlib import Path

import pytest

from benchmark.harbor_v4flash.image_cache import (
    MAX_IMAGE_PULL_CONCURRENCY,
    TaskImageError,
    _pull_image,
    prefetch_task_images,
    task_image_reference,
)


def test_task_image_reference_requires_prebuilt_image(tmp_path: Path) -> None:
    task = tmp_path / "task"
    task.mkdir()
    (task / "task.toml").write_text(
        """
schema_version = "1.1"
[task]
name = "test/task"
[environment]
docker_image = "example/task:fixed"
""".strip(),
        encoding="utf-8",
    )

    assert task_image_reference(task) == "example/task:fixed"


def test_task_image_reference_rejects_missing_image(tmp_path: Path) -> None:
    task = tmp_path / "task"
    task.mkdir()
    (task / "task.toml").write_text(
        """
schema_version = "1.1"
[task]
name = "test/task"
[environment]
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(TaskImageError, match="docker_image"):
        task_image_reference(task)


def test_pull_image_reuses_complete_local_image(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = {
        "reference": "example/task:fixed",
        "id": "sha256:image",
        "repo_digests": ["example/task@sha256:repo"],
        "platform": "linux/amd64",
        "size_bytes": 123,
    }
    monkeypatch.setattr(
        "benchmark.harbor_v4flash.image_cache._inspect_image",
        lambda _: identity,
    )

    assert _pull_image("example/task:fixed") == {
        **identity,
        "cache_hit": True,
        "pull_attempts": 0,
    }


def test_prefetch_limits_registry_concurrency(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    active = 0
    peak = 0
    lock = threading.Lock()
    release = threading.Event()
    tasks = []
    for index in range(4):
        task = tmp_path / f"task-{index}"
        task.mkdir()
        (task / "task.toml").write_text(
            f"""
schema_version = "1.1"
[task]
name = "test/task-{index}"
[environment]
docker_image = "example/task-{index}:fixed"
""".strip(),
            encoding="utf-8",
        )
        tasks.append(task)

    def pull(reference: str) -> dict[str, object]:
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
            if peak == MAX_IMAGE_PULL_CONCURRENCY:
                release.set()
        assert release.wait(timeout=5)
        with lock:
            active -= 1
        return {"reference": reference, "id": "sha256:image"}

    monkeypatch.setattr(
        "benchmark.harbor_v4flash.image_cache._pull_image",
        pull,
    )

    result = asyncio.run(prefetch_task_images(tasks))

    assert peak == MAX_IMAGE_PULL_CONCURRENCY
    assert set(result) == {str(path.resolve()) for path in tasks}


def test_prefetch_pulls_shared_reference_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tasks = []
    for index in range(2):
        task = tmp_path / f"task-{index}"
        task.mkdir()
        (task / "task.toml").write_text(
            f"""
schema_version = "1.1"
[task]
name = "test/task-{index}"
[environment]
docker_image = "example/shared:fixed"
""".strip(),
            encoding="utf-8",
        )
        tasks.append(task)
    calls: list[str] = []

    def pull(reference: str) -> dict[str, object]:
        calls.append(reference)
        return {"reference": reference, "id": "sha256:image"}

    monkeypatch.setattr(
        "benchmark.harbor_v4flash.image_cache._pull_image",
        pull,
    )

    result = asyncio.run(prefetch_task_images(tasks))

    assert calls == ["example/shared:fixed"]
    assert set(result) == {str(path.resolve()) for path in tasks}

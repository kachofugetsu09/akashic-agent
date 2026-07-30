from __future__ import annotations

import asyncio
import json
import subprocess
from pathlib import Path
from typing import Any

from harbor.models.task.config import TaskConfig as HarborTaskConfig

MAX_IMAGE_PULL_CONCURRENCY = 2
IMAGE_PULL_ATTEMPTS = 3
IMAGE_PULL_TIMEOUT_SEC = 1800


class TaskImageError(RuntimeError):
    pass


def task_image_reference(task_dir: Path) -> str:
    """读取 Harbor task 声明的预构建 Docker image。"""

    config_path = task_dir.resolve() / "task.toml"
    config = HarborTaskConfig.model_validate_toml(
        config_path.read_text(encoding="utf-8")
    )
    reference = (config.environment.docker_image or "").strip()
    if not reference:
        raise TaskImageError(f"{config_path} 未声明 [environment].docker_image")
    return reference


def _inspect_image(reference: str) -> dict[str, object] | None:
    result = subprocess.run(
        ["docker", "image", "inspect", reference],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        return None
    payload: object = json.loads(result.stdout)
    if not isinstance(payload, list) or len(payload) != 1:
        raise TaskImageError("docker image inspect 必须返回一个 image")
    raw = payload[0]
    if not isinstance(raw, dict):
        raise TaskImageError("docker image inspect 元素必须是对象")
    image: dict[str, Any] = raw
    image_id = str(image.get("Id") or "")
    platform = f"{image.get('Os')}/{image.get('Architecture')}"
    if not image_id.startswith("sha256:"):
        raise TaskImageError(f"task image 缺少不可变 ID：{reference}")
    if platform not in {"linux/amd64", "linux/arm64"}:
        raise TaskImageError(f"task image 平台不受支持：{reference} {platform}")
    size = image.get("Size")
    if not isinstance(size, int) or size <= 0:
        raise TaskImageError(f"task image 缺少有效 size：{reference}")
    return {
        "reference": reference,
        "id": image_id,
        "repo_digests": sorted(
            str(value) for value in image.get("RepoDigests") or []
        ),
        "platform": platform,
        "size_bytes": size,
    }


def _pull_image(reference: str) -> dict[str, object]:
    """命中本地 image cache，或在模型运行前有限重试拉取一次。"""

    # 1. 已冻结到本机的完整 image 直接复用，不再次访问 registry。
    cached = _inspect_image(reference)
    if cached is not None:
        return {**cached, "cache_hit": True, "pull_attempts": 0}

    # 2. 只重试尚未创建 trial 的幂等下载；最后一次错误保持原义。
    last_output = ""
    for attempt in range(1, IMAGE_PULL_ATTEMPTS + 1):
        try:
            result = subprocess.run(
                ["docker", "pull", reference],
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=IMAGE_PULL_TIMEOUT_SEC,
            )
        except subprocess.TimeoutExpired as error:
            last_output = f"docker pull 超过 {IMAGE_PULL_TIMEOUT_SEC}s"
            if attempt == IMAGE_PULL_ATTEMPTS:
                raise TaskImageError(
                    f"task image 拉取超时：{reference}；{last_output}"
                ) from error
        else:
            last_output = "\n".join((result.stdout, result.stderr)).strip()
            if result.returncode == 0:
                inspected = _inspect_image(reference)
                if inspected is None:
                    raise TaskImageError(
                        f"docker pull 成功但 image 不可 inspect：{reference}"
                    )
                return {
                    **inspected,
                    "cache_hit": False,
                    "pull_attempts": attempt,
                }
            if attempt == IMAGE_PULL_ATTEMPTS:
                raise TaskImageError(
                    f"task image 拉取失败：{reference}\n{last_output[-4000:]}"
                )
    raise AssertionError("image pull retry loop 没有终止")


async def prefetch_task_images(
    task_dirs: list[Path],
) -> dict[str, dict[str, object]]:
    """最多两路预拉取 task images，并按 task 路径返回不可变身份。"""

    # 1. 相同 reference 只拉取一次，避免并发 trial 重复访问 registry。
    task_references = {
        str(task_dir.resolve()): task_image_reference(task_dir.resolve())
        for task_dir in task_dirs
    }
    references = sorted(set(task_references.values()))

    # 2. semaphore 独占 registry 并发，trial 不再同时承担 image 下载。
    semaphore = asyncio.Semaphore(MAX_IMAGE_PULL_CONCURRENCY)

    async def prefetch(reference: str) -> tuple[str, dict[str, object]]:
        async with semaphore:
            image = await asyncio.to_thread(_pull_image, reference)
        return reference, image

    # 3. 任一 image 失败则整个批次在创建 trial 前失败。
    images = dict(await asyncio.gather(*(prefetch(ref) for ref in references)))
    return {
        task_path: images[reference]
        for task_path, reference in task_references.items()
    }

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


class CampaignGateError(RuntimeError):
    pass


MAX_CAMPAIGN_CONCURRENCY = 6


def task_slug(task_dir: Path) -> str:
    value = re.sub(r"[^a-z0-9-]", "-", task_dir.name.lower()).strip("-")
    if not value:
        raise ValueError(f"task 目录无法生成 slug：{task_dir}")
    return value[:48]


def validate_campaign_request(task_dirs: list[Path], max_concurrent: int) -> None:
    """验证 diagnostic campaign 的任务身份和并发硬上限。"""

    # 1. 当前阶段只允许用户确认过的最多六并发。
    if not 1 <= max_concurrent <= MAX_CAMPAIGN_CONCURRENCY:
        raise ValueError(
            f"max_concurrent 必须在 1 到 {MAX_CAMPAIGN_CONCURRENCY} 之间"
        )

    # 2. 一个 case 对应一个独立 task 实例，不接受重复路径。
    resolved = [path.resolve() for path in task_dirs]
    if len(resolved) < 2:
        raise ValueError("campaign 至少需要两个 task；单 task 请使用 smoke")
    if len(set(resolved)) != len(resolved):
        raise ValueError("campaign 不允许重复 task")
    missing = [str(path) for path in resolved if not path.is_dir()]
    if missing:
        raise FileNotFoundError(f"task 目录不存在：{missing}")


def find_open_concurrency_gate(
    runs_root: Path,
    *,
    expected_source_digest: str,
) -> dict[str, Any]:
    """读取当前源码已完成 smoke 的并发授权证据。"""

    candidates: list[tuple[float, Path, dict[str, Any]]] = []
    for path in runs_root.glob("akasic-bench-v4flash-*/campaign-manifest.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        gate = payload.get("concurrency_gate")
        online = payload.get("online")
        docker = payload.get("docker")
        source = payload.get("source")
        if (
            payload.get("state") == "completed"
            and isinstance(gate, dict)
            and gate.get("opened") is True
            and gate.get("max_concurrent") == MAX_CAMPAIGN_CONCURRENCY
            and isinstance(online, dict)
            and online.get("status") == "passed"
            and isinstance(docker, dict)
            and docker.get("all_stopped") is True
            and isinstance(source, dict)
            and source.get("digest_after") == expected_source_digest
        ):
            candidates.append((path.stat().st_mtime, path, payload))
    if not candidates:
        raise CampaignGateError(
            "未找到当前源码完成且隔离验证通过的 "
            f"concurrency={MAX_CAMPAIGN_CONCURRENCY} smoke Gate："
            f"{expected_source_digest}"
        )
    _, path, payload = max(candidates, key=lambda item: item[0])
    return {
        "manifest": str(path),
        "trial_name": payload["trial_name"],
        "source_digest": payload["source"]["digest_after"],
    }

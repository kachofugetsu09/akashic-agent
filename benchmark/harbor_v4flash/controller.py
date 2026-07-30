from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import subprocess
import time
import tomllib
from pathlib import Path
from typing import Any, cast

from harbor.models.environment_type import EnvironmentType
from harbor.models.task.config import TaskConfig as HarborTaskConfig
from harbor.models.trial.config import (
    AgentConfig,
    EnvironmentConfig,
    TaskConfig as TrialTaskConfig,
    TrialConfig,
)
from harbor.trial.trial import Trial

from benchmark.harbor_v4flash import HARNESS_VERSION
from benchmark.harbor_v4flash.agent import AkashicHarborAgent
from benchmark.harbor_v4flash.campaign import (
    find_open_concurrency_gate,
    task_slug,
    validate_campaign_request,
)
from benchmark.harbor_v4flash.isolation import (
    BENCHMARK_PREFIX,
    IsolationError,
    artifact_digests,
    atomic_json,
    compose_project_name,
    create_source_bundle,
    inspect_compose_project,
    online_process_snapshot,
    reserve_compose_network,
    source_tree_digest,
    validate_online_processes_unchanged,
)
from benchmark.harbor_v4flash.runtime_volume import (
    inspect_runtime_volume,
    runtime_compose_overlay,
)

DEFAULT_FORBIDDEN_PATHS = (
    Path("/mnt/data/coding/akasic-agent"),
    Path("/home/huashen/.akashic/workspace"),
    Path("/home/huashen/.akashic-plugin/cache"),
)
HARNESS_CLEANUP_RESERVE_SEC = 120.0


def _git_output(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.strip()


def _credential_env(profile_path: Path) -> dict[str, str]:
    """从本机 profile 只提取当前实验所需密钥，不持久化其值。"""

    data = tomllib.loads(profile_path.read_text(encoding="utf-8"))
    llm = data.get("llm")
    memory = data.get("memory")
    if not isinstance(llm, dict) or not isinstance(memory, dict):
        raise ValueError("credential profile 缺少 llm 或 memory")
    main = llm.get("main")
    embedding = memory.get("embedding")
    if not isinstance(main, dict) or not isinstance(embedding, dict):
        raise ValueError("credential profile 缺少 llm.main 或 memory.embedding")
    deepseek = str(main.get("api_key") or "").strip()
    dashscope = str(embedding.get("api_key") or "").strip()
    if not deepseek or not dashscope:
        raise ValueError("credential profile 缺少 DeepSeek 或 embedding API key")
    return {
        "DEEPSEEK_API_KEY": deepseek,
        "DASHSCOPE_API_KEY": dashscope,
    }


def _task_agent_timeout_sec(task_dir: Path) -> float:
    """读取并校验 Harbor task 声明的 agent 执行预算。"""

    # 1. 复用 Harbor 的 task schema 解析权威配置。
    config_path = task_dir / "task.toml"
    config = HarborTaskConfig.model_validate_toml(
        config_path.read_text(encoding="utf-8")
    )

    # 2. 本 harness 要求显式、有限且为正的 task 预算。
    timeout_sec = config.agent.timeout_sec
    if timeout_sec is None or not math.isfinite(timeout_sec) or timeout_sec <= 0:
        raise ValueError(f"{config_path} 缺少有效的 [agent].timeout_sec")
    return timeout_sec


def _safe_trial_result(result: Any) -> dict[str, object]:
    verifier = getattr(result, "verifier_result", None)
    rewards = getattr(verifier, "rewards", None) if verifier is not None else None
    exception = getattr(result, "exception_info", None)
    return {
        "trial_name": result.trial_name,
        "task_name": result.task_name,
        "rewards": rewards,
        "exception": (
            None
            if exception is None
            else {
                "type": getattr(exception, "exception_type", None),
                "message": getattr(exception, "exception_message", None),
            }
        ),
    }


def _inspect_finished_project(
    result: Any,
    project_name: str,
) -> tuple[list[dict[str, object]], str | None]:
    """保留 Harbor 前置失败，同时让成功 trial 的容器缺失继续 fail-loud。"""

    try:
        return inspect_compose_project(project_name), None
    except IsolationError as error:
        if getattr(result, "exception_info", None) is None:
            raise
        return [], str(error)


async def run_trial(
    args: argparse.Namespace,
    task_dir: Path,
    *,
    run_kind: str,
) -> dict[str, object]:
    """运行一个保留容器的独立 task，并生成不可含密钥的 manifest。"""

    # 1. 冻结源码、依赖、任务与线上 owner 证据。
    source_root = args.source_root.resolve()
    task_dir = task_dir.resolve()
    runs_root = args.runs_dir.resolve()
    task_agent_timeout_sec = _task_agent_timeout_sec(task_dir)
    uv_binary = args.uv_binary.resolve()
    runtime_volume = inspect_runtime_volume(
        args.runtime_volume,
        source_root=source_root,
        uv_binary=uv_binary,
    )
    runtime_manifest = cast(dict[str, Any], runtime_volume["manifest"])
    runtime_recipe = cast(dict[str, Any], runtime_manifest["recipe"])
    runtime_lock = cast(dict[str, Any], runtime_recipe["resolved_lock"])
    runtime_python = cast(dict[str, Any], runtime_recipe["python"])
    timestamp = (
        time.strftime("%Y%m%d-%H%M%S", time.gmtime())
        + f"-{time.time_ns() % 1_000_000:06d}"
    )
    trial_name = f"{BENCHMARK_PREFIX}{run_kind}-{task_slug(task_dir)}-{timestamp}"
    trial_dir = runs_root / trial_name
    trial_dir.mkdir(parents=True, exist_ok=False)
    project = compose_project_name(f"{trial_name}__env")
    before_source = source_tree_digest(source_root)
    before_online = online_process_snapshot()
    credential_env = _credential_env(args.credential_profile.resolve())
    source_bundle = create_source_bundle(
        source_root,
        trial_dir / "inputs" / "source.bundle",
    )
    network = reserve_compose_network(project)
    runtime_compose_path = (
        trial_dir / "inputs" / "runtime-network-compose.json"
    )
    compose_overlay = runtime_compose_overlay(args.runtime_volume)
    compose_overlay["networks"] = {
        "default": {
            "external": True,
            "name": network["name"],
        }
    }
    atomic_json(
        runtime_compose_path,
        compose_overlay,
    )

    manifest_path = trial_dir / "campaign-manifest.json"
    initial_source: dict[str, object] = {
        "path": str(source_root),
        "git_head": source_bundle["head"],
        "git_status": _git_output(source_root, "status", "--short"),
        "digest_before": before_source,
        "bundle": source_bundle,
    }
    initial_manifest: dict[str, object] = {
        "schema": "akasic.harbor-trial.v1",
        "state": "prepared",
        "harness_version": HARNESS_VERSION,
        "trial_name": trial_name,
        "task": {
            "name": f"terminal-bench/{task_dir.name}",
            "path": str(task_dir),
            "digest": source_tree_digest(task_dir),
        },
        "model": {
            "provider": "deepseek",
            "name": "deepseek-v4-flash",
            "reasoning_effort": "high",
            "max_output_tokens": None,
            "max_output_policy": "provider_default",
        },
        "timeouts": {
            "task_agent_sec": task_agent_timeout_sec,
            "turn_sec": task_agent_timeout_sec,
            "harness_cleanup_reserve_sec": HARNESS_CLEANUP_RESERVE_SEC,
            "harbor_agent_sec": (
                task_agent_timeout_sec + HARNESS_CLEANUP_RESERVE_SEC
            ),
        },
        "source": initial_source,
        "runtime_cache": runtime_volume,
        "harbor": {
            "root": str(args.harbor_root.resolve()),
            "git_head": _git_output(args.harbor_root.resolve(), "rev-parse", "HEAD"),
            "version": "0.16.1",
        },
        "credentials": {
            "source": str(args.credential_profile.resolve()),
            "injected_names": sorted(credential_env),
            "persisted_values": False,
        },
        "online_before": before_online,
        "docker": {
            "project": project,
            "network": network,
        },
    }
    atomic_json(manifest_path, initial_manifest)

    # 2. Harbor 负责启动环境、执行 agent、外部 verifier 和停止但保留容器。
    config = TrialConfig(
        task=TrialTaskConfig(path=task_dir),
        trial_name=trial_name,
        trials_dir=runs_root,
        agent=AgentConfig(
            import_path=AkashicHarborAgent.import_path(),
            model_name="deepseek/deepseek-v4-flash",
            override_setup_timeout_sec=900,
            override_timeout_sec=(
                task_agent_timeout_sec + HARNESS_CLEANUP_RESERVE_SEC
            ),
            env=credential_env,
            kwargs={
                "source_root": str(source_root),
                "source_bundle": source_bundle["path"],
                "source_head": source_bundle["head"],
                "allowed_bind_root": str(trial_dir),
                "forbidden_host_paths": [
                    str(path) for path in DEFAULT_FORBIDDEN_PATHS
                ],
                "source_digest": before_source,
                "runtime_volume_name": args.runtime_volume,
                "runtime_digest": runtime_manifest["runtime_digest"],
                "runtime_manifest_digest": runtime_manifest["manifest_digest"],
                "runtime_lock_digest": runtime_lock["digest"],
                "runtime_python_version": runtime_python["version"],
                "bootstrap_timeout_sec": 900,
                "turn_timeout_sec": task_agent_timeout_sec,
            },
        ),
        environment=EnvironmentConfig(
            type=EnvironmentType.DOCKER,
            delete=True,
            extra_docker_compose=[runtime_compose_path],
            kwargs={"keep_containers": True},
        ),
    )
    trial = await Trial.create(config)
    result = await trial.run()

    # 3. 只有完整 trace、外部 verifier、停止容器和线上 owner 不变才算 trial 完成。
    after_source = source_tree_digest(source_root)
    after_online = online_process_snapshot()
    online_report = validate_online_processes_unchanged(
        before_online,
        after_online,
    )
    containers, inspection_error = _inspect_finished_project(result, project)
    stopped = bool(containers) and all(
        not bool(container.get("running")) for container in containers
    )
    agent_dir = trial_dir / "agent"
    trace_path = agent_dir / "trace.jsonl"
    turn_result_path = agent_dir / "turn-result.json"
    required_artifacts = [
        agent_dir / "isolation.preflight.json",
        trace_path,
        turn_result_path,
        trial_dir / "verifier" / "reward.txt",
        trial_dir / "result.json",
    ]
    missing_artifacts = [
        str(path.relative_to(trial_dir))
        for path in required_artifacts
        if not path.is_file()
    ]
    trial_completed = (
        not missing_artifacts
        and stopped
        and after_source == before_source
        and online_report["status"] == "passed"
        and getattr(result, "exception_info", None) is None
    )
    final_manifest = {
        **initial_manifest,
        "state": "completed" if trial_completed else "failed",
        "source": {
            **initial_source,
            "digest_after": after_source,
            "unchanged": after_source == before_source,
        },
        "online": online_report,
        "docker": {
            "project": project,
            "retained": bool(containers),
            "all_stopped": stopped,
            "containers": containers,
            "inspection_error": inspection_error,
            "network": network,
        },
        "result": _safe_trial_result(result),
        "artifacts": {
            "missing": missing_artifacts,
            "digests": artifact_digests(
                trial_dir,
                exclude={manifest_path},
            ),
        },
        "concurrency_gate": {
            "max_concurrent": 3 if trial_completed else 1,
            "opened": trial_completed,
        },
    }
    atomic_json(manifest_path, final_manifest)
    print(
        json.dumps(
            {
                "state": final_manifest["state"],
                "trial_dir": str(trial_dir),
                "manifest": str(manifest_path),
                "trace": str(trace_path),
                "containers_stopped": stopped,
                "concurrency_gate": final_manifest["concurrency_gate"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return {
        "state": final_manifest["state"],
        "trial_name": trial_name,
        "trial_dir": str(trial_dir),
        "manifest": str(manifest_path),
        "trace": str(trace_path),
        "task": initial_manifest["task"],
        "reward": final_manifest["result"]["rewards"],
        "containers_stopped": stopped,
        "concurrency_gate": final_manifest["concurrency_gate"],
    }


async def run_smoke(args: argparse.Namespace, task_dir: Path) -> int:
    outcome = await run_trial(args, task_dir, run_kind="smoke")
    return 0 if outcome["state"] == "completed" else 1


async def run_campaign(
    args: argparse.Namespace,
    task_dirs: list[Path],
) -> int:
    """按硬上限三并发运行 diagnostic tasks，并冻结 campaign 汇总。"""

    # 1. 只有已完成 smoke 能打开并发，且整个 campaign 再次冻结源码和线上 owner。
    validate_campaign_request(task_dirs, args.max_concurrent)
    runs_root = args.runs_dir.resolve()
    gate = find_open_concurrency_gate(runs_root)
    source_root = args.source_root.resolve()
    before_source = source_tree_digest(source_root)
    before_online = online_process_snapshot()
    campaign_id = (
        f"{BENCHMARK_PREFIX}campaign-"
        + time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    )
    campaign_dir = runs_root / "_campaigns" / campaign_id
    campaign_dir.mkdir(parents=True, exist_ok=False)
    manifest_path = campaign_dir / "manifest.json"
    initial = {
        "schema": "akasic.harbor-campaign.v1",
        "state": "running",
        "campaign_id": campaign_id,
        "max_concurrent": args.max_concurrent,
        "gate": gate,
        "source_digest_before": before_source,
        "tasks": [str(path.resolve()) for path in task_dirs],
        "online_before": before_online,
    }
    atomic_json(manifest_path, initial)

    # 2. semaphore 是唯一并发 owner；每个 task 仍创建独立 Trial/Docker project。
    semaphore = asyncio.Semaphore(args.max_concurrent)

    async def guarded(task_dir: Path) -> dict[str, object]:
        async with semaphore:
            try:
                return await run_trial(args, task_dir, run_kind="diagnostic")
            except Exception as error:
                return {
                    "state": "controller_failed",
                    "task": str(task_dir.resolve()),
                    "error": {
                        "type": type(error).__name__,
                        "message": str(error),
                    },
                }

    outcomes = await asyncio.gather(*(guarded(path) for path in task_dirs))

    # 3. campaign 完成只表示生命周期证据齐全；reward 单独统计，不把失败题伪装通过。
    after_source = source_tree_digest(source_root)
    after_online = online_process_snapshot()
    online_report = validate_online_processes_unchanged(before_online, after_online)
    lifecycle_complete = (
        all(outcome["state"] == "completed" for outcome in outcomes)
        and after_source == before_source
        and online_report["status"] == "passed"
    )
    passed = sum(
        1
        for outcome in outcomes
        if isinstance(outcome.get("reward"), dict)
        and outcome["reward"].get("reward") == 1.0
    )
    final = {
        **initial,
        "state": "completed" if lifecycle_complete else "failed",
        "source_digest_after": after_source,
        "source_unchanged": after_source == before_source,
        "online": online_report,
        "outcomes": outcomes,
        "score": {
            "passed": passed,
            "total": len(outcomes),
            "pass_rate": passed / len(outcomes),
        },
    }
    atomic_json(manifest_path, final)
    print(json.dumps(final, ensure_ascii=False, indent=2))
    return 0 if lifecycle_complete else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--harbor-root", type=Path, required=True)
    parser.add_argument("--task-dir", type=Path, action="append", required=True)
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument("--credential-profile", type=Path, required=True)
    parser.add_argument("--max-concurrent", type=int, default=1)
    parser.add_argument(
        "--uv-binary",
        type=Path,
        default=Path(os.environ.get("AKASIC_BENCH_UV", "/home/huashen/.local/bin/uv")),
    )
    parser.add_argument(
        "--runtime-volume",
        default=os.environ.get("AKASIC_BENCH_RUNTIME_VOLUME"),
    )
    args = parser.parse_args()
    if not args.runtime_volume:
        parser.error(
            "--runtime-volume 或 AKASIC_BENCH_RUNTIME_VOLUME 是必填项；"
            "harness 不会在 trial 内冷安装"
        )
    if len(args.task_dir) == 1:
        return asyncio.run(run_smoke(args, args.task_dir[0]))
    return asyncio.run(run_campaign(args, args.task_dir))


if __name__ == "__main__":
    raise SystemExit(main())

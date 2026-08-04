from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import subprocess
import time
from pathlib import Path
from typing import Any, cast, override

from harbor.models.environment_type import EnvironmentType
from harbor.models.task.config import TaskConfig as HarborTaskConfig
from harbor.models.trial.config import (
    AgentConfig,
    EnvironmentConfig,
    TrialConfig,
)
from harbor.models.trial.config import (
    TaskConfig as TrialTaskConfig,
)
from harbor.models.verifier.result import VerifierResult
from harbor.trial.errors import VerifierTimeoutError
from harbor.trial.hooks import TrialEvent, TrialHookEvent
from harbor.trial.single_step import SingleStepTrial
from harbor.trial.trial import Trial

from benchmark.harbor_v4flash import HARNESS_VERSION
from benchmark.harbor_v4flash.agent import (
    _VERIFIER_PREPARE_TIMEOUT_SEC,
    AkashicHarborAgent,
    _prepare_verifier_runtime,
)
from benchmark.harbor_v4flash.campaign import (
    MAX_CAMPAIGN_CONCURRENCY,
    find_open_concurrency_gate,
    task_slug,
    validate_campaign_request,
)
from benchmark.harbor_v4flash.credentials import credential_scope
from benchmark.harbor_v4flash.git_volume import inspect_git_volume
from benchmark.harbor_v4flash.image_cache import prefetch_task_images
from benchmark.harbor_v4flash.isolation import (
    BENCHMARK_PREFIX,
    IsolationError,
    artifact_digests,
    atomic_json,
    cleanup_compose_project,
    compose_project_name,
    create_source_bundle,
    inspect_compose_project,
    online_process_snapshot,
    require_storage_capacity,
    reserve_compose_network,
    source_tree_digest,
    stop_and_cleanup_compose_project,
    validate_online_processes_unchanged,
)
from benchmark.harbor_v4flash.resource_evidence import (
    RESOURCE_EVIDENCE_FILENAME,
    load_resource_evidence,
)
from benchmark.harbor_v4flash.runtime_driver import (
    _turn_was_account_limited,
    _turn_was_rate_limited,
    _turn_was_transient_provider_failure,
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
_VERIFIER_CONCURRENCY = 1
_VERIFIER_GATE = asyncio.Semaphore(_VERIFIER_CONCURRENCY)
_VERIFIER_PREPARE_CONCURRENCY = 2
_VERIFIER_PREPARE_GATE = asyncio.Semaphore(_VERIFIER_PREPARE_CONCURRENCY)
_PROVIDER_INVALID_STATUSES = {"rate_limited", "provider_transient", "account_limited"}
_CANDIDATE_DIGEST_COMMAND = (
    "set -eu; pwd -P; "
    "tar --sort=name --mtime='UTC 1970-01-01' --owner=0 --group=0 "
    "--numeric-owner --format=gnu -cf - . | sha256sum | awk '{print $1}'"
)


class _SerializedVerifierTrial(SingleStepTrial):
    """Run official verifiers through one process-wide admission lane."""

    @override
    async def _run_verifier(self) -> None:
        if self.config.verifier.disable:
            await super()._run_verifier()
            return
        driver_outcome_path = self.paths.agent_dir / "driver-outcome.json"
        driver_outcome = json.loads(driver_outcome_path.read_text(encoding="utf-8"))
        driver_status = str(driver_outcome.get("status") or "")
        if driver_status in _PROVIDER_INVALID_STATUSES:
            atomic_json(
                self.paths.agent_dir / "verifier-skipped.json",
                {
                    "schema": "akasic.verifier-skipped.v1",
                    "reason": driver_status,
                    "official_verifier_timeout_started": False,
                },
            )
            return
        async with _VERIFIER_PREPARE_GATE:
            expected_uv_version = self.config.agent.kwargs["runtime_uv_version"]
            test_path = self.task.paths.discovered_test_path_for(
                self.agent_environment.os
            )
            if test_path is None:
                raise FileNotFoundError("找不到当前 task 的官方 verifier 脚本")
            evidence = await _prepare_verifier_runtime(
                self.agent_environment,
                expected_uv_version=str(expected_uv_version),
                test_script=test_path.read_text(encoding="utf-8"),
            )
            atomic_json(
                self.paths.agent_dir / "verifier-bootstrap.json",
                evidence,
            )
        async with _VERIFIER_GATE:
            await super()._run_verifier()


async def _create_serialized_trial(config: TrialConfig) -> SingleStepTrial:
    """Load a single-step task and bind it to the serialized verifier trial."""

    # 1. Reuse Harbor's task and skill resolution without changing task inputs.
    Trial._resolve_agent_skills(config)
    task = await Trial._load_task(config)

    # 2. This harness only supports the Terminal-Bench single-step contract.
    if task.has_steps:
        raise ValueError("V4 Flash harness 不支持 multi-step verifier")
    return _SerializedVerifierTrial(config, _task=task)


async def _capture_candidate_digest(
    trial: SingleStepTrial,
    trial_dir: Path,
    _event: TrialHookEvent,
) -> None:
    """Freeze the candidate filesystem identity before verification starts."""

    # 1. Hash the task worktree after the agent has stopped mutating it.
    result = await trial.agent_environment.exec(
        command=_CANDIDATE_DIGEST_COMMAND,
        timeout_sec=300,
        user="root",
    )
    if result.return_code != 0 or not (result.stdout or "").strip():
        raise RuntimeError("无法冻结 verifier replay 的 /app 候选摘要")

    # 2. Persist the pre-verifier identity outside the disposable container.
    lines = (result.stdout or "").strip().splitlines()
    if len(lines) < 2:
        raise RuntimeError("verifier replay 候选摘要输出结构无效")
    root, digest = lines[-2:]
    atomic_json(
        trial_dir / "agent" / "candidate-identity.json",
        {
            "schema": "akasic.verifier-candidate.v1",
            "root": root,
            "digest": f"sha256:{digest}",
        },
    )


def _verifier_timeout(result: Any) -> bool:
    exception = getattr(result, "exception_info", None)
    return (
        exception is not None
        and getattr(exception, "exception_type", None) == VerifierTimeoutError.__name__
        and getattr(result, "verifier_result", None) is None
    )


async def _run_process(
    command: list[str],
    *,
    timeout_sec: float,
) -> tuple[int, str, bool]:
    """Run one replay command and expose timeout as an explicit outcome."""

    process = await asyncio.create_subprocess_exec(
        *command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    try:
        stdout, _ = await asyncio.wait_for(
            process.communicate(),
            timeout=timeout_sec,
        )
    except TimeoutError:
        process.terminate()
        try:
            await asyncio.wait_for(process.wait(), timeout=5)
        except TimeoutError:
            process.kill()
            await process.wait()
        return -1, "", True
    return process.returncode or 0, stdout.decode(errors="replace"), False


async def _docker_command(
    *args: str,
    timeout_sec: float = 60,
) -> str:
    return_code, output, timed_out = await _run_process(
        ["docker", *args],
        timeout_sec=timeout_sec,
    )
    if timed_out:
        raise RuntimeError(f"docker {' '.join(args)} 超时")
    if return_code != 0:
        raise RuntimeError(
            f"docker {' '.join(args)} 失败 ({return_code}): {output.strip()}"
        )
    return output


def _read_replay_reward(verifier_dir: Path) -> float:
    reward_path = verifier_dir / "reward.txt"
    if not reward_path.is_file():
        raise RuntimeError("verifier replay 未生成 reward.txt")
    reward = float(reward_path.read_text(encoding="utf-8").strip())
    if reward not in {0.0, 1.0}:
        raise RuntimeError(f"verifier replay reward 无效：{reward}")
    return reward


async def _replay_timed_out_verifier(
    result: Any,
    *,
    trial_dir: Path,
    containers: list[dict[str, object]],
    verifier_timeout_sec: float,
) -> dict[str, object] | None:
    """Replay one timed-out official verifier against the unchanged candidate."""

    # 1. Only infrastructure timeouts with one retained main container qualify.
    if not _verifier_timeout(result):
        return None
    if len(containers) != 1:
        raise RuntimeError("verifier replay 要求恰好一个冻结的 task 容器")
    candidate_path = trial_dir / "agent" / "candidate-identity.json"
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    container_id = str(containers[0]["id"])
    replay_dir = trial_dir / "verifier-replay"
    replay_dir.mkdir(parents=True, exist_ok=False)
    original_result_path = trial_dir / "result.json"
    (replay_dir / "original-result.json").write_bytes(original_result_path.read_bytes())

    # 2. Restart the exact container and reject any candidate mutation.
    verifier_dir = trial_dir / "verifier"
    for name in ("reward.txt", "reward.json"):
        (verifier_dir / name).unlink(missing_ok=True)
    await _docker_command("start", container_id)
    try:
        current_lines = (
            (
                await _docker_command(
                    "exec",
                    "--user",
                    "root",
                    container_id,
                    "bash",
                    "-lc",
                    _CANDIDATE_DIGEST_COMMAND,
                    timeout_sec=300,
                )
            )
            .strip()
            .splitlines()
        )
        if len(current_lines) < 2:
            raise RuntimeError("verifier replay 候选复核输出结构无效")
        current_root, current_digest = current_lines[-2:]
        if (
            current_root != candidate["root"]
            or f"sha256:{current_digest}" != candidate["digest"]
        ):
            raise RuntimeError("首次 verifier 已改变候选工作目录，禁止重放")

        # 3. Keep the official script and timeout unchanged, but run alone.
        async with _VERIFIER_GATE:
            return_code, output, timed_out = await _run_process(
                [
                    "docker",
                    "exec",
                    container_id,
                    "bash",
                    "-lc",
                    "(/tests/test.sh)",
                ],
                timeout_sec=verifier_timeout_sec,
            )
        (replay_dir / "test-stdout.txt").write_text(output, encoding="utf-8")
        replay: dict[str, object] = {
            "schema": "akasic.verifier-replay.v1",
            "candidate": candidate,
            "container_id": container_id,
            "official_timeout_sec": verifier_timeout_sec,
            "command": "/tests/test.sh",
            "timed_out": timed_out,
            "return_code": return_code,
        }
        if timed_out:
            atomic_json(replay_dir / "evidence.json", replay)
            return replay
        reward = _read_replay_reward(verifier_dir)
        replay["reward"] = reward
        atomic_json(replay_dir / "evidence.json", replay)
        result.verifier_result = VerifierResult(rewards={"reward": reward})
        result.exception_info = None
        original_result_path.write_text(result.model_dump_json(indent=4))
        return replay
    finally:
        await _docker_command("stop", "--time", "30", container_id)


def _append_campaign_event(path: Path, payload: dict[str, object]) -> None:
    """追加并刷盘一个 campaign 事件，进程中断后可据此续跑。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"timestamp": time.time(), **payload}
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _accepted_campaign_outcomes(path: Path) -> dict[str, dict[str, object]]:
    """从 append-only ledger 恢复已完整验收的 task 结果。"""

    accepted: dict[str, dict[str, object]] = {}
    if not path.is_file():
        return accepted
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        try:
            event = json.loads(line)
        except json.JSONDecodeError as error:
            raise RuntimeError(f"campaign ledger 第 {line_number} 行损坏") from error
        if event.get("event") != "accepted":
            continue
        task = event.get("task")
        outcome = event.get("outcome")
        if not isinstance(task, str) or not isinstance(outcome, dict):
            raise RuntimeError(f"campaign ledger 第 {line_number} 行结构无效")
        if task in accepted:
            raise RuntimeError(f"campaign ledger 重复接受 task：{task}")
        accepted[task] = outcome
    return accepted


def _seed_campaign_outcomes(
    seed_campaign_dir: Path,
    task_dirs: list[Path],
) -> tuple[dict[str, dict[str, object]], dict[str, object]]:
    """核验旧 campaign 的有效结果，并排除可重试的 provider 基础设施失败。"""

    # 1. seed 必须覆盖完全相同的 task 集合，避免把部分数据误投影成全量结果。
    seed_dir = seed_campaign_dir.resolve()
    manifest_path = seed_dir / "manifest.json"
    ledger_path = seed_dir / "events.jsonl"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_tasks = [str(path.resolve()) for path in task_dirs]
    if manifest.get("tasks") != expected_tasks:
        raise RuntimeError("seed campaign 与当前 task 集合不一致")
    accepted = _accepted_campaign_outcomes(ledger_path)

    # 2. 旧版可能把 provider 5xx 记成 agent_failed；回看权威 turn 终态后再接纳。
    included: dict[str, dict[str, object]] = {}
    excluded: list[dict[str, str]] = []
    for task in expected_tasks:
        outcome = accepted.get(task)
        if outcome is None:
            continue
        trial_dir_raw = outcome.get("trial_dir")
        if not isinstance(trial_dir_raw, str):
            raise TypeError(f"seed outcome 缺少 trial_dir：{task}")
        trial_dir = Path(trial_dir_raw).resolve()
        turn_result_path = trial_dir / "agent" / "turn-result.json"
        driver_outcome_path = trial_dir / "agent" / "driver-outcome.json"
        turn_result = (
            json.loads(turn_result_path.read_text(encoding="utf-8"))
            if turn_result_path.is_file()
            else None
        )
        driver_outcome = json.loads(driver_outcome_path.read_text(encoding="utf-8"))
        terminal = (
            turn_result.get("terminal") if isinstance(turn_result, dict) else None
        )
        failure_class = None
        if driver_outcome.get("status") == "rate_limited":
            failure_class = "provider_rate_limited"
        elif driver_outcome.get("status") == "provider_transient":
            failure_class = "provider_transient"
        elif driver_outcome.get("status") == "account_limited" or (
            isinstance(terminal, dict) and _turn_was_account_limited(terminal)
        ):
            failure_class = "provider_account_limited"
        elif isinstance(terminal, dict) and _turn_was_rate_limited(terminal):
            failure_class = "provider_rate_limited"
        elif isinstance(terminal, dict) and _turn_was_transient_provider_failure(
            terminal
        ):
            failure_class = "provider_transient"
        if failure_class is not None:
            excluded.append({"task": task, "failure_class": failure_class})
            continue
        included[task] = outcome

    # 3. report 冻结跨源码阶段的来源和每个排除理由，供最终结果审计。
    report: dict[str, object] = {
        "campaign_dir": str(seed_dir),
        "campaign_id": manifest.get("campaign_id"),
        "source_digest": manifest.get("source_digest_before"),
        "accepted_seen": len(accepted),
        "included": len(included),
        "excluded": excluded,
    }
    return included, report


def _write_campaign_results(
    path: Path,
    task_dirs: list[Path],
    accepted: dict[str, dict[str, object]],
) -> None:
    """从 append-only WAL 原子生成当前 accepted dataset 投影。"""

    outcomes = [
        accepted[key]
        for task_dir in task_dirs
        if (key := str(task_dir.resolve())) in accepted
    ]
    passed = sum(
        1
        for outcome in outcomes
        if isinstance(outcome.get("reward"), dict)
        and outcome["reward"].get("reward") == 1.0
    )
    atomic_json(
        path,
        {
            "schema": "akasic.harbor-campaign-results.v1",
            "accepted": len(outcomes),
            "expected": len(task_dirs),
            "score": {
                "passed": passed,
                "total": len(outcomes),
                "pass_rate": passed / len(outcomes) if outcomes else None,
            },
            "outcomes": outcomes,
        },
    )


def _rate_limit_backoff_sec(task_key: str, attempt: int, base_sec: float) -> float:
    """生成带稳定抖动的指数退避，避免三个 slot 同时再次冲击 provider。"""

    if attempt < 1 or base_sec <= 0:
        raise ValueError("rate-limit backoff 参数无效")
    jitter_unit = int.from_bytes(hashlib.sha256(task_key.encode()).digest()[:2]) / 65535
    return base_sec * (2 ** (attempt - 1)) + base_sec * 0.25 * jitter_unit


def _greedy_task_schedule(
    task_dirs: list[Path],
) -> tuple[list[Path], list[dict[str, object]]]:
    """只按官方 agent 时限降序生成 LPT 队列，让长预算任务优先占 slot。"""

    # 1. task.toml 是唯一估算 owner；历史运行结果不参与本次顺序。
    estimates: list[tuple[float, str, Path]] = []
    for task_dir in task_dirs:
        resolved = task_dir.resolve()
        estimates.append((_task_agent_timeout_sec(resolved), resolved.name, resolved))

    # 2. Longest Processing Time first；同估算按 task 名稳定排序。
    estimates.sort(key=lambda item: (-item[0], item[1]))
    schedule = [
        {
            "rank": rank,
            "task": str(task_dir),
            "estimated_duration_sec": estimate,
            "basis": "task_agent_timeout",
        }
        for rank, (estimate, _, task_dir) in enumerate(estimates, 1)
    ]
    return [item[2] for item in estimates], schedule


def _restore_greedy_schedule(
    task_dirs: list[Path],
    schedule: object,
) -> list[Path]:
    """从 campaign manifest 恢复冻结队列，拒绝缺失、重复或额外 task。"""

    if not isinstance(schedule, list):
        raise RuntimeError("resume campaign 缺少冻结 schedule")
    selected = {str(path.resolve()): path.resolve() for path in task_dirs}
    ordered: list[Path] = []
    for item in schedule:
        task = item.get("task") if isinstance(item, dict) else None
        if not isinstance(task, str) or task not in selected:
            raise RuntimeError("resume campaign schedule 含无效 task")
        ordered.append(selected[task])
    if len(ordered) != len(selected) or len(set(ordered)) != len(ordered):
        raise RuntimeError("resume campaign schedule 与 task 集合不一致")
    return ordered


def _task_set_identity(
    task_dirs: list[Path],
    *,
    dataset_dir: Path | None,
    dataset_ref: str | None,
) -> dict[str, object]:
    """冻结 campaign 的有序 task 集合与本地内容身份。"""

    tasks = [
        {
            "name": path.name,
            "path": str(path.resolve()),
            "digest": source_tree_digest(path.resolve()),
        }
        for path in task_dirs
    ]
    encoded = json.dumps(
        tasks,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return {
        "path": None if dataset_dir is None else str(dataset_dir.resolve()),
        "declared_ref": dataset_ref,
        "provenance": "declared" if dataset_ref else "unverified_local_copy",
        "task_count": len(tasks),
        "task_set_digest": f"sha256:{hashlib.sha256(encoded).hexdigest()}",
        "tasks": tasks,
    }


def _git_output(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.strip()


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


def _task_verifier_timeout_sec(task_dir: Path) -> float:
    """Read and validate the official task verifier budget."""

    # 1. Reuse Harbor's task schema as the timeout owner.
    config_path = task_dir / "task.toml"
    config = HarborTaskConfig.model_validate_toml(
        config_path.read_text(encoding="utf-8")
    )

    # 2. Replay must preserve the same explicit finite budget.
    timeout_sec = config.verifier.timeout_sec
    if timeout_sec is None or not math.isfinite(timeout_sec) or timeout_sec <= 0:
        raise ValueError(f"{config_path} 缺少有效的 [verifier].timeout_sec")
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
    task_verifier_timeout_sec = _task_verifier_timeout_sec(task_dir)
    storage_before = require_storage_capacity(
        runs_root,
        min_runs_free_gib=args.min_runs_free_gib,
        min_tmp_free_gib=args.min_tmp_free_gib,
        min_docker_free_gib=args.min_docker_free_gib,
    )
    task_image = args.task_images[str(task_dir)]
    uv_binary = args.uv_binary.resolve()
    runtime_volume = inspect_runtime_volume(
        args.runtime_volume,
        source_root=source_root,
        uv_binary=uv_binary,
    )
    runtime_manifest = cast(dict[str, Any], runtime_volume["manifest"])
    git_volume = inspect_git_volume(args.git_volume)
    git_manifest = cast(dict[str, Any], git_volume["manifest"])
    git_recipe = cast(dict[str, Any], git_manifest["recipe"])
    git_packages = cast(dict[str, Any], git_recipe["packages"])
    runtime_recipe = cast(dict[str, Any], runtime_manifest["recipe"])
    runtime_lock = cast(dict[str, Any], runtime_recipe["resolved_lock"])
    runtime_python = cast(dict[str, Any], runtime_recipe["python"])
    runtime_uv = cast(dict[str, Any], runtime_recipe["uv"])
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
    credential_names = cast(tuple[str, ...], args.credential_names)
    source_bundle = create_source_bundle(
        source_root,
        trial_dir / "inputs" / "source.bundle",
    )
    network = reserve_compose_network(project)
    runtime_compose_path = trial_dir / "inputs" / "runtime-network-compose.json"
    compose_overlay = runtime_compose_overlay(
        args.runtime_volume,
        task_image_id=str(task_image["id"]),
        git_volume_name=args.git_volume,
    )
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
            "reasoning_effort": "max",
            "max_output_tokens": None,
            "max_output_policy": "provider_default",
        },
        "timeouts": {
            "task_agent_sec": task_agent_timeout_sec,
            "task_verifier_sec": task_verifier_timeout_sec,
            "verifier_dependency_prepare_sec": _VERIFIER_PREPARE_TIMEOUT_SEC,
            "agent_execution_sec": task_agent_timeout_sec,
            "turn_max_sec": task_agent_timeout_sec,
            "harness_cleanup_reserve_sec": HARNESS_CLEANUP_RESERVE_SEC,
            "harbor_agent_sec": (task_agent_timeout_sec + HARNESS_CLEANUP_RESERVE_SEC),
        },
        "source": initial_source,
        "runtime_cache": runtime_volume,
        "git_cache": git_volume,
        "harbor": {
            "root": str(args.harbor_root.resolve()),
            "git_head": _git_output(args.harbor_root.resolve(), "rev-parse", "HEAD"),
            "version": "0.16.1",
        },
        "credentials": {
            "source": str(args.credential_profile.resolve()),
            "injected_names": list(credential_names),
            "persisted_values": False,
        },
        "storage_before": storage_before,
        "online_before": before_online,
        "docker": {
            "project": project,
            "network": network,
            "task_image": task_image,
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
            override_timeout_sec=(task_agent_timeout_sec + HARNESS_CLEANUP_RESERVE_SEC),
            kwargs={
                "source_root": str(source_root),
                "source_bundle": source_bundle["path"],
                "source_head": source_bundle["head"],
                "allowed_bind_root": str(trial_dir),
                "forbidden_host_paths": [str(path) for path in DEFAULT_FORBIDDEN_PATHS],
                "source_digest": before_source,
                "runtime_volume_name": args.runtime_volume,
                "runtime_digest": runtime_manifest["runtime_digest"],
                "runtime_manifest_digest": runtime_manifest["manifest_digest"],
                "runtime_lock_digest": runtime_lock["digest"],
                "runtime_python_version": runtime_python["version"],
                "runtime_uv_digest": runtime_uv["digest"],
                "runtime_uv_version": runtime_uv["version"],
                "git_volume_name": args.git_volume,
                "git_runtime_digest": git_manifest["runtime_digest"],
                "git_manifest_digest": git_manifest["manifest_digest"],
                "git_version": git_packages["git_version"],
                "bootstrap_timeout_sec": 900,
                "turn_timeout_sec": task_agent_timeout_sec,
                "credential_names": credential_names,
            },
        ),
        environment=EnvironmentConfig(
            type=EnvironmentType.DOCKER,
            delete=True,
            extra_docker_compose=[runtime_compose_path],
            kwargs={"keep_containers": True},
        ),
    )
    trial = await _create_serialized_trial(config)

    async def capture_candidate(event: TrialHookEvent) -> None:
        await _capture_candidate_digest(trial, trial_dir, event)

    trial.add_hook(TrialEvent.AGENT_END, capture_candidate)
    try:
        result = await trial.run()
    except asyncio.CancelledError as error:
        cleanup = stop_and_cleanup_compose_project(
            project,
            network=network,
        )
        interrupted_manifest = {
            **initial_manifest,
            "state": "interrupted",
            "docker": {
                "project": project,
                "network": network,
                "cleanup": cleanup,
                "retained": False,
            },
            "interruption": {
                "type": type(error).__name__,
                "message": "campaign cancellation",
            },
        }
        atomic_json(manifest_path, interrupted_manifest)
        raise

    # 3. 只有完整 trace、外部 verifier、停止容器和线上 owner 不变才算 trial 完成。
    after_source = source_tree_digest(source_root)
    after_online = online_process_snapshot()
    online_report = validate_online_processes_unchanged(
        before_online,
        after_online,
    )
    containers, inspection_error = _inspect_finished_project(result, project)
    verifier_replay = await _replay_timed_out_verifier(
        result,
        trial_dir=trial_dir,
        containers=containers,
        verifier_timeout_sec=task_verifier_timeout_sec,
    )
    if verifier_replay is not None:
        containers, inspection_error = _inspect_finished_project(result, project)
    stopped = bool(containers) and all(
        not bool(container.get("running")) for container in containers
    )
    agent_dir = trial_dir / "agent"
    trace_path = agent_dir / "trace.jsonl"
    turn_result_path = agent_dir / "turn-result.json"
    driver_outcome_path = agent_dir / "driver-outcome.json"
    resource_evidence_path = agent_dir / RESOURCE_EVIDENCE_FILENAME
    resource_evidence = load_resource_evidence(resource_evidence_path)
    driver_outcome = (
        json.loads(driver_outcome_path.read_text(encoding="utf-8"))
        if driver_outcome_path.is_file()
        else {"status": "missing"}
    )
    provider_failure_classes = {
        "rate_limited": "provider_rate_limited",
        "provider_transient": "provider_transient",
        "account_limited": "provider_account_limited",
    }
    provider_failure_class = provider_failure_classes.get(
        str(driver_outcome.get("status") or "")
    )
    required_artifacts = [
        agent_dir / "isolation.preflight.json",
        agent_dir / "candidate-identity.json",
        trace_path,
        driver_outcome_path,
        resource_evidence_path,
        trial_dir / "result.json",
    ]
    if provider_failure_class is None:
        required_artifacts.extend(
            [
                agent_dir / "verifier-bootstrap.json",
                trial_dir / "verifier" / "reward.txt",
            ]
        )
    else:
        required_artifacts.append(agent_dir / "verifier-skipped.json")
    missing_artifacts = [
        str(path.relative_to(trial_dir))
        for path in required_artifacts
        if not path.is_file()
    ]
    turn_result_required = driver_outcome.get("status") in {
        "completed",
        *_PROVIDER_INVALID_STATUSES,
    }
    if turn_result_required and not turn_result_path.is_file():
        missing_artifacts.append(str(turn_result_path.relative_to(trial_dir)))
    lifecycle_evidence_complete = (
        not missing_artifacts
        and resource_evidence["status"] == "collected"
        and stopped
        and after_source == before_source
        and online_report["status"] == "passed"
        and getattr(result, "exception_info", None) is None
    )
    trial_completed = lifecycle_evidence_complete and provider_failure_class is None
    trial_state = (
        "invalid_infra"
        if lifecycle_evidence_complete and provider_failure_class is not None
        else "completed" if trial_completed else "failed"
    )
    final_manifest = {
        **initial_manifest,
        "state": trial_state,
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
        "agent_execution": driver_outcome,
        "resource_evidence": {
            "artifact": str(resource_evidence_path.relative_to(trial_dir)),
            **resource_evidence,
        },
        "verifier_replay": verifier_replay,
        "artifacts": {
            "missing": missing_artifacts,
            "digests": artifact_digests(
                trial_dir,
                exclude={manifest_path},
            ),
        },
        "concurrency_gate": {
            "max_concurrent": (MAX_CAMPAIGN_CONCURRENCY if trial_completed else 1),
            "opened": trial_completed,
        },
    }
    atomic_json(manifest_path, final_manifest)

    # 4. 按冻结策略删除终态 Docker 现场；冷证据始终保留。
    reward = final_manifest["result"]["rewards"]
    passed = isinstance(reward, dict) and reward.get("reward") == 1.0
    should_cleanup = args.retention == "none" or (
        args.retention == "failures" and passed
    )
    if should_cleanup and containers:
        try:
            cleanup = cleanup_compose_project(
                project,
                expected_containers=containers,
                network=network,
            )
        except Exception as error:
            cleanup = {
                "status": "failed",
                "error": {"type": type(error).__name__, "message": str(error)},
            }
            final_manifest["state"] = "failed_cleanup"
        final_manifest["docker"]["cleanup"] = cleanup
        final_manifest["docker"]["retained"] = cleanup["status"] != "removed"
        atomic_json(manifest_path, final_manifest)
    print(
        json.dumps(
            {
                "state": final_manifest["state"],
                "trial_dir": str(trial_dir),
                "manifest": str(manifest_path),
                "trace": str(trace_path),
                "containers_stopped": stopped,
                "resource_classification": resource_evidence["classification"],
                "concurrency_gate": final_manifest["concurrency_gate"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    outcome = {
        "state": final_manifest["state"],
        "trial_name": trial_name,
        "trial_dir": str(trial_dir),
        "manifest": str(manifest_path),
        "trace": str(trace_path),
        "task": initial_manifest["task"],
        "reward": final_manifest["result"]["rewards"],
        "containers_stopped": stopped,
        "resource_classification": resource_evidence["classification"],
        "concurrency_gate": final_manifest["concurrency_gate"],
        "failure_class": provider_failure_class,
    }
    return outcome


async def run_smoke(args: argparse.Namespace, task_dir: Path) -> int:
    outcome = await run_trial(args, task_dir, run_kind="smoke")
    return 0 if outcome["state"] == "completed" else 1


async def run_campaign(
    args: argparse.Namespace,
    task_dirs: list[Path],
) -> int:
    """按硬上限四并发和 LPT 贪心队列运行 tasks，并冻结 campaign 汇总。"""

    # 1. 只有已完成 smoke 能打开并发，且整个 campaign 再次冻结源码和线上 owner。
    validate_campaign_request(task_dirs, args.max_concurrent)
    runs_root = args.runs_dir.resolve()
    source_root = args.source_root.resolve()
    selected_task_dirs = list(task_dirs)
    before_source = source_tree_digest(source_root)
    gate = find_open_concurrency_gate(
        runs_root,
        expected_source_digest=before_source,
    )
    before_online = online_process_snapshot()
    if args.resume_campaign_dir is None:
        campaign_id = f"{BENCHMARK_PREFIX}campaign-" + time.strftime(
            "%Y%m%d-%H%M%S", time.gmtime()
        )
        campaign_dir = runs_root / "_campaigns" / campaign_id
        campaign_dir.mkdir(parents=True, exist_ok=False)
    else:
        campaign_dir = args.resume_campaign_dir.resolve()
        campaign_id = campaign_dir.name
        if campaign_dir.parent != runs_root / "_campaigns":
            raise ValueError("resume campaign 必须位于当前 runs-dir/_campaigns")
    manifest_path = campaign_dir / "manifest.json"
    ledger_path = campaign_dir / "events.jsonl"
    results_path = campaign_dir / "accepted-results.json"
    if args.resume_campaign_dir is None:
        scheduled_task_dirs, schedule = _greedy_task_schedule(selected_task_dirs)
    else:
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        schedule = previous.get("schedule")
        scheduled_task_dirs = _restore_greedy_schedule(
            selected_task_dirs,
            schedule,
        )
    seeded: dict[str, dict[str, object]] = {}
    seed_report: dict[str, object] | None = None
    if args.seed_campaign_dir is not None:
        seeded, seed_report = _seed_campaign_outcomes(
            args.seed_campaign_dir,
            selected_task_dirs,
        )
    initial = {
        "schema": "akasic.harbor-campaign.v1",
        "state": "running",
        "campaign_id": campaign_id,
        "max_concurrent": args.max_concurrent,
        "retry_policy": {
            "provider_rate_limit_max_attempts": args.rate_limit_max_attempts,
            "base_backoff_sec": args.rate_limit_backoff_sec,
        },
        "gate": gate,
        "source_digest_before": before_source,
        "tasks": [str(path.resolve()) for path in selected_task_dirs],
        "scheduling_policy": "longest_processing_time_first",
        "schedule": schedule,
        "dataset": _task_set_identity(
            selected_task_dirs,
            dataset_dir=args.dataset_dir,
            dataset_ref=args.dataset_ref,
        ),
        "online_before": before_online,
    }
    if seed_report is not None:
        initial["seed"] = seed_report
    if args.resume_campaign_dir is None:
        atomic_json(manifest_path, initial)
        _append_campaign_event(
            ledger_path,
            {"event": "campaign_started", "campaign_id": campaign_id},
        )
        for task_dir in selected_task_dirs:
            task_key = str(task_dir.resolve())
            if task_key not in seeded:
                continue
            _append_campaign_event(
                ledger_path,
                {
                    "event": "accepted",
                    "task": task_key,
                    "attempt": 0,
                    "seeded_from": seed_report["campaign_id"],
                    "outcome": seeded[task_key],
                },
            )
    else:
        if (
            previous.get("source_digest_before") != before_source
            or previous.get("tasks") != initial["tasks"]
            or previous.get("dataset") != initial["dataset"]
            or previous.get("max_concurrent") != args.max_concurrent
            or previous.get("retry_policy") != initial["retry_policy"]
        ):
            raise RuntimeError("resume campaign 的源码、任务或并发协议已变化")
        initial = previous
        initial["state"] = "running"
        atomic_json(manifest_path, initial)
        _append_campaign_event(ledger_path, {"event": "campaign_resumed"})

    # 2. semaphore 是唯一并发 owner；每个 task 仍创建独立 Trial/Docker project。
    semaphore = asyncio.Semaphore(args.max_concurrent)

    accepted = _accepted_campaign_outcomes(ledger_path)
    _write_campaign_results(results_path, selected_task_dirs, accepted)

    async def guarded(task_dir: Path) -> dict[str, object]:
        task_key = str(task_dir.resolve())
        if task_key in accepted:
            return accepted[task_key]
        for attempt in range(1, args.rate_limit_max_attempts + 1):
            async with semaphore:
                _append_campaign_event(
                    ledger_path,
                    {"event": "attempt_started", "task": task_key, "attempt": attempt},
                )
                try:
                    outcome = await run_trial(args, task_dir, run_kind="diagnostic")
                except Exception as error:
                    outcome = {
                        "state": "controller_failed",
                        "task": task_key,
                        "error": {
                            "type": type(error).__name__,
                            "message": str(error),
                        },
                    }
            event = "accepted" if outcome["state"] == "completed" else "attempt_failed"
            _append_campaign_event(
                ledger_path,
                {
                    "event": event,
                    "task": task_key,
                    "attempt": attempt,
                    "outcome": outcome,
                },
            )
            if event == "accepted":
                accepted[task_key] = outcome
                _write_campaign_results(results_path, selected_task_dirs, accepted)
                return outcome
            if (
                outcome.get("failure_class")
                not in {"provider_rate_limited", "provider_transient"}
                or attempt >= args.rate_limit_max_attempts
            ):
                return outcome
            delay = _rate_limit_backoff_sec(
                task_key,
                attempt,
                args.rate_limit_backoff_sec,
            )
            _append_campaign_event(
                ledger_path,
                {
                    "event": "retry_scheduled",
                    "task": task_key,
                    "attempt": attempt + 1,
                    "reason": outcome["failure_class"],
                    "delay_sec": delay,
                },
            )
            await asyncio.sleep(delay)
        raise AssertionError("rate-limit attempt loop 未返回")

    try:
        outcomes = await asyncio.gather(
            *(guarded(path) for path in scheduled_task_dirs)
        )
    except asyncio.CancelledError:
        interrupted = {**initial, "state": "interrupted"}
        atomic_json(manifest_path, interrupted)
        _append_campaign_event(
            ledger_path,
            {"event": "campaign_interrupted", "state": "interrupted"},
        )
        raise

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
    _append_campaign_event(
        ledger_path,
        {"event": "campaign_finished", "state": final["state"]},
    )
    print(json.dumps(final, ensure_ascii=False, indent=2))
    return 0 if lifecycle_complete else 1


async def _run_selected(args: argparse.Namespace) -> int:
    """预拉取冻结 task images，再进入 smoke 或 campaign lifecycle。"""

    args.task_images = await prefetch_task_images(args.task_dir)
    if len(args.task_dir) == 1:
        return await run_smoke(args, args.task_dir[0])
    return await run_campaign(args, args.task_dir)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--harbor-root", type=Path, required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--task-dir", type=Path, action="append")
    selection.add_argument(
        "--dataset-dir",
        type=Path,
        help="自动发现直接子目录中的 task.toml，并按目录名稳定排序",
    )
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument("--credential-profile", type=Path, required=True)
    parser.add_argument("--max-concurrent", type=int, default=4)
    parser.add_argument("--rate-limit-max-attempts", type=int, default=3)
    parser.add_argument("--rate-limit-backoff-sec", type=float, default=30.0)
    parser.add_argument(
        "--retention",
        choices=("all", "failures", "none"),
        default="none",
        help="none 删除全部终态容器；failures 仅保留未通过题；all 保留全部",
    )
    parser.add_argument("--min-runs-free-gib", type=float, default=20.0)
    parser.add_argument("--min-tmp-free-gib", type=float, default=2.0)
    parser.add_argument("--min-docker-free-gib", type=float, default=20.0)
    parser.add_argument("--resume-campaign-dir", type=Path)
    parser.add_argument(
        "--seed-campaign-dir",
        type=Path,
        help="新 campaign 从旧 campaign 接纳已核验结果，并重跑 provider 基础设施失败",
    )
    parser.add_argument(
        "--dataset-ref",
        help="记录外部已核对的 dataset revision；不提供时 manifest 明示本地来源未验证",
    )
    parser.add_argument(
        "--uv-binary",
        type=Path,
        default=Path(os.environ.get("AKASIC_BENCH_UV", "/home/huashen/.local/bin/uv")),
    )
    parser.add_argument(
        "--runtime-volume",
        default=os.environ.get("AKASIC_BENCH_RUNTIME_VOLUME"),
    )
    parser.add_argument(
        "--git-volume",
        default=os.environ.get("AKASIC_BENCH_GIT_VOLUME"),
    )
    args = parser.parse_args()
    if args.resume_campaign_dir is not None and args.seed_campaign_dir is not None:
        parser.error("--resume-campaign-dir 与 --seed-campaign-dir 不能同时使用")
    if args.rate_limit_max_attempts < 1:
        parser.error("--rate-limit-max-attempts 必须大于等于 1")
    if args.rate_limit_backoff_sec <= 0:
        parser.error("--rate-limit-backoff-sec 必须大于 0")
    if args.dataset_dir is not None:
        dataset_dir = args.dataset_dir.resolve()
        args.task_dir = sorted(
            (path.parent for path in dataset_dir.glob("*/task.toml")),
            key=lambda path: path.name,
        )
        if not args.task_dir:
            parser.error(f"dataset-dir 没有发现 task.toml：{dataset_dir}")
    if not args.runtime_volume:
        parser.error(
            "--runtime-volume 或 AKASIC_BENCH_RUNTIME_VOLUME 是必填项；"
            "harness 不会在 trial 内冷安装"
        )
    if not args.git_volume:
        parser.error(
            "--git-volume 或 AKASIC_BENCH_GIT_VOLUME 是必填项；"
            "harness 不会在 trial 内安装 Git"
        )
    with credential_scope(args.credential_profile.resolve()) as credential_names:
        args.credential_names = credential_names
        return asyncio.run(_run_selected(args))


if __name__ == "__main__":
    raise SystemExit(main())

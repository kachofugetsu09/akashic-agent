from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from benchmark.harbor_v4flash.campaign import MAX_CAMPAIGN_CONCURRENCY
from benchmark.harbor_v4flash.isolation import (
    BENCHMARK_PREFIX,
    atomic_json,
    stop_and_cleanup_compose_project,
)


def _terminal_task_names(ledger_path: Path) -> set[str]:
    terminal: set[str] = set()
    for line in ledger_path.read_text(encoding="utf-8").splitlines():
        event = json.loads(line)
        if event.get("event") not in {"accepted", "attempt_failed"}:
            continue
        task = event.get("task")
        if isinstance(task, str):
            terminal.add(Path(task).name)
    return terminal


def _wait_for_cohort(
    ledger_path: Path,
    cohort: set[str],
    *,
    poll_interval_sec: float,
) -> None:
    """等待冻结 cohort 全部形成 campaign 终态事件。"""

    while not cohort.issubset(_terminal_task_names(ledger_path)):
        time.sleep(poll_interval_sec)


def _unit_active(unit: str) -> bool:
    result = subprocess.run(
        ["systemctl", "--user", "is-active", "--quiet", unit],
        check=False,
    )
    return result.returncode == 0


def _signal_old_controller(unit: str) -> None:
    """若旧 controller 仍运行，以 SIGINT 停止调度并触发任务取消。"""

    if not _unit_active(unit):
        return

    subprocess.run(
        [
            "systemctl",
            "--user",
            "kill",
            "--kill-who=main",
            "--signal=SIGINT",
            unit,
        ],
        check=True,
    )


def _wait_for_old_controller(unit: str, *, timeout_sec: float) -> None:
    """等待旧 controller 在资源回收后退出，超时则明确失败。"""

    deadline = time.monotonic() + timeout_sec
    while _unit_active(unit):
        if time.monotonic() >= deadline:
            raise TimeoutError(f"旧 controller 未在 {timeout_sec:g}s 内停止")
        time.sleep(1)


def _old_source_projects(runs_dir: Path, source_digest: str) -> list[str]:
    """只枚举仍运行且 manifest 属于旧源码阶段的 benchmark projects。"""

    result = subprocess.run(
        [
            "docker",
            "ps",
            "--format",
            "{{.Label \"com.docker.compose.project\"}}",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    projects: list[str] = []
    for project in sorted(set(result.stdout.splitlines())):
        if not project.startswith(BENCHMARK_PREFIX) or not project.endswith("__env"):
            continue
        trial_dir = runs_dir / project.removesuffix("__env")
        manifest_path = trial_dir / "campaign-manifest.json"
        if not manifest_path.is_file():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("source", {}).get("digest_before") == source_digest:
            projects.append(project)
    return projects


def _cleanup_old_projects(runs_dir: Path, source_digest: str) -> None:
    """按 compose project 与 managed network 双重身份回收旧阶段抢跑资源。"""

    for project in _old_source_projects(runs_dir, source_digest):
        network_result = subprocess.run(
            [
            "docker",
            "network",
            "ls",
            "--no-trunc",
            "--filter",
                f"label=com.docker.compose.project={project}",
                "--format",
                "{{.ID}}\t{{.Name}}",
            ],
            check=True,
            text=True,
            capture_output=True,
        )
        networks = [line.split("\t", 1) for line in network_result.stdout.splitlines()]
        if len(networks) != 1 or len(networks[0]) != 2:
            raise RuntimeError(f"旧 project 的 network 身份不唯一：{project}")
        network_id, network_name = networks[0]
        stop_and_cleanup_compose_project(
            project,
            network={"id": network_id, "name": network_name},
        )


def _controller_command(args: argparse.Namespace, *selection: str) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "benchmark.harbor_v4flash.controller",
        "--source-root",
        str(args.source_root),
        "--harbor-root",
        str(args.harbor_root),
        *selection,
        "--runs-dir",
        str(args.runs_dir),
        "--credential-profile",
        str(args.credential_profile),
        "--max-concurrent",
        str(args.max_concurrent),
        "--rate-limit-max-attempts",
        "3",
        "--rate-limit-backoff-sec",
        "30",
        "--retention",
        "none",
        "--min-runs-free-gib",
        "20",
        "--min-tmp-free-gib",
        "2",
        "--min-docker-free-gib",
        "20",
        "--runtime-volume",
        args.runtime_volume,
        "--git-volume",
        args.git_volume,
    ]
    return command


def _controller_environment(source_root: Path) -> dict[str, str]:
    """为 controller 固定源码与 SDK import 路径，同时保留既有环境。"""

    environment = os.environ.copy()
    required = [str(source_root), str(source_root / "sdk" / "python" / "src")]
    inherited = environment.get("PYTHONPATH", "").split(os.pathsep)
    environment["PYTHONPATH"] = os.pathsep.join(
        dict.fromkeys([*required, *(item for item in inherited if item)])
    )
    return environment


def _open_fixed_gate(args: argparse.Namespace) -> None:
    """用修复后的源码完成 smoke；临时 provider 抖动最多重试三次。"""

    command = _controller_command(args, "--task-dir", str(args.smoke_task_dir))
    for attempt in range(1, 4):
        result = subprocess.run(
            command,
            check=False,
            env=_controller_environment(args.source_root),
        )
        if result.returncode == 0:
            return
        if attempt < 3:
            time.sleep(30 * attempt)
    raise RuntimeError("修正版 smoke 连续三次未打开并发 Gate")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-unit", required=True)
    parser.add_argument("--old-campaign-dir", type=Path, required=True)
    parser.add_argument("--cohort-task", action="append", required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--harbor-root", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument("--credential-profile", type=Path, required=True)
    parser.add_argument("--smoke-task-dir", type=Path, required=True)
    parser.add_argument("--runtime-volume", required=True)
    parser.add_argument("--git-volume", required=True)
    parser.add_argument("--max-concurrent", type=int, default=4)
    parser.add_argument("--poll-interval-sec", type=float, default=15.0)
    args = parser.parse_args()
    if args.max_concurrent != MAX_CAMPAIGN_CONCURRENCY:
        parser.error(f"transition campaign 并发必须为 {MAX_CAMPAIGN_CONCURRENCY}")

    # 1. 等用户指定的当前 cohort 全部完成，不提前牺牲正在执行的题。
    old_campaign_dir = args.old_campaign_dir.resolve()
    state_path = old_campaign_dir / "harness-repair-transition.json"
    old_manifest = json.loads(
        (old_campaign_dir / "manifest.json").read_text(encoding="utf-8")
    )
    atomic_json(
        state_path,
        {"state": "waiting_for_cohort", "cohort": sorted(args.cohort_task)},
    )
    _wait_for_cohort(
        old_campaign_dir / "events.jsonl",
        set(args.cohort_task),
        poll_interval_sec=args.poll_interval_sec,
    )

    # 2. 中断旧调度并精确回收它在 cohort 之后抢占的资源。
    atomic_json(state_path, {"state": "stopping_old_controller"})
    _signal_old_controller(args.old_unit)
    atomic_json(state_path, {"state": "cleaning_old_projects"})
    _cleanup_old_projects(
        args.runs_dir.resolve(),
        str(old_manifest["source_digest_before"]),
    )
    atomic_json(state_path, {"state": "waiting_for_old_controller"})
    _wait_for_old_controller(args.old_unit, timeout_sec=180)

    # 3. 修正版先重新打开 Gate，再 exec 为全量 seeded continuation owner。
    atomic_json(state_path, {"state": "opening_fixed_gate"})
    _open_fixed_gate(args)
    command = _controller_command(args, "--dataset-dir", str(args.dataset_dir))
    command.extend(["--seed-campaign-dir", str(old_campaign_dir)])
    atomic_json(
        state_path,
        {"state": "starting_seeded_continuation", "command": command[1:]},
    )
    os.execve(
        sys.executable,
        command,
        _controller_environment(args.source_root),
    )


if __name__ == "__main__":
    raise SystemExit(main())

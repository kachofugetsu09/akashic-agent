import asyncio
import json
import os
import shutil
import subprocess
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest
from harbor.environments.base import ExecResult

from benchmark.harbor_v4flash.agent import (
    _ENDPOINT,
    _WORKSPACE,
    _build_driver_command,
    _build_gateway_command,
    _prepare_verifier_runtime,
    _run_driver_and_shutdown,
    _start_gateway_with_resource_evidence,
    _verifier_dependency_command,
)
from benchmark.harbor_v4flash.controller import (
    _accepted_campaign_outcomes,
    _append_campaign_event,
    _capture_candidate_digest,
    _greedy_task_schedule,
    _inspect_finished_project,
    _rate_limit_backoff_sec,
    _replay_timed_out_verifier,
    _restore_greedy_schedule,
    _seed_campaign_outcomes,
    _task_agent_timeout_sec,
    _task_set_identity,
    _task_verifier_timeout_sec,
    _verifier_timeout,
    _write_campaign_results,
)
from benchmark.harbor_v4flash.credentials import credential_scope
from benchmark.harbor_v4flash.isolation import IsolationError, create_source_bundle
from benchmark.harbor_v4flash.resource_evidence import (
    RESOURCE_EVIDENCE_FILENAME,
    resource_probe_command,
)
from benchmark.harbor_v4flash.runtime_driver import _write_driver_outcome


class _ScriptedEnvironment:
    def __init__(self, *outcomes: ExecResult | BaseException) -> None:
        self._outcomes = list(outcomes)
        self.commands: list[str] = []

    async def exec(
        self,
        *,
        command: str,
        timeout_sec: float,
        user: str | int | None = None,
    ) -> ExecResult:
        self.commands.append(command)
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class _ReplayResult:
    def __init__(self) -> None:
        self.exception_info = SimpleNamespace(
            exception_type="VerifierTimeoutError",
            exception_message="Verifier execution timed out after 900.0 seconds",
        )
        self.verifier_result = None

    def model_dump_json(self, *, indent: int) -> str:
        return json.dumps(
            {
                "exception": (
                    None
                    if self.exception_info is None
                    else self.exception_info.exception_type
                ),
                "reward": (
                    None
                    if self.verifier_result is None
                    else self.verifier_result.rewards
                ),
            },
            indent=indent,
        )


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.strip()


def _resource_result(*, oom_kill: int = 0) -> ExecResult:
    return ExecResult(
        return_code=0,
        stdout=(
            "cgroup_version=2\n"
            "@@memory.max\n4294967296\n"
            "@@memory.current\n268435456\n"
            "@@memory.events\n"
            f"low 0\nhigh 0\nmax 1\noom {oom_kill}\n"
            f"oom_kill {oom_kill}\noom_group_kill 0\n"
            "@@memory.peak\n4294967296\n"
        ),
    )


def test_v4flash_uses_deepseek_max_and_provider_output_limit() -> None:
    config_path = (
        Path(__file__).parents[2] / "benchmark" / "harbor_v4flash" / "config.toml"
    )
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))

    runtime = config["llm"]["runtimes"]["main"]
    assert runtime["provider"] == "deepseek"
    assert runtime["base_url"] == "https://api.deepseek.com/v1"
    assert runtime["api_key"] == "${DEEPSEEK_API_KEY}"
    assert config["llm"]["runtimes"]["main"]["reasoning_effort"] == "max"
    assert config["llm"]["runtimes"]["main"]["max_output_tokens"] == 0
    assert config["agent"]["max_tokens"] == 0
    assert config["agent"]["max_iterations"] == 0


def test_benchmark_commands_inherit_terminal_task_workdir() -> None:
    gateway_command = _build_gateway_command()
    driver_command = _build_driver_command(900)

    assert _WORKSPACE == "/opt/akashic-workspace"
    assert _ENDPOINT == "/opt/akashic-workspace/akashic.sock"
    assert "cd /app" not in gateway_command
    assert "cd /app" not in driver_command
    assert gateway_command.startswith("mkdir -p /opt/akashic-workspace && env ")
    assert "main.py veda-reset" in gateway_command
    assert driver_command.startswith("PYTHONPATH=/opt/akashic/src:")
    assert "--outcome /logs/agent/driver-outcome.json" in driver_command


def test_campaign_ledger_recovers_only_accepted_outcomes(tmp_path: Path) -> None:
    ledger = tmp_path / "events.jsonl"
    accepted = {"state": "completed", "reward": {"reward": 1.0}}
    _append_campaign_event(
        ledger,
        {"event": "attempt_failed", "task": "/tasks/one", "outcome": {}},
    )
    _append_campaign_event(
        ledger,
        {"event": "accepted", "task": "/tasks/two", "outcome": accepted},
    )

    assert _accepted_campaign_outcomes(ledger) == {"/tasks/two": accepted}


def test_campaign_results_are_derived_from_accepted_wal(tmp_path: Path) -> None:
    tasks = [tmp_path / "one", tmp_path / "two"]
    for task in tasks:
        task.mkdir()
    accepted = {
        str(tasks[1].resolve()): {
            "state": "completed",
            "reward": {"reward": 1.0},
        }
    }

    path = tmp_path / "accepted-results.json"
    _write_campaign_results(path, tasks, accepted)

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["accepted"] == 1
    assert payload["expected"] == 2
    assert payload["score"] == {"passed": 1, "total": 1, "pass_rate": 1.0}
    assert payload["outcomes"] == [accepted[str(tasks[1].resolve())]]


def test_seed_campaign_requeues_provider_error_500_and_keeps_valid_zero(
    tmp_path: Path,
) -> None:
    tasks = [tmp_path / "caffe", tmp_path / "valid-zero"]
    for task in tasks:
        task.mkdir()
    seed = tmp_path / "seed"
    seed.mkdir()
    (seed / "manifest.json").write_text(
        json.dumps(
            {
                "campaign_id": "old-campaign",
                "source_digest_before": "sha256:old",
                "tasks": [str(task.resolve()) for task in tasks],
            }
        )
    )
    for task in tasks:
        trial = tmp_path / f"trial-{task.name}"
        agent = trial / "agent"
        agent.mkdir(parents=True)
        terminal = {
            "status": "failed" if task.name == "caffe" else "completed",
            "error": (
                {
                    "type": "provider_error",
                    "message": "Error code: 500 - Router.Unavailable",
                    "retryable": True,
                }
                if task.name == "caffe"
                else None
            ),
        }
        (agent / "turn-result.json").write_text(json.dumps({"terminal": terminal}))
        (agent / "driver-outcome.json").write_text(
            json.dumps(
                {"status": "agent_failed" if task.name == "caffe" else "completed"}
            )
        )
        _append_campaign_event(
            seed / "events.jsonl",
            {
                "event": "accepted",
                "task": str(task.resolve()),
                "outcome": {
                    "state": "completed",
                    "trial_dir": str(trial),
                    "reward": {"reward": 0.0},
                },
            },
        )

    included, report = _seed_campaign_outcomes(seed, tasks)

    assert set(included) == {str(tasks[1].resolve())}
    assert report["included"] == 1
    assert report["excluded"] == [
        {
            "task": str(tasks[0].resolve()),
            "failure_class": "provider_transient",
        }
    ]


def test_rate_limit_backoff_is_exponential_with_stable_jitter() -> None:
    first = _rate_limit_backoff_sec("/tasks/one", 1, 30)
    second = _rate_limit_backoff_sec("/tasks/one", 2, 30)

    assert 30 <= first <= 37.5
    assert 60 <= second <= 67.5
    assert first == _rate_limit_backoff_sec("/tasks/one", 1, 30)


def test_greedy_schedule_runs_long_official_budgets_first(tmp_path: Path) -> None:
    tasks = [tmp_path / "short", tmp_path / "long", tmp_path / "medium"]
    for task, timeout in zip(tasks, (600, 3600, 1200), strict=True):
        task.mkdir()
        (task / "task.toml").write_text(f"[agent]\ntimeout_sec={timeout}\n")

    ordered, schedule = _greedy_task_schedule(tasks)

    assert [path.name for path in ordered] == ["long", "medium", "short"]
    assert [item["estimated_duration_sec"] for item in schedule] == [3600, 1200, 600]
    assert {item["basis"] for item in schedule} == {"task_agent_timeout"}
    assert _restore_greedy_schedule(tasks, schedule) == ordered


def test_greedy_schedule_ignores_historical_artifacts(tmp_path: Path) -> None:
    task = tmp_path / "official"
    task.mkdir()
    (task / "task.toml").write_text("[agent]\ntimeout_sec=900\n")
    (tmp_path / "campaign-manifest.json").write_text(
        json.dumps({"historical_duration_sec": 99999})
    )

    _, schedule = _greedy_task_schedule([task])

    assert schedule[0]["estimated_duration_sec"] == 900
    assert schedule[0]["basis"] == "task_agent_timeout"


def test_task_set_identity_freezes_order_and_marks_local_provenance(
    tmp_path: Path,
) -> None:
    tasks = [tmp_path / "one", tmp_path / "two"]
    for task in tasks:
        task.mkdir()
        (task / "task.toml").write_text("[agent]\ntimeout_sec=900\n")

    identity = _task_set_identity(
        tasks,
        dataset_dir=tmp_path,
        dataset_ref=None,
    )

    assert identity["task_count"] == 2
    assert identity["provenance"] == "unverified_local_copy"
    assert str(identity["task_set_digest"]).startswith("sha256:")


def test_driver_timeout_outcome_is_machine_readable(tmp_path: Path) -> None:
    outcome = tmp_path / "driver-outcome.json"
    _write_driver_outcome(
        outcome,
        status="timed_out",
        error=TimeoutError("official budget exhausted"),
    )

    payload = json.loads(outcome.read_text(encoding="utf-8"))
    assert payload["status"] == "timed_out"
    assert payload["error"]["type"] == "TimeoutError"


def test_driver_success_keeps_command_order_and_logs(tmp_path: Path) -> None:
    environment = _ScriptedEnvironment(
        ExecResult(return_code=0, stdout="driver complete\n"),
        _resource_result(),
        ExecResult(return_code=0, stdout="shutdown complete\n"),
    )
    result = asyncio.run(
        _run_driver_and_shutdown(
            environment,  # type: ignore[arg-type]
            driver_command="driver",
            driver_timeout_sec=5,
            shutdown_command="shutdown",
            logs_dir=tmp_path,
        )
    )

    assert result.stdout == "driver complete\n"
    assert environment.commands == ["driver", resource_probe_command(), "shutdown"]
    assert (tmp_path / "driver.stdout.log").read_text(
        encoding="utf-8"
    ) == "driver complete\n"
    assert (tmp_path / "runtime.shutdown.log").read_text(
        encoding="utf-8"
    ) == "shutdown complete\n"
    assert (
        json.loads((tmp_path / RESOURCE_EVIDENCE_FILENAME).read_text(encoding="utf-8"))[
            "classification"
        ]
        == "none"
    )
    assert not (tmp_path / "driver.exception.log").exists()


def test_verifier_uv_is_prepared_after_agent_with_frozen_version() -> None:
    test_script = """#!/bin/bash
apt-get update
uvx \\
  -p 3.13 \\
  -w torch==2.7.0 \\
  pytest /tests/test_outputs.py
"""
    digest = "/app\n" + "a" * 64 + "\n"
    environment = _ScriptedEnvironment(
        ExecResult(return_code=0),
        ExecResult(return_code=0, stdout=digest),
        ExecResult(return_code=0, stdout="Resolved 5 packages\n"),
        ExecResult(return_code=0, stdout=digest),
    )

    evidence = asyncio.run(
        _prepare_verifier_runtime(
            environment,  # type: ignore[arg-type]
            expected_uv_version="uv 0.9.5",
            test_script=test_script,
        )
    )

    assert len(environment.commands) == 4
    command = environment.commands[0]
    assert "/opt/akashic-runtime/uv" in command
    assert "/root/.local/bin/uvx" in command
    assert "uv tool run" in command
    assert "uv 0.9.5" in command
    assert "torch==2.7.0" in environment.commands[2]
    assert "pytest /tests/test_outputs.py" not in environment.commands[2]
    assert "python -c 'pass'" in environment.commands[2]
    assert evidence["official_verifier_timeout_started"] is False
    assert evidence["candidate_digest_before"] == evidence["candidate_digest_after"]


def test_verifier_dependency_command_keeps_pip_setup_outside_pytest() -> None:
    command = _verifier_dependency_command(
        "#!/bin/bash\npip install pytest==8.4.1\npython -m pytest /tests/test.py\n"
    )

    assert command is not None
    assert "pip install pytest==8.4.1" in command
    assert "python -m pytest" not in command


def test_driver_timeout_still_shuts_down_gateway_and_persists_evidence(
    tmp_path: Path,
) -> None:
    environment = _ScriptedEnvironment(
        TimeoutError("driver exceeded deadline"),
        _resource_result(oom_kill=1),
        ExecResult(return_code=0, stdout="gateway stopped\n"),
    )

    with pytest.raises(TimeoutError, match="driver exceeded deadline"):
        asyncio.run(
            _run_driver_and_shutdown(
                environment,  # type: ignore[arg-type]
                driver_command="driver",
                driver_timeout_sec=5,
                shutdown_command="shutdown",
                logs_dir=tmp_path,
            )
        )

    assert environment.commands == ["driver", resource_probe_command(), "shutdown"]
    assert (tmp_path / "driver.stdout.log").read_text(encoding="utf-8") == ""
    assert (tmp_path / "driver.stderr.log").read_text(encoding="utf-8") == ""
    assert (tmp_path / "driver.exception.log").read_text(
        encoding="utf-8"
    ) == "TimeoutError: driver exceeded deadline\n"
    assert (tmp_path / "runtime.shutdown.log").read_text(
        encoding="utf-8"
    ) == "gateway stopped\n"
    assert (
        json.loads((tmp_path / RESOURCE_EVIDENCE_FILENAME).read_text(encoding="utf-8"))[
            "classification"
        ]
        == "resource_limit"
    )


def test_driver_cancellation_still_shuts_down_gateway(
    tmp_path: Path,
) -> None:
    environment = _ScriptedEnvironment(
        asyncio.CancelledError("trial cancelled"),
        _resource_result(),
        ExecResult(return_code=0),
    )

    with pytest.raises(asyncio.CancelledError, match="trial cancelled"):
        asyncio.run(
            _run_driver_and_shutdown(
                environment,  # type: ignore[arg-type]
                driver_command="driver",
                driver_timeout_sec=5,
                shutdown_command="shutdown",
                logs_dir=tmp_path,
            )
        )

    assert environment.commands == ["driver", resource_probe_command(), "shutdown"]
    assert (tmp_path / "driver.exception.log").read_text(
        encoding="utf-8"
    ) == "CancelledError: trial cancelled\n"
    assert (tmp_path / "runtime.shutdown.log").read_text(encoding="utf-8") == ""


def test_driver_failure_remains_primary_when_shutdown_also_fails(
    tmp_path: Path,
) -> None:
    environment = _ScriptedEnvironment(
        ExecResult(return_code=1, stdout="driver failed\n"),
        RuntimeError("cgroup probe failed"),
        ExecResult(return_code=2),
    )

    with pytest.raises(RuntimeError, match="执行 SDK turn") as caught:
        asyncio.run(
            _run_driver_and_shutdown(
                environment,  # type: ignore[arg-type]
                driver_command="driver",
                driver_timeout_sec=5,
                shutdown_command="shutdown",
                logs_dir=tmp_path,
            )
        )

    assert any("gateway cleanup also failed" in note for note in caught.value.__notes__)
    assert any(
        "resource evidence collection also failed" in note
        for note in caught.value.__notes__
    )
    assert (
        (tmp_path / "driver.exception.log")
        .read_text(encoding="utf-8")
        .startswith("RuntimeError: 执行 SDK turn")
    )
    assert (tmp_path / "runtime.shutdown.log").read_text(encoding="utf-8") == "exit=2\n"
    resource = json.loads(
        (tmp_path / RESOURCE_EVIDENCE_FILENAME).read_text(encoding="utf-8")
    )
    assert resource["status"] == "collection_failed"
    assert resource["classification"] == "unknown"


def test_resource_probe_failure_fails_loud_after_successful_driver(
    tmp_path: Path,
) -> None:
    environment = _ScriptedEnvironment(
        ExecResult(return_code=0, stdout="driver complete\n"),
        ExecResult(return_code=23, stderr="required cgroup file missing\n"),
        ExecResult(return_code=0, stdout="gateway stopped\n"),
    )

    with pytest.raises(RuntimeError, match="采集容器资源证据"):
        asyncio.run(
            _run_driver_and_shutdown(
                environment,  # type: ignore[arg-type]
                driver_command="driver",
                driver_timeout_sec=5,
                shutdown_command="shutdown",
                logs_dir=tmp_path,
            )
        )

    assert environment.commands == ["driver", resource_probe_command(), "shutdown"]
    resource = json.loads(
        (tmp_path / RESOURCE_EVIDENCE_FILENAME).read_text(encoding="utf-8")
    )
    assert resource["status"] == "collection_failed"
    assert resource["classification"] == "unknown"
    assert (tmp_path / "runtime.shutdown.log").read_text(
        encoding="utf-8"
    ) == "gateway stopped\n"


def test_secure_gateway_failure_keeps_primary_error_and_resource_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_error = RuntimeError("secure docker exec failed")
    environment = _ScriptedEnvironment(_resource_result(oom_kill=1))

    async def fail_secure_exec(*args: object, **kwargs: object) -> ExecResult:
        raise startup_error

    monkeypatch.setattr(
        "benchmark.harbor_v4flash.agent.secure_docker_exec",
        fail_secure_exec,
    )

    with pytest.raises(RuntimeError, match="secure docker exec failed") as caught:
        asyncio.run(
            _start_gateway_with_resource_evidence(
                environment,  # type: ignore[arg-type]
                gateway_command="start-gateway",
                credential_names=("DEEPSEEK_API_KEY", "DASHSCOPE_API_KEY"),
                logs_dir=tmp_path,
            )
        )

    assert caught.value is startup_error
    assert environment.commands == [resource_probe_command()]
    resource = json.loads(
        (tmp_path / RESOURCE_EVIDENCE_FILENAME).read_text(encoding="utf-8")
    )
    assert resource["status"] == "collected"
    assert resource["classification"] == "resource_limit"


def test_task_agent_timeout_uses_harbor_task_budget(tmp_path: Path) -> None:
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    (task_dir / "task.toml").write_text(
        "[agent]\ntimeout_sec = 3600.0\n",
        encoding="utf-8",
    )

    assert _task_agent_timeout_sec(task_dir) == 3600.0


def test_task_verifier_timeout_uses_official_task_budget(tmp_path: Path) -> None:
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    (task_dir / "task.toml").write_text(
        "[verifier]\ntimeout_sec = 900.0\n",
        encoding="utf-8",
    )

    assert _task_verifier_timeout_sec(task_dir) == 900.0


def test_candidate_digest_is_persisted_before_verifier(tmp_path: Path) -> None:
    environment = _ScriptedEnvironment(
        ExecResult(return_code=0, stdout="/workspace\nabc123\n")
    )
    trial = SimpleNamespace(agent_environment=environment)

    asyncio.run(
        _capture_candidate_digest(trial, tmp_path, SimpleNamespace())  # type: ignore[arg-type]
    )

    identity = json.loads(
        (tmp_path / "agent" / "candidate-identity.json").read_text(encoding="utf-8")
    )
    assert identity == {
        "schema": "akasic.verifier-candidate.v1",
        "root": "/workspace",
        "digest": "sha256:abc123",
    }


def test_verifier_timeout_replays_same_candidate_without_model_sampling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _ReplayResult()
    (tmp_path / "agent").mkdir()
    (tmp_path / "verifier").mkdir()
    (tmp_path / "result.json").write_text("original\n", encoding="utf-8")
    (tmp_path / "agent" / "candidate-identity.json").write_text(
        json.dumps(
            {
                "schema": "akasic.verifier-candidate.v1",
                "root": "/app",
                "digest": "sha256:abc123",
            }
        ),
        encoding="utf-8",
    )
    docker_calls: list[tuple[str, ...]] = []

    async def docker_command(*args: str, timeout_sec: float = 60) -> str:
        docker_calls.append(args)
        if args[0] == "exec":
            return "/app\nabc123\n"
        return ""

    async def run_process(
        command: list[str], *, timeout_sec: float
    ) -> tuple[int, str, bool]:
        assert command[-1] == "(/tests/test.sh)"
        assert timeout_sec == 900.0
        (tmp_path / "verifier" / "reward.txt").write_text("1\n", encoding="utf-8")
        return 0, "6 passed\n", False

    monkeypatch.setattr(
        "benchmark.harbor_v4flash.controller._docker_command",
        docker_command,
    )
    monkeypatch.setattr(
        "benchmark.harbor_v4flash.controller._run_process",
        run_process,
    )

    replay = asyncio.run(
        _replay_timed_out_verifier(
            result,
            trial_dir=tmp_path,
            containers=[{"id": "container-id"}],
            verifier_timeout_sec=900.0,
        )
    )

    assert _verifier_timeout(result) is False
    assert result.exception_info is None
    assert result.verifier_result.rewards == {"reward": 1.0}
    assert replay is not None and replay["reward"] == 1.0
    assert docker_calls[0] == ("start", "container-id")
    assert docker_calls[-1] == ("stop", "--time", "30", "container-id")
    assert (tmp_path / "verifier-replay" / "original-result.json").read_text(
        encoding="utf-8"
    ) == "original\n"


def test_credential_scope_keeps_values_out_of_host_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = tmp_path / "config.toml"
    profile.write_text(
        """
[llm]
main = "opencode_go_main"

[llm.runtimes.deepseek_main]
provider = "deepseek"
api_key = "deepseek-sentinel"

[memory.embedding]
api_key = "dashscope-sentinel"
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("DEEPSEEK_API_KEY", "previous")
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)

    with credential_scope(profile) as names:
        assert names == ("DASHSCOPE_API_KEY", "DEEPSEEK_API_KEY")
        assert os.environ["DEEPSEEK_API_KEY"] == "previous"
        assert "DASHSCOPE_API_KEY" not in os.environ

    assert os.environ["DEEPSEEK_API_KEY"] == "previous"
    assert "DASHSCOPE_API_KEY" not in os.environ


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf"])
def test_task_agent_timeout_rejects_invalid_budget(
    tmp_path: Path,
    value: str,
) -> None:
    task_dir = tmp_path / value
    task_dir.mkdir()
    (task_dir / "task.toml").write_text(
        f"[agent]\ntimeout_sec = {value}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"\[agent\]\.timeout_sec"):
        _task_agent_timeout_sec(task_dir)


def test_harbor_startup_failure_keeps_original_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = type("Result", (), {"exception_info": object()})()

    def missing_project(project_name: str) -> list[dict[str, object]]:
        raise IsolationError(f"未找到 compose project：{project_name}")

    monkeypatch.setattr(
        "benchmark.harbor_v4flash.controller.inspect_compose_project",
        missing_project,
    )

    containers, error = _inspect_finished_project(result, "akasic-bench-missing")

    assert containers == []
    assert error == "未找到 compose project：akasic-bench-missing"


def test_success_without_compose_project_fails_loud(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = type("Result", (), {"exception_info": None})()

    def missing_project(project_name: str) -> list[dict[str, object]]:
        raise IsolationError(f"未找到 compose project：{project_name}")

    monkeypatch.setattr(
        "benchmark.harbor_v4flash.controller.inspect_compose_project",
        missing_project,
    )

    with pytest.raises(IsolationError, match="未找到 compose project"):
        _inspect_finished_project(result, "akasic-bench-missing")


def test_source_bundle_restores_history_and_keeps_worktree_overlay(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    _git(source, "init")
    _git(source, "config", "user.name", "Benchmark Test")
    _git(source, "config", "user.email", "benchmark@localhost")
    tracked = source / "tracked.txt"
    tracked.write_text("baseline\n", encoding="utf-8")
    _git(source, "add", "tracked.txt")
    _git(source, "commit", "-m", "baseline")
    baseline = _git(source, "rev-parse", "HEAD")
    tracked.write_text("head\n", encoding="utf-8")
    _git(source, "commit", "-am", "head")
    head = _git(source, "rev-parse", "HEAD")
    tracked.write_text("dirty overlay\n", encoding="utf-8")

    bundle = tmp_path / "inputs" / "source.bundle"
    info = create_source_bundle(source, bundle)

    restored = tmp_path / "restored"
    restored.mkdir()
    _git(restored, "init")
    shutil.copyfile(tracked, restored / "tracked.txt")
    _git(
        restored,
        "fetch",
        str(bundle),
        "+refs/heads/*:refs/remotes/benchmark/*",
    )
    _git(restored, "reset", "--mixed", head)
    assert _git(restored, "cat-file", "-t", baseline) == "commit"
    assert _git(restored, "rev-parse", "HEAD") == head
    assert (restored / "tracked.txt").read_text(encoding="utf-8") == "dirty overlay\n"
    assert _git(restored, "status", "--short") == "M tracked.txt"
    assert info["head"] == head

import asyncio
import os
import shutil
import subprocess
import tomllib
from pathlib import Path

import pytest
from harbor.environments.base import ExecResult

from benchmark.harbor_v4flash.agent import (
    _ENDPOINT,
    _WORKSPACE,
    _run_driver_and_shutdown,
)
from benchmark.harbor_v4flash.controller import (
    _credential_templates,
    _inspect_finished_project,
    _task_agent_timeout_sec,
)
from benchmark.harbor_v4flash.isolation import IsolationError, create_source_bundle


class _ScriptedEnvironment:
    def __init__(self, *outcomes: ExecResult | BaseException) -> None:
        self._outcomes = list(outcomes)
        self.commands: list[str] = []

    async def exec(self, *, command: str, timeout_sec: float) -> ExecResult:
        self.commands.append(command)
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.strip()


def test_v4flash_high_uses_provider_output_limit() -> None:
    config_path = (
        Path(__file__).parents[2]
        / "benchmark"
        / "harbor_v4flash"
        / "config.toml"
    )
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))

    assert config["llm"]["runtimes"]["main"]["max_output_tokens"] == 0
    assert config["agent"]["max_tokens"] == 0
    assert config["agent"]["max_iterations"] == 0


def test_benchmark_workspace_matches_terminal_task_root() -> None:
    assert _WORKSPACE == "/app"
    assert _ENDPOINT == "/app/akashic.sock"


def test_driver_success_keeps_command_order_and_logs(tmp_path: Path) -> None:
    environment = _ScriptedEnvironment(
        ExecResult(return_code=0, stdout="driver complete\n"),
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
    assert environment.commands == ["driver", "shutdown"]
    assert (
        tmp_path / "driver.stdout.log"
    ).read_text(encoding="utf-8") == "driver complete\n"
    assert (
        tmp_path / "runtime.shutdown.log"
    ).read_text(encoding="utf-8") == "shutdown complete\n"
    assert not (tmp_path / "driver.exception.log").exists()


def test_driver_timeout_still_shuts_down_gateway_and_persists_evidence(
    tmp_path: Path,
) -> None:
    environment = _ScriptedEnvironment(
        TimeoutError("driver exceeded deadline"),
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

    assert environment.commands == ["driver", "shutdown"]
    assert (tmp_path / "driver.stdout.log").read_text(encoding="utf-8") == ""
    assert (tmp_path / "driver.stderr.log").read_text(encoding="utf-8") == ""
    assert (
        tmp_path / "driver.exception.log"
    ).read_text(encoding="utf-8") == "TimeoutError: driver exceeded deadline\n"
    assert (
        tmp_path / "runtime.shutdown.log"
    ).read_text(encoding="utf-8") == "gateway stopped\n"


def test_driver_cancellation_still_shuts_down_gateway(
    tmp_path: Path,
) -> None:
    environment = _ScriptedEnvironment(
        asyncio.CancelledError("trial cancelled"),
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

    assert environment.commands == ["driver", "shutdown"]
    assert (
        tmp_path / "driver.exception.log"
    ).read_text(encoding="utf-8") == "CancelledError: trial cancelled\n"
    assert (tmp_path / "runtime.shutdown.log").read_text(encoding="utf-8") == ""


def test_driver_failure_remains_primary_when_shutdown_also_fails(
    tmp_path: Path,
) -> None:
    environment = _ScriptedEnvironment(
        ExecResult(return_code=1, stdout="driver failed\n"),
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

    assert any(
        "gateway cleanup also failed" in note for note in caught.value.__notes__
    )
    assert (
        tmp_path / "driver.exception.log"
    ).read_text(encoding="utf-8").startswith("RuntimeError: 执行 SDK turn")
    assert (
        tmp_path / "runtime.shutdown.log"
    ).read_text(encoding="utf-8") == "exit=2\n"


def test_task_agent_timeout_uses_harbor_task_budget(tmp_path: Path) -> None:
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    (task_dir / "task.toml").write_text(
        "[agent]\ntimeout_sec = 3600.0\n",
        encoding="utf-8",
    )

    assert _task_agent_timeout_sec(task_dir) == 3600.0


def test_credential_templates_persist_names_and_restore_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = tmp_path / "config.toml"
    profile.write_text(
        """
[llm.main]
api_key = "deepseek-sentinel"

[memory.embedding]
api_key = "dashscope-sentinel"
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("DEEPSEEK_API_KEY", "previous")
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)

    with _credential_templates(profile) as templates:
        assert templates == {
            "DEEPSEEK_API_KEY": "${DEEPSEEK_API_KEY}",
            "DASHSCOPE_API_KEY": "${DASHSCOPE_API_KEY}",
        }
        assert os.environ["DEEPSEEK_API_KEY"] == "deepseek-sentinel"
        assert os.environ["DASHSCOPE_API_KEY"] == "dashscope-sentinel"

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
    info = create_source_bundle(
        source,
        bundle,
        migration_baseline=baseline,
    )

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
    assert info["migration_baseline"] == baseline

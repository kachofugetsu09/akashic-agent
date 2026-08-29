from __future__ import annotations

import json
import re
import shlex
import time
from pathlib import Path

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment, ExecResult
from harbor.models.agent.context import AgentContext

from benchmark.harbor_v4flash import HARNESS_VERSION
from benchmark.harbor_v4flash.credentials import secure_docker_exec
from benchmark.harbor_v4flash.git_volume import GIT_BIN_PATH, GIT_MOUNT_PATH
from benchmark.harbor_v4flash.isolation import (
    atomic_json,
    compose_project_name,
    inspect_compose_project,
    validate_isolation,
)
from benchmark.harbor_v4flash.resource_evidence import (
    RESOURCE_EVIDENCE_FILENAME,
    parse_resource_probe_output,
    resource_probe_command,
    resource_probe_failure,
)
from benchmark.harbor_v4flash.result_projection import project_agent_context
from benchmark.harbor_v4flash.runtime_volume import (
    RUNTIME_MOUNT_PATH,
    RUNTIME_UV_PATH,
    RUNTIME_VENV_PATH,
)

_RUNTIME_ROOT = "/opt/akashic"
_SOURCE_ROOT = f"{_RUNTIME_ROOT}/src"
_WORKSPACE = "/opt/akashic-workspace"
_ENDPOINT = f"{_WORKSPACE}/akashic.sock"
_AGENT_LOGS = "/logs/agent"
_VERIFIER_PREPARE_TIMEOUT_SEC = 14_400
_CANDIDATE_DIGEST_COMMAND = (
    "pwd -P; "
    "tar --sort=name --mtime='UTC 1970-01-01' --owner=0 --group=0 "
    "--numeric-owner --format=gnu -cf - . | sha256sum | awk '{print $1}'"
)


def _require_success(label: str, return_code: int, output: str) -> None:
    if return_code != 0:
        raise RuntimeError(f"{label} 失败，exit={return_code}\n{output[-4000:]}")


def _write_driver_evidence(
    logs_dir: Path,
    completed: ExecResult | None,
    error: BaseException | None,
) -> None:
    (logs_dir / "driver.stdout.log").write_text(
        "" if completed is None else completed.stdout or "",
        encoding="utf-8",
    )
    (logs_dir / "driver.stderr.log").write_text(
        "" if completed is None else completed.stderr or "",
        encoding="utf-8",
    )
    if error is not None:
        (logs_dir / "driver.exception.log").write_text(
            f"{type(error).__name__}: {error}\n",
            encoding="utf-8",
        )


def _write_shutdown_evidence(
    logs_dir: Path,
    completed: ExecResult | None,
    error: BaseException | None,
) -> None:
    if completed is not None:
        output = (completed.stdout or "") + (completed.stderr or "")
        if completed.return_code != 0:
            output = f"exit={completed.return_code}\n{output}"
    else:
        assert error is not None
        output = f"{type(error).__name__}: {error}\n"
    (logs_dir / "runtime.shutdown.log").write_text(output, encoding="utf-8")


def _build_gateway_command() -> str:
    """生成继承题目镜像 WORKDIR 的 gateway 启动命令。"""

    return (
        f"mkdir -p {_WORKSPACE} && "
        f"env PYTHONPATH={_SOURCE_ROOT}:{_SOURCE_ROOT}/sdk/python/src "
        f"{RUNTIME_VENV_PATH}/bin/python {_SOURCE_ROOT}/main.py veda-reset "
        f"--config {_SOURCE_ROOT}/benchmark/harbor_v4flash/config.toml "
        f"--workspace {_WORKSPACE} >/dev/null && "
        f"env PATH={GIT_MOUNT_PATH}/bin:$PATH "
        f"PYTHONPATH={_SOURCE_ROOT}:{_SOURCE_ROOT}/sdk/python/src "
        "PYTHONDONTWRITEBYTECODE=1 "
        f"{RUNTIME_VENV_PATH}/bin/python {_SOURCE_ROOT}/main.py gateway "
        f"--config {_SOURCE_ROOT}/benchmark/harbor_v4flash/config.toml "
        f"--workspace {_WORKSPACE} "
        f">{_AGENT_LOGS}/runtime.stdout.log "
        f"2>{_AGENT_LOGS}/runtime.stderr.log & "
        f"echo $! > {_AGENT_LOGS}/runtime.pid; "
        f"env PYTHONPATH={_SOURCE_ROOT}:{_SOURCE_ROOT}/sdk/python/src "
        f"{RUNTIME_VENV_PATH}/bin/python "
        f"{_SOURCE_ROOT}/docker/debug/model_plugin_fixture.py "
        "--settings-url http://127.0.0.1:2236/api/settings/model "
        "--connection harbor-chat --endpoint https://api.deepseek.com/v1 "
        "--api-key-env DEEPSEEK_API_KEY --chat-model deepseek-v4-flash "
        "--context-window 1000000 --reasoning-effort max && "
        f"env PYTHONPATH={_SOURCE_ROOT}:{_SOURCE_ROOT}/sdk/python/src "
        f"{RUNTIME_VENV_PATH}/bin/python "
        f"{_SOURCE_ROOT}/docker/debug/model_plugin_fixture.py "
        "--settings-url http://127.0.0.1:2236/api/settings/model "
        "--connection harbor-embedding "
        "--endpoint https://dashscope.aliyuncs.com/compatible-mode/v1 "
        "--api-key-env DASHSCOPE_API_KEY --embedding-model text-embedding-v3"
    )


def _build_driver_command(turn_timeout_sec: float) -> str:
    """生成继承题目镜像 WORKDIR 的 SDK driver 命令。"""

    return " ".join(
        [
            f"PYTHONPATH={_SOURCE_ROOT}:{_SOURCE_ROOT}/sdk/python/src",
            "PYTHONDONTWRITEBYTECODE=1",
            f"{RUNTIME_VENV_PATH}/bin/python",
            f"{_SOURCE_ROOT}/benchmark/harbor_v4flash/runtime_driver.py",
            "--endpoint",
            shlex.quote(_ENDPOINT),
            "--instruction-file",
            f"{_AGENT_LOGS}/instruction.md",
            "--trace",
            f"{_AGENT_LOGS}/trace.jsonl",
            "--result",
            f"{_AGENT_LOGS}/turn-result.json",
            "--outcome",
            f"{_AGENT_LOGS}/driver-outcome.json",
            "--turn-timeout",
            str(turn_timeout_sec),
        ]
    )


async def _capture_resource_evidence(
    environment: BaseEnvironment,
    logs_dir: Path,
) -> tuple[BaseException | None, BaseException | None]:
    """采集并持久化容器资源证据，分别返回采集与写入失败。"""

    # 1. 容器仍运行时读取固定 cgroup 白名单。
    resource_error: BaseException | None = None
    try:
        result = await environment.exec(
            command=resource_probe_command(),
            timeout_sec=10,
        )
        _require_success(
            "采集容器资源证据",
            result.return_code,
            (result.stdout or "") + (result.stderr or ""),
        )
        evidence = parse_resource_probe_output(result.stdout or "")
    except BaseException as error:
        resource_error = error
        evidence = resource_probe_failure(error)

    # 2. 即使采集失败也写明确未知证据；写入失败单独返回。
    try:
        atomic_json(logs_dir / RESOURCE_EVIDENCE_FILENAME, evidence)
    except BaseException as error:
        return resource_error, error
    return resource_error, None


def _raise_lifecycle_errors(
    *,
    driver_error: BaseException | None,
    resource_error: BaseException | None,
    resource_write_error: BaseException | None,
    shutdown_error: BaseException | None,
) -> None:
    """按 driver、资源采集、证据写入、shutdown 顺序传播主失败。"""

    if driver_error is not None:
        if resource_error is not None:
            driver_error.add_note(
                f"resource evidence collection also failed: {resource_error}"
            )
        if resource_write_error is not None:
            driver_error.add_note(
                f"resource evidence persistence also failed: {resource_write_error}"
            )
        if shutdown_error is not None:
            driver_error.add_note(f"gateway cleanup also failed: {shutdown_error}")
        raise driver_error
    if resource_error is not None:
        if resource_write_error is not None:
            resource_error.add_note(
                f"resource evidence persistence also failed: {resource_write_error}"
            )
        if shutdown_error is not None:
            resource_error.add_note(f"gateway cleanup also failed: {shutdown_error}")
        raise resource_error
    if resource_write_error is not None:
        if shutdown_error is not None:
            resource_write_error.add_note(
                f"gateway cleanup also failed: {shutdown_error}"
            )
        raise resource_write_error
    if shutdown_error is not None:
        raise shutdown_error


async def _start_gateway_with_resource_evidence(
    environment: BaseEnvironment,
    *,
    gateway_command: str,
    credential_names: tuple[str, ...],
    logs_dir: Path,
) -> None:
    """安全启动 gateway，并在启动失败时保留资源证据和原始错误。"""

    # 1. 仅 gateway 启动进程接收 benchmark 凭据。
    try:
        started = await secure_docker_exec(
            environment,
            command=gateway_command,
            credential_names=credential_names,
        )
        _require_success(
            "启动 Akasic gateway",
            started.return_code,
            (started.stdout or "") + (started.stderr or ""),
        )
    except BaseException as startup_error:
        # 2. 启动失败仍采集 cgroup；原始启动错误始终保持主失败。
        resource_error, resource_write_error = await _capture_resource_evidence(
            environment,
            logs_dir,
        )
        _raise_lifecycle_errors(
            driver_error=startup_error,
            resource_error=resource_error,
            resource_write_error=resource_write_error,
            shutdown_error=None,
        )
        raise AssertionError("unreachable")


async def _run_driver_and_shutdown(
    environment: BaseEnvironment,
    *,
    driver_command: str,
    driver_timeout_sec: float,
    shutdown_command: str,
    logs_dir: Path,
) -> ExecResult:
    """运行 driver、采集资源证据并关闭 gateway，同时保留主失败。"""

    # 1. 捕获 driver 的成功结果或原始异常，不提前跳过 cleanup。
    driver_result: ExecResult | None = None
    driver_error: BaseException | None = None
    try:
        driver_result = await environment.exec(
            command=driver_command,
            timeout_sec=driver_timeout_sec,  # pyright: ignore[reportArgumentType]
        )
        _require_success(
            "执行 SDK turn",
            driver_result.return_code,
            (driver_result.stdout or "") + (driver_result.stderr or ""),
        )
    except BaseException as error:
        driver_error = error

    # 2. driver 任意终态都先读取固定 cgroup 白名单，再停止容器内进程。
    resource_error, resource_write_error = await _capture_resource_evidence(
        environment,
        logs_dir,
    )

    # 3. 资源采集成功或失败后都尝试关闭 gateway。
    shutdown_result: ExecResult | None = None
    shutdown_error: BaseException | None = None
    try:
        shutdown_result = await environment.exec(
            command=shutdown_command,
            timeout_sec=70,
        )
        _require_success(
            "收束 Akasic gateway",
            shutdown_result.return_code,
            (shutdown_result.stdout or "") + (shutdown_result.stderr or ""),
        )
    except BaseException as error:
        shutdown_error = error

    # 4. 先写全生命周期证据，再按 driver→资源→shutdown 的优先级传播。
    _write_driver_evidence(logs_dir, driver_result, driver_error)
    _write_shutdown_evidence(logs_dir, shutdown_result, shutdown_error)
    _raise_lifecycle_errors(
        driver_error=driver_error,
        resource_error=resource_error,
        resource_write_error=resource_write_error,
        shutdown_error=shutdown_error,
    )
    assert driver_result is not None
    return driver_result


def _verifier_dependency_command(test_script: str) -> str | None:
    """Extract dependency setup while replacing the verifier body with a no-op."""

    # 1. Find the first supported test runner boundary in the official script.
    lines = test_script.splitlines()
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped == "uvx \\" or stripped.startswith("uvx "):
            end = index
            while lines[end].rstrip().endswith("\\"):
                end += 1
            invocation = " ".join(
                item.rstrip().removesuffix("\\").strip()
                for item in lines[index : end + 1]
            )
            match = re.search(r"\s+pytest(?:\s|$)", invocation)
            if match is None:
                raise RuntimeError("官方 uvx verifier 不以 pytest 为执行边界")
            dependency_invocation = invocation[: match.start()]
            preamble = "\n".join(lines[:index])
            return f"{preamble}\n{dependency_invocation} python -c 'pass'"
        if re.match(r"(?:python(?:3)?\s+-m\s+)?pytest(?:\s|$)", stripped):
            return "\n".join(lines[:index])
    return None


async def _candidate_digest(environment: BaseEnvironment) -> str:
    """Return the frozen `/app` identity around verifier preparation."""

    result = await environment.exec(
        command=_CANDIDATE_DIGEST_COMMAND,
        timeout_sec=300,
        user="root",
    )
    _require_success(
        "计算 verifier 候选摘要",
        result.return_code,
        (result.stdout or "") + (result.stderr or ""),
    )
    lines = (result.stdout or "").strip().splitlines()
    if len(lines) < 2:
        raise RuntimeError("verifier 候选摘要输出无效")
    root, digest = lines[-2:]
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise RuntimeError("verifier 候选摘要输出无效")
    return f"{root}|sha256:{digest}"


async def _prepare_verifier_runtime(
    environment: BaseEnvironment,
    *,
    expected_uv_version: str,
    test_script: str,
    timeout_sec: float = _VERIFIER_PREPARE_TIMEOUT_SEC,
) -> dict[str, object]:
    """在官方计时前下载 verifier 依赖并证明候选未被改变。"""

    # 1. 只在 turn 完成后复制工具，避免改变 Agent 可见能力。
    command = (
        "mkdir -p /root/.local/bin && "
        f"install -m 0755 {RUNTIME_UV_PATH} /root/.local/bin/uv && "
        "printf '%s\\n' '#!/bin/sh' "
        "'exec /root/.local/bin/uv tool run \"$@\"' "
        "> /root/.local/bin/uvx && "
        "chmod 0755 /root/.local/bin/uvx && "
        "printf '%s\\n' 'export PATH=\"$HOME/.local/bin:$PATH\"' "
        "> /root/.local/bin/env && "
        f'test "$(/root/.local/bin/uv --version)" = '
        f"{shlex.quote(expected_uv_version)}"
    )
    prepared = await environment.exec(
        command=command,
        timeout_sec=30,
        user="root",
    )
    _require_success(
        "准备 verifier uv",
        prepared.return_code,
        (prepared.stdout or "") + (prepared.stderr or ""),
    )

    # 2. 从官方脚本抽出安装段，uvx 只解析和下载同一组依赖。
    dependency_command = _verifier_dependency_command(test_script)
    before_digest = await _candidate_digest(environment)
    started = time.monotonic()
    if dependency_command is not None:
        dependency_result = await environment.exec(
            command=f"set -eu\n{dependency_command}",
            timeout_sec=timeout_sec,
            user="root",
        )
        output = (dependency_result.stdout or "") + (dependency_result.stderr or "")
        _require_success("准备 verifier 依赖", dependency_result.return_code, output)
    else:
        output = ""
    duration_sec = time.monotonic() - started

    # 3. 依赖阶段只准改变 verifier runtime；触碰候选立即失败。
    after_digest = await _candidate_digest(environment)
    if after_digest != before_digest:
        raise RuntimeError("verifier 依赖准备改变了 /app 候选，禁止进入评分")
    return {
        "schema": "akasic.verifier-bootstrap.v1",
        "status": "prepared" if dependency_command is not None else "not_required",
        "official_verifier_timeout_started": False,
        "timeout_sec": timeout_sec,
        "duration_sec": duration_sec,
        "candidate_digest_before": before_digest,
        "candidate_digest_after": after_digest,
        "stdout_tail": output[-4000:],
    }


class AkashicHarborAgent(BaseAgent):
    """在 Harbor task 容器内运行完整 Akasic runtime。"""

    @staticmethod
    def name() -> str:
        return "akasic-v4flash"

    def __init__(
        self,
        logs_dir: Path,
        model_name: str | None = None,
        *,
        source_root: str,
        source_bundle: str,
        source_head: str,
        allowed_bind_root: str,
        forbidden_host_paths: list[str],
        source_digest: str,
        runtime_volume_name: str,
        runtime_digest: str,
        runtime_manifest_digest: str,
        runtime_lock_digest: str,
        runtime_python_version: str,
        runtime_uv_digest: str,
        runtime_uv_version: str,
        git_volume_name: str,
        git_runtime_digest: str,
        git_manifest_digest: str,
        git_version: str,
        bootstrap_timeout_sec: int = 900,
        turn_timeout_sec: float,
        credential_names: tuple[str, ...],
        **kwargs,
    ) -> None:
        super().__init__(logs_dir=logs_dir, model_name=model_name, **kwargs)
        self._source_root = Path(source_root).resolve()
        self._source_bundle = Path(source_bundle).resolve()
        self._source_head = source_head
        self._allowed_bind_root = Path(allowed_bind_root).resolve()
        self._forbidden_host_paths = [
            Path(path).resolve() for path in forbidden_host_paths
        ]
        self._source_digest = source_digest
        self._runtime_volume_name = runtime_volume_name
        self._runtime_digest = runtime_digest
        self._runtime_manifest_digest = runtime_manifest_digest
        self._runtime_lock_digest = runtime_lock_digest
        self._runtime_python_version = runtime_python_version
        self._runtime_uv_digest = runtime_uv_digest
        self._runtime_uv_version = runtime_uv_version
        self._git_volume_name = git_volume_name
        self._git_runtime_digest = git_runtime_digest
        self._git_manifest_digest = git_manifest_digest
        self._git_version = git_version
        self._bootstrap_timeout_sec = bootstrap_timeout_sec
        if turn_timeout_sec <= 0:
            raise ValueError("turn_timeout_sec 必须大于 0")
        self._turn_timeout_sec = turn_timeout_sec
        if not credential_names:
            raise ValueError("credential_names 不能为空")
        self._credential_names = credential_names

    def version(self) -> str:
        return HARNESS_VERSION

    def _runtime_check_command(self) -> str:
        """生成只读取固定 runtime manifest 和 lock 的容器内校验命令。"""

        expected = {
            "volume_name": self._runtime_volume_name,
            "runtime_digest": self._runtime_digest,
            "manifest_digest": self._runtime_manifest_digest,
            "lock_digest": self._runtime_lock_digest,
            "python_version": self._runtime_python_version,
            "uv_digest": self._runtime_uv_digest,
            "uv_version": self._runtime_uv_version,
        }
        code = (
            "import hashlib,json,platform,pathlib,subprocess;"
            f"root=pathlib.Path({RUNTIME_MOUNT_PATH!r});"
            "manifest=json.loads((root/'manifest.json').read_text());"
            f"expected=json.loads({json.dumps(json.dumps(expected))});"
            "assert manifest['volume_name']==expected['volume_name'];"
            "assert manifest['runtime_digest']==expected['runtime_digest'];"
            "assert manifest['manifest_digest']==expected['manifest_digest'];"
            "lock=hashlib.sha256((root/'resolved.lock').read_bytes()).hexdigest();"
            "assert 'sha256:'+lock==expected['lock_digest'];"
            "uv=hashlib.sha256((root/'uv').read_bytes()).hexdigest();"
            "assert 'sha256:'+uv==expected['uv_digest'];"
            "uv_version=subprocess.run([str(root/'uv'),'--version'],"
            "check=True,text=True,stdout=subprocess.PIPE).stdout.strip();"
            "assert uv_version==expected['uv_version'];"
            "assert platform.python_version()==expected['python_version']"
        )
        return f"{RUNTIME_VENV_PATH}/bin/python -c {shlex.quote(code)}"

    def _git_check_command(self) -> str:
        """生成只读取固定 Git manifest 并执行版本探针的校验命令。"""

        expected = {
            "volume_name": self._git_volume_name,
            "runtime_digest": self._git_runtime_digest,
            "manifest_digest": self._git_manifest_digest,
            "git_version": self._git_version,
        }
        code = (
            "import json,pathlib,subprocess;"
            f"root=pathlib.Path({GIT_MOUNT_PATH!r});"
            "manifest=json.loads((root/'manifest.json').read_text());"
            f"expected=json.loads({json.dumps(json.dumps(expected))});"
            "assert manifest['volume_name']==expected['volume_name'];"
            "assert manifest['runtime_digest']==expected['runtime_digest'];"
            "assert manifest['manifest_digest']==expected['manifest_digest'];"
            f"version=subprocess.run([{GIT_BIN_PATH!r},'--version'],"
            "check=True,text=True,stdout=subprocess.PIPE).stdout.strip();"
            "assert version==expected['git_version']"
        )
        return f"{RUNTIME_VENV_PATH}/bin/python -c {shlex.quote(code)}"

    async def setup(self, environment: BaseEnvironment) -> None:
        """复制只读源码，验证共享 runtime，并证明容器隔离。"""

        # 1. 只向当前 Harbor task 容器复制源码，不建立主机源码挂载。
        prepare = await environment.exec(
            command=f"mkdir -p {_SOURCE_ROOT} {_AGENT_LOGS}",
            user="root",
        )
        _require_success("创建 runtime 目录", prepare.return_code, prepare.stdout or "")
        await environment.upload_dir(self._source_root, _SOURCE_ROOT)
        await environment.upload_file(
            self._source_bundle,
            f"{_RUNTIME_ROOT}/source.bundle",
        )

        # 2. 模型和网络安装前，拒绝错误或可写的共享 volume。
        project = compose_project_name(environment.session_id)
        containers = inspect_compose_project(project)
        report = validate_isolation(
            containers,
            project_name=project,
            allowed_bind_root=self._allowed_bind_root,
            forbidden_host_paths=self._forbidden_host_paths,
            allowed_volume_mounts=[
                (self._runtime_volume_name, RUNTIME_MOUNT_PATH),
                (self._git_volume_name, GIT_MOUNT_PATH),
            ],
        )
        checked_runtime = await environment.exec(
            command=self._runtime_check_command(),
            user="root",
        )
        _require_success(
            "校验 Akasic runtime volume",
            checked_runtime.return_code,
            (checked_runtime.stdout or "") + (checked_runtime.stderr or ""),
        )
        checked_git = await environment.exec(
            command=self._git_check_command(),
            user="root",
        )
        _require_success(
            "校验 Akasic Git volume",
            checked_git.return_code,
            (checked_git.stdout or "") + (checked_git.stderr or ""),
        )
        report["source_digest"] = self._source_digest
        report["runtime_volume"] = {
            "name": self._runtime_volume_name,
            "mount_path": RUNTIME_MOUNT_PATH,
            "runtime_digest": self._runtime_digest,
            "manifest_digest": self._runtime_manifest_digest,
            "resolved_lock_digest": self._runtime_lock_digest,
            "python_version": self._runtime_python_version,
            "uv_digest": self._runtime_uv_digest,
            "uv_version": self._runtime_uv_version,
        }
        report["git_volume"] = {
            "name": self._git_volume_name,
            "mount_path": GIT_MOUNT_PATH,
            "runtime_digest": self._git_runtime_digest,
            "manifest_digest": self._git_manifest_digest,
            "git_version": self._git_version,
        }
        atomic_json(self.logs_dir / "isolation.preflight.json", report)

        # 3. 使用只读 Git 基础设施恢复真实历史，不在 trial 内联网安装。
        install_command = (
            f"rm -rf {_SOURCE_ROOT}/.git && "
            f"{GIT_BIN_PATH} -C {_SOURCE_ROOT} init && "
            f"{GIT_BIN_PATH} -C {_SOURCE_ROOT} fetch {_RUNTIME_ROOT}/source.bundle "
            "'+refs/heads/*:refs/remotes/benchmark/*' "
            "'+refs/tags/*:refs/tags/*' && "
            f"{GIT_BIN_PATH} -C {_SOURCE_ROOT} reset --mixed "
            f"{shlex.quote(self._source_head)} && "
            f"{GIT_BIN_PATH} -C {_SOURCE_ROOT} cat-file -e "
            "012e37c8b51df045353972bb551d8e868ab52455^{commit} && "
            f'test "$({GIT_BIN_PATH} -C {_SOURCE_ROOT} rev-parse HEAD)" = '
            f"{shlex.quote(self._source_head)} && "
            f"chmod -R a-w {_SOURCE_ROOT}"
        )
        installed = await environment.exec(
            command=install_command,
            timeout_sec=self._bootstrap_timeout_sec,
            user="root",
        )
        _require_success(
            "准备 Akasic source checkout",
            installed.return_code,
            (installed.stdout or "") + (installed.stderr or ""),
        )

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        """启动完整 gateway，通过公开 SDK 完成任务并保留 trace。"""

        # 1. task 官方 agent 预算从 Agent.run 入口开始，包含 gateway readiness。
        agent_started = time.monotonic()
        instruction_path = self.logs_dir / "instruction.md"
        instruction_path.write_text(instruction, encoding="utf-8")
        await _start_gateway_with_resource_evidence(
            environment,
            gateway_command=_build_gateway_command(),
            credential_names=self._credential_names,
            logs_dir=self.logs_dir,
        )

        # 2. SDK driver 只获得官方预算的剩余部分，不额外赠送 readiness 时间。
        remaining_sec = self._turn_timeout_sec - (time.monotonic() - agent_started)
        if remaining_sec <= 0:
            raise TimeoutError("gateway 启动已耗尽 task 官方 agent 预算")
        driver_error: RuntimeError | None = None
        try:
            _ = await _run_driver_and_shutdown(
                environment,
                driver_command=_build_driver_command(remaining_sec),
                driver_timeout_sec=remaining_sec + 5,
                shutdown_command=(
                    f"pid=$(cat {_AGENT_LOGS}/runtime.pid) && "
                    'if ! kill -0 "$pid" 2>/dev/null; then exit 0; fi; '
                    'kill -TERM "$pid" && '
                    "for _ in $(seq 1 120); do "
                    'if ! kill -0 "$pid" 2>/dev/null; then exit 0; fi; '
                    "sleep 0.5; "
                    "done; "
                    "exit 1"
                ),
                logs_dir=self.logs_dir,
            )
        except RuntimeError as error:
            driver_error = error

        # 3. 题目原始时限只终止 Agent；已完成 cleanup 后仍交给官方 verifier 打分。
        driver_outcome = json.loads(
            (self.logs_dir / "driver-outcome.json").read_text(encoding="utf-8")
        )
        if driver_error is not None and (
            driver_outcome.get("status")
            not in {
                "timed_out",
                "agent_failed",
                "rate_limited",
                "provider_transient",
                "account_limited",
            }
            or getattr(driver_error, "__notes__", None)
        ):
            raise driver_error

        # 3. 把可机读终态投影到 Harbor AgentContext。
        turn_result_path = self.logs_dir / "turn-result.json"
        if turn_result_path.is_file():
            turn_result = json.loads(turn_result_path.read_text(encoding="utf-8"))
            project_agent_context(
                context,
                turn_result,
                harness_name=self.name(),
                harness_version=self.version(),
                source_digest=self._source_digest,
            )
        else:
            context.metadata = {
                **(context.metadata or {}),
                "harness": self.name(),
                "harness_version": self.version(),
                "source_digest": self._source_digest,
                "turn_status": driver_outcome["status"],
                "trace": "agent/trace.jsonl",
            }

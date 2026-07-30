from __future__ import annotations

import json
import shlex
from pathlib import Path

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

from benchmark.harbor_v4flash import HARNESS_VERSION
from benchmark.harbor_v4flash.isolation import (
    atomic_json,
    compose_project_name,
    inspect_compose_project,
    validate_isolation,
)
from benchmark.harbor_v4flash.git_volume import GIT_BIN_PATH, GIT_MOUNT_PATH
from benchmark.harbor_v4flash.result_projection import project_agent_context
from benchmark.harbor_v4flash.runtime_volume import (
    RUNTIME_MOUNT_PATH,
    RUNTIME_VENV_PATH,
)

_RUNTIME_ROOT = "/opt/akashic"
_SOURCE_ROOT = f"{_RUNTIME_ROOT}/src"
_WORKSPACE = "/app"
_ENDPOINT = f"{_WORKSPACE}/akashic.sock"
_AGENT_LOGS = "/logs/agent"


def _require_success(label: str, return_code: int, output: str) -> None:
    if return_code != 0:
        raise RuntimeError(f"{label} 失败，exit={return_code}\n{output[-4000:]}")


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
        git_volume_name: str,
        git_runtime_digest: str,
        git_manifest_digest: str,
        git_version: str,
        bootstrap_timeout_sec: int = 900,
        turn_timeout_sec: float,
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
        self._git_volume_name = git_volume_name
        self._git_runtime_digest = git_runtime_digest
        self._git_manifest_digest = git_manifest_digest
        self._git_version = git_version
        self._bootstrap_timeout_sec = bootstrap_timeout_sec
        if turn_timeout_sec <= 0:
            raise ValueError("turn_timeout_sec 必须大于 0")
        self._turn_timeout_sec = turn_timeout_sec

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
        }
        code = (
            "import hashlib,json,platform,pathlib;"
            f"root=pathlib.Path({RUNTIME_MOUNT_PATH!r});"
            "manifest=json.loads((root/'manifest.json').read_text());"
            f"expected=json.loads({json.dumps(json.dumps(expected))});"
            "assert manifest['volume_name']==expected['volume_name'];"
            "assert manifest['runtime_digest']==expected['runtime_digest'];"
            "assert manifest['manifest_digest']==expected['manifest_digest'];"
            "lock=hashlib.sha256((root/'resolved.lock').read_bytes()).hexdigest();"
            "assert 'sha256:'+lock==expected['lock_digest'];"
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
            f"test \"$({GIT_BIN_PATH} -C {_SOURCE_ROOT} rev-parse HEAD)\" = "
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

        # 1. Harbor 提供的 task instruction 通过证据挂载传入，不进入 shell 参数。
        instruction_path = self.logs_dir / "instruction.md"
        instruction_path.write_text(instruction, encoding="utf-8")
        gateway_command = (
            f"mkdir -p {_WORKSPACE} && cd /app || exit $?; "
            f"env PATH={GIT_MOUNT_PATH}/bin:$PATH "
            f"PYTHONPATH={_SOURCE_ROOT}:{_SOURCE_ROOT}/sdk/python/src "
            "PYTHONDONTWRITEBYTECODE=1 "
            f"{RUNTIME_VENV_PATH}/bin/python {_SOURCE_ROOT}/main.py gateway "
            f"--config {_SOURCE_ROOT}/benchmark/harbor_v4flash/config.toml "
            f"--workspace {_WORKSPACE} "
            f">{_AGENT_LOGS}/runtime.stdout.log "
            f"2>{_AGENT_LOGS}/runtime.stderr.log & "
            f"echo $! > {_AGENT_LOGS}/runtime.pid"
        )
        started = await environment.exec(command=gateway_command)
        _require_success(
            "启动 Akasic gateway",
            started.return_code,
            (started.stdout or "") + (started.stderr or ""),
        )

        # 2. SDK driver 记录全部 turn 通知并核对持久化终态。
        driver_command = " ".join(
            [
                f"cd /app && PYTHONPATH={_SOURCE_ROOT}:{_SOURCE_ROOT}/sdk/python/src",
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
                "--turn-timeout",
                str(self._turn_timeout_sec),
            ]
        )
        completed = await environment.exec(
            command=driver_command,
            timeout_sec=self._turn_timeout_sec + 5,
        )
        shutdown = await environment.exec(
            command=(
                f"pid=$(cat {_AGENT_LOGS}/runtime.pid) && "
                "if ! kill -0 \"$pid\" 2>/dev/null; then exit 0; fi; "
                "kill -TERM \"$pid\" && "
                "for _ in $(seq 1 120); do "
                "if ! kill -0 \"$pid\" 2>/dev/null; then exit 0; fi; "
                "sleep 0.5; "
                "done; "
                "exit 1"
            ),
            timeout_sec=70,
        )
        (self.logs_dir / "driver.stdout.log").write_text(
            completed.stdout or "",
            encoding="utf-8",
        )
        (self.logs_dir / "driver.stderr.log").write_text(
            completed.stderr or "",
            encoding="utf-8",
        )
        (self.logs_dir / "runtime.shutdown.log").write_text(
            (shutdown.stdout or "") + (shutdown.stderr or ""),
            encoding="utf-8",
        )
        _require_success(
            "收束 Akasic gateway",
            shutdown.return_code,
            (shutdown.stdout or "") + (shutdown.stderr or ""),
        )
        _require_success(
            "执行 SDK turn",
            completed.return_code,
            (completed.stdout or "") + (completed.stderr or ""),
        )

        # 3. 把可机读终态投影到 Harbor AgentContext。
        turn_result = json.loads(
            (self.logs_dir / "turn-result.json").read_text(encoding="utf-8")
        )
        project_agent_context(
            context,
            turn_result,
            harness_name=self.name(),
            harness_version=self.version(),
            source_digest=self._source_digest,
        )

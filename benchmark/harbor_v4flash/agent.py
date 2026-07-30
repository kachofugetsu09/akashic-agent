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
from benchmark.harbor_v4flash.result_projection import project_agent_context

_RUNTIME_ROOT = "/opt/akashic"
_SOURCE_ROOT = f"{_RUNTIME_ROOT}/src"
_VENV = f"{_RUNTIME_ROOT}/venv"
_WORKSPACE = "/tmp/akashic-workspace"
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
        uv_binary: str,
        allowed_bind_root: str,
        forbidden_host_paths: list[str],
        source_digest: str,
        install_timeout_sec: int = 900,
        turn_timeout_sec: float,
        **kwargs,
    ) -> None:
        super().__init__(logs_dir=logs_dir, model_name=model_name, **kwargs)
        self._source_root = Path(source_root).resolve()
        self._source_bundle = Path(source_bundle).resolve()
        self._source_head = source_head
        self._uv_binary = Path(uv_binary).resolve()
        self._allowed_bind_root = Path(allowed_bind_root).resolve()
        self._forbidden_host_paths = [
            Path(path).resolve() for path in forbidden_host_paths
        ]
        self._source_digest = source_digest
        self._install_timeout_sec = install_timeout_sec
        if turn_timeout_sec <= 0:
            raise ValueError("turn_timeout_sec 必须大于 0")
        self._turn_timeout_sec = turn_timeout_sec

    def version(self) -> str:
        return HARNESS_VERSION

    async def setup(self, environment: BaseEnvironment) -> None:
        """复制只读 harness，安装独立 Python runtime，并证明容器隔离。"""

        # 1. 只向当前 Harbor task 容器复制源码与 uv，不建立主机源码挂载。
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
        await environment.upload_file(self._uv_binary, f"{_RUNTIME_ROOT}/uv")

        # 2. 补齐官方 agent setup 所需基础设施，并恢复真实 Git 历史。
        install_command = (
            "if ! command -v git >/dev/null 2>&1; then "
            "if command -v apk >/dev/null 2>&1; then "
            "apk add --no-cache git ca-certificates; "
            "elif command -v apt-get >/dev/null 2>&1; then "
            "apt-get update && DEBIAN_FRONTEND=noninteractive "
            "apt-get install -y --no-install-recommends git ca-certificates; "
            "elif command -v yum >/dev/null 2>&1; then "
            "yum install -y git ca-certificates; "
            "else echo '任务镜像缺少 git 且无受支持的包管理器' >&2; exit 2; fi; fi && "
            f"rm -rf {_SOURCE_ROOT}/.git && "
            f"git -C {_SOURCE_ROOT} init && "
            f"git -C {_SOURCE_ROOT} fetch {_RUNTIME_ROOT}/source.bundle "
            "'+refs/heads/*:refs/remotes/benchmark/*' "
            "'+refs/tags/*:refs/tags/*' && "
            f"git -C {_SOURCE_ROOT} reset --mixed {shlex.quote(self._source_head)} && "
            f"git -C {_SOURCE_ROOT} cat-file -e "
            "012e37c8b51df045353972bb551d8e868ab52455^{commit} && "
            f"test \"$(git -C {_SOURCE_ROOT} rev-parse HEAD)\" = "
            f"{shlex.quote(self._source_head)} && "
            f"chmod 755 {_RUNTIME_ROOT}/uv && "
            f"{_RUNTIME_ROOT}/uv venv --python 3.13 {_VENV} && "
            f"{_RUNTIME_ROOT}/uv pip install --python {_VENV}/bin/python "
            f"-r {_SOURCE_ROOT}/requirements.txt tzdata && "
            f"chmod -R a-w {_SOURCE_ROOT}"
        )
        installed = await environment.exec(
            command=install_command,
            timeout_sec=self._install_timeout_sec,
            user="root",
        )
        _require_success(
            "安装 Akasic runtime",
            installed.return_code,
            (installed.stdout or "") + (installed.stderr or ""),
        )

        # 3. 模型调用前，从宿主 Docker 控制面拒绝任何线上路径或端口泄漏。
        project = compose_project_name(environment.session_id)
        containers = inspect_compose_project(project)
        report = validate_isolation(
            containers,
            project_name=project,
            allowed_bind_root=self._allowed_bind_root,
            forbidden_host_paths=self._forbidden_host_paths,
        )
        report["source_digest"] = self._source_digest
        atomic_json(self.logs_dir / "isolation.preflight.json", report)

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
            f"env PYTHONPATH={_SOURCE_ROOT}:{_SOURCE_ROOT}/sdk/python/src "
            "PYTHONDONTWRITEBYTECODE=1 "
            f"{_VENV}/bin/python {_SOURCE_ROOT}/main.py gateway "
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
                f"{_VENV}/bin/python",
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

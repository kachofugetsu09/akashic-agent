from __future__ import annotations

import asyncio
import os
import subprocess
import tomllib
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

from harbor.environments.base import BaseEnvironment, ExecResult

from benchmark.harbor_v4flash.isolation import (
    BENCHMARK_PREFIX,
    IsolationError,
    compose_project_name,
)

_CREDENTIAL_VALUES: ContextVar[dict[str, str] | None] = ContextVar(
    "benchmark_credential_values",
    default=None,
)


def _credential_values(profile_path: Path) -> dict[str, str]:
    """从本机 profile 只提取当前实验所需密钥。"""

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


@contextmanager
def credential_scope(profile_path: Path) -> Iterator[tuple[str, ...]]:
    """在 controller 内存中暴露密钥，并在 trial 结束后恢复原作用域。"""

    values = _credential_values(profile_path)
    token = _CREDENTIAL_VALUES.set(values)
    try:
        yield tuple(sorted(values))
    finally:
        _CREDENTIAL_VALUES.reset(token)


def _main_container_id(project_name: str) -> str:
    """解析唯一运行中的 benchmark main 容器。"""

    if not project_name.startswith(BENCHMARK_PREFIX):
        raise IsolationError(f"compose project 缺少 benchmark 前缀：{project_name}")
    result = subprocess.run(
        [
            "docker",
            "ps",
            "-q",
            "--filter",
            f"label=com.docker.compose.project={project_name}",
            "--filter",
            "label=com.docker.compose.service=main",
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    container_ids = [item for item in result.stdout.splitlines() if item]
    if len(container_ids) != 1:
        raise IsolationError(
            f"benchmark project 必须恰有一个运行中的 main 容器："
            f"{project_name}，实际 {len(container_ids)}"
        )
    return container_ids[0]


def _credential_values_for(names: Sequence[str]) -> dict[str, str]:
    values = _CREDENTIAL_VALUES.get()
    if values is None:
        raise RuntimeError("benchmark credential scope 未激活")
    requested = tuple(names)
    if not requested:
        raise ValueError("credential_names 不能为空")
    missing = [name for name in requested if name not in values]
    if missing:
        raise RuntimeError(f"credential scope 缺少变量：{missing}")
    return {name: values[name] for name in requested}


def _docker_exec_argv(
    *,
    container_id: str,
    command: str,
    credential_names: Sequence[str],
    cwd: str | None,
    user: str | int | None,
) -> list[str]:
    argv = ["docker", "exec"]
    if cwd:
        argv.extend(["--workdir", cwd])
    for name in credential_names:
        argv.extend(["--env", name])
    if user is not None:
        argv.extend(["--user", str(user)])
    argv.extend([container_id, "bash", "-c", command])
    return argv


async def secure_docker_exec(
    environment: BaseEnvironment,
    *,
    command: str,
    credential_names: Sequence[str],
    timeout_sec: float | None = None,
) -> ExecResult:
    """仅把密钥放入 docker 客户端环境，并以变量名注入 main 进程。"""

    # 1. 从隔离 project 解析唯一 main 容器并构造不含密钥值的 argv。
    values = _credential_values_for(credential_names)
    project_name = compose_project_name(environment.session_id)
    container_id = _main_container_id(project_name)
    argv = _docker_exec_argv(
        container_id=container_id,
        command=command,
        credential_names=credential_names,
        cwd=environment.task_env_config.workdir,
        user=environment.default_user,
    )

    # 2. 真实值只进入当前 docker 客户端进程环境，由 `--env NAME` 复制。
    child_env = dict(os.environ)
    child_env.update(values)
    process = await asyncio.create_subprocess_exec(
        *argv,
        env=child_env,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        if timeout_sec is None:
            stdout, stderr = await process.communicate()
        else:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout_sec,
            )
    except asyncio.TimeoutError:
        process.kill()
        _ = await process.wait()
        raise RuntimeError(f"安全注入命令超时：{timeout_sec} 秒") from None
    except BaseException:
        if process.returncode is None:
            process.kill()
            _ = await process.wait()
        raise
    return ExecResult(
        stdout=stdout.decode(errors="replace") or None,
        stderr=stderr.decode(errors="replace") or None,
        return_code=process.returncode or 0,
    )

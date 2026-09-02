import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from benchmark.harbor_v4flash.credentials import (
    credential_scope,
    secure_docker_exec,
)


class _CompletedProcess:
    returncode = 0

    async def communicate(self) -> tuple[bytes, bytes]:
        return b"credentials received\n", b""

    def kill(self) -> None:
        raise AssertionError("成功路径不应终止 docker 客户端")

    async def wait(self) -> int:
        return self.returncode


def test_secure_exec_argv_hides_values_and_subprocess_receives_them(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = tmp_path / "config.toml"
    profile.write_text(
        """
[credentials]
DEEPSEEK_API_KEY = "deepseek-sentinel"
DASHSCOPE_API_KEY = "dashscope-sentinel"
""".strip(),
        encoding="utf-8",
    )
    captured: dict[str, Any] = {}

    async def create_subprocess_exec(
        *argv: str,
        **kwargs: object,
    ) -> _CompletedProcess:
        captured["argv"] = argv
        captured["env"] = kwargs["env"]
        return _CompletedProcess()

    monkeypatch.setattr(
        "benchmark.harbor_v4flash.credentials._main_container_id",
        lambda project: "container-id",
    )
    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess_exec)
    environment = SimpleNamespace(
        session_id="akasic-bench-v4flash-secret__env",
        default_user="root",
        task_env_config=SimpleNamespace(workdir="/app"),
    )

    with credential_scope(profile) as names:
        result = asyncio.run(
            secure_docker_exec(
                environment,  # type: ignore[arg-type]
                command="run-gateway",
                credential_names=names,
            )
        )

    argv = tuple(str(value) for value in captured["argv"])
    joined_argv = "\0".join(argv)
    child_env = captured["env"]
    assert result.return_code == 0
    assert "deepseek-sentinel" not in joined_argv
    assert "dashscope-sentinel" not in joined_argv
    assert not any(value.startswith("DEEPSEEK_API_KEY=") for value in argv)
    assert not any(value.startswith("DASHSCOPE_API_KEY=") for value in argv)
    assert argv.count("--env") == 2
    assert "DEEPSEEK_API_KEY" in argv
    assert "DASHSCOPE_API_KEY" in argv
    assert child_env["DEEPSEEK_API_KEY"] == "deepseek-sentinel"
    assert child_env["DASHSCOPE_API_KEY"] == "dashscope-sentinel"


def test_secure_exec_fails_loud_without_credential_scope() -> None:
    environment = SimpleNamespace(
        session_id="akasic-bench-v4flash-secret__env",
        default_user=None,
        task_env_config=SimpleNamespace(workdir="/app"),
    )

    with pytest.raises(RuntimeError, match="scope 未激活"):
        asyncio.run(
            secure_docker_exec(
                environment,  # type: ignore[arg-type]
                command="run-gateway",
                credential_names=("DEEPSEEK_API_KEY",),
            )
        )

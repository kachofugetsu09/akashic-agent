from __future__ import annotations

import asyncio
import json
import re
import time
import tomllib
from pathlib import Path
from typing import Any, cast

from agent.mcp.watcher import WorkspaceMcpWatcher
from infra.persistence.json_store import atomic_write_text

_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")
_MAX_COMMAND_ITEMS = 32
_MAX_ENV_ITEMS = 64
_MAX_WATCH_PATHS = 64


class WorkspaceMcpAdmin:
    """持久化 workspace MCP 声明并通过 watcher 原子发布结果。"""

    def __init__(self, workspace: Path, watcher: WorkspaceMcpWatcher) -> None:
        self._mcp_root = workspace / "mcp"
        self._declarations_dir = self._mcp_root / "servers"
        self._backup_root = self._mcp_root / "backups"
        self._watcher = watcher
        self._mutation_lock = asyncio.Lock()

    async def apply(
        self,
        *,
        name: str,
        command: list[str],
        cwd: str | None,
        env: dict[str, str],
        watch_paths: list[str],
    ) -> dict[str, Any]:
        """写入一个声明，确认候选可用后返回已发布代际。"""

        # 1. 在工具边界校验结构，路径边界交给声明加载器统一拥有。
        self._validate_spec(name, command, cwd, env, watch_paths)
        content = _render_declaration(name, command, cwd, env, watch_paths)
        path = self._declaration_path(name)

        # 2. 备份并原子替换声明，再由 watcher 完成完整连接和发布。
        async with self._mutation_lock:
            previous = path.read_text(encoding="utf-8") if path.exists() else None
            backup = self._backup(name, previous) if previous is not None else None
            atomic_write_text(path, content, domain="workspace_mcp_admin")
            try:
                _ = await self._watcher.reconcile()
            except (OSError, ValueError, RuntimeError) as error:
                await self._rollback(path, previous)
                raise RuntimeError(
                    f"workspace MCP 发布失败，声明已回滚: {error}"
                ) from error

        return {
            "status": "active",
            "name": name,
            "declaration": str(path),
            "backup": str(backup) if backup is not None else None,
            "runtime": self._watcher.status(),
            "effectiveFrom": "next_turn",
        }

    async def remove(self, name: str) -> dict[str, Any]:
        """删除一个声明，确认空缺代际发布后返回排空状态。"""

        self._validate_name(name)
        path = self._declaration_path(name)
        async with self._mutation_lock:
            previous = path.read_text(encoding="utf-8")
            backup = self._backup(name, previous)
            path.unlink()
            try:
                _ = await self._watcher.reconcile()
            except (OSError, ValueError, RuntimeError) as error:
                await self._rollback(path, previous)
                raise RuntimeError(
                    f"workspace MCP 删除发布失败，声明已恢复: {error}"
                ) from error

        return {
            "status": "removed",
            "name": name,
            "backup": str(backup),
            "runtime": self._watcher.status(),
            "effectiveFrom": "next_turn",
        }

    def status(self, name: str | None = None) -> dict[str, Any]:
        """列出声明及已发布代际，环境变量只暴露名称。"""

        if name is not None:
            self._validate_name(name)
        active = self._watcher.status()
        active_servers = set(active["servers"])
        declarations: list[dict[str, Any]] = []
        if name is not None:
            paths = [self._declaration_path(name)]
        elif self._declarations_dir.is_dir():
            paths = sorted(self._declarations_dir.glob("*.toml"))
        else:
            paths = []
        for path in paths:
            if not path.exists():
                continue
            try:
                raw = tomllib.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, tomllib.TOMLDecodeError) as error:
                declarations.append(
                    {
                        "name": path.stem,
                        "path": str(path),
                        "state": "invalid",
                        "error": str(error),
                    }
                )
                continue
            declared_name = str(raw.get("name", path.stem))
            env_value = raw.get("env", {})
            env = cast(dict[str, object], env_value) if isinstance(env_value, dict) else {}
            declarations.append(
                {
                    "name": declared_name,
                    "path": str(path),
                    "state": "active" if declared_name in active_servers else "declared",
                    "command": raw.get("command"),
                    "cwd": raw.get("cwd"),
                    "watchPaths": raw.get("watch_paths", []),
                    "envKeys": sorted(env) if isinstance(env, dict) else [],
                }
            )
        return {"declarations": declarations, "runtime": active}

    async def _rollback(self, path: Path, previous: str | None) -> None:
        """恢复声明文件，并重新确认恢复后的代际可发布。"""

        if previous is None:
            path.unlink(missing_ok=True)
        else:
            atomic_write_text(path, previous, domain="workspace_mcp_admin")
        try:
            _ = await self._watcher.reconcile()
        except (OSError, ValueError, RuntimeError) as rollback_error:
            raise RuntimeError(
                f"workspace MCP 声明回滚后仍不可发布: {rollback_error}"
            ) from rollback_error

    def _backup(self, name: str, content: str) -> Path:
        stamp = f"{time.strftime('%Y%m%d-%H%M%S')}-{time.time_ns()}"
        path = self._backup_root / name / f"{stamp}.toml"
        atomic_write_text(path, content, domain="workspace_mcp_admin_backup")
        return path

    def _declaration_path(self, name: str) -> Path:
        return self._declarations_dir / f"{name}.toml"

    @staticmethod
    def _validate_name(name: str) -> None:
        if not _NAME_PATTERN.fullmatch(name):
            raise ValueError("MCP name 必须以小写字母开头，且只含小写字母、数字、_、-")

    @classmethod
    def _validate_spec(
        cls,
        name: str,
        command: list[str],
        cwd: str | None,
        env: dict[str, str],
        watch_paths: list[str],
    ) -> None:
        cls._validate_name(name)
        if not command or len(command) > _MAX_COMMAND_ITEMS or not all(
            isinstance(item, str) and item for item in command
        ):
            raise ValueError("MCP command 必须包含 1-32 个非空字符串")
        if cwd is not None and (not isinstance(cwd, str) or not cwd):
            raise ValueError("MCP cwd 必须是非空字符串")
        if len(env) > _MAX_ENV_ITEMS or not all(
            isinstance(key, str) and key and isinstance(value, str)
            for key, value in env.items()
        ):
            raise ValueError("MCP env 最多 64 项，键必须非空且值必须是字符串")
        if len(watch_paths) > _MAX_WATCH_PATHS or not all(
            isinstance(item, str) and item for item in watch_paths
        ):
            raise ValueError("MCP watch_paths 最多 64 项且必须是非空字符串")


def _render_declaration(
    name: str,
    command: list[str],
    cwd: str | None,
    env: dict[str, str],
    watch_paths: list[str],
) -> str:
    """把已校验参数编码成稳定 TOML 声明。"""

    lines = [
        "schema_version = 1",
        f"name = {_toml_string(name)}",
        f"command = [{', '.join(_toml_string(item) for item in command)}]",
    ]
    if cwd is not None:
        lines.append(f"cwd = {_toml_string(cwd)}")
    if watch_paths:
        rendered = ", ".join(_toml_string(item) for item in watch_paths)
        lines.append(f"watch_paths = [{rendered}]")
    if env:
        lines.extend(["", "[env]"])
        lines.extend(
            f"{_toml_string(key)} = {_toml_string(value)}"
            for key, value in sorted(env.items())
        )
    return "\n".join(lines) + "\n"


def _toml_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)

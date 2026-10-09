from __future__ import annotations

import logging
import os
from pathlib import Path

from agent.control.context import running_turn_id
from agent.process_runtime import ExecutionResult
from core.common.diagnostic_log import log_event

logger = logging.getLogger(__name__)

_PLUGIN_ROLLOUT_OWNER_TURN_ENV = "AKASHIC_PLUGIN_ROLLOUT_OWNER_TURN"
_PLUGIN_ROLLOUT_CAPABILITY_ENV = "AKASHIC_PLUGIN_ROLLOUT_CAPABILITY"
_UNIFIED_EXEC_ENV = {
    "NO_COLOR": "1",
    "TERM": "dumb",
    "LANG": "C.UTF-8",
    "LC_CTYPE": "C.UTF-8",
    "LC_ALL": "C.UTF-8",
    "COLORTERM": "",
    "PAGER": "cat",
    "GIT_PAGER": "cat",
    "GH_PAGER": "cat",
}


def _execution_outcome(result: ExecutionResult) -> str:
    if result.execution_id is not None:
        return "running"
    if result.finish_reason == "timeout":
        return "timed_out"
    return "succeeded" if result.exit_code == 0 else "failed"


def _log_shell_execution(
    event: str,
    *,
    operation_id: str,
    description: str,
    command_fp: str,
    command_bytes: int,
    cwd: str,
    shell_kind: str,
    login: bool,
    tty: bool,
    session: str,
    result: ExecutionResult | None = None,
) -> None:
    """Emit one bounded Shell lifecycle event without command text."""

    if result is None:
        log_event(
            logger,
            logging.INFO,
            event,
            operation_id=operation_id,
            description=description,
            command_fp=command_fp,
            command_bytes=command_bytes,
            cwd=cwd,
            shell_kind=shell_kind,
            login=login,
            tty=tty,
            session=session,
        )
        return
    log_event(
        logger,
        logging.INFO,
        event,
        operation_id=operation_id,
        description=description,
        command_fp=command_fp,
        command_bytes=command_bytes,
        cwd=cwd,
        shell_kind=shell_kind,
        login=login,
        tty=tty,
        session=session,
        duration_ms=result.wall_time_ms,
        execution_id=result.execution_id,
        exit_code=result.exit_code,
        finish_reason=result.finish_reason,
        outcome=_execution_outcome(result),
        output_bytes=len(result.output),
        output_omitted_bytes=result.output_omitted_bytes,
    )


def _shell_env() -> dict[str, str]:
    env = os.environ.copy()
    env.pop(_PLUGIN_ROLLOUT_CAPABILITY_ENV, None)
    turn_id = running_turn_id.get()
    if turn_id:
        env[_PLUGIN_ROLLOUT_OWNER_TURN_ENV] = turn_id
    else:
        env.pop(_PLUGIN_ROLLOUT_OWNER_TURN_ENV, None)
    _prepend_existing_path_entries(env, _discover_user_path_entries(env))
    env.update(_UNIFIED_EXEC_ENV)
    return env


_user_path_cache: tuple[tuple[str, str, str | None, int | None], tuple[Path, ...]] | None = None


def _discover_user_path_entries(env: dict[str, str]) -> list[Path]:
    """nvm 版本目录枚举按 (HOME, NVM_DIR, NVM_BIN, node_root mtime) 缓存；安装新版本后 mtime 变化自动失效。"""
    home_text = env.get("HOME")
    if not home_text:
        return []
    home = Path(home_text).expanduser()
    nvm_dir = Path(env.get("NVM_DIR") or home / ".nvm").expanduser()
    nvm_bin = env.get("NVM_BIN")
    node_root = nvm_dir / "versions" / "node"
    try:
        stamp: int | None = node_root.stat().st_mtime_ns
    except OSError:
        stamp = None
    key = (home_text, str(nvm_dir), nvm_bin, stamp)
    global _user_path_cache
    if _user_path_cache is not None and _user_path_cache[0] == key:
        return list(_user_path_cache[1])
    entries = [home / ".local" / "bin"]
    if nvm_bin:
        entries.append(Path(nvm_bin).expanduser())
    entries.extend(_discover_nvm_node_bins(nvm_dir))
    _user_path_cache = (key, tuple(entries))
    return entries


def _discover_nvm_node_bins(nvm_dir: Path) -> list[Path]:
    node_root = nvm_dir / "versions" / "node"
    try:
        version_dirs = [path for path in node_root.iterdir() if path.is_dir()]
    except OSError:
        return []
    return [
        version_dir / "bin"
        for version_dir in sorted(
            version_dirs,
            key=lambda path: _node_version_key(path.name),
            reverse=True,
        )
        if (version_dir / "bin").is_dir()
    ]


def _node_version_key(version: str) -> tuple[int, int, int]:
    parts = version.removeprefix("v").split(".")
    numbers = [int(part) if part.isdigit() else 0 for part in parts[:3]]
    numbers.extend([0] * (3 - len(numbers)))
    return (numbers[0], numbers[1], numbers[2])


def _prepend_existing_path_entries(env: dict[str, str], entries: list[Path]) -> None:
    current = [path for path in env.get("PATH", "").split(os.pathsep) if path]
    seen = set(current)
    prepend: list[str] = []
    for entry in entries:
        text = str(entry)
        if text in seen or not entry.is_dir():
            continue
        prepend.append(text)
        seen.add(text)
    env["PATH"] = os.pathsep.join([*prepend, *current])

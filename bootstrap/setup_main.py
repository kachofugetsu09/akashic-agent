from __future__ import annotations

import json
import os
import re
import tempfile
import tomllib
from datetime import datetime, timezone
from pathlib import Path

import click

from agent.config import Config
from agent.model_runtime.auth import Credential, CredentialStore
from bootstrap.setup_wizard import (
    WizardAnswers,
    _atomic_write_with_backup,
    _phase_main_llm,
)

_HEADER_RE = re.compile(r"^\s*\[([^\[\]]+)\]\s*(?:#.*)?$")
_MANAGED_RUNTIME_KEYS = {
    "provider",
    "auth",
    "model",
    "base_url",
    "reasoning_effort",
    "enable_thinking",
    "context_window",
    "max_output_tokens",
    "input_modalities",
}


def run_main_model_setup(config_path: Path) -> None:
    """交互式切换主模型，同时保留其余配置和 runtime。"""

    # 1. 只收集主模型，不进入频道、插件、memory 或其他角色流程。
    if not config_path.is_file():
        raise click.ClickException(f"配置文件不存在: {config_path}")
    click.echo(click.style("\n══ akashic 主模型切换 ══\n", bold=True))
    answers = WizardAnswers()
    _phase_main_llm(
        answers,
        configure_vl=False,
        prompt_memory_window=False,
        reuse_codex_auth=True,
    )
    if answers.provider != "codex":
        _persist_main_api_key(answers)

    # 2. 在内存中定点更新并先完成 TOML/schema 验证。
    original = config_path.read_text(encoding="utf-8")
    updated = patch_main_model_config(original, answers)
    _validate_candidate(config_path, updated)

    # 3. 明确备份后原子替换，不触碰任何外部配置文件。
    mode = config_path.stat().st_mode & 0o777
    _atomic_write_with_backup(
        config_path,
        updated,
        mode=mode,
        backup_name=f"{config_path.name}.before-setup-main.bak",
    )
    click.echo(f"主模型已更新，备份位于 {config_path}.before-setup-main.bak")


def patch_main_model_config(original: str, answers: WizardAnswers) -> str:
    """只更新主角色、稳定主 runtime 和主上下文窗口。"""

    # 1. 旧版 llm.main table 与新版 main 字符串冲突，原文归档为注释。
    text = _archive_table(original, "llm.main")
    runtime_id = "codex_main" if answers.provider == "codex" else "api_main"

    # 2. 更新稳定 runtime；未管理字段和其他 table 原样保留。
    modalities = ["text", "image"] if answers.multimodal else ["text"]
    values = {
        "provider": _toml_string(answers.provider),
        "auth": _toml_string(answers.auth_id),
        "model": _toml_string(answers.model),
        "base_url": _toml_string(answers.base_url),
        "context_window": str(answers.context_window),
        "max_output_tokens": str(answers.max_output_tokens),
        "input_modalities": json.dumps(modalities, ensure_ascii=False),
    }
    if answers.reasoning_effort:
        values["reasoning_effort"] = _toml_string(answers.reasoning_effort)
    if answers.enable_thinking:
        values["enable_thinking"] = "true"
    text = _replace_table_keys(
        text,
        f"llm.runtimes.{runtime_id}",
        values,
        _MANAGED_RUNTIME_KEYS,
    )

    # 3. 角色只切 main；历史窗口根据新上下文自动更新。
    text = _upsert_table_key(text, "llm", "main", _toml_string(runtime_id))
    text = _upsert_table_key(
        text,
        "agent.context",
        "memory_window",
        str(answers.memory_window),
    )
    return text


def _persist_main_api_key(answers: WizardAnswers) -> None:
    if not answers.auth_id or not answers.api_key:
        raise click.BadParameter("主模型 API key 不能为空")
    CredentialStore().put(
        answers.auth_id,
        Credential(
            driver="api_key",
            access_token=answers.api_key,
            updated_at=datetime.now(timezone.utc).isoformat(),
        ),
    )


def _validate_candidate(config_path: Path, content: str) -> None:
    """在替换正式配置前验证 TOML 和完整 Config schema。"""
    _ = tomllib.loads(content)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{config_path.name}.setup-main-",
        suffix=".toml",
        dir=config_path.parent,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        _ = Config.load(temp_name)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def _archive_table(text: str, table: str) -> str:
    lines = text.splitlines(keepends=True)
    span = _find_table(lines, table)
    if span is None:
        return text
    start, end = span
    archived = [
        line if not line.strip() else f"# setup-main archived: {line}"
        for line in lines[start:end]
    ]
    return "".join([*lines[:start], *archived, *lines[end:]])


def _replace_table_keys(
    text: str,
    table: str,
    values: dict[str, str],
    managed_keys: set[str],
) -> str:
    lines = text.splitlines(keepends=True)
    newline = _newline(text)
    span = _find_table(lines, table)
    rendered = [f"{key} = {value}{newline}" for key, value in values.items()]
    if span is None:
        return _append_table(text, table, rendered)
    start, end = span
    kept = [
        line
        for line in lines[start + 1 : end]
        if _assignment_key(line) not in managed_keys
    ]
    return "".join([*lines[: start + 1], *rendered, *kept, *lines[end:]])


def _upsert_table_key(text: str, table: str, key: str, value: str) -> str:
    lines = text.splitlines(keepends=True)
    newline = _newline(text)
    span = _find_table(lines, table)
    if span is None:
        return _insert_missing_table(text, table, [f"{key} = {value}{newline}"])
    start, end = span
    for index in range(start + 1, end):
        if _assignment_key(lines[index]) == key:
            indent = lines[index][: len(lines[index]) - len(lines[index].lstrip())]
            comment = _inline_comment(lines[index].rstrip("\r\n"))
            lines[index] = f"{indent}{key} = {value}{comment}{newline}"
            return "".join(lines)
    lines.insert(start + 1, f"{key} = {value}{newline}")
    return "".join(lines)


def _find_table(lines: list[str], table: str) -> tuple[int, int] | None:
    for index, line in enumerate(lines):
        match = _HEADER_RE.match(line.rstrip("\r\n"))
        if match is None or match.group(1).strip() != table:
            continue
        end = index + 1
        while end < len(lines) and _HEADER_RE.match(lines[end].rstrip("\r\n")) is None:
            end += 1
        return index, end
    return None


def _insert_missing_table(text: str, table: str, body: list[str]) -> str:
    if table == "llm":
        lines = text.splitlines(keepends=True)
        for index, line in enumerate(lines):
            match = _HEADER_RE.match(line.rstrip("\r\n"))
            if match is not None and match.group(1).strip().startswith("llm."):
                newline = _newline(text)
                block = [f"[llm]{newline}", *body, newline]
                return "".join([*lines[:index], *block, *lines[index:]])
    return _append_table(text, table, body)


def _append_table(text: str, table: str, body: list[str]) -> str:
    newline = _newline(text)
    prefix = text
    if prefix and not prefix.endswith(("\n", "\r")):
        prefix += newline
    if prefix and not prefix.endswith(newline * 2):
        prefix += newline
    return f"{prefix}[{table}]{newline}{''.join(body)}"


def _assignment_key(line: str) -> str | None:
    match = re.match(r"^\s*([A-Za-z0-9_-]+)\s*=", line)
    return match.group(1) if match is not None else None


def _inline_comment(line: str) -> str:
    quoted = False
    escaped = False
    for index, char in enumerate(line):
        if escaped:
            escaped = False
            continue
        if char == "\\" and quoted:
            escaped = True
            continue
        if char == '"':
            quoted = not quoted
            continue
        if char == "#" and not quoted:
            return " " + line[index:].lstrip()
    return ""


def _newline(text: str) -> str:
    return "\r\n" if "\r\n" in text else "\n"


def _toml_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


__all__ = ["patch_main_model_config", "run_main_model_setup"]

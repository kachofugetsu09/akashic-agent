from __future__ import annotations

import os
import tempfile
import tomllib
from collections.abc import MutableMapping
from pathlib import Path
from typing import Any

import click
import tomlkit

from agent.config import Config
from bootstrap.setup_wizard import WizardAnswers, _atomic_write_with_backup, _phase_main_llm

_MANAGED_KEYS = {
    "provider",
    "auth",
    "api_key",
    "model",
    "base_url",
    "reasoning_effort",
    "enable_thinking",
    "context_window",
    "max_context_window",
    "effective_context_percent",
    "max_output_tokens",
    "input_modalities",
    "use_responses_lite",
    "supports_parallel_tool_calls",
    "reasoning_summary",
}


def run_main_model_setup(config_path: Path, workspace: Path) -> None:
    """交互式切换主模型，并原子保留其余 TOML 配置。"""

    # 1. 只收集主模型答案和对应凭据。
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
    # 2. 由保留注释的 TOML 文档模型定点更新并验证。
    updated = patch_main_model_config(
        config_path.read_text(encoding="utf-8"), answers
    )
    _validate_candidate(config_path, updated, workspace)

    # 3. 明确备份后原子替换，不触碰其他配置文件。
    _atomic_write_with_backup(
        config_path,
        updated,
        mode=config_path.stat().st_mode & 0o777,
        backup_name=f"{config_path.name}.before-setup-main.bak",
    )
    click.echo(f"主模型已更新，备份位于 {config_path}.before-setup-main.bak")


def patch_main_model_config(original: str, answers: WizardAnswers) -> str:
    """只替换主 runtime 和自动推导的历史窗口。"""
    document = tomlkit.parse(original)
    llm = _table(document, "llm")
    runtimes = _table(llm, "runtimes")
    runtime_id = _main_runtime_id(answers.provider)
    runtime = _table(runtimes, runtime_id)
    existing_api_key = str(runtime.get("api_key") or "")

    # 1. 清除旧后端字段，避免切换认证后残留明文或不兼容参数。
    for key in _MANAGED_KEYS:
        runtime.pop(key, None)
    values: dict[str, object] = {
        "provider": answers.provider,
        "model": answers.model,
        "base_url": answers.base_url,
        "context_window": answers.context_window,
        "effective_context_percent": answers.effective_context_percent,
        "max_output_tokens": answers.max_output_tokens,
        "input_modalities": ["text", "image"] if answers.multimodal else ["text"],
    }
    if answers.provider == "codex":
        values["auth"] = answers.auth_id
    else:
        api_key = answers.api_key or existing_api_key
        if not api_key:
            raise ValueError(f"provider {answers.provider!r} 缺少 API key")
        values["api_key"] = api_key
    runtime.update(values)
    if answers.reasoning_effort:
        runtime["reasoning_effort"] = answers.reasoning_effort
    if answers.enable_thinking:
        runtime["enable_thinking"] = True
    if answers.use_responses_lite:
        runtime["use_responses_lite"] = True
    if not answers.supports_parallel_tool_calls:
        runtime["supports_parallel_tool_calls"] = False
    if answers.reasoning_summary != "none":
        runtime["reasoning_summary"] = answers.reasoning_summary

    # 2. 角色只切 main；旧 inline table 被文档模型直接替换。
    llm["main"] = runtime_id
    _table(_table(document, "agent"), "context")["memory_window"] = (
        answers.memory_window
    )
    return tomlkit.dumps(document)


def _main_runtime_id(provider: str) -> str:
    """为内建 Provider 生成稳定、可并存的主 runtime ID。"""
    normalized = provider.strip().lower().replace("-", "_")
    if not normalized or not normalized.replace("_", "").isalnum():
        raise ValueError(f"provider 无法生成 runtime ID: {provider!r}")
    return f"{normalized}_main"


def _table(
    parent: MutableMapping[str, Any], key: str
) -> MutableMapping[str, Any]:
    value = parent.get(key)
    if isinstance(value, MutableMapping):
        return value
    table = tomlkit.table()
    parent[key] = table
    return table


def _validate_candidate(config_path: Path, content: str, workspace: Path) -> None:
    """在正式替换前验证 TOML 与完整配置边界。"""
    tomllib.loads(content)
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
        Config.load(temp_name, workspace=workspace)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)

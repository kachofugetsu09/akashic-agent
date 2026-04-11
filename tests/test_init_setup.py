from __future__ import annotations

from pathlib import Path

from agent.config import load_config
from bootstrap.init_setup import (
    build_default_config_dict,
    initialize_workspace,
    run_init,
    setup_telegram_channel,
)


def test_build_default_config_minimal_openai() -> None:
    config = build_default_config_dict(
        provider="openai",
        model="gpt-4o-mini",
        api_key="${OPENAI_API_KEY}",
        enable_memory_v2=True,
        embed_model="text-embedding-v3",
        enable_proactive=False,
    )
    assert config["provider"] == "openai"
    assert config["model"] == "gpt-4o-mini"
    assert "base_url" not in config
    assert config["memory_v2"] == {
        "enabled": True,
        "embed_model": "text-embedding-v3",
    }


def test_build_default_config_supports_separate_light_and_embed_urls() -> None:
    config = build_default_config_dict(
        provider="qwen",
        model="qwen3.5-plus",
        api_key="${QWEN_API_KEY}",
        base_url="https://coding.dashscope.aliyuncs.com/v1",
        light_model="qwen-flash",
        light_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        enable_memory_v2=True,
        embed_model="text-embedding-v3",
        embed_base_url="https://embedding.example.com/v1",
    )
    assert config["light_model"] == "qwen-flash"
    assert config["light_base_url"] == "https://dashscope.aliyuncs.com/compatible-mode/v1"
    assert config["memory_v2"]["embed_model"] == "text-embedding-v3"
    assert config["memory_v2"]["base_url"] == "https://embedding.example.com/v1"


def test_build_default_config_custom_provider_requires_base_url() -> None:
    try:
        build_default_config_dict(
            provider="custom",
            model="local-model",
            api_key="",
            base_url="",
        )
    except ValueError as exc:
        assert "base_url" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_build_default_config_requires_embed_model_when_memory_v2_enabled() -> None:
    try:
        build_default_config_dict(
            provider="openai",
            model="gpt-4o-mini",
            api_key="k",
            enable_memory_v2=True,
            embed_model="",
        )
    except ValueError as exc:
        assert "embed_model" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_initialize_workspace_creates_required_artifacts(tmp_path: Path) -> None:
    created = initialize_workspace(
        tmp_path,
        enable_memory_v2=True,
        enable_proactive=True,
    )
    created_set = {path.name for path in created}
    assert "sessions.db" in created_set
    assert "memory2.db" in created_set
    assert "observe.db" in created_set
    assert "proactive_state.json" in created_set
    assert "proactive_quota.json" in created_set
    assert "PROACTIVE_CONTEXT.md" in created_set
    assert (tmp_path / "memory" / "NOW.md").exists()
    assert (tmp_path / "mcp_servers.json").exists()
    assert (tmp_path / "schedules.json").exists()
    assert (tmp_path / "memes" / "manifest.json").exists()


def test_run_init_generates_loadable_config_and_workspace(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    workspace = tmp_path / "workspace"
    answers = iter(
        [
            "openai",
            "gpt-4o-mini",
            "${OPENAI_API_KEY}",
            "",
            "",
            "y",
            "text-embedding-v3",
            "",
            "",
            "y",
            "telegram",
            "123456",
        ]
    )
    printed: list[str] = []

    run_init(
        config_path=config_path,
        workspace=workspace,
        input_fn=lambda _prompt: next(answers),
        output_fn=printed.append,
    )

    cfg = load_config(config_path)
    assert cfg.provider == "openai"
    assert cfg.model == "gpt-4o-mini"
    assert cfg.memory_v2.enabled is True
    assert cfg.memory_v2.embed_model == "text-embedding-v3"
    assert cfg.proactive.enabled is True
    assert cfg.proactive.default_chat_id == "123456"
    assert (workspace / "sessions.db").exists()
    assert (workspace / "memory" / "MEMORY.md").exists()
    assert (workspace / "PROACTIVE_CONTEXT.md").exists()
    assert (workspace / "proactive_quota.json").exists()
    assert (workspace / "memes" / "manifest.json").exists()
    assert any("已生成配置" in line for line in printed)


def test_setup_telegram_channel_updates_existing_config(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        """
{
  "provider": "openai",
  "model": "gpt-4o-mini",
  "api_key": "${OPENAI_API_KEY}",
  "system_prompt": "You are a helpful assistant.",
  "proactive": {
    "enabled": true,
    "preset": "daily"
  }
}
""".strip()
        + "\n",
        encoding="utf-8",
    )
    answers = iter(
        [
            "${TELEGRAM_BOT_TOKEN}",
            "alice,bob",
            "y",
            "424242",
        ]
    )
    printed: list[str] = []

    data = setup_telegram_channel(
        config_path=config_path,
        input_fn=lambda _prompt: next(answers),
        output_fn=printed.append,
    )

    assert data["channels"]["telegram"]["token"] == "${TELEGRAM_BOT_TOKEN}"
    assert data["channels"]["telegram"]["allowFrom"] == ["alice", "bob"]
    assert data["proactive"]["default_channel"] == "telegram"
    assert data["proactive"]["default_chat_id"] == "424242"
    assert any("Telegram" in line for line in printed)

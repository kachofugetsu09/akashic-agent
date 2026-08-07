from __future__ import annotations

from pathlib import Path

import pytest

from agent.config import load_config

_BASE = """
[llm]
main = "test_main"

[llm.runtimes.test_main]
provider = "openai"
model = "test-model"
api_key = "test-key"
context_window = 64000

[agent]
system_prompt = "test"

[channels.chat]
enabled = true
channel_name = "web"
"""


def _write_config(tmp_path: Path, mobile: str) -> Path:
    path = tmp_path / "config.toml"
    path.write_text(_BASE + mobile, encoding="utf-8")
    return path


def test_mobile_realtime_defaults_to_disabled(tmp_path: Path) -> None:
    config = load_config(_write_config(tmp_path, ""), workspace=tmp_path)

    assert config.mobile_realtime.enabled is False
    assert config.mobile_realtime.port == 6323
    assert str(config.mobile_realtime.key_encryption.keyset_manifest) == (
        "data/mobile/keys/current.json"
    )


def test_mobile_realtime_loads_strict_wss_and_key_encryption(tmp_path: Path) -> None:
    config = load_config(
        _write_config(
            tmp_path,
            """
[mobile_realtime]
enabled = true
host = "0.0.0.0"
port = 6323
database = "data/mobile.db"
lan_hostname = "agent.local"
public_url = "wss://agent.example.com/ws"
max_attachment_mb = 64
inbox_retention_days = 9

[mobile_realtime.key_encryption]
provider = "secret_service"
master_key_namespace = "akasic/mobile-test"
keyset_manifest = "data/mobile/keys/current.json"
            """,
        ),
        workspace=tmp_path,
    )

    assert config.mobile_realtime.enabled is True
    assert config.mobile_realtime.public_url == "wss://agent.example.com/ws"
    assert config.mobile_realtime.max_attachment_mb == 64
    assert config.mobile_realtime.inbox_retention.days == 9


@pytest.mark.parametrize(
    ("mobile", "message"),
    [
        (
            """
[mobile_realtime]
enabled = true
public_url = "ws://agent.example.com/ws"
""",
            "public_url",
        ),
        (
            """
[mobile_realtime]
enabled = true
database = "../outside.db"
""",
            "安全相对路径",
        ),
        (
            """
[mobile_realtime]
enabled = true
[mobile_realtime.key_encryption]
provider = "plaintext"
""",
            "只支持 secret_service",
        ),
    ],
)
def test_mobile_realtime_rejects_unsafe_configuration(
    tmp_path: Path,
    mobile: str,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        load_config(_write_config(tmp_path, mobile), workspace=tmp_path)


def test_mobile_realtime_requires_enabled_webchat_pairing_entry(
    tmp_path: Path,
) -> None:
    config = _BASE.replace(
        "[channels.chat]\nenabled = true",
        "[channels.chat]\nenabled = false",
    )
    path = tmp_path / "config.toml"
    path.write_text(
        config + """
[mobile_realtime]
enabled = true
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="配对入口"):
        load_config(path, workspace=tmp_path)

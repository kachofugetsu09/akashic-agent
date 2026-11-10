"""只复制旧控制配置；监听 owner 切换前保留完整源文件。"""
from __future__ import annotations

import tomllib

from yoyo import step

from agent.migrations.context import current_migration_context
from agent.plugin_composition.config_input import CONFIG_INPUT, load_config, save_config
from .helpers.settings import GatewayConfig

__depends__ = set()
__transactional__ = False


def copy_config(connection: object) -> None:
    """写入唯一 bundle owner 的固定输入；冲突失败，不覆盖用户选择。"""
    context = current_migration_context()
    source = context.config_path
    if source.is_symlink() or not source.is_file():
        raise ValueError("Gateway 配置源必须是普通文件")
    document = tomllib.loads(source.read_text(encoding="utf-8"))
    raw = document.get("app_server")
    if raw is None:
        return
    if not isinstance(raw, dict):
        raise ValueError("app_server 必须是 TOML table")
    expected = GatewayConfig.from_legacy(raw).model_dump()
    data_root = context.bundle_data_roots["gateway_config"]
    current, _ = load_config(data_root)
    if (data_root / CONFIG_INPUT).exists():
        if GatewayConfig.model_validate(current).model_dump() != expected:
            raise ValueError("Gateway 固定配置与旧 app_server 不同，拒绝覆盖")
        return
    save_config(data_root, expected)


steps = [step(copy_config)]

"""退役配置键：本地配置输入不得再携带已删除的字段。

Akasha 4.1.0 删除了 `index_path` 与 `frozen_history_path`，而 Config 是
`extra="forbid"`。已安装 workspace 的 `config.input.json` 仍带旧键时，插件会在
apply 阶段抛 ValidationError，组合拓扑无法就绪，插件更新也就无法发布。
本迁移只改自己这个 bundle 的数据根，删除退役键并保留可恢复备份。
"""
from __future__ import annotations

import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from yoyo import step

from agent.migrations.context import current_migration_context

__depends__ = {"20260918_03_register_replay_without_memory_config"}
__transactional__ = False

_BUNDLE_ID = "akasha"
_CONFIG_FILE = "config.input.json"
_RETIRED_KEYS = ("index_path", "frozen_history_path")


def _retire_config_keys(_connection: object) -> None:
    """从本地配置输入删除退役字段；缺文件或已清理时保持原样。"""

    context = current_migration_context()
    data_root = context.bundle_data_roots[_BUNDLE_ID]
    config_path = data_root / _CONFIG_FILE
    if not config_path.is_file():
        return
    document = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"Akasha 配置输入必须是 object: {config_path}")
    raw = document.get("config")
    if not isinstance(raw, list) or len(raw) != 2 or not isinstance(raw[1], dict):
        raise ValueError(f"Akasha 配置输入结构无效: {config_path}")
    values = dict(raw[1])
    removed = [key for key in _RETIRED_KEYS if key in values]
    if not removed:
        return

    # 1. 先留可读备份，再原子替换；失败不改变原文件。
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    backup = config_path.with_name(f"{config_path.name}.pre-retire-keys-{stamp}")
    backup.write_bytes(config_path.read_bytes())
    for key in removed:
        values.pop(key)
    document["config"] = [raw[0], values]
    descriptor, name = tempfile.mkstemp(prefix=".config-retire-", dir=data_root)
    os.close(descriptor)
    temporary = Path(name)
    temporary.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, config_path)


steps = [step(_retire_config_keys)]

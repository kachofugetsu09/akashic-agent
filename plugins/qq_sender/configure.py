"""显式配置 Sender；明文只传给私有凭据写入边界。"""
from __future__ import annotations

import argparse
import getpass
import os
from pathlib import Path
import tomllib

from agent.plugin_composition.config_input import save_config, save_credential, upgrade_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upgrade", action="store_true")
    args = parser.parse_args()
    data_dir = Path(os.environ["AKASHIC_PLUGIN_DATA_DIR"])
    if args.upgrade:
        backup = upgrade_config(data_dir, lambda content: _convert(data_dir, tomllib.loads(content.decode("utf-8"))))
        print(f"配置已升级；恢复点：{backup}")
        return
    enabled = input("启用此 Sender？[y/N] ").strip().lower() == "y"
    values: dict[str, object] = {"enabled": enabled}
    if enabled:
        values["token"] = getpass.getpass("Token（QQ 可留空）: ")
        values["endpoint"] = input("OneBot WS API endpoint: ").strip()
    save_config(data_dir, _convert(data_dir, values))


def _convert(data_dir: Path, values: dict[str, object]) -> dict[str, object]:
    token = values.pop("token", None)
    if token is not None and not isinstance(token, str):
        raise ValueError("token 必须是字符串")
    if token:
        values["token"] = save_credential(data_dir, token)
    return values


if __name__ == "__main__":
    main()

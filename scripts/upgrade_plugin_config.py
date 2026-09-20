#!/usr/bin/env python3
"""离线升级一个明确不含凭据的插件配置；含秘密的配置由所属 configure.py 升级。"""
from __future__ import annotations

import argparse
from pathlib import Path
import tomllib

from agent.plugin_composition.config_input import upgrade_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--no-secrets", action="store_true", required=True,
                        help="操作者确认此插件旧配置没有明文凭据")
    args = parser.parse_args()
    backup = upgrade_config(args.data_dir.resolve(), lambda content: tomllib.loads(content.decode("utf-8")))
    print(f"配置已升级；恢复点：{backup}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import os
from pathlib import Path
import tomllib
from typing import cast

import toml


def _section(data: dict[str, object], name: str) -> dict[str, object]:
    value = data.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"配置缺少 [{name}] section")
    return cast(dict[str, object], value)


def prepare_config(source: Path, target: Path, public_url: str, lan_hostname: str) -> None:
    """从真实模型配置生成无外发能力的隔离 Mobile Lab 配置。"""

    # 1. 读取真实 provider 配置，但固定所有运行数据到 sandbox
    data = cast(dict[str, object], tomllib.loads(source.read_text(encoding="utf-8")))
    _section(data, "runtime")["workspace"] = "/sandbox/workspace"

    # 2. 关闭外部频道和主动任务，只保留本机配对页面
    _ = _section(data, "channels")
    data["channels"] = {
        "chat": {
            "enabled": True,
            "host": "127.0.0.1",
            "port": 6322,
            "channel_name": "web-lab",
        }
    }
    proactive = _section(data, "proactive")
    proactive["enabled"] = False
    drift = proactive.get("drift")
    if isinstance(drift, dict):
        drift["enabled"] = False
    agent = _section(data, "agent")
    maintenance = agent.setdefault("maintenance", {})
    if not isinstance(maintenance, dict):
        raise ValueError("配置 [agent.maintenance] 不是对象")
    maintenance["memory_optimizer_enabled"] = False

    # 3. 使用独立端口、数据库、密钥命名空间和公网 Tunnel
    mobile = _section(data, "mobile_realtime")
    mobile.update(
        {
            "enabled": True,
            "host": "0.0.0.0",
            "port": 16323,
            "database": "data/mobile_realtime.db",
            "lan_hostname": lan_hostname,
            "public_url": public_url,
        }
    )
    key_encryption = mobile.get("key_encryption")
    if not isinstance(key_encryption, dict):
        raise ValueError("配置缺少 [mobile_realtime.key_encryption] section")
    key_encryption["master_key_namespace"] = "akasic/mobile-realtime-lab"
    key_encryption["keyset_manifest"] = "data/mobile/keys/current.json"

    # 4. 原子替换容器内配置，并限制其中 provider secret 的权限
    target.parent.mkdir(parents=True, exist_ok=True)
    pending = target.with_suffix(".toml.pending")
    _ = pending.write_text(toml.dumps(data), encoding="utf-8")
    os.chmod(pending, 0o600)
    _ = pending.replace(target)


def main() -> None:
    parser = argparse.ArgumentParser()
    _ = parser.add_argument("--source", type=Path, required=True)
    _ = parser.add_argument("--target", type=Path, required=True)
    _ = parser.add_argument("--public-url", required=True)
    _ = parser.add_argument("--lan-hostname", required=True)
    args = parser.parse_args()
    prepare_config(args.source, args.target, args.public_url, args.lan_hostname)


if __name__ == "__main__":
    main()

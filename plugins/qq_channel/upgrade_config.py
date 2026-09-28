"""离线转换本插件旧配置；不运行首次配置问答。"""
from pathlib import Path
import argparse
import tomllib
from agent.plugin_composition.config_input import save_credential, upgrade_config


def convert(content: bytes, data_dir: Path) -> dict[str, object]:
    values = tomllib.loads(content.decode("utf-8"))
    token = values.pop("token", None)
    if token is not None and not isinstance(token, str):
        raise ValueError("token 必须是字符串")
    if token:
        values["token"] = save_credential(data_dir, token)
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    args = parser.parse_args()
    backup = upgrade_config(args.data_dir, lambda content: convert(content, args.data_dir))
    print(f"配置已升级；恢复点：{backup}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast


def _empty_strings() -> list[str]:
    return []


@dataclass(frozen=True)
class FeishuConfig:
    app_id: str = ""
    app_secret: str = ""
    allow_from: list[str] = field(default_factory=_empty_strings)
    domain: str = "https://open.feishu.cn"


def load_feishu_config(*, plugin_dir: Path | None = None) -> FeishuConfig:
    root = plugin_dir or Path(__file__).resolve().parent
    payload = _read_toml(root / "config.local.toml")
    return FeishuConfig(
        app_id=str(payload.get("app_id") or "").strip(),
        app_secret=str(payload.get("app_secret") or "").strip(),
        allow_from=_string_list(payload.get("allow_from")),
        domain=str(payload.get("domain") or "https://open.feishu.cn").strip(),
    )


def _read_toml(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    return cast(dict[str, object], tomllib.loads(path.read_text(encoding="utf-8")))


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for item in cast(list[object], value):
        text = str(item).strip()
        if text:
            result.append(text)
    return result

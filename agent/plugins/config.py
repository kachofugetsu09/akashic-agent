from __future__ import annotations

from typing import Any


class PluginConfig:
    def __init__(self, values: dict[str, Any]) -> None:
        self._values = dict(values)

    def get(self, key: str, default: Any = None) -> Any:
        return self._values.get(key, default)

    def as_dict(self) -> dict[str, Any]:
        return dict(self._values)

    def __getattr__(self, key: str) -> Any:
        try:
            return self._values[key]
        except KeyError as e:
            raise AttributeError(key) from e

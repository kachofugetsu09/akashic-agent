from __future__ import annotations

import tomllib
from collections.abc import Callable, Mapping
from pathlib import Path

from agent.plugin_composition.channels import CredentialRef, ProviderClient
from agent.plugins.config import read_config_source


class CoreProviderClient:
    """只暴露本次租约取得的凭据；关闭后清空并释放 factory 记录。"""

    def __init__(self, values: Mapping[tuple[str, ...], str], on_close: Callable[[CoreProviderClient], None]) -> None:
        self._values = dict(values)
        self._closed = False
        self._on_close = on_close

    def credential(self, ref: CredentialRef) -> str:
        if self._closed:
            raise RuntimeError("provider credential client 已关闭")
        try:
            return self._values[ref.path]
        except KeyError as error:
            raise RuntimeError("CredentialRef 不属于当前 provider client") from error

    async def aclose(self) -> None:
        self._values.clear()
        self._closed = True
        self._on_close(self)


class CoreProviderClientFactory:
    """按已冻结配置读取声明的凭据，统一拥有 Channel 与普通插件的租约。"""

    def __init__(
        self,
        config_path: Path,
        credential_paths: tuple[str, ...],
        raw_config_revision: str,
    ) -> None:
        self._config_path = config_path
        self._allowed = frozenset(tuple(path.split(".")) for path in credential_paths)
        self._raw_config_revision = raw_config_revision
        self._clients: set[CoreProviderClient] = set()
        self._closed = False

    async def create(
        self,
        credentials: Mapping[str, CredentialRef],
    ) -> ProviderClient:
        """Resolve the requested refs only after the raw-config revision fence."""

        if self._closed:
            raise RuntimeError("provider client factory 已关闭")
        content, revision = read_config_source(self._config_path)
        if revision != self._raw_config_revision:
            raise RuntimeError("plugin credential config revision 已漂移")
        raw = {} if content is None else tomllib.loads(content.decode("utf-8"))
        values: dict[tuple[str, ...], str] = {}
        for name, ref in credentials.items():
            if not isinstance(name, str) or not isinstance(ref, CredentialRef):
                raise TypeError("credentials 必须映射到 CredentialRef")
            if ref.path not in self._allowed or tuple(name.split(".")) != ref.path:
                raise RuntimeError("CredentialRef 不属于 frozen plugin 声明")
            value = _resolve_path(raw, ref.path)
            if not isinstance(value, str) or not value:
                raise RuntimeError(f"plugin credential 必须是非空字符串: {name}")
            values[ref.path] = value
        client = CoreProviderClient(values, self._clients.discard)
        self._clients.add(client)
        return client

    async def aclose(self) -> None:
        """Close every credential lease before releasing the formal binding."""

        for client in tuple(self._clients):
            await client.aclose()
        self._clients.clear()
        self._closed = True


def _resolve_path(value: object, path: tuple[str, ...]) -> object:
    current = value
    for segment in path:
        if not isinstance(current, Mapping) or segment not in current:
            raise RuntimeError(f"plugin credential 不存在: {'.'.join(path)}")
        current = current[segment]
    return current


__all__ = ["CoreProviderClientFactory"]

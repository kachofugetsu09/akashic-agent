from __future__ import annotations

import tomllib
from collections.abc import Mapping
from pathlib import Path

from agent.plugin_composition.channels import CredentialRef, ProviderClient
from agent.plugins.config import read_config_source


class CoreProviderClient:
    """Expose only the credential values leased for one formal channel start."""

    def __init__(self, values: Mapping[tuple[str, ...], str]) -> None:
        self._values = dict(values)
        self._closed = False

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


class CoreProviderClientFactory:
    """Resolve frozen refs from the exact formal config and own every lease."""

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
            raise RuntimeError("channel credential config revision 已漂移")
        raw = {} if content is None else tomllib.loads(content.decode("utf-8"))
        values: dict[tuple[str, ...], str] = {}
        for name, ref in credentials.items():
            if not isinstance(name, str) or not isinstance(ref, CredentialRef):
                raise TypeError("credentials 必须映射到 CredentialRef")
            if ref.path not in self._allowed or tuple(name.split(".")) != ref.path:
                raise RuntimeError("CredentialRef 不属于 frozen channel 声明")
            value = _resolve_path(raw, ref.path)
            if not isinstance(value, str) or not value:
                raise RuntimeError(f"channel credential 必须是非空字符串: {name}")
            values[ref.path] = value
        client = CoreProviderClient(values)
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
            raise RuntimeError(f"channel credential 不存在: {'.'.join(path)}")
        current = current[segment]
    return current


__all__ = ["CoreProviderClientFactory"]

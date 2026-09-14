from __future__ import annotations

import hashlib
import json

from collections.abc import Callable, Mapping
from pathlib import Path

from agent.plugin_composition.channels import CredentialRef, ProviderClient
from agent.plugin_composition.config_input import config_refs, load_config, _credential_path


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
    """只解析固定输入授予的引用，统一拥有 Channel 与普通插件的租约。"""

    def __init__(
        self,
        data_dir: Path,
        config: Mapping[str, object],
        raw_config_revision: str,
    ) -> None:
        self._data_dir = data_dir
        self._allowed = config_refs(config)
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
        _, revision = load_config(self._data_dir)
        if revision != self._raw_config_revision:
            raise RuntimeError("plugin credential config revision 已漂移")
        values: dict[tuple[str, ...], str] = {}
        for name, ref in credentials.items():
            if not isinstance(name, str) or not isinstance(ref, CredentialRef):
                raise TypeError("credentials 必须映射到 CredentialRef")
            if ref not in self._allowed:
                raise RuntimeError("CredentialRef 不属于 frozen plugin 输入")
            value = _read_credential(self._data_dir, ref)
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


def _read_credential(data_dir: Path, ref: CredentialRef) -> str:
    """只供已授权租约读取指定版本，损坏或撤销明确失败。"""
    path = _credential_path(data_dir, ref)
    revoked = path.with_suffix(".revoked")
    if revoked.exists() or revoked.is_symlink():
        raise PermissionError("凭据版本已撤销")
    if path.is_symlink() or path.stat().st_mode & 0o077:
        raise PermissionError("凭据必须是权限 0600 的实际文件")
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != ref.path[1]:
        raise RuntimeError("凭据固定版本内容已漂移")
    raw = json.loads(content)
    if (not isinstance(raw, dict) or set(raw) != {"owner", "id", "value"}
            or raw["owner"] != data_dir.name or raw["id"] != ref.path[0]
            or not isinstance(raw["value"], str) or not raw["value"]):
        raise ValueError("凭据版本 owner 或内容无效")
    return raw["value"]


__all__ = ["CoreProviderClientFactory"]

"""web module 字节经内容寻址端点发布:bootstrap 只携带描述符,摘要围栏拒绝过期请求。"""
import hashlib
import json
from types import SimpleNamespace
from typing import cast

import pytest

from agent.plugin_composition import Context
from plugins.ui.contract import WebModuleAsset, WebModuleDescriptor
from plugins.ui.plugin import Registration, Ui
from plugins.ui.web import build_web_ui_catalog, encode_web_ui_bootstrap

_SOURCE = "export function activate(ctx) {}\n"


def _descriptor(plugin_id: str = "notes-ui@lab", source: str = _SOURCE) -> WebModuleDescriptor:
    return WebModuleDescriptor(
        plugin_id=plugin_id,
        registration_uuid="0" * 32,
        generation_id="generation-1",
        asset=WebModuleAsset(
            module=source,
            module_sha256=hashlib.sha256(source.encode()).hexdigest(),
            module_bytes=len(source.encode()),
            stylesheet="",
            stylesheet_sha256=None,
            stylesheet_bytes=0,
        ),
    )


def test_bootstrap_keeps_inline_bytes_and_descriptors():
    """冷启动网络合同不变:模块字节随 bootstrap 内联,描述符供内容寻址端点核对。"""
    descriptor = _descriptor()
    catalog = build_web_ui_catalog((descriptor,))
    payload = json.loads(encode_web_ui_bootstrap(catalog, "snapshot-1"))
    (module,) = payload["modules"]
    assert module["module"] == _SOURCE
    assert module["moduleSha256"] == descriptor.asset.module_sha256
    assert module["moduleBytes"] == descriptor.asset.module_bytes


class _StubScope:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return False


class _StubContext:
    generation_id = "generation-1"

    def runtime_scope(self):
        return _StubScope()

    def require_runtime_identity(self, *_args):
        return None


def _registry_with(entry: Registration | None) -> Ui:
    registry = Ui(cast(Context, _StubContext()), ())
    if entry is not None:
        registry._entries[entry.web.plugin_id] = entry
    return registry


def _registration(descriptor: WebModuleDescriptor, *, initialized: bool = True,
                  active: bool = True) -> Registration:
    from agent.plugin_composition import FiberState

    contributor = SimpleNamespace(
        fiber=SimpleNamespace(state=FiberState.ACTIVE if active else FiberState.DISPOSED))
    return Registration(ctx=cast(Context, contributor), web=descriptor, resources=None,
                        registration_uuid=descriptor.registration_uuid,
                        initialized=initialized)


@pytest.mark.asyncio
async def test_module_asset_serves_only_current_digest():
    descriptor = _descriptor()
    registry = _registry_with(_registration(descriptor))
    assert await registry.module_asset(descriptor.plugin_id,
                                       descriptor.asset.module_sha256) == _SOURCE
    with pytest.raises(KeyError):
        await registry.module_asset(descriptor.plugin_id, "f" * 64)
    with pytest.raises(KeyError):
        await registry.module_asset("unknown@lab", descriptor.asset.module_sha256)


@pytest.mark.asyncio
async def test_module_asset_rejects_uninitialized_or_inactive():
    descriptor = _descriptor()
    with pytest.raises(KeyError):
        await _registry_with(_registration(descriptor, initialized=False)).module_asset(
            descriptor.plugin_id, descriptor.asset.module_sha256)
    with pytest.raises(KeyError):
        await _registry_with(_registration(descriptor, active=False)).module_asset(
            descriptor.plugin_id, descriptor.asset.module_sha256)

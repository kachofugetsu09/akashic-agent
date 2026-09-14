"""UI 定义、目录封存和 Dashboard 资源由普通插件拥有。"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import ModuleType, FunctionType

from agent.plugin_composition import Context, Effect, SNAPSHOT_SEALING, SnapshotSealing
from agent.plugin_composition.ui import UI, WEB_UI, DashboardBinding, WebModuleDescriptor, WebUiCatalog

from agent.plugin_composition.ui_slots import UI_SLOTS

from .mobile import MobileUiSlots
from .dashboard import DashboardImportError, DashboardResources, _core_routes, _require_routes_available
from .web import freeze_web_ui_catalog, resolve_web_module

api_version = 3
name = "ui"
version = "1.0.0"
desc = "注册并封存插件的 Web 与 Dashboard UI"

logger = logging.getLogger(__name__)


@dataclass
class Registration:
    ctx: Context
    web: WebModuleDescriptor | None
    resources: DashboardResources | None
    effect: Effect | None = None
    binding: DashboardBinding | None = None
    unavailable: bool = False


class Ui:
    """一个 Root 的唯一 UI 注册表；关闭不删除代码或插件数据。"""

    def __init__(self, ctx: Context) -> None:
        self._ctx = ctx
        self._entries: dict[str, Registration] = {}
        self._catalog: WebUiCatalog | None = None

    @property
    def root_instance_token(self) -> object:
        return self._ctx.root_instance_token

    async def register(
        self, ctx: Context, *, web: str | None = None, dashboard: Callable[[], ModuleType] | None = None,
        requires: tuple[str, ...] = (), provides: tuple[str, ...] = (),
        contract_digests: Mapping[str, str] | None = None,
    ) -> Effect:
        """从实际贡献 Context 固定身份和代码路径，再取得注册 Effect。"""
        # 1. provider 只接受同一 Root 的贡献方，不接受自报 owner 或 generation。
        if ctx.root_instance_token is not self._ctx.root_instance_token or ctx.require(UI) is not self:
            raise ValueError("UI 注册不能跨 composition Root")
        if self._catalog is not None:
            raise RuntimeError("UI 目录已封存")
        owner = ctx.runtime.plugin_id
        if owner in self._entries:
            raise ValueError(f"UI owner 重复注册: {owner}")
        if web is None and dashboard is None:
            raise ValueError("UI 注册必须包含 Web 或 Dashboard")
        requires = _contracts(requires)
        provides = _contracts(provides)
        digests = {} if contract_digests is None else dict(contract_digests)
        if any(not isinstance(key, str) or not isinstance(value, str)
               or re.fullmatch(r"[0-9a-f]{64}", value) is None for key, value in digests.items()):
            raise ValueError("Web contract digest 必须是 SHA-256")
        if set(digests) - (set(requires) | set(provides)):
            raise ValueError("Web contract digest 没有对应声明")
        if web is None and (requires or provides or digests):
            raise ValueError("Web contract 必须属于 Web 模块")

        # 2. JS/CSS 与合同由 provider 校验；Dashboard 资源在宿主准备时建立。
        root = ctx.runtime.plugin_dir.resolve(strict=True)
        module = None
        if web is not None:
            _path(root, web, ".js")
            asset = resolve_web_module(root, web, requires=requires, provides=provides,
                                       contract_digests=tuple(sorted(digests.items())))
            assert asset is not None
            module = WebModuleDescriptor(owner, ctx.runtime.generation_id, asset)
        if dashboard is not None:
            if not isinstance(dashboard, FunctionType):
                raise ValueError("Dashboard loader 必须是贡献插件的函数")
            loader_path = Path(dashboard.__code__.co_filename).resolve(strict=True)
            if not loader_path.is_relative_to(root):
                raise ValueError("Dashboard loader 不属于贡献方代码制品")
        resources = None if dashboard is None else DashboardResources(
            ctx, dashboard, has_web=module is not None,
        )
        registration = Registration(ctx, module, resources)

        def setup():
            self._entries[owner] = registration

            async def cleanup() -> None:
                if resources is not None:
                    await resources.aclose()
                del self._entries[owner]

            return cleanup

        registration.effect = await ctx.effect(setup, label="ui")
        return registration.effect

    def seal(self, _event: SnapshotSealing) -> None:
        """只封存本 Root 的实际注册，校验失败不发布目录。"""
        if self._catalog is not None:
            raise RuntimeError("UI 目录不能重复封存")
        self._catalog = freeze_web_ui_catalog(tuple(
            entry.web for _, entry in sorted(self._entries.items()) if entry.web is not None
        ))

    def catalog(self) -> WebUiCatalog:
        if self._catalog is None:
            raise RuntimeError("UI 目录尚未封存")
        return self._catalog

    def bindings(self) -> tuple[DashboardBinding, ...]:
        return tuple(entry.binding for _, entry in sorted(self._entries.items()) if entry.binding is not None)

    def contributors(self) -> tuple[Context, ...]:
        return tuple(entry.ctx for _, entry in sorted(self._entries.items()))

    async def bootstrap(self) -> bytes:
        self._ctx.require_runtime_owner(WEB_UI, self)
        scope = self._ctx.capture_runtime_scope()
        async with scope:
            return self.catalog().encode_bootstrap(scope.snapshot_id)

    async def state(self) -> dict[str, str]:
        self._ctx.require_runtime_owner(WEB_UI, self)
        scope = self._ctx.capture_runtime_scope()
        async with scope:
            return {"snapshotId": scope.snapshot_id, "catalogId": self.catalog().identity}

    def prepare_dashboard(
        self, *, core_routes: tuple[object, ...],
        workload_urls: Callable[[str], Mapping[tuple[str, str], str]],
        validation_owners: frozenset[str], tolerate_failures: bool,
    ) -> None:
        """为本 Root 准备实际 Dashboard，失败资源仍由原注册 Effect 关闭。"""
        self.catalog()
        occupied = list(_core_routes(core_routes))
        for owner, entry in sorted(self._entries.items()):
            resources = entry.resources
            if resources is None or entry.unavailable:
                continue
            if entry.binding is None:
                try:
                    entry.binding = resources.build(
                        occupied=occupied, workload_urls=workload_urls(entry.ctx.runtime.generation_id),
                        validation=owner in validation_owners,
                    )
                except DashboardImportError as error:
                    if not tolerate_failures or entry.web is not None:
                        raise
                    entry.unavailable = True
                    logger.warning("初始插件 dashboard 挂载失败 (%s): %s", owner, error)
                    continue
            else:
                _require_routes_available(entry.binding, occupied)
            occupied.extend(entry.binding.routes)

    async def release_validation(self) -> None:
        """候选晋升前关闭隔离资源，失败仍留在原 Effect。"""
        for entry in reversed(tuple(self._entries.values())):
            if entry.binding is not None and entry.binding.validation:
                assert entry.effect is not None
                await entry.effect.aclose()


def _contracts(value: tuple[str, ...]) -> tuple[str, ...]:
    if not isinstance(value, tuple) or any(not isinstance(item, str) or not item or item.strip() != item for item in value):
        raise ValueError("Web contracts 必须是不重复的非空字符串 tuple")
    if len(value) != len(set(value)):
        raise ValueError("Web contracts 不得重复")
    return value


def _path(root: Path, relative_path: str, suffix: str) -> Path:
    """拒绝绝对路径和跨制品链接，路径只能来自贡献方固定代码。"""
    if not isinstance(relative_path, str) or not relative_path or relative_path.strip() != relative_path:
        raise ValueError("UI 模块必须是相对文件路径")
    relative = PurePosixPath(relative_path)
    if relative.is_absolute() or ".." in relative.parts or "\\" in relative_path:
        raise ValueError("UI 模块路径越过贡献方代码制品")
    path = (root / relative).resolve(strict=True)
    if not path.is_relative_to(root) or path.suffix != suffix or not path.is_file():
        raise ValueError("UI 模块不属于贡献方代码制品或类型不符")
    return path


async def apply(ctx: Context) -> None:
    registry = Ui(ctx)
    await ctx.provide(UI, registry, binding_contributors=registry.contributors)
    await ctx.provide(WEB_UI, registry, binding_contributors=registry.contributors)
    await ctx.on(SNAPSHOT_SEALING, registry.seal)
    mobile = MobileUiSlots(ctx)
    await ctx.provide(UI_SLOTS, mobile, binding_contributors=mobile.contributors)
    await ctx.on(SNAPSHOT_SEALING, mobile.seal)

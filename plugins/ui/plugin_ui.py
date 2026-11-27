"""普通 UI provider 拥有插件界面注册、资源校验与目录封存。"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
from collections.abc import Callable, Mapping
from pathlib import Path
from types import MappingProxyType
from typing import cast
from uuid import uuid4

from agent.plugin_composition import Context, CompositionError, Effect, FiberState
from plugins.ui.contract import (
    UI_SLOTS, PLUGIN_UI_SLOTS, PluginUiAsset, PluginUiAssetFile, PluginUiBinding, PluginUiDefinition,
    PluginUiDescriptor, PluginUiFollowHandler, PluginUiQueryHandler, PluginUiSlot,
)

_MAX_ENTRY_BYTES = 240 * 1024
_MAX_CHUNK_FILE_BYTES = 1024 * 1024
_MAX_CHUNK_FILES = 128
_MAX_CHUNK_TOTAL_BYTES = 12 * 1024 * 1024
_ALLOWED_CHUNK_MEDIA: Mapping[str, str] = MappingProxyType({
    ".js": "text/javascript",
    ".css": "text/css",
})


class PluginUiSlots:
    """Collect the live Plugin UI registrations for one composition Root."""

    def __init__(self, ctx: Context) -> None:
        self._ctx = ctx
        self._registrations: dict[str, PluginUiBinding] = {}
        self._changed = asyncio.Event()
        self._slot_epochs: dict[str, int] = {}
        self._slot_changed = asyncio.Event()

    def contributors(self) -> tuple[Context, ...]:
        self._ctx._require_current()  # pyright: ignore[reportPrivateUsage]
        for binding in self._registrations.values():
            binding.context._require_current()  # pyright: ignore[reportPrivateUsage]
        return tuple(binding.context for binding in self._registrations.values())

    @property
    def root_instance_token(self) -> object:
        self._ctx._require_current()  # pyright: ignore[reportPrivateUsage]
        return self._ctx.root_instance_token

    def changed(self) -> asyncio.Event:
        return self._changed

    def slot_epochs(self) -> Mapping[str, int]:
        return dict(self._slot_epochs)

    def slot_changed(self) -> asyncio.Event:
        return self._slot_changed

    def invalidate_slot(self, ctx: Context, slot: PluginUiSlot) -> None:
        """Record a slot-level invalidation for one registered plugin owner."""
        ctx._require_current()  # pyright: ignore[reportPrivateUsage]
        provider_token = self.root_instance_token
        if ctx.root_instance_token is not provider_token or ctx.require(UI_SLOTS) is not self:
            raise ValueError("Plugin UI 失效通知不能跨实际 Root 或 provider")
        if not isinstance(slot, str) or slot not in PLUGIN_UI_SLOTS:
            raise ValueError(f"Plugin UI slot 无效: {slot}")
        owner = ctx.runtime.plugin_id
        binding = self._registrations.get(owner)
        if binding is None or binding.context is not ctx or slot not in binding.descriptor.slots:
            raise ValueError(f"插件 {owner} 未登记 slot: {slot}")
        self._slot_epochs[slot] = self._slot_epochs.get(slot, 0) + 1
        self._slot_changed.set()
        self._slot_changed = asyncio.Event()

    # 唤醒等待者并换一个新事件，下一次变化重新等待。
    def _notify(self) -> None:
        self._changed.set()
        self._changed = asyncio.Event()

    def bindings(self) -> tuple[PluginUiBinding, ...]:
        """Return the current registration projection without running callbacks."""

        self._ctx._require_current()  # pyright: ignore[reportPrivateUsage]
        for binding in self._registrations.values():
            binding.context._require_current()  # pyright: ignore[reportPrivateUsage]
        return tuple(
            binding for _owner, binding
            in sorted(self._registrations.items())
        )

    async def register_plugin_ui(
        self,
        ctx: Context,
        definition: PluginUiDefinition,
        *,
        query: PluginUiQueryHandler,
        available: Callable[[], bool] | None = None,
        follow: PluginUiFollowHandler | None = None,
    ) -> Effect:
        """Register one plugin UI declaration as an Effect of the calling Fiber."""

        ctx._require_current()  # pyright: ignore[reportPrivateUsage]
        provider_token = self.root_instance_token
        if ctx.root_instance_token is not provider_token:
            raise ValueError("Plugin UI 注册不能跨实际 Root 或 provider")
        if ctx.require(UI_SLOTS) is not self:
            raise ValueError("Plugin UI 注册不能跨实际 Root 或 provider")
        # 1. Validate the public ABI before any registration becomes visible.
        if not isinstance(definition, PluginUiDefinition):
            raise TypeError("插件 Plugin UI 声明必须是 PluginUiDefinition")
        if not callable(query):
            raise TypeError("插件 Plugin UI query 必须可调用")
        if follow is not None and not callable(follow):
            raise TypeError("插件 Plugin UI follow 必须可调用")
        if available is not None:
            _validate_sync_callable(available, "available")
        runtime = ctx.runtime
        if ctx.fiber.state is not FiberState.LOADING and ctx.fiber.state is not FiberState.ACTIVE:
            raise CompositionError(
                "INACTIVE_FIBER",
                f"{runtime.plugin_id} 当前 Fiber 不允许登记 Plugin UI",
            )
        asset = resolve_plugin_ui_asset(
            runtime.plugin_dir,
            module=definition.module,
            stylesheet=definition.stylesheet,
            slots=tuple(definition.slots),
            assets_dir=definition.assets_dir,
        )
        binding = PluginUiBinding(
            descriptor=PluginUiDescriptor(
                owner=runtime.plugin_id,
                module_sha256=asset.module_sha256,
                module_bytes=asset.module_bytes,
                stylesheet_sha256=asset.stylesheet_sha256,
                stylesheet_bytes=asset.stylesheet_bytes,
                slots=asset.slots,
                bundle_sha256=asset.bundle_sha256,
                module_path=asset.module_path,
                stylesheet_path=asset.stylesheet_path,
            ),
            asset=asset,
            query=cast(PluginUiQueryHandler, query),
            available=_always_available if available is None else available,
            context=ctx,
            registration_uuid=uuid4().hex,
            follow=follow,
        )
        undo = await ctx.effect(
            lambda: self._register(ctx, binding),
            label=f"ui-slot:plugin:{definition.module}",
        )
        # LOADING 期间登记的界面在 activation 就绪后才可见；就绪时再通知一次，跟随方据此开流。
        if ctx.fiber.state is FiberState.LOADING:
            _ = await ctx.spawn(self._announce_active(), name="plugin-ui-active")
        return undo

    async def _announce_active(self) -> None:
        self._notify()

    def _register(
        self,
        ctx: Context,
        binding: PluginUiBinding,
    ) -> Callable[[], None]:
        """Add one declaration and return its exact inverse."""

        ctx._require_current()  # pyright: ignore[reportPrivateUsage]
        provider_token = self.root_instance_token
        if ctx.root_instance_token is not provider_token:
            raise CompositionError(
                "STALE_ACTIVATION",
                f"插件 Plugin UI 注册已脱离当前 Composition Root: {ctx.runtime.plugin_id}",
            )
        plugin_id = ctx.runtime.plugin_id
        # 1. One plugin owns one current registration; the Effect is its owner.
        if plugin_id in self._registrations:
            raise CompositionError(
                "DUPLICATE_PLUGIN_UI",
                f"插件只能声明一个 Plugin UI: {plugin_id}",
            )

        # 2. 每个 owner 只持有一条注册，Effect 关闭后解除归属。
        self._registrations[plugin_id] = binding
        self._notify()

        def cleanup() -> None:
            del self._registrations[plugin_id]
            self._notify()

        return cleanup


def resolve_plugin_ui_asset(
    plugin_dir: Path,
    *,
    module: str,
    stylesheet: str | None,
    slots: tuple[str, ...],
    assets_dir: str | None = None,
) -> PluginUiAsset:
    """Validate and freeze plugin-owned plugin UI static assets."""

    # 1. Validate metadata and the protocol slot namespace.
    _validate_plugin_ui_metadata(
        module=module,
        stylesheet=stylesheet,
        slots=slots,
        assets_dir=assets_dir,
    )

    # 2. Resolve symlinks before enforcing plugin-source containment.
    plugin_root = plugin_dir.resolve(strict=True)
    module_path = _resolve_asset_path(plugin_root, module, suffix=".js", kind="module")
    stylesheet_path = (
        None
        if stylesheet is None
        else _resolve_asset_path(
            plugin_root,
            stylesheet,
            suffix=".css",
            kind="stylesheet",
        )
    )
    extra_dir = (
        None
        if assets_dir is None
        else _resolve_assets_dir(plugin_root, assets_dir)
    )
    return _build_plugin_ui_asset(
        plugin_root,
        module_path,
        stylesheet_path,
        extra_dir,
        slots=slots,
    )


def _validate_plugin_ui_metadata(
    *,
    module: str,
    stylesheet: str | None,
    slots: tuple[str, ...],
    assets_dir: str | None = None,
) -> None:
    if (
        not isinstance(module, str)
        or not module
        or module != module.strip()
        or Path(module).is_absolute()
    ):
        raise RuntimeError("插件 UI module 必须是非空相对路径")
    if stylesheet is not None and (
        not isinstance(stylesheet, str)
        or not stylesheet
        or stylesheet != stylesheet.strip()
        or Path(stylesheet).is_absolute()
    ):
        raise RuntimeError("插件 UI stylesheet 必须是非空相对路径")
    if assets_dir is not None and (
        not isinstance(assets_dir, str)
        or not assets_dir
        or assets_dir != assets_dir.strip()
        or Path(assets_dir).is_absolute()
    ):
        raise RuntimeError("插件 UI assets_dir 必须是非空相对路径")
    if not isinstance(slots, tuple) or any(
        not isinstance(slot, str) or slot not in PLUGIN_UI_SLOTS for slot in slots
    ):
        raise RuntimeError("插件 UI slots 无效")
    if len(set(slots)) != len(slots):
        raise RuntimeError("插件 UI slots 无效")


def _build_plugin_ui_asset(
    plugin_root: Path,
    module_path: Path,
    stylesheet_path: Path | None,
    assets_dir: Path | None,
    *,
    slots: tuple[str, ...],
) -> PluginUiAsset:
    """Read validated assets and attach content hashes and byte sizes."""

    module_stat = module_path.stat()
    sheet_stat_size = 0 if stylesheet_path is None else stylesheet_path.stat().st_size
    if module_stat.st_size + sheet_stat_size > _MAX_ENTRY_BYTES:
        raise RuntimeError("插件 UI 资产超过协议安全预算")
    module_content = module_path.read_text(encoding="utf-8")
    stylesheet_content = (
        "" if stylesheet_path is None else stylesheet_path.read_text(encoding="utf-8")
    )
    module_encoded = module_content.encode("utf-8")
    stylesheet_encoded = stylesheet_content.encode("utf-8")
    if len(module_encoded) + len(stylesheet_encoded) > _MAX_ENTRY_BYTES:
        raise RuntimeError("插件 UI 资产超过协议安全预算")

    module_rel = module_path.relative_to(plugin_root).as_posix()
    module_sha256 = hashlib.sha256(module_encoded).hexdigest()
    stylesheet_rel = (
        None
        if stylesheet_path is None
        else stylesheet_path.relative_to(plugin_root).as_posix()
    )
    stylesheet_sha256 = (
        hashlib.sha256(stylesheet_encoded).hexdigest()
        if stylesheet_content
        else None
    )

    files: dict[str, PluginUiAssetFile] = {
        module_rel: PluginUiAssetFile(
            rel_path=module_rel,
            content=module_encoded,
            sha256=module_sha256,
            media_type="text/javascript",
        ),
    }
    if stylesheet_rel is not None and stylesheet_sha256 is not None:
        if stylesheet_rel in files:
            raise RuntimeError(f"插件 UI 资产路径冲突: {stylesheet_rel}")
        files[stylesheet_rel] = PluginUiAssetFile(
            rel_path=stylesheet_rel,
            content=stylesheet_encoded,
            sha256=stylesheet_sha256,
            media_type="text/css",
        )

    if assets_dir is not None:
        chunk_count = 0
        chunk_total = 0
        for candidate in sorted(assets_dir.rglob("*")):
            if candidate.is_dir():
                resolved_dir = candidate.resolve(strict=True)
                if not resolved_dir.is_relative_to(assets_dir):
                    raise RuntimeError(f"插件 UI assets_dir 包含越界目录: {candidate}")
                continue
            resolved = candidate.resolve(strict=True)
            if not resolved.is_relative_to(assets_dir) or not resolved.is_file():
                raise RuntimeError(f"插件 UI assets_dir 包含非法文件: {candidate}")
            media_type = _ALLOWED_CHUNK_MEDIA.get(resolved.suffix)
            if media_type is None:
                raise RuntimeError(f"插件 UI assets_dir 不支持的文件类型: {resolved.name}")
            size = resolved.stat().st_size
            if size > _MAX_CHUNK_FILE_BYTES:
                raise RuntimeError(f"插件 UI 分块文件超过单文件上限: {resolved.name}")
            chunk_count += 1
            chunk_total += size
            if chunk_count > _MAX_CHUNK_FILES or chunk_total > _MAX_CHUNK_TOTAL_BYTES:
                raise RuntimeError("插件 UI 分块资产超过协议安全预算")
            rel = resolved.relative_to(plugin_root).as_posix()
            if rel in files:
                raise RuntimeError(f"插件 UI 资产路径重复: {rel}")
            raw = resolved.read_bytes()
            if len(raw) > _MAX_CHUNK_FILE_BYTES:
                raise RuntimeError(f"插件 UI 分块文件超过单文件上限: {resolved.name}")
            files[rel] = PluginUiAssetFile(
                rel_path=rel,
                content=raw,
                sha256=hashlib.sha256(raw).hexdigest(),
                media_type=media_type,
            )

    manifest_encoded = json.dumps(
        {
            "module": module_rel,
            "stylesheet": stylesheet_rel if stylesheet_sha256 is not None else None,
            "files": [[rel, entry.sha256] for rel, entry in sorted(files.items())],
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    bundle_sha256 = hashlib.sha256(manifest_encoded).hexdigest()

    return PluginUiAsset(
        module=module_content,
        module_sha256=module_sha256,
        module_bytes=len(module_encoded),
        stylesheet=stylesheet_content,
        stylesheet_sha256=stylesheet_sha256,
        stylesheet_bytes=len(stylesheet_encoded),
        slots=slots,
        module_path=module_rel,
        stylesheet_path=stylesheet_rel if stylesheet_sha256 is not None else None,
        bundle_sha256=bundle_sha256,
        files=MappingProxyType(files),
    )


def _resolve_assets_dir(plugin_root: Path, relative_path: str) -> Path:
    try:
        path = (plugin_root / relative_path).resolve(strict=True)
    except FileNotFoundError as error:
        raise RuntimeError(f"插件 UI assets_dir 无效: {relative_path}") from error
    if not path.is_relative_to(plugin_root) or path == plugin_root or not path.is_dir():
        raise RuntimeError(f"插件 UI assets_dir 无效: {relative_path}")
    return path


def _resolve_asset_path(
    plugin_root: Path,
    relative_path: str,
    *,
    suffix: str,
    kind: str,
) -> Path:
    """Resolve one asset and reject missing, wrong-type, and escaped paths."""

    try:
        path = (plugin_root / relative_path).resolve(strict=True)
    except FileNotFoundError as error:
        raise RuntimeError(f"插件 UI {kind} 无效: {relative_path}") from error
    if (
        not path.is_relative_to(plugin_root)
        or path.suffix != suffix
        or not path.is_file()
    ):
        raise RuntimeError(f"插件 UI {kind} 无效: {relative_path}")
    return path


def _always_available() -> bool:
    return True


def _validate_sync_callable(value: object, field_name: str) -> None:
    if not callable(value):
        raise TypeError(f"插件 Plugin UI {field_name} 必须可调用")
    if inspect.iscoroutinefunction(value) or inspect.iscoroutinefunction(
        getattr(value, "__call__", None)
    ):
        raise TypeError(f"插件 Plugin UI {field_name} 必须是同步函数")

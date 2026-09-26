"""资产目录由贡献插件登记，生命周期随注册 Effect。"""

from collections.abc import AsyncIterator, Callable
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import PurePosixPath

from agent.plugin_composition import Context, Effect
from agent.plugin_composition.assets import INSTALLED_ASSETS, InstalledAsset

api_version = 3
name = "assets"
version = "1.0.0"
desc = "保存当前组合的固定代码资产目录"


class Assets:
    """拥有一个 Root 的目录注册，不解析内容或复制代码制品。"""

    def __init__(self, ctx: Context) -> None:
        self._ctx = ctx
        self._entries: dict[InstalledAsset, Context] = {}

    async def register(
        self, ctx: Context, category: str, relative_path: str,
    ) -> Effect:
        """只登记调用方固定代码目录，注销只移除内存记录。"""
        # 1. 身份从实际 Context 取得，不接受调用方指定其他 owner 或 Root。
        if ctx.root_instance_token is not self._ctx.root_instance_token:
            raise ValueError("资产注册不能跨 composition Root")
        if ctx.require(INSTALLED_ASSETS) is not self:
            raise ValueError("资产服务不是贡献方绑定的 provider")
        if (
            not isinstance(relative_path, str)
            or not relative_path
            or relative_path.strip() != relative_path
        ):
            raise ValueError("资产路径必须是非空相对路径")
        relative = PurePosixPath(relative_path)
        if relative.is_absolute() or ".." in relative.parts or "\\" in relative_path:
            raise ValueError("资产路径必须位于贡献方代码制品内")
        root = ctx.runtime.plugin_dir.resolve(strict=True)
        path = (root / relative).resolve(strict=True)
        if not path.is_relative_to(root) or not path.is_dir():
            raise ValueError("资产目录必须位于贡献方代码制品内")
        # 目录中的链接也不得把其他 owner 的资源带入此注册。
        if any(
            not child.resolve(strict=True).is_relative_to(root)
            for child in path.rglob("*")
        ):
            raise ValueError("资产资源链接越过贡献方代码制品")
        asset = InstalledAsset(ctx.runtime.plugin_id, category, path)

        # 2. Effect 由贡献方持有，provider 保留唯一注册表和归档贡献关系。
        def setup() -> Callable[[], None]:
            if asset in self._entries:
                raise ValueError(f"资产目录重复注册: {asset}")
            self._entries[asset] = ctx

            def cleanup() -> None:
                del self._entries[asset]

            return cleanup

        return await ctx.effect(setup, label=f"asset:{category}:{relative_path}")

    def __call__(self, consumer: Context) -> tuple[InstalledAsset, ...]:
        consumer.require_runtime_owner(INSTALLED_ASSETS, self)
        return tuple(sorted(
            self._entries,
            key=lambda item: (item.owner_id, item.category, str(item.root_dir)),
        ))

    @asynccontextmanager
    async def open(
        self, consumer: Context, *, category: str,
    ) -> AsyncIterator[tuple[InstalledAsset, ...]]:
        """异步读取跨越换代时，持有选中资产的原贡献者。"""
        async with self._ctx.runtime_scope(), AsyncExitStack() as stack:
            # 取得目录与接纳原 owner 之间没有可挂起的业务操作。
            assets = tuple(item for item in self(consumer) if item.category == category)
            owners = tuple(dict.fromkeys(self._entries[item] for item in assets))
            for owner in owners:
                await stack.enter_async_context(owner.runtime_scope())
            yield assets

    def contributors(self) -> tuple[Context, ...]:
        return tuple(dict.fromkeys(self._entries.values()))


async def apply(ctx: Context) -> None:
    registry = Assets(ctx)
    await ctx.provide(INSTALLED_ASSETS, registry, binding_contributors=registry.contributors)

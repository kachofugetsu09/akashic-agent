"""收集各插件声明的设置表单，由一个通用 Web 模块渲染、一组路由读写。"""

from collections.abc import Callable
from importlib import import_module

from agent.plugin_composition import Context, Effect
from plugins.ui.contract import UI, Configuration, SETTINGS_SLOT

from .contract import SETTINGS_FORMS, SettingsEntry, SettingsForm

api_version = 3
name = "plugin_settings"
version = "1.0.0"
desc = "声明式插件设置表单：插件声明字段，宿主统一渲染与保存"
inject = (UI,)


class Forms:
    """一个 Root 的表单注册表；键由贡献插件身份派生，不接受自报。"""

    def __init__(self, ctx: Context) -> None:
        self._ctx = ctx
        self._entries: dict[str, SettingsEntry] = {}

    # 1. 只接受同一 Root 的贡献方；2. 键就是完整插件 ID（无损，不同插件不会撞键）；3. Effect 撤销即移除。
    async def register(self, ctx: Context, form: SettingsForm, settings: Configuration) -> Effect:
        if ctx.root_instance_token is not self._ctx.root_instance_token or ctx.require(SETTINGS_FORMS) is not self:
            raise ValueError("设置表单注册不能跨 composition Root")
        owner = ctx.runtime.plugin_id
        key = owner
        entry = SettingsEntry(key, owner, form, settings, ctx)

        def setup() -> Callable[[], None]:
            if key in self._entries:
                raise ValueError(f"设置表单重复注册: {owner}")
            self._entries[key] = entry

            def cleanup() -> None:
                del self._entries[key]

            return cleanup

        return await ctx.effect(setup, label=f"settings-form:{key}")

    def entries(self) -> tuple[SettingsEntry, ...]:
        return tuple(self._entries[key] for key in sorted(self._entries))

    def entry(self, key: str) -> SettingsEntry:
        return self._entries[key]

    def contributors(self) -> tuple[Context, ...]:
        return tuple(entry.context for entry in self._entries.values())


async def apply(ctx: Context) -> None:
    forms = Forms(ctx)
    await ctx.provide(SETTINGS_FORMS, forms, binding_contributors=forms.contributors)
    _ = await ctx.require(UI).register(ctx, web="web_module.js", dashboard=lambda: import_module(".dashboard", __package__),
                                       requires=("shell.settings-plugins.v1",), contract_digests=SETTINGS_SLOT)

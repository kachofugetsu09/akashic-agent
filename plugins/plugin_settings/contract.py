"""声明式设置表单：插件只声明字段，宿主统一渲染与保存；外置插件不用自带前端。"""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Literal, Protocol

from pydantic import BaseModel

from agent.plugin_composition.context import Context
from agent.plugin_composition.effect import Effect
from agent.plugin_composition.model import ServiceKey
from plugins.models.contract import ModelUse, model_options
from plugins.onboarding.contract import ONBOARDING, Ability, Step
from plugins.ui.contract import Configuration, ModelSettings

# 插件自己报告的状态：ready/reason 等键覆盖表单默认的"已就绪"，比如还没填凭据、还没授权。
StatusReader = Callable[[], Awaitable[Mapping[str, object]]]

# text 单行文本、lines 每行一项的列表、number 数字、toggle 开关、model 插件自己的模型选择、
# secret 凭据（存成 CredentialRef，读回只说有没有，不回显原文）。
FieldKind = Literal["text", "lines", "number", "toggle", "model", "secret"]
_KEY = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


@dataclass(frozen=True, slots=True)
class SettingsField:
    name: str
    label: str
    kind: FieldKind = "text"
    hint: str = ""
    # 只有 model 字段需要：声明这个用途能跟随什么、对模型有什么要求。
    use: ModelUse | None = None

    def __post_init__(self) -> None:
        if not _KEY.fullmatch(self.name):
            raise ValueError(f"设置字段名无效: {self.name}")
        if (self.kind == "model") != (self.use is not None):
            raise ValueError("只有 model 字段带 ModelUse，且 model 字段必须带")

    def body(self) -> dict[str, object]:
        return {"name": self.name, "label": self.label, "kind": self.kind, "hint": self.hint}


# 表单同时作为引导里的一步：按分组出现在能力卡上，内嵌的就是这张表单。
@dataclass(frozen=True, slots=True)
class Setup:
    group: str
    title: str
    ability: Ability | None = None


# 粘贴回调网址式授权：start 给出授权页地址；用户同意后浏览器跳到打不开的回调页，
# 把地址栏的完整网址交给 finish 换取令牌。finish 用 ValueError 说明用户能改正的错误。
@dataclass(frozen=True, slots=True)
class Authorize:
    label: str
    hint: str
    start: Callable[[], Awaitable[str]]
    finish: Callable[[str], Awaitable[None]]

    def body(self) -> dict[str, object]:
        return {"label": self.label, "hint": self.hint}


@dataclass(frozen=True, slots=True)
class SettingsForm:
    title: str
    description: str
    fields: tuple[SettingsField, ...]
    setup: Setup | None = None

    def __post_init__(self) -> None:
        names = [field.name for field in self.fields]
        if not names or len(set(names)) != len(names):
            raise ValueError("设置表单至少一个字段，且字段名不能重复")
        if sum(field.kind == "secret" for field in self.fields) > 1:
            raise ValueError("设置表单最多一个 secret 字段")


@dataclass(frozen=True, slots=True)
class SettingsEntry:
    key: str
    owner: str
    form: SettingsForm
    settings: Configuration
    context: Context
    authorize: Authorize | None = None

    def body(self) -> dict[str, object]:
        form = self.form
        return {"id": self.key, "owner": self.owner, "title": form.title, "description": form.description,
                "fields": [field.body() for field in form.fields],
                "authorize": None if self.authorize is None else self.authorize.body()}


class SettingsForms(Protocol):
    # 注册随贡献插件的 Effect 撤销；同一插件只能有一张声明式表单。
    async def register(self, ctx: Context, form: SettingsForm, settings: Configuration,
                       authorize: Authorize | None = None) -> Effect: ...

    def entries(self) -> tuple[SettingsEntry, ...]: ...

    # 键不存在时抛 KeyError。
    def entry(self, key: str) -> SettingsEntry: ...


SETTINGS_FORMS = ServiceKey[SettingsForms]("plugin_settings.forms.v1")


# 常见用法：字段取自插件自己的配置模型，读写与校验复用 ModelSettings；有 model 字段时需注入 MODEL_CATALOG。
# 字段的 use 只决定下拉里列哪些选项；配置也可能从别处写入，插件的配置模型要自己用 use.check 校验该字段。
async def declare_settings(ctx: Context, config: type[BaseModel], form: SettingsForm, *,
                           status: StatusReader | None = None, authorize: Authorize | None = None) -> Effect:
    return await ctx.require(SETTINGS_FORMS).register(ctx, form, _settings(ctx, config, form, status), authorize)


# 表单声明了 setup 时登记引导步骤；需注入 ONBOARDING。引导内嵌的仍是 plugin_settings 渲染的这张表单，
# 步骤是否完成看 status 报告的 ready，所以必填项要由插件在 status 里说明。
async def declare_setup(ctx: Context, config: type[BaseModel], form: SettingsForm, *,
                        status: StatusReader | None = None) -> None:
    setup = form.setup
    if setup is None:
        raise ValueError("表单没有声明 setup")
    onboarding = ctx.require(ONBOARDING)
    _ = await onboarding.group(ctx, setup.group, setup.title, setup.ability)
    route = f"{SETTINGS_ROUTE}/{ctx.runtime.plugin_id}"
    _ = await onboarding.register(ctx, Step("configure", form.title, setup.group, route,
                                            _settings(ctx, config, form, status).read))


# plugin_settings 卡片的 route；后缀插件 ID 时只渲染该插件的表单。
SETTINGS_ROUTE = "plugin-settings"


def _settings(ctx: Context, config: type[BaseModel], form: SettingsForm, status: StatusReader | None) -> ModelSettings:
    uses = {field.name: field.use for field in form.fields if field.use is not None}
    readers = [reader for reader in (model_options(ctx, uses) if uses else None, status) if reader is not None]

    async def merged() -> Mapping[str, object]:
        result: dict[str, object] = {}
        for reader in readers:
            result.update(await reader())
        return result

    secret = next((field.name for field in form.fields if field.kind == "secret"), None)
    return ModelSettings(ctx, config, tuple(field.name for field in form.fields if field.name != secret),
                         secret=secret, status=merged if readers else None)

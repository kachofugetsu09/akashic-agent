"""声明式设置表单：插件只声明字段，宿主统一渲染与保存；外置插件不用自带前端。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal, Protocol

from pydantic import BaseModel

from agent.plugin_composition.context import Context
from agent.plugin_composition.effect import Effect
from agent.plugin_composition.model import ServiceKey
from plugins.models.contract import ModelUse, model_options
from plugins.ui.contract import Configuration, ModelSettings

# text 单行文本、lines 每行一项的列表、number 数字、toggle 开关、model 插件自己的模型选择。
FieldKind = Literal["text", "lines", "number", "toggle", "model"]
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


@dataclass(frozen=True, slots=True)
class SettingsForm:
    title: str
    description: str
    fields: tuple[SettingsField, ...]
    # 插件配置页里的分组；同组的表单排在一张卡里。
    family: str = "plugins"
    family_label: str = "其他插件"

    def __post_init__(self) -> None:
        names = [field.name for field in self.fields]
        if not names or len(set(names)) != len(names):
            raise ValueError("设置表单至少一个字段，且字段名不能重复")
        if not _KEY.fullmatch(self.family):
            raise ValueError(f"设置分组无效: {self.family}")


@dataclass(frozen=True, slots=True)
class SettingsEntry:
    key: str
    owner: str
    form: SettingsForm
    settings: Configuration
    context: Context

    def body(self) -> dict[str, object]:
        form = self.form
        return {"id": self.key, "owner": self.owner, "title": form.title, "description": form.description,
                "family": form.family, "family_label": form.family_label,
                "fields": [field.body() for field in form.fields]}


class SettingsForms(Protocol):
    # 注册随贡献插件的 Effect 撤销；同一插件只能有一张声明式表单。
    async def register(self, ctx: Context, form: SettingsForm, settings: Configuration) -> Effect: ...

    def entries(self) -> tuple[SettingsEntry, ...]: ...

    # 键不存在时抛 KeyError。
    def entry(self, key: str) -> SettingsEntry: ...


SETTINGS_FORMS = ServiceKey[SettingsForms]("plugin_settings.forms.v1")


# 常见用法：字段取自插件自己的配置模型，读写与校验复用 ModelSettings；有 model 字段时需注入 MODEL_CATALOG。
async def declare_settings(ctx: Context, config: type[BaseModel], form: SettingsForm) -> Effect:
    uses = {field.name: field.use for field in form.fields if field.use is not None}
    settings = ModelSettings(ctx, config, tuple(field.name for field in form.fields),
                             status=model_options(ctx, uses) if uses else None)
    return await ctx.require(SETTINGS_FORMS).register(ctx, form, settings)

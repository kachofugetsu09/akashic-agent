# 0130 · 插件声明设置字段，宿主统一渲染

- 状态：proposed / implemented for review
- 日期：2026-10-22
- 关联：ADR 0129；PLG-014

## 背景

插件的设置页要自带 Web 模块：写 React、打包、注册到「插件配置」。内置插件可以这样做，外置插件（如 github-watch）
只有 Python，ADR 0129 之后它要能选模型，却没有地方放这个下拉框。

## 决定

1. 新内置插件 `plugin_settings` 提供 `SETTINGS_FORMS`。插件声明 `SettingsForm`（标题、说明、字段），字段类型只有
   `text`、`lines`、`number`、`toggle`、`model`；`model` 字段带 `ModelUse`。
2. 常见用法 `declare_settings(ctx, Config, FORM)`：读写与校验复用 `ModelSettings`，值写回插件自己的配置，
   空值即回到配置模型的默认值。注册随贡献插件的 Effect 撤销，一个插件一张表单。
3. 依赖是可选的：插件用 `ctx.inject((SETTINGS_FORMS, …))` 声明，`plugin_settings` 未启用时插件照常工作，只是没有设置页。
4. 前端是一个通用 Web 模块，按字段类型选现成控件（`Field`、`Check`、`ModelField`、多行列表）。

## 取舍

- **一张卡片，不分组。** 宿主 Web 合同只接受激活时同步登记，表单列表却要异步读取；因此模块登记一张固定的「插件设置」卡，
  打开时再读列表，每张表单各自保存。代价是外置插件不能进自己的分组卡片；需要的插件仍可自带 Web 模块。
- **字段类型有限。** 不覆盖 OAuth 授权、预览、依赖字段联动等交互；这些继续由插件自带前端。
- **键由插件身份派生。** 表单 ID 取插件 ID 去掉来源后缀，不接受插件自报，避免两个插件互相覆盖。

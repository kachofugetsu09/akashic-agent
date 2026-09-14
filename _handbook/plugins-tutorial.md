# 插件开发入口

旧版 `class DemoPlugin(Plugin)` 教程已退役，不能用于当前插件。

请从[项目索引](../docs/INDEX.md)进入[插件能力手册](../docs/design/plugin-v3-capabilities.md)。
当前唯一入口是 `apply(ctx)`；插件从 `ctx.config` 读取本次装配的输入，自行解析配置，
通过服务组合能力并登记所属资源。

整体换代和 stable 的目标合同见 [0071](../docs/decisions/0071-plugin-composition-and-whole-runtime-updates.md)。
该重构仍在实施，代码提交不代表运行验收或部署完成。

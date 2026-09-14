# Akashic 插件编写参考

插件系统正在按 ADR 0071 简化。本文件不复制第二份 API 表或历史示例：

- [能力手册](../../../docs/design/plugin-v3-capabilities.md)：当前入口、ServiceKey、资源与能力合同。
- [插件底座职责](../../../docs/decisions/0071-plugin-composition-and-whole-runtime-updates.md)：已批准的目标与边界。
- [实施合同](../../../docs/design/plugin-whole-runtime-simplification.md)：尚在分层实施的行为。
- [工作流](../../../docs/WORKFLOW.md)：目标 checkout 的实际交付和验证规则。

唯一入口为 apply(ctx)。插件用 ctx.config 取得该组合固定的输入，自己解析和校验。
参数名不构成加载合同；入口必须能用一个位置参数调用。

Skill 等静态资产目前通过 asset_roots 按类别声明。MCP、process、Workload 配置只在插件代码中
构造一次。不要复制旧版 apply(ctx, config)、Config 特殊导出、skill_roots、drift_skill_roots
或 TOML MCP/process 示例。最终 manifest 删除与完整换代仍以当前实施合同和真实代码为准。

实际验证前检查用户授权。候选无正式凭据和正式可写数据；已安装、已编译和已提交 PR 都不是
业务验证或成功晋升的证据。

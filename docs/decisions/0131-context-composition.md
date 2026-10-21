# 0131 · 上下文构成按装配来源记账

- 状态：proposed / implemented for review
- 日期：2026-10-22
- 关联：ADR 0128；WEBUI-008

## 背景

用量面板只有 provider 实报的输入总量，看不出上下文里是什么占了地方：系统提示、工具定义、记忆、历史还是工具结果。
按消息角色猜不准：记忆和环境材料以 user 行注入，看起来和用户消息一样。

## 决定

1. **在装配处记出处，而不是事后猜。** `ModelRequest.sources` 是一组 `RequestSource(category, label, rows | tools)`，不发给 provider：
   - System Prompt 按材料贡献者拆（`default_prompt`、`markdown_memory`、`skills`…），材料汇总时保留各段；
   - 本轮材料按提醒块拆（`prompt · environment`、akasha 召回…），提醒带上实际 owner；
   - 历史由投影拆：用户、助手、按工具名的工具结果、回放的旧提醒；历史摘要单列；
   - 工具定义按工具。
2. **models 统一估算并存档。** 调用成功后用该模型 driver 的估算器逐来源估算，按调用记录 ID 写入
   `plugin-data/models-release/composition.sqlite3`，保留 31 天。界面按 provider 实报输入等比换算。
3. 旁路调用（标题、看图、记忆整理、压缩摘要）不经装配，没有 `sources`，按消息角色粗分。

## 取舍

- **是估算份额，不是精确 token。** 每段单独估算会重复计入估算器的固定开销，所以只用比例，总量以实报为准。
- **独立文件。** 不改 `model_calls` 表，也不写进消息里的 `model.facts`：回滚到旧版本时旧代码不读它，不会因未知字段拒绝消息。
- **只有上线之后的调用有构成**，之前的轮次在趋势里画成灰柱。
- 回放的旧提醒只能归为「历史提醒」，原始分块在当轮装配后就不再保留。

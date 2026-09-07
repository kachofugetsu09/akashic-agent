# 0061 · 已停止的旧执行完整归档，不自动续跑

- 状态：accepted
- 日期：2026-09-08
- 关联条款：SES-001、SES-003、WSP-001、BAK-001
- 设计：[Message 重构与迁移](../design/0902-reviewed-v4.md)

## 背景

真实 hua-home 快照迁移遇到 38 条历史末尾执行链：20 条 failed、13 条 interrupted、5 条 cancelled，时间为 2026-08-05 至 2026-08-29；queued/in_progress 为 0。其 1097 条工具轨迹只有展示状态和预览，没有可用于安全恢复的领域回执。旧迁移因此阻断整库转换。

## 决定与理由

维护者确认采用推荐处理：完整保留已停止旧链的原消息、工具轨迹和状态，作为历史迁入，不自动续跑。`turn_messages` 增加原行 `history.record` 及同事务迁移回执 `archived_without_tool_receipts`；不把 success 文本升级成 ToolResult，不补造用户输入或成功状态，不执行旧工具，也不发送旧回复。

这替换持久化地图中对所有无回执旧工具链一律阻断的选择。真正 queued/in_progress 的链仍阻断迁移；缺失、循环、跨 Session 父链和损坏数据仍报错。后续工作通过新的明确输入发起，不能把归档当成已有安全续跑能力。

## 验收与恢复

在禁网隔离的真实副本执行自动迁移，核对所有原 Message、旧 turns、附件和向量保全；38 条归档原因可审计，无新增原生 ToolCall/ToolResult，无外部执行，第二次启动不重复归档。原生 SQLite 备份和原行 digest 提供恢复点。仅授权本地演练和代码交付，不授权正式部署或修改正式 workspace。

# 0146 · 移除漫游（Drift），主动联系只剩 Content 与 Alert

- 状态：accepted / implemented for review
- 日期：2026-11-07
- 关联条款：PRO-002、PLG-006、TST-001
- 修订：[0145](0145-message-provenance-is-decided-once.md) 之后 Wake 的职责只剩 `content` 与 `alert`

## 背景

`drift` 插件为 Wake 提供“漫游职责”：来源写入提案，到期后由 Wake 领取、让模型决定是否联系用户。
生产实例的提案表最后一次更新在 2026-08-26，全部已结束（await_change / invalidated）；
Wake 从未以 drift 职责运行过，消息账本里也没有 drift 写入的消息。fleet 与插件组织里没有任何插件写提案或读取 `drift.*` 合同。
它仍让 Wake 多一条领取、失败关闭、结算分支，让用量面板硬编码一个插件名。

## 决定

1. 删除 `plugins/drift` 与两套 bundle 中的 `drift` 行；Wake 去掉 drift 的准入、领取、阶段、工具组与技能章节。
2. 已有安装按插件退役处理：当前发行版没有该插件源码即视为退役，`plugin-data/drift-release/` 与旧 `workspace/drift/` 原样保留，不删数据。
3. Wake 状态库 `wake_runs.owner` 的 CHECK 约束仍包含 `'drift'`：表结构按落盘 SQL 逐字校验，历史行不改写；新代码只写 `alert` / `content`。
4. 用量面板的“主动消息”不再由 models 插件点名 `wake`、`drift`：来源插件经 `USAGE_SOURCES.declare_proactive` 声明自己的 lane，
   声明随插件生命周期撤销；检查点记录声明集合，集合变化即丢弃重算。
5. 只服务于 drift 或早已无法运行的 debug 脚本一并删除（`drift_io_isolation`、`drift_proposal_io`、`wake_state_io_isolation`、
   `wake_v3_provider_e2e`、`content_wake_h5_e2e`、`source_write_io`）。

## 代价与边界

- 历史设计文档里的 Drift 描述保持原样，作为当时的记录；以本记录为准。
- `scripts/proactive_island` 是旧布局数据的离线交接工具，读取旧 `workspace/drift/drift.db` 只读，不依赖插件，保留。
- 升级后第一次打开用量面板会全量重算一次（检查点版本 2 → 3）。

## 验收

隔离实例以选择了 `drift@release` 的旧版本数据升级到新版本：启动、插件页与 Wake 正常，drift 数据目录未被删除；
用量面板中 Wake 回合仍归入“主动消息”。

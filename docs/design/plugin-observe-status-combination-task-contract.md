# Observe 与 status_commands 组合迁移任务合同

- 状态：accepted / implementation approved
- 日期：2026-08-15
- 目标分支：`codex/plugin-turn-committed-event` 之后的 Draft stacked PR
- 恢复点：`backup/plugin-observe-status-combination-before-20260815@92bd39e`
- 上游：[0004](../decisions/0004-cross-repository-evidence-is-an-immutable-combination.md)、[0036](../decisions/0036-plugin-composition-keeps-promotion-owner.md)、[0037](../decisions/0037-plugin-services-name-capabilities-not-categories.md)、[0038](../decisions/0038-human-commands-are-not-model-tools.md)

## Goal

用一个 exact-commit 跨仓组合 Gate 证明 Observe 和 status_commands 可以按能力 owner 混合迁移。Core 只提供 generation、Commands、Session Read、UI Slots 与旧 lifecycle module 挂载点；插件自己拥有命令实现、投影和派生数据库。

```text
locked commits
      │
      ▼
┌──────────────────────────┐
│ real PluginManager       │
│ Observe v2 + status v3   │
└────────────┬─────────────┘
             │
      ┌──────┴───────┐
      ▼              ▼
Observe owns      status owns
/kvcache + DB      /memory_status + UI
      │              │
      └──────┬───────┘
             ▼
   immutable evidence report
```

## Ownership and invariants

- 锁文件是候选组合身份，不是安装清单、stable 指针或发布审批；本 PR 不安装、晋升或切换正式插件。
- Observe 独占 `/kvcache`、`observe/observe.db` 和 `turn.after_answer`；status_commands 不再读取 Observe 数据。
- status_commands 独占 `/memory_status` 和 `drawer.panel`；只通过 Core 的 Commands、Session Read 与 UI Slots 能力接入。
- Core 独占 namespace、generation、命令目录组合、UI slot 组合、生命周期挂载和晋升验证，不实现任何插件领域逻辑。
- 外部源码必须由完整 commit SHA、Git tree 和 `plugin.py` SHA-256 共同绑定；不接受 branch、tag、cache 或宿主 checkout。
- Telegram 目录必须精确包含 `kvcache` 与 `memorystatus`，两个插件的 UI slot 必须分别保持 `turn.after_answer` 与 `drawer.panel`。
- `/memory_status` 路径不得改变 `sessions.db*`；一次 `TurnCommitted` 只能在临时 Observe 数据库产生一行 `300 / 260` 证据。
- Gate 结束后 `loaded_count == 0` 且没有 current snapshot；临时目录随进程清理。

## Change and persistence

```yaml
change_type: test
semantic_delta: none
capability_owner: core
consumer_scope:
  - observe migration candidate
  - status_commands migration candidate
runtime_patch: none
authoritative_state_owner: "Core owns composition; each plugin owns its implementation and derived data."
protected_state:
  - formal workspace and plugin-data
  - plugin manifest, stable/latest pointers and promotion decisions
  - SessionDB contents and write set
  - existing v2/v3 plugin runtime behavior
allowed_effects:
  - exact-commit network checkout in an isolated temporary directory
  - temporary SessionDB and Observe database rows
  - reproducible JSON evidence under docker/debug/reports
forbidden_effects:
  - formal install, reload, promotion or deployment
  - editing plugin cache or canonical source
  - channel delivery and external API calls
rollback: "Revert this adjacent PR or return to backup/plugin-observe-status-combination-before-20260815."
```

本 PR 不增加、更新、逻辑失效或物理减少任何权威持久记录。CI artifact 只保存可重建的源码身份与临时运行观察值。

## Verification

- 严格 schema 测试拒绝浮动或漂移的组合锁；
- 真实 `PluginManager` 加载锁定的 Observe v2 与 status_commands v3；
- 命令目录、UI slots、命令 owner、Observe row 和 SessionDB write set 同时通过；
- manager terminate 后 generation 与 snapshot 清零；
- Core worktree dirty 时 CI 模式 fail-loud，防止证据与提交身份分离。

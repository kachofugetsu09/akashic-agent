# 插件 Mobile UI generation 绑定任务合同

- 状态：accepted / implementation approved
- 日期：2026-08-15
- 目标分支：`codex/plugin-session-read` 之后的 Draft stacked PR
- 恢复点：`backup/plugin-mobile-ui-slots-before-20260815@10e52743`
- 上游：[0036](../decisions/0036-plugin-composition-keeps-promotion-owner.md)、[0037](../decisions/0037-plugin-services-name-capabilities-not-categories.md)、[插件 UI Slots 组合能力合同](plugin-ui-slots-service-task-contract.md)

## Goal

先把现有 Mobile UI asset、query handler 与动态可用性编译成同一个 generation contribution，使查询 host 不再把任意 generation instance 强制解释成旧 v2 `Plugin`。这一步只收敛 Core 内部 owner，为下一张 v3 `core.ui_slots.register_mobile` PR 提供可复用落点，不新增插件公共入口。

```text
┌────────────────────┐   collect once   ┌────────────────────┐
│ v2 legacy plugin   │ ───────────────▶ │ generation contrib│
│ declaration/methods│                  │ asset/handler/check│
└────────────────────┘                  └─────────┬──────────┘
                                                  │ snapshot lease
                                                  ▼
                                        ┌────────────────────┐
                                        │ Mobile UI provider │
                                        │ no v2 class cast   │
                                        └────────────────────┘
```

## Ownership and invariants

- 插件仍拥有 JS/CSS、`mobile_ui_query` 与 `mobile_ui_available` 实现；本 PR 不改变 v2 class method 的调用时机。
- Core 在 candidate generation 收集时把 asset、handler 和 available callback 一起写入 `PluginContributions`。asset 与 callback 必须成组存在；不完整的内部状态 fail-loud。
- 静态资产校验抽到一个 Core owner：只接受插件 source 内的 `.js`/`.css`，限制 navigation、slot 和 240 KiB 总大小，并固化内容摘要。旧 resolver 的合法输入与错误分类保持兼容。
- `PluginMobileUiProvider` 只消费当前 snapshot generation contribution；revision、lease、queue、timeout、结果大小和传输错误语义不变。
- 下一张 PR 才允许 v3 Fiber 向这些字段登记实现。本 PR 的 committed head 上，外部插件仍只能使用 v2 API。

## Change intent

```yaml
change_type: refactor
semantic_delta: none
capability_owner: "Plugin owns Mobile UI implementation; Core generation owns the immutable runtime binding."
consumer_scope:
  - existing v2 Mobile UI plugins and provider
runtime_patch: required
runtime_patch_reason: "The generation snapshot and its lease are Core-owned; a client cannot replace the server-side binding."
authoritative_state_owner: "Existing plugin generation snapshot; plugin business stores remain unchanged."
client_only_alternative: "Not applicable because this is server-side generation ownership."
protected_state:
  - Mobile UI catalog, asset and query protocol
  - v2 plugin lifecycle and dynamic availability behavior
  - stable/latest pointers, formal workspace and plugin-data
allowed_effects:
  - in-memory generation contribution callbacks
  - temporary plugin roots and workspaces in tests
forbidden_effects:
  - formal plugin migration or runtime switch
  - workspace, Session, plugin-data, channel or external API writes
rollback: "Revert this adjacent PR or return to backup/plugin-mobile-ui-slots-before-20260815."
```

本 PR 不增加、更新、逻辑失效或物理减少任何权威持久记录。

## Verification

- 现有 v2 plugin manager 真实收集 asset、query 与 available callback；
- catalog、内容寻址 asset、动态 unavailable、同步 worker offload、queue bound、timeout 后 lease drain、无效结果和错误隔离回归保持不变；
- legacy hot reload 失败继续保留旧 active snapshot；
- public Gate 回归 Mobile WebUI、plugin generation、publication 与正式状态零写入。

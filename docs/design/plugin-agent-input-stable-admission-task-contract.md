# 插件 Agent Input stable admission 任务合同

- 状态：accepted / implementation approved
- 日期：2026-08-15
- 目标分支：`codex/plugin-timer-snapshot-scheduling` 之后的 Draft stacked PR
- 恢复点：`backup/plugin-agent-input-stable-admission-before-20260815@a832477c`
- 上游：[0036](../decisions/0036-plugin-composition-keeps-promotion-owner.md)、[插件 Agent Input 组合能力合同](plugin-agent-input-service-task-contract.md)、[插件 Timer stable snapshot 调度合同](plugin-timer-snapshot-scheduling-task-contract.md)
- 首个真实消费者：GitHub Watcher

## Goal

允许已经由 stable snapshot 准入并持有精确 lease 的 Timer callback 在 pointer 切换后完成同步等待中的 Agent Input。这个权限只属于原 callback task 和原 Root；latest candidate、reload 后新建的 detached task、没有绑定 lease 的 retired Context 继续拒绝。

```text
stable S0 timer enqueue
        │ claim lease(stable_at_claim=true)
        ▼
┌──────────────────┐       promote       ┌──────────────────┐
│ S0 callback task │ ───────────────────▶ │ current = S1     │
│ bound S0 lease   │                      │ S0 is retired    │
└────────┬─────────┘                      └──────────────────┘
         │ same task + same Root
         ▼
 Agent Input allowed and forks S0 lease

latest lease / detached child / bare retired Context ── denied
```

## Ownership and invariants

- `RuntimeSnapshotLease` 记录 `stable_at_claim`。该事实只在首次 claim 时由 `RuntimeSnapshotStore` 计算，fork 原样继承，不能在 pointer 切换后重新计算。
- `stable_at_claim=true` 只表示这次工作在准入时来自 stable，不表示 snapshot 永远 current，也不授权新 task 继承外部效果权限。
- Agent Input 优先读取当前 task 绑定的 runtime lease。只有 lease active、snapshot 的 composition Root 与调用 Root 相同且 `stable_at_claim=true` 时才 fork 并继续。
- 没有满足上述条件时，保留现有 current stable + accepting leases 检查。latest、paused、retired 和 foreign Root 继续 fail-loud，并在对应 composition audit 中记录 denied。
- `get_current_runtime_lease()` 的 owner-task fence 保持不变。callback 创建的 detached child 即使继承 Python ContextVar，也不能取得父 task 的 lease 权限。
- Core 只授权 Session/Turn admission；插件继续拥有事件账本、幂等、重试和外部客户端。busy、缺失 Session、metadata 错误及后端失败原样传播。

## Change and persistence

```yaml
change_type: fix
semantic_delta: compatible
capability_owner: core
consumer_scope:
  - composition plugin API v3 Agent Input callers admitted by stable snapshot work
  - first planned consumer: github-watch
runtime_patch: required
runtime_patch_reason: "Only RuntimeSnapshotStore can attest whether one lease was claimed from stable and preserve that fact across retirement."
authoritative_state_owner: "Core owns snapshot admission and Session/Turn commit; plugins own domain input and dedupe state."
client_only_alternative: "A plugin cannot safely infer stable admission from generation paths or current pointers after the callback has started."
protected_state:
  - latest candidate external-effect denial
  - owner-task runtime binding fence
  - current stable admission and busy behavior
  - formal workspace, plugin-data and SessionDB
allowed_effects:
  - in-memory immutable lease admission metadata
  - fake Agent Input calls and temporary workspaces in tests
forbidden_effects:
  - formal Session/Turn creation during verification
  - formal plugin install, promotion, runtime switch or external API calls
rollback: "Revert this adjacent PR or return to backup/plugin-agent-input-stable-admission-before-20260815."
```

本 PR 不增加、更新、逻辑失效或物理减少权威持久记录。生产中被允许的 Agent Input 仍走既有 `ControlService` 提交协议；本 PR 只修正已经 stable-admitted 的在途工作跨 pointer 切换时的授权判断。

## Verification

- stable lease 在 claim 后固定 `stable_at_claim=true`，fork 与 retirement 后保持不变；
- latest candidate lease 固定为 false，绑定后调用 Agent Input 仍留下 denied audit，不能晋升；
- stable Timer callback 阻塞期间发布新 generation，旧 callback 在同一 task 中仍能提交一次 fake Agent Input；
- 同一旧 callback 创建并等待的 child task 无法取得父 task lease，调用被拒绝；
- 没有绑定 lease 的 retired/disposed Context、foreign Root 和 inactive Fiber 现有 oracle 保持通过；
- Plugin Manager、hot reload、Timer、Agent Input targeted regressions 与公开 Gate 通过。

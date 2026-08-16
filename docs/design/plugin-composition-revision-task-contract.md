# 插件组合结构身份与 revision 任务合同（R3a）

- 状态：implemented / reviewed
- 日期：2026-08-15
- 实现基线：`7d8f1c8fcd81fb6b3cac6b56e8a7d8de73891a4f`
- 关联条款：PLG-001～PLG-004、PLG-008～PLG-010、PLG-014
- 上游：[0036](../decisions/0036-plugin-composition-keeps-promotion-owner.md)、[R2b candidate Root 隔离](plugin-candidate-root-isolation-task-contract.md)

## 1. 目标

把“当前结构是什么”和“封存后是否发生过结构变化”拆成两个事实，避免 Fiber 状态、历史错误或普通 Effect 改变 immutable snapshot identity，也避免结构移除后原样恢复绕过 candidate promotion 校验。

```text
Fiber/Service/Listener add/remove ──→ composition_revision + 1
                │
                └───────────────→ structural content hash

Fiber state / Health / Incident / generic Effect
                └───────────────→ 不进入结构 hash，不递增 revision
```

## 2. 合同

- `TopologyView.identity` 只摘要 Fiber name/parent/required/dependencies、当前 Service 和有序 typed listener；不摘要 generation id、Fiber state、错误或普通 Effect。Root Context 直系 Fiber 的 parent 为 `None`，nested Fiber 使用 Root 内全局唯一的 parent name。
- `TopologyView.effects` 暂保留为一次性诊断视图，但不参与 identity 或 revision；R3b 的 Health/Incident 另有 owner。
- Root 内 `composition_revision` 从零单调增加；Fiber、Service、listener 的实际注册或注销各增加一次。
- `ctx.fiber` 与 `ctx.mount()/inject()` 返回只读 `FiberHandle`；插件保留 `name/state/restart()/dispose()`，不能直接改写 Core-owned dependencies/effects/children/state。
- snapshot compile 同时封存 content identity 与 revision。后续即使结构恢复为同一 hash，只要 revision 变化，candidate seal/promotion 必须 fail-loud，并要求 fresh rebuild。
- stable/latest、lease、validation、promotion 和 drain 仍由 `RuntimeSnapshotStore` 与 `PluginManager` 拥有；本 PR 不新增插件领域 API。

## 3. 验证

- Fiber `ACTIVE → PENDING → ACTIVE` 与普通 Effect add/remove 不改变 topology identity/revision。
- 两棵 Root 的 Fiber 集合、mount 次数与 revision 相同但 parent ownership 不同时，topology identity 与 snapshot id 必须不同；candidate/formal rebuild 不得接受这类生命周期漂移。
- provider/Fiber 移除并按原结构重建后 content hash 相同、revision 更大，旧 candidate 无法 seal；fresh snapshot 可以 promotion。
- listener 注册顺序仍进入 content identity。
- targeted：composition kernel/events、v3 loader 与 hot reload。
- cumulative：R2a/R2b publication 与公开 Change Gate。
- 本地证据：composition kernel/events/loader/hot reload `210 passed`；Basedpyright `0 errors`；compileall、`git diff --check` 与公开 Change Gate 通过。
- Terra xhigh 只读复审无 P0/P1；Fiber、Service、parent ownership、snapshot id 与 generation identity mutant 已分别固化为独立 oracle。
- 停止条件：mutable state 改变 snapshot id、结构恢复可绕过 revision、fresh candidate 无法 promotion、stable Root 被错误 dispose。
- 回滚点：Git tag `backup/plugin-parent-edge-before-20260816`。

## 4. v2 清理关联

R3a 不新增 v2 compatibility。snapshot 的 legacy payload 可以继续与 composition topology 并存；最终删除 v2 时只移除 legacy generation/contribution 编译路径，不回退 identity/revision 分离。

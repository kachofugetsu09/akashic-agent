# 插件 v3 admission 与 lifecycle 收口任务合同

> 2026-09-15 勘误：[0071](../decisions/0071-plugin-composition-and-whole-runtime-updates.md) 已取代本文的两参数入口、精确参数名和 Core 配置模型校验。现行入口为 `apply(ctx)`，插件自行解析 `ctx.config`；下文相关描述仅保留历史背景。凭据脱敏及正式解析授权边界不因此取消。

- 状态：candidate / independently reviewed
- 日期：2026-08-16
- 实现基线：`33fd3542`
- 精确实现：`4ba266ad`
- 关联条款：PLG-001～PLG-004、PLG-008～PLG-010、PLG-014、TST-001～TST-008
- 上游：[v3 production readiness checklist](plugin-v3-production-readiness-checklist.md)、[v3 loader 合同](plugin-v3-loader-task-contract.md)

## 1. 目标与边界

本 PR 关闭四个真实可达的 v3 信任边界：listener 必须可调用、被拒绝的 `ctx.spawn()` 不遗留
coroutine、模块 `apply` 必须精确声明 `(ctx, config)`、继承了错误 task binding 的 lifecycle 不再静默
跳过。`semantic_delta: compatible`；合法 v2/v3 插件、无 composition Root 的 bootstrap/legacy snapshot 和同 task
lifecycle 顺序不变。

```text
plugin namespace ── validate apply(ctx, config) ──► ComposablePlugin
                                                           │
ctx.on ── validate callable ──► Effect-owned listener      │
ctx.spawn ── transfer ownership ──► Effect-owned task       │
                                                           ▼
request task lease ── strict lifecycle binding ──► frozen CompositionRoot
```

不新增兼容层，不迁移插件，不写 workspace/plugin-data/数据库，不发送渠道消息或外部请求。

## 2. 合同与失败语义

1. `EventRegistry.register()` 在修改 event contract、listener、Effect 或 topology revision 前拒绝非 callable，
   返回 `CompositionError(INVALID_EVENT_LISTENER)`。
2. `Context.spawn()` 只在 task 尚未创建、Effect 尚未取得 coroutine ownership 时关闭 coroutine；task 已创建后
   仍由 Effect cleanup 唯一回收。原始异常必须原样传播。
3. `ComposablePlugin.from_module()` 在 Manager 建 generation、Root 或 data directory 前拒绝除精确
   `apply(ctx, config)` 外的签名。两个参数必须无默认值、可按位置传递、名称分别为 `ctx` 和 `config`；
   sync/async 均合法。
4. lifecycle 没有绑定 snapshot 或绑定 snapshot 没有 CompositionRoot 时保持 no-op。存在绑定但 lease 已失效，
   或 ContextVar 被非 owner child task 继承时 fail-loud；插件 listener 不允许被静默跳过。

## 3. 验证与停止条件

- event 五种 dispatch mode 参数化验证 non-callable 在零 mutation 状态失败；
- disposed Fiber 保存的 Context 调用 `spawn()` 后 coroutine 已关闭、Effect/task 为空且无 warning；
- malformed/valid sync/async `apply` 签名 admission matrix；
- Manager admission 在任何 config 读取前校验 plugin-data 路径，并在成功后才创建目录；
- lifecycle unbound/no Root no-op、same-task dispatch、wrong-task/inactive lease fail-loud；
- composition events/kernel/lifecycle/loader 定向回归、Basedpyright error-level、compileall、`git diff --check`。

任何合法插件 admission 变化、lifecycle 顺序变化、双重关闭 coroutine/task、或失败后 topology/Effect 残留都停止交付。

## 4. 回滚

代码恢复点为分支 `codex/plugin-v3-full-migration` 的 `677bedd6`。本任务没有运行数据变更；回滚只需撤销
本 PR 的源码、测试和合同提交。

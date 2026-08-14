# 插件 lifecycle 接入点任务合同

- 状态：implemented / verified
- 日期：2026-08-14
- 实现基线：`294af5819ffad323f10fc2150b64c2992847b18d`
- 关联条款：PLG-002～PLG-004、PLG-008、PLG-014
- 上游：[0036](../decisions/0036-plugin-composition-keeps-promotion-owner.md)、[TopologyView 任务合同](plugin-topology-view-task-contract.md)

## 1. 目标与边界

本 PR 只在现有 Prompt 和回答提交链增加三个 typed serial 接入点。事件 key 由拥有精确阶段的 phase 模块声明；`agent.lifecycle.composition` 只从当前 request 已绑定的 `RuntimeSnapshot` 取得 composition Root 并执行 serial dispatch，不维护 `LifecycleEvents` Service 或事件目录。因此 stable/latest、lease、晋升和排空 owner 不变。

`semantic_delta: none`。本 PR 不加载 v3 插件，不迁移 Citation/Meme，不删除 phase/slot/EventBus，也不修改数据库、workspace、plugin-data、渠道或外部 API。

```text
Prompt ctx ─ legacy EventBus ─ composition serial ─ legacy modules ─ render

Answer ctx ─ legacy pre modules ─ composition preprocess ─ legacy EventBus
           ─ legacy post modules ─ composition cleanup ─ persist/outbound
```

## 2. 接入点

| owner | event | payload / mode | generation scope 内的精确位置 | 失败语义 |
|---|---|---|---|---|
| `prompt_render` | `turn.prompt_render.after_event_bus` | `PromptRenderCtx` / serial | legacy `EventBus` 之后、插件 phase module 与 render 之前 | listener 失败立即传播；`Bail` fail-loud |
| `after_reasoning` | `turn.after_reasoning.before_event_bus` | `AfterReasoningCtx` / serial | legacy pre phase module 之后、legacy `EventBus` 之前 | listener 失败立即传播；`Bail` fail-loud |
| `after_reasoning` | `turn.after_reasoning.before_persist` | `AfterReasoningCtx` / serial | legacy `EventBus` 与 post phase module 之后、SessionDB persist 之前 | listener 失败立即传播；`Bail` fail-loud |

三者使用 generation 内稳定注册顺序并逐个等待。它们只允许原位转换，不允许通过 `Bail` 终止 Core turn；返回 `Bail` 时 fail-loud。没有 snapshot 或 composition Root 的旧路径保持 no-op。

## 3. 验证与回滚

- targeted：新接入点的 snapshot 绑定、顺序、错误和 Bail；旧 Prompt/after-reasoning phase 顺序。
- cumulative：composition、PluginManager、hot reload、turn rollout 与全部 lifecycle phase tests。
- static：compileall、Basedpyright、`git diff --check`。
- Gate：`python docker/debug/gate.py run --base origin/main`。
- 停止条件：旧 phase module 顺序、持久化 write set、EventBus 或 snapshot publication 发生变化。
- 回滚：`/mnt/data/coding/akasic-agent/.backups/20260814-pre-plugin-lifecycle-seam-294af581.bundle`。

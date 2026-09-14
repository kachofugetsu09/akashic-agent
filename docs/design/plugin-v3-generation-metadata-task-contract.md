# 插件 v3 generation metadata 收口任务合同

> 2026-09-15 勘误：[0071](../decisions/0071-plugin-composition-and-whole-runtime-updates.md) 已取代本文的两参数入口、精确参数名和 Core 配置模型校验。现行入口为 `apply(ctx)`，插件自行解析 `ctx.config`；下文相关描述仅保留历史背景。凭据脱敏及正式解析授权边界不因此取消。

- 状态：candidate / independently reviewed
- 日期：2026-08-16
- 实现基线：`6d6b0d97`
- 精确实现：`2d9fb408`
- 关联条款：PLG-001～PLG-004、PLG-008～PLG-010、PLG-014、TST-001～TST-008
- 上游：[v3 production readiness checklist](plugin-v3-production-readiness-checklist.md)、[v3 data root 合同](plugin-data-root-task-contract.md)

## 1. 目标与边界

本 PR 把 v3 的插件目录、配置、数据目录和 generation identity 收回 `PluginGeneration` 与
`PluginRuntime`。`ComposablePlugin` 只冻结模块声明并执行 `apply(ctx, ctx.runtime.config)`，不再保存或读取
v2 `PluginContext`。`semantic_delta: compatible`；v2 的 `PluginContext` 行为本轮不变，后续由明确的 v2
删除任务物理移除。

```text
plugin.py namespace ──► ComposablePlugin ──► apply(Context)
                              ▲                   │
                              │                   ▼
PluginGeneration ── Core mount owner ──► PluginRuntime
  plugin_dir / config / data_dir / generation_id

PluginContext ──► 只归 legacy v2 Manager 分支所有
```

本任务不迁移插件、不改数据库 schema、不写正式 Akashic workspace、不发送渠道消息或外部请求，也不新增
一份与 `PluginGeneration` 并行的 v3 metadata DTO。

## 2. 合同与失败语义

1. `PluginGeneration` 是 Core 对 plugin source、config 与 data identity 的唯一 generation owner；正式 Root、
   candidate clone、Dashboard/Skill/pointer helpers 都从同一 generation 事实构建。
2. v3 namespace adapter 不暴露 `context` 属性。插件只通过 composition `Context` 读取 `runtime.config`、
   `data_root`、`workspace_root()` 与 service；不能取得 legacy event bus、KV、session manager、memory engine 或
   LLM 全功能句柄。
3. candidate clone 重新导入 namespace，并从隔离 data root 加载自己的 config；正式 rebuild 继续使用被冻结的
   generation config。candidate/formal topology 和 snapshot identity 不得因 metadata owner 迁移而变化。
4. v2 分支继续创建和使用 `PluginContext`。所有 Manager 公共路径必须先按 generation 类型分流；禁止用
   `getattr`、默认值或兼容壳伪装 v3 仍有 legacy context。
5. admission、candidate discard、promotion failure 和 terminate 后，clone module、attempt data、Root、Effect、
   scope 与 pointer 的既有清理语义不变。

## 3. 验证与停止条件

- v3 stable load：实例没有 `context`，`apply` 收到冻结 config，正式 `data_root/workspace` 正确；
- installed/builtin candidate：clone 与 formal rebuild 均不读取或写入 `PluginContext`，隔离 config/data 正确；
- v2 stable/hot reload/KV 回滚继续通过，证明 legacy owner 未被误删；
- candidate drift/失败/取消后 stable pointer、正式 plugin-data、module tree 与 Root cleanup 不变；
- composition loader、Manager、hot reload 定向累计回归，Basedpyright error-level、compileall、`git diff --check`。

任何 v3 路径仍构造/读取 `PluginContext`、v2 KV 行为变化、candidate data 进入正式目录、或 formal rebuild identity
漂移都停止交付。

## 4. 回滚

代码恢复点为分支 `backup/plugin-v3-c16-closeout-20260816` 的 `6d6b0d97`。本任务不修改正式 workspace；
回滚只需撤销本 PR 的源码、测试和合同提交。

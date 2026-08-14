# 插件 Tool 组合能力任务合同

- 状态：accepted / implementation approved
- 日期：2026-08-14
- 目标分支：`codex/plugin-timer` 之后的 Draft stacked PR
- 恢复点：`backup/plugin-tools-before-20260814@2ac553f4`
- 上游：[0036](../decisions/0036-plugin-composition-keeps-promotion-owner.md)、[0037](../decisions/0037-plugin-services-name-capabilities-not-categories.md)、[插件元能力底座与测试先行合同](plugin-meta-capability-task-contract.md)
- 参考实现：`/mnt/data/source-code/deepseek-harness@47f943859bef60e4160492346772ded9b24f765a`

## Goal

给 v3 插件提供 Fiber-owned Tool 声明入口，并把声明编译进 Akashic 既有、snapshot-aware 的 `ToolRegistry`。Core 继续独占 schema 校验、搜索可见性、执行上下文、候选隔离和晋升；插件只实现 Tool 领域行为，并可在 Core 明确拥有的执行位置观察或拒绝调用。

```text
┌──────────────────┐  inject core.tools  ┌──────────────────────┐
│ v3 plugin Fiber  │ ───────────────────▶ │ PluginTools collector│
│ owns Tool impl   │   Effect register    │ candidate Root only  │
└────────┬─────────┘                      └──────────┬───────────┘
         │ dispose removes declaration               │ freeze
         ▼                                           ▼
┌──────────────────┐                      ┌──────────────────────┐
│ Fiber lifecycle  │                      │ existing ToolRegistry│
└──────────────────┘                      │ candidate snapshot   │
                                          └──────────┬───────────┘
                                                     │ committed lease
                                                     ▼
                                          ┌──────────────────────┐
                                          │ ToolExecutor pipeline │
                                          │ hooks → events → tool │
                                          └──────────────────────┘
```

## Ownership and invariants

- `PluginTools` 只收集声明；不执行 Tool、不复制 registry、不拥有晋升。
- 插件身份来自 `ctx.runtime.plugin_id`，插件不能伪造 `source_name`。
- 每次注册是所属 Fiber 的 Effect；激活失败、reload 或 dispose 精确撤销。
- 同一候选 Root 内 Tool 名称重复立即失败；与 builtin、v2 或 MCP 的冲突在候选 snapshot 编译时失败，不修改稳定 registry。
- Core 未配置 `ToolRegistry` 时，包含 Tool 声明的候选失败；声明不能静默消失。
- `PluginTools.freeze()` 保留注册顺序并返回不可变 snapshot 输入；冻结后禁止新增声明。
- `ToolRegistry` 继续独占 schema、risk metadata、搜索授权和真实执行。
- `ToolExecutor` 继续独占 `legacy pre hook → invoker → legacy post hook` 顺序，并在这个真实 owner 内声明两个 typed event。

## Public seam

- `PLUGIN_TOOLS = ServiceKey("core.tools")`
- `PluginTools.register(ctx, tool, *, risk, always_on=False, preloadable=True, requires_turn_search=False, search_hint=None)`
- `risk` 第一版只接受 `read-only`、`write`、`external-side-effect`。
- `agent.tool_hooks.executor.TOOL_EXECUTION_BEFORE`：执行 owner 声明的串行事件 `tool.execute.before_invoker`，位于 legacy pre hook 完成改参之后、invoker 之前；payload 参数为只读映射。listener 只能返回 `None` 或 `Bail(non-empty reason)`，不能改写参数。
- `agent.tool_hooks.executor.TOOL_EXECUTION_AFTER`：执行 owner 声明的串行事件 `tool.execute.after_pipeline`，在 success、denied、error 三种结果上各触发一次；不接受 Bail。listener 失败转为明确 Tool error，使候选验证不能把失败当成功。

现有 `ToolExecutor.preflight()` 不执行 invoker，因此不发出这两个执行事件。没有绑定 RuntimeSnapshot 或 snapshot 没有 CompositionRoot 时保持旧行为。

## Selective DSH translation

吸收 DSH `ToolRuntime.register()` 的 scoped disposer、声明与执行分离、pre/after 执行事件思想；不转译它的第二套 registry、input waterfall、presentation/output mode、code-mode、并发工具限制或 scope restriction。Akashic 已经有这些问题的 owner 或尚无真实消费者，整套照搬会形成双 owner。

旧 v2 `@tool` 和 tool hook 保持原样。本 PR 是底座实验，不迁移 Meme、Citation、GitHub Watcher 或其他正式插件。

## Change intent

```yaml
change_type: additive
semantic_delta: none for existing plugins
capability_owner: "Core owns registry compilation, execution order and promotion; plugin owns Tool implementation and listeners."
consumer_scope:
  - v3 composition plugins
protected_state:
  - stable RuntimeSnapshot and ToolRegistry
  - formal workspace and plugin-data
  - existing v2 tools and hooks
allowed_effects:
  - temporary namespace plugins and isolated tool calls in tests
forbidden_effects:
  - formal plugin migration or promotion
  - external network, channel delivery or persistent workspace writes
rollback: "Revert this adjacent PR or return to backup/plugin-tools-before-20260814; v2 tools remain unchanged."
```

## Verification

- real namespace fixture proves metadata compilation, source ownership and snapshot-only registration;
- real RuntimeSnapshot lease proves legacy rewrite precedes the read-only before event, then invoker and after event run in exact order;
- deny prevents invoker, listener failure becomes error, and after fires exactly once for every settled status;
- name conflict rejects the candidate while the base registry identity and contents remain unchanged;
- duplicate activation rollback and a leaked-disposer mutant prove Effect cleanup oracle sensitivity;
- public plugin generation Gate binds these observations to the exact source digest.

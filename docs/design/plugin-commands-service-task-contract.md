# 插件 Commands 组合能力任务合同

- 状态：accepted / implementation approved
- 日期：2026-08-14
- 目标分支：`codex/plugin-event-import-contracts` 之后的 Draft stacked PR
- 恢复点：`backup/plugin-commands-before-20260814@435bbaf1e6f777951b4622702d7a7796665ed67a`
- 上游：[0038](../decisions/0038-human-commands-are-not-model-tools.md)、[0037](../decisions/0037-plugin-services-name-capabilities-not-categories.md)、[插件元能力底座与测试先行合同](plugin-meta-capability-task-contract.md)
- 参考实现：`/mnt/data/source-code/deepseek-harness@47f943859bef60e4160492346772ded9b24f765a/packages/interaction/commands`

## Goal

给 v3 插件提供 Fiber-owned 人类命令注册入口，并把不可变目录编译进 `RuntimeSnapshot`。Core 在 Session acquisition 和模型之前执行已知命令；插件只实现命令行为，不取得模型 Tool registry、`SessionManager`、持久 repository 或任意 SQL。

```text
┌──────────────────┐ inject core.commands ┌────────────────────┐
│ v3 plugin Fiber  │ ────────────────────▶ │ PluginCommands     │
│ command handler  │   Effect register     │ candidate Root only│
└────────┬─────────┘                       └─────────┬──────────┘
         │ dispose removes exact names                │ freeze
         ▼                                            ▼
┌──────────────────┐                       ┌────────────────────┐
│ Fiber lifecycle  │                       │ RuntimeSnapshot    │
└──────────────────┘                       │ CommandRegistry    │
                                           └─────────┬──────────┘
                                                     │ stable lease
                                                     ▼
                                           ┌────────────────────┐
                                           │ PassiveTurn Phase 0│
                                           │ hit → direct reply  │
                                           │ miss → BeforeTurn   │
                                           └────────────────────┘
```

## Ownership and invariants

- `PluginCommands` 只拥有候选 Root 内的名称注册与冲突；不执行命令、不拥有晋升。
- `CommandRegistry` 是已冻结 snapshot 的不可变视图；descriptor 按 canonical name 排序，alias 只用于兼容准入。
- canonical name 与 alias 使用同一 namespace；重复立即失败，候选 Fiber 回滚已登记 Effect。
- 插件身份来自 `ctx.runtime.plugin_id`，注册、reload 和 dispose 都由所属 Fiber 回收。
- 已知命令只在绑定 stable RuntimeSnapshot 的普通 passive input 上执行；命中发生在 Session acquisition 前。
- 未知名称和非命令输入不调用 handler、不产生目录状态，继续进入现有 passive lifecycle。
- handler 返回值在 Core 边界必须是 `CommandResult`；抛出、非法结果或 command stage 失败不得静默转成成功。
- 第一版不记录命令生命周期，不写 `sessions.db/messages`、Session metadata、workspace 或 plugin-data。
- Telegram discovery adapter 读取 stable snapshot 的 canonical descriptor；Mobile discovery 留给后续 UI 合同，不从 alias 推断目录项。

## Public seam

- `COMMANDS = ServiceKey("core.commands")`
- `PluginCommands.register(ctx, CommandDefinition(...))`
- `CommandDefinition(name, description, handler, aliases=(), input_hint=None)`
- `CommandInvocation(name, raw_input, session_key, channel, chat_id, sender)`
- `CommandResult(kind, text)`，`kind` 为 `success` 或 `error`，第一版的 `text` 必须非空。
- `CommandRegistry.descriptors` 提供排序后的 immutable discovery catalog。
- `CommandRegistry.execute(line, *, session_key, channel, chat_id, sender)` 对未知命令返回 `None`，对已知命令返回 `CommandExecution`。

名称使用小写 `[a-z][a-z0-9_-]*`。为保持现有 Akashic 行为，输入解析接受命令大小写差异、首尾空白和 Telegram `@botname` 后缀；handler 收到 canonical name 与命令头之后的原始参数。

## Selective DSH translation

吸收 DSH 独立 Commands service、插件 scoped disposer、名称冲突、不可变排序目录、未知命令 admission miss 和模型外执行。第一版不转译 per-agent shadow layer、remote service、abort signal、change observer、`commandId` 或 `command/run`/`command/done` Session log。Akashic 当前只有全局插件 generation，且正式命令的行为等价要求零持久写入；这些能力在出现真实 consumer 前没有 owner。

## Change intent

```yaml
change_type: additive
semantic_delta: none for existing plugins
capability_owner: "Core owns command namespace, stable admission and result boundary; plugin owns command behavior and text."
consumer_scope:
  - v3 human-command plugins
  - existing Telegram command discovery adapter
runtime_patch: required
runtime_patch_reason: "已知命令必须在共享 Session/LLM 准入前按 stable snapshot 执行；插件内 lifecycle 模块无法独占跨插件名称或保证不先创建 Session。"
authoritative_state_owner: "No durable command state in v1; RuntimeSnapshot owns only the immutable active registry view."
protected_state:
  - formal workspace, sessions and plugin-data
  - stable and latest plugin generations
  - existing v2 lifecycle commands and model Tools
allowed_effects:
  - temporary namespace plugins and in-memory command registrations
  - recording outbound adapter in tests
forbidden_effects:
  - formal plugin migration or promotion
  - Session/database/file writes
  - real channel, model, GitHub or external API calls
rollback: "Revert this adjacent PR or return to backup/plugin-commands-before-20260814; v2 command modules remain unchanged."
```

## Verification

- unit fixture 覆盖 canonical/alias namespace、大小写、`@botname`、raw input、排序目录与未知命令 miss；
- real namespace plugin 经 `PluginManager` 进入 candidate Root 和 RuntimeSnapshot；
- stable lease 下的真实 `PassiveTurnPipeline` 命中命令，证明 Session acquire、Context prepare、Reasoner 与 append 均未调用，且只发送一个 short-circuit outbound；
- 未知命令继续进入原有 `BeforeTurn` 路径；
- duplicate alias 触发真实 Fiber rollback，泄漏 disposer mutant 被同一 oracle 杀死；
- manager teardown 后 effect、service 与 command catalog 归零；
- public change-impact Gate 绑定 exact source digest。

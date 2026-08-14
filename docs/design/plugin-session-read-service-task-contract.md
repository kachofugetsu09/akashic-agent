# 插件 Session Read 组合能力任务合同

- 状态：accepted / implementation approved
- 日期：2026-08-14
- 目标分支：`codex/plugin-commands` 之后的 Draft stacked PR
- 恢复点：`backup/plugin-session-read-before-20260814@8c45ab78`
- 上游：[0037](../decisions/0037-plugin-services-name-capabilities-not-categories.md)、[0038](../decisions/0038-human-commands-are-not-model-tools.md)、[Cordis 插件迁移能力等价验收](cordis-plugin-capability-parity.md)

## Goal

为 `status_commands` 提供一个只读取既有 Session 的组合能力。Core 保留 Session 身份、持久化和缺失语义；插件只得到脱离 Core 缓存的消息与整理游标快照，自行实现记忆状态 projection、命令文本和 Mobile DTO。

```text
┌──────────────────┐  inject core.session_read  ┌──────────────────┐
│ v3 plugin Fiber  │ ─────────────────────────▶ │ SessionReadService│
│ owns projection  │                            │ owns read boundary│
└────────┬─────────┘                            └────────┬─────────┘
         │ read(session_key)                              │ get_existing
         ▼                                                ▼
┌──────────────────┐                            ┌──────────────────┐
│ detached snapshot│ ◀───────────────────────── │ SessionManager   │
│ no write methods │                            │ persistence owner│
└──────────────────┘                            └──────────────────┘
```

## Ownership and public seam

- `SESSION_READ = ServiceKey("core.session_read")`。
- `SessionReadService.read(session_key) -> SessionReadSnapshot | None`。
- Core 只向 Service 注入 `SessionManager.get_existing`；缺失 Session 返回 `None`，其他存储损坏或读取失败继续 fail-loud。
- `SessionReadSnapshot` 只包含 `session_key`、脱离缓存的 `messages` 和 `last_consolidated`。第一版不暴露保存、删除、任意 SQL、Session metadata、turn 或附件接口。
- 消息顶层映射不可修改，嵌套对象经过深复制；插件修改自己的 projection 输入不能反向改变 Session cache 或数据库。
- 这项 Service 独占“插件只读既有 Session 且不能偶然创建”的边界，因此不是旧插件类别的翻译。DeepSeek Harness 的 `ctx.sessions` 是完整 Session host；Akashic 只转译当前真实 consumer 所需的最小读取面，不复制写入、fork、flush 或 event log。

## Persistence and effects

```yaml
change_type: additive
semantic_delta: none for existing plugins
capability_owner: "Core owns Session identity and reads; plugins own derived projections."
consumer_scope:
  - status_commands memory command and Mobile projection
runtime_patch: required
runtime_patch_reason: "Only Core can enforce get_existing-only access without exposing SessionManager or database ownership."
authoritative_state_owner: "SessionManager and SessionStore remain the sole Session persistence owners."
client_only_alternative: "A client cannot derive last_consolidated or enforce server-side non-creation from a local cache."
protected_state:
  - sessions.db and Session cache
  - formal workspace and plugin-data
  - existing v2 lifecycle and Mobile UI behavior
allowed_effects:
  - detached in-memory snapshots
  - temporary SessionDB and plugin roots in tests
forbidden_effects:
  - Session creation, save, update or deletion
  - arbitrary SQL or repository exposure
  - formal workspace, channel or external API writes
rollback: "Revert this adjacent PR or return to backup/plugin-session-read-before-20260814; v2 status_commands remains unchanged."
```

`sessions.db/messages` 正常路径继续只增加；本能力不 INSERT、UPDATE 或 DELETE。`sessions` 元数据也保持不变。缺失会话不创建空 metadata。恢复证据是读取前后的 `sessions.db*` 字节摘要、隔离数据库可重新打开，以及旧 v2 插件仍可由 legacy host 承载。

## Verification

- 单元 fixture 证明快照与 Core 缓存脱离，插件侧修改不会反向污染 Session；
- create-on-read mutant 与正确实现运行同一缺失会话 fixture，并被“返回 `None`、`get_or_create` 零调用”oracle 杀死；
- 真实 namespace loader 证明 `core.session_read` 注入、Session 内容可读、SessionDB 字节不变；
- Root dispose 后 Service 与 Effect 归零；
- public change-impact Gate 绑定 exact source digest 并回归 Session persistence、Mobile UI 和 plugin generation。

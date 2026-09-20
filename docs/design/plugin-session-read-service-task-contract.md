# 插件 Session Read 组合能力任务合同

- 状态：historical / superseded（2026-09-13）
- 日期：2026-08-17
- 首个 consumer：Status Commands
- 恢复点：`backup/c26-session-read-pre-integration-20260817@4ca22a5a`
- 上游：[持久化状态地图](persistence-state-map.md)、[插件 v3 生产替代清单](plugin-v3-production-readiness-checklist.md)

> `SESSION_READ` 的 Core 导出与注册已退役。当前插件通过 `MESSAGE_CATALOG` 和普通 Turn projection 读取消息，见[插件 V3 能力手册](plugin-v3-capabilities.md#42-messagesession-与上下文)。下文保留历史设计及其状态保护依据，不是当前 API 用法。

## Goal

为 v3 插件提供一个只读取既有 Session 的窄能力。Core 保留 Session 身份、缓存和持久化 owner；插件只得到脱离 Core 缓存的消息与 active compaction 边界，自行完成状态投影。

```text
┌──────────────────┐  inject core.session_read  ┌──────────────────┐
│ committed Fiber  │ ─────────────────────────▶ │ SessionReadService│
└────────┬─────────┘                            └────────┬─────────┘
         │ read(existing key)                            │ get_existing
         ▼                                               ▼
┌──────────────────┐                            ┌──────────────────┐
│ detached snapshot│ ◀───────────────────────── │ SessionManager   │
│ no write methods │                            │ persistence owner│
└──────────────────┘                            └──────────────────┘

candidate Root ── same Service key ──▶ read() fail-loud, zero Session access
```

## Ownership and behavior

- `SESSION_READ = ServiceKey("core.session_read")`。
- `SessionReadService.read(session_key) -> SessionReadSnapshot | None`。
- formal Root 只注入 `SessionManager.get_existing`；缺失 Session 返回 `None`，其他读取错误继续 fail-loud。
- candidate Root 只保留同名 Service 的拓扑接线。插件可在 `apply()` 捕获能力，但任何读取正式 Session 的调用都立即失败，使 candidate 无法观察正式消息或数据库。
- `SessionReadSnapshot` 只包含 `session_key`、深复制后的 `messages`、`compaction_generation` 和 `consolidated_through_seq`，不暴露保存、删除、任意 SQL、metadata、turn 或附件接口。
- `last_consolidated` 是 compaction generation，不是消息数组下标。Core 同时读取 active ledger row 并核对 generation；插件只能用 `consolidated_through_seq` 判断消息是否已被整理。
- 顶层消息映射不可修改；嵌套对象与 Core cache 脱离。插件修改自己的 projection 输入不会反向改变 Session cache 或数据库。

## Persistence and failure

这项能力不拥有持久化状态。`sessions.db/messages`、Session metadata 和 cache 仍由 `SessionManager`/`SessionStore` 独占；正常读取不 INSERT、UPDATE 或 DELETE，缺失查询也不创建空 Session。

当前恢复范围只包含两类：

1. 进程内读取失败或取消：不捕获为成功，不留下写入或插件持有的可变 Session 引用；
2. 进程崩溃后重启：没有待恢复日志或中间状态，重新从既有 `sessions.db` 读取。

断电中途、任意磁盘断点和文件系统损坏不属于本能力的恢复协议。

## Verification

- 正式 Root 读取前后 `sessions.db*` 摘要一致，缺失 key 不创建 Session；
- 快照与 Core cache 脱离，插件侧修改不反向污染；
- candidate 调用 `read()` 在任何 Session lookup 前 fail-loud，stable snapshot 与数据库摘要保持不变；
- Manager terminate 后 Root Service/Effect 均为空；
- Status Commands 真实 Manager Gate 证明 committed command 与 Mobile query 都只消费这项只读能力。

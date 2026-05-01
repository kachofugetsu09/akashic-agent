# status_commands 插件

内置诊断命令拦截器。在 BeforeTurn 管道的早期阶段识别 `/memory_status` 和 `/kvcache` 命令，直接返回诊断报告，绕过后续的记忆检索和 LLM 推理。

---

## 接入点

| 接入方式 | 阶段 |
|---|---|
| `before_turn_modules_early()` | BeforeTurn 管道早期模块——命令识别与 abort |

---

## 运作逻辑

两个命令各对应一个 PhaseModule，均插入在记忆检索（`_PrepareContextModule`）之前。任意一个命中时，向 `session:ctx` slot 写入一个 `abort=True` 的 `BeforeTurnCtx`，后续管道模块及 LLM 推理全部跳过，直接返回该 slot 的内容作为本轮回复。

### MemoryStatusCommandModule（`/memory_status` / `/compact_status`）

读取当前 session 的 `messages` 列表和 `last_consolidated` 指针，统计：

- 已整理到的用户消息数量（`last_consolidated` 之前）。
- 尚未整理的用户消息数量。
- 最后一条已整理用户消息的预览。
- 当前会话总消息数。

格式化为可读文本后作为 abort_reply 返回。只统计"真实用户消息"（role=user 且非 context frame 占位符）。

### KVCacheCommandModule（`/kvcache` / `/cache_status`）

查询 observe 数据库（`observe/observe.db`），从 `turns` 表取最近 N 轮（默认 5，可追加参数覆盖，最大 30）的 KVCache 统计字段：

- `react_cache_prompt_tokens`：本轮送入的 prompt tokens 总量。
- `react_cache_hit_tokens`：命中缓存的 tokens 数量。

计算每轮命中率和总体命中率，格式化为表格后返回。若 observe 数据库不存在则返回提示信息。

# recall_inspector 插件

记忆召回诊断记录器。把每轮 turn 的上下文准备结果和 `recall_memory` 工具调用结果以 JSONL 追加写入本地文件，供离线分析和 dashboard 可视化。

---

## 接入点

| 接入方式 | 阶段 |
|---|---|
| `before_turn_modules_late()` | BeforeTurn 管道晚期模块——记录上下文准备结果 |
| `@on_tool_result()` TAP | 工具返回后——记录 recall_memory 结果 |

---

## 运作逻辑

### 1. 初始化（initialize）

创建 `plugins/recall_inspector/.data/recall_turns.jsonl` 文件及目录，初始化写入锁（`threading.RLock`）和 `_active_turns` 字典（用于将同一轮 turn 的多条记录关联到同一个 `turn_id`）。

### 2. 记录上下文准备（ContextPrepareRecordModule）

插入在 BeforeTurn 管道的 late 位置（EventBus emit 之后），此时 `BeforeTurnCtx` 已由记忆检索模块填充完毕。读取以下字段并写一条 `kind=context_prepare` 记录：

- `retrieved_memory_block`：注入给 LLM 的记忆文本块，解析出 item_id 列表。
- `retrieval_trace_raw`：检索过程的原始 trace（包含每条命中的 score、memory_type、injected 标志）。
- `session_key`、`content`、`timestamp` 等基本字段。

同时把 `turn_id`（由 session_key + timestamp + content 的 SHA-1 前 16 位生成）存入 `_active_turns[session_key]`，供后续工具记录关联使用。

### 3. 记录工具调用结果（record_recall_memory）

TAP 钩子监听所有工具返回事件，过滤出 `tool_name == "recall_memory"` 的调用。从 `event.result`（JSON 字符串）解析出 `items` 列表，对每条 item 提取 id、memory_type、summary、score 等字段，写一条 `kind=recall_memory` 记录，并通过 `_active_turns` 关联到对应的 `turn_id`。

### 4. 数据结构

所有记录追加写入 `recall_turns.jsonl`，每行一个 JSON 对象。`dashboard.py` 提供 FastAPI 路由，从该文件读取数据并聚合成 per-turn 视图，用于分析记忆召回质量。

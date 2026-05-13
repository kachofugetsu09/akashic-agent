# 记忆系统——Markdown 文件层

akashic 的记忆分为两层：**Markdown 文件层**（人类可读，LLM 直接写入）和**向量数据库层**（`memory2.db`，语义检索）。本文档只讲 Markdown 层——哪几个文件、各自干什么、consolidation 怎么把对话变成记忆。

---

## 五个 Markdown 文件

都在 `~/.akashic/workspace/memory/` 下：

| 文件 | 写者 | 读方 | 用途 |
|------|------|------|------|
| **MEMORY.md** | Optimizer（主 agent 自动维护） | 被动/主动 agent 的 system prompt | 长期记忆——用户的稳定事实、偏好、身份 |
| **SELF.md** | Optimizer（主 agent 自动维护） | 被动/主动 agent 的 system prompt | Akashic 的自我认知——形象、对用户的理解、关系定义 |
| **HISTORY.md** | Consolidation worker 自动追加 | 被动 agent（检索时 grep）、consolidation 自身（取最近 3 条做上下文） | 按时间线的事件日志，只追加不修改 |
| **RECENT_CONTEXT.md** | Consolidation worker 自动维护 | 被动/主动 agent 的 system prompt | 近期上下文摘要——最近在聊什么、关注什么 |
| **PENDING.md** | Consolidation worker 追加 → Optimizer 消费后清空 | Optimizer | 缓冲队列——从对话中提取的待归档事实 |

还有一个 `NOW.md` 在工作区根目录下（不是 `memory/` 里），是纯用户维护的备忘录，agent 不自动读写。

---

## Consolidation：对话怎么变成记忆

每次 agent 回复完后会触发一次 consolidation 检查。不是每条消息都触发——有一个最小新消息数的门槛。

### 什么时候触发

```
TurnCommitted 事件
  → MarkdownMemoryMaintenance._should_consolidate_session()
    → 检查：(新消息数 - 保留数 - 上次 consolidate 的位置) >= min
       min = max(5, keep_count // 2)
  → 够了就调 consolidation，不够就只刷新 RECENT_CONTEXT.md 的 Recent Turns 块
```

### Consolidation 干了什么

一次 consolidation 会调用 LLM 做两件事，然后写三个文件：

```
最近 N 条对话消息（文本格式）
    ↓
  LLM 提取（一段 prompt，一次调用）
    ↓
  产出两样东西：
    1. history_entries[]  — 时间线事件，每条 {summary: "1-2句摘要", emotional_weight: 0-10}
    2. pending_items[]    — 可归档的长期事实，每条 {tag: "identity|preference|...", content: "..."}
    ↓
  写入三个 markdown 文件：
    • HISTORY.md    — 追加 history_entries（幂等，按 source_ref 去重）
    • PENDING.md    — 追加 pending_items（幂等，按 source_ref 去重）
    • RECENT_CONTEXT.md — 用第二次 LLM 调用生成 Compression 摘要
    • journal/YYYY-MM-DD.md — 同 history_entries，按天分文件
```

**HISTORY.md 格式**：每条一行，前面有不可见的 consolidation 标记（`<!-- consolidation:["msg_id1","msg_id2"]:history_entry -->`），正常渲染看不到。

```
[2026-05-09 14:30] 用户开始学习 Rust 语言，购买了《Rust程序设计》第二版。
[2026-05-09 15:00] 用户表示不喜欢悬疑压抑风格的游戏。
```

**PENDING.md 格式**：每条 `- [tag] 内容`，支持 6 种 tag：

| Tag | 含义 | 例子 |
|-----|------|------|
| `identity` | 稳定身份事实 | `- [identity] 用户是互联网公司产品经理` |
| `preference` | 稳定偏好/禁忌 | `- [preference] 用户不喜欢悬疑压抑风格的游戏` |
| `key_info` | 密钥/账号/ID | `- [key_info] 用户的 GitHub 用户名是 example` |
| `health_long_term` | 长期健康事实 | `- [health_long_term] 用户有慢性偏头痛` |
| `requested_memory` | 用户明确要求记住 | `- [requested_memory] 项目 deadline 是 6 月 15 日` |
| `correction` | 更正已有记忆 | `- [correction] 更正：用户不是学生，已毕业` |

**RECENT_CONTEXT.md 格式**：

```markdown
# Recent Context

## Compression
until: 2026-05-09T15:00:00
- 最近持续关注：用户最近在讨论记忆检索架构重构
- 最近明确偏好：偏好低压力、能长期坚持的创作方式
- 最近待延续话题：上次未完成的代码讨论
- 最近避免事项：不要在不适当的时机讨论技术架构

## Ongoing Threads
- 用户最近持续受睡眠问题影响，情绪低落

## Recent Turns
[user] 我头有点疼，可能昨晚没睡好
[a-preview] 先休息一下，不要勉强
```

Compression 部分由第二次 LLM 调用生成，有严格规则：**只从 USER 消息里提取，不把 assistant 的建议当事实**。Recent Turns 部分是轻量的——每次 turn 后单独刷新，不触发完整 LLM 调用。

### 幂等性保证

`consolidation_writes.db`（SQLite）用 `source_ref`（一条 JSON 数组，比如 `["msg_001","msg_002"]`）做主键，同一批消息不会写两次。HISTORY.md 和 PENDING.md 内部的隐藏标记也做第二层保护。

PENDING.md 还有两阶段提交：`snapshot_pending()` → Optimizer 处理 → `commit_pending_snapshot()` 或 `rollback_pending_snapshot()`。启动时如果发现残留 snapshot 会自动回滚合并，防止崩溃丢数据。

---

## PENDING → MEMORY：Optimizer 怎么归档

PENDING.md 只是缓冲——consolidation 把新事实写进 PENDING，但 PENDING 不注入 system prompt。真正的长期记忆 MEMORY.md 交给一个定时任务（Optimizer）来更新。

**为什么要隔一层？为了缓存。**

MEMORY.md 是全文注入 system prompt 的。如果每次 consolidation 都直接改 MEMORY.md，那每轮对话的 system prompt 都不一样 → DeepSeek 的 prompt cache 永远命中不了 → 每轮都多花几秒和一份 cache_write token。Optimizer 把 MEMORY.md 的更新频率降到 18 小时一次，中间攒在 PENDING.md 里不动它，让 prompt cache 能稳定命中几十上百轮。

```
consolidation (每 N 条消息触发一次)
    ↓
  写入 PENDING.md（高频追加，不改 MEMORY.md）
    ↓
  MEMORY.md 保持不变 → prompt cache 持续命中
    ↓
Optimizer (定时触发)
    ↓
  读 MEMORY.md + PENDING.md → LLM 归档 → 一次性地更新 MEMORY.md → 清空 PENDING.md
    ↓
  MEMORY.md 变化一次 → prompt cache miss 一次 → 下一轮重新建立缓存
```

### Optimizer 怎么工作的

```
PENDING.md（增量事实缓冲区）
    ↓
  Optimizer 定时触发（默认 memory_optimizer_interval_seconds = 64800）
    ↓
  读 MEMORY.md + PENDING.md → LLM 做归档决策：
    • 新事实 → 写入 MEMORY.md 对应分类
    • 与已有条目冲突 → 更新或替换
    • 重复 → 忽略
    • 更正 → 用 correction tag 的内容覆盖旧条目
    ↓
  写入新的 MEMORY.md
  clear_pending() → 清空 PENDING.md
```

Optimizer 在主 agent 启动时注册为后台任务（`memory_optimizer_enabled = true`）。

---

## 这些文件怎么进入 System Prompt

每次回复前构建 system prompt 时，按 priority 顺序依次渲染：

| Priority | 块名 | 来源文件 | 注入形式 |
|----------|------|---------|---------|
| 30 | SelfModel | `SELF.md` | `## Akashic 自我认知\n\n{全文}` |
| 35 | LongTermMemory | `MEMORY.md` | `## Long-term Memory\n\n{全文}` |
| 45 | RecentContext | `RECENT_CONTEXT.md` | Compression + Ongoing Threads（不含 Recent Turns，那个有独立的滑动窗口） |
| 55 | MemoryBlock | 向量检索结果 | `recall_memory` 的语义召回块（带 [id] 前缀和时间戳） |

**注意**：MEMORY.md 和 SELF.md 是**全文注入**的，不做截断或检索——这也是为什么 Optimizer 需要保持它们紧凑。HISTORY.md 不直接注入 prompt，只用于 grep 检索和 consolidation 自身的上下文。

---

## 文件流转总览

```
用户发消息
    ↓
system prompt 注入：SELF.md + MEMORY.md + RECENT_CONTEXT.md + 向量检索块
    ↓
LLM 回复（记忆在上下文里）
    ↓
TurnCommitted
    ↓
┌─ 新消息不够 → 只刷新 RECENT_CONTEXT.md 的 Recent Turns 块
└─ 新消息够了 →
    ├─ LLM 提取 history_entries + pending_items
    ├─ 写入 HISTORY.md（追加，幂等）
    ├─ 写入 PENDING.md（追加，幂等）
    ├─ 写入 RECENT_CONTEXT.md（LLM 生成 Compression + 刷新 Recent Turns）
    └─ 写入 journal/YYYY-MM-DD.md（追加）
    ↓
Optimizer 定时任务
    ├─ 读 PENDING.md + MEMORY.md → LLM 归档 → 更新 MEMORY.md
    └─ clear_pending()
```

两层记忆的分工：**Markdown 层**管"人类能看懂的全景"——你是谁、你喜欢什么、最近发生了什么。**向量层**管"机器能搜到的细节"——语义检索、时间过滤、去重计数。两个层各自独立更新，通过 `ConsolidationCommitted` 事件桥接——markdown 层写完 HISTORY 后发事件，向量层收到后把同批数据 embed 写入 `memory2.db`。

---

## 向量记忆 API——谁在调它

`core/memory/engine.py` 定义了一套抽象协议（`MemoryEngine`），由四个子协议组成。引擎本身是一个 **plugin**——`[memory].engine` 配置项指定用哪个实现，留空 = `default_memory` 插件。协议与实现解耦。

### API 协议一览

**MemoryIngestApi** — 程序化内容摄入

| 方法 | 用途 |
|------|------|
| `ingest(request)` | 摄入一条内容（目前无生产级调用方，仅 default_memory 内部使用） |

**MemoryRetrievalApi** — 语义检索

| 方法 | 调用方 |
|------|--------|
| `retrieve(request)` | 被动回复每轮自动检索（注入 system prompt） |
| `retrieve_explicit(request)` | `recall_memory` 工具（LLM 主动调用） |
| `retrieve_interest_block(request)` | proactive tick 中评估内容候选时 |

**MemoryWriteApi** — 写入/删除

| 方法 | 调用方 |
|------|--------|
| `remember(request)` | `memorize` 工具（用户要求"记住..."） |
| `forget(request)` | `forget_memory` 工具（用户纠正错误记忆） |
| `reinforce_items_batch(ids)` | 仅 default_memory 内部调用，无外部调用点 |

**MemoryAdminApi** — 管理面板

| 方法 | 调用方 |
|------|--------|
| `list_items_for_dashboard(...)` | Dashboard `GET /api/dashboard/memories` |
| `get_item_for_dashboard(id)` | Dashboard `GET /api/dashboard/memories/{id}` |
| `update_item_for_dashboard(id, ...)` | Dashboard `PATCH /api/dashboard/memories/{id}` |
| `delete_item(id)` | Dashboard `DELETE /api/dashboard/memories/{id}` |
| `delete_items_batch(ids)` | Dashboard `POST /api/dashboard/memories/batch-delete` |
| `find_similar_items_for_dashboard(id)` | Dashboard `GET /api/dashboard/memories/{id}/similar` |
| `describe()` | 引擎描述（仅 default_memory 内部） |
| `keyword_match_procedures(tokens)` | 关键词匹配过程（仅 default_memory 内部） |
| `list_events_by_time_range(start, end)` | 时间范围事件列表（仅 default_memory 内部） |

### 调用点汇总

所有不在 `plugins/default_memory/` 下的真实调用点：

| 位置 | 行 | 方法 | 说明 |
|------|-----|------|------|
| `agent/retrieval/default_pipeline.py` | 31 | `retrieve()` | 被动 turn 每轮的语义检索入口。`DefaultContextStore.prepare()` → `retrieval_pipeline.retrieve()` → `engine.retrieve(MemoryEngineRetrieveRequest)` |
| `agent/tools/recall_memory.py` | 138 | `retrieve_explicit()` | `recall_memory` 工具。LLM 传入 query/memory_type/time_filter/limit，工具调 `facade.retrieve_explicit(ExplicitRetrievalRequest)` |
| `proactive_v2/tools.py` | 220 | `retrieve_interest_block()` | proactive tick 中评估内容候选时调用，查询用户对某条内容是否可能感兴趣 |
| `agent/tools/memorize.py` | 71 | `remember()` | `memorize` 工具。用户要求"记住..."时调 `engine.remember(RememberRequest)` |
| `agent/tools/forget_memory.py` | 55 | `forget()` | `forget_memory` 工具。用户纠正错误记忆时调 `memory.forget(ForgetRequest)` |
| `bootstrap/dashboard_api.py` | 1032 | `list_items_for_dashboard()` | Dashboard 记忆列表页 |
| `bootstrap/dashboard_api.py` | 1063 | `find_similar_items_for_dashboard()` | Dashboard 相似记忆查询 |
| `bootstrap/dashboard_api.py` | 1085 | `get_item_for_dashboard()` | Dashboard 单条记忆详情 |
| `bootstrap/dashboard_api.py` | 1099 | `update_item_for_dashboard()` | Dashboard 编辑记忆 |
| `bootstrap/dashboard_api.py` | 1115 | `delete_item()` | Dashboard 删除单条 |
| `bootstrap/dashboard_api.py` | 1122 | `delete_items_batch()` | Dashboard 批量删除 |
| `core/memory/runtime.py` | 51/57/63 | `retrieve()` / `retrieve_explicit()` / `retrieve_interest_block()` | `MemoryRuntime` 薄封装层，统一入口 |
| `core/memory/plugin.py` | 80-126 | 全部方法 | `DisabledMemoryEngine` — 记忆关闭时的空实现，所有方法都返回空/报错 |

### 引擎如何注入

```
bootstrap/memory.py:116 → engine = plugin_runtime.engine
    ↓
  bootstrap/tools.py:231 → memory_engine 注入 ToolsetDeps
    ↓
    ├── 被动 reply: DefaultMemoryRetrievalPipeline(memory=...) → engine.retrieve()
    ├── 工具注册: memorize/forget/recall_memory 拿到 engine 引用
    └── PluginManager(memory_engine=...) → 所有插件都能通过 self.context.memory_engine 访问
    ↓
  bootstrap/app.py:145 → memory_admin=engine → Dashboard API
  proactive_v2/tools.py:38 → ToolDeps.memory → engine.retrieve_interest_block()
```

调用方拿到的都是协议类型（`MemoryRetrievalApi` / `MemoryWriteApi` / `MemoryAdminApi`），不是 `DefaultMemoryEngine`。换引擎实现只需要改 `config.toml` 的 `[memory].engine`。

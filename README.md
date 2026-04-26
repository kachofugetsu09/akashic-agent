# akashic Agent

---

## Quickstart

**1. 初始化**

```bash
git clone <this-repo>
cd akashic-agent
python main.py init
```

`init` 会做两件事：把 `config.example.toml` 复制为 `config.toml`，并在 `~/.akashic/workspace/` 下创建运行时所需的全部文件和数据库：

```
~/.akashic/workspace/
  memory/
    MEMORY.md          # 长期记忆（空）
    SELF.md            # 自我认知（空）
    HISTORY.md         # 事件日志（空）
    RECENT_CONTEXT.md  # 近期上下文摘要（空）
    PENDING.md         # 待提取事实（空）
    NOW.md             # 近期进行中 / 待确认事项（模板）
    memory2.db         # 语义记忆数据库（memory.enabled=true 时）
    consolidation_writes.db  # 归档写入记录
  PROACTIVE_CONTEXT.md # 主动推送规则文件（模板）
  mcp_servers.json     # MCP server 注册表
  schedules.json       # 定时任务列表
  proactive_sources.json  # 信息源列表
  memes/manifest.json  # 表情包清单
  skills/              # 用户自定义 skill 目录
  drift/skills/        # 用户自定义 Drift skill 目录
  sessions.db          # 会话存储
  observe/observe.db   # trace 数据库
  proactive.db         # proactive 状态数据库
  proactive_quota.json # proactive 配额
```

**2. 填写配置**

编辑 `config.toml`，至少要改 API key 和频道 token。推荐配置（DeepSeek 主模型 + Qwen 轻量/视觉）：
推荐如果非多模态模型可以用deepseek-v4-flash,他的agent能力是nextlevel的
```toml
[llm]
provider = "deepseek"

[llm.main]
model = "deepseek-v4-flash"     # 主模型：推理能力强、速度快、价格低
api_key = "sk-..."              # DeepSeek API key
base_url = "https://api.deepseek.com/v1"
enable_thinking = true          # 开启 reasoning（R1 模式）
multimodal = false              # DeepSeek 不支持图片，用 VL 工具补

[llm.fast]
model = "qwen-flash"            # 轻量模型：memory gate / query rewrite / HyDE
api_key = "sk-..."              # Qwen API key（DashScope）
base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"

[llm.vl]
model = "qwen-vl-plus"          # 视觉模型：主模型 multimodal=false 时自动启用
api_key = "sk-..."              # 同 Qwen key
base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"

[channels.telegram]
token = "123456:ABC..."         # BotFather 给的 bot token
allow_from = ["your_username"]  # 你的 Telegram 用户名（不带 @）
```

也可以用 Qwen 全家桶（主模型换 `qwen3.5-plus`，`multimodal = true`，`llm.vl` 留空），或者用 OpenAI 兼容的任意 provider。

**图像能力**

| 路线 | 配置 | 效果 |
|------|------|------|
| A: 主模型多模态 | `multimodal = true`，`llm.vl.model = ""` | 图片直接 `image_url` 进主模型 |
| B: 主模型 + VL 工具 | `multimodal = false`，`llm.vl.model = "qwen-vl-plus"` | 图片转路径提示，模型按需调用 `read_image_vision` |
| C: 纯文本 | `multimodal = false`，`llm.vl.model = ""` | 图片不可理解，只保留路径 |

路线 B 是推荐方案：主模型用 DeepSeek 这类纯文本强模型，图片理解交给专门的 VL 模型，性价比最高。

**3. 启动并发消息**

```bash
python main.py
```

打开 Telegram，找到你的 bot，发一条消息，就可以开始对话。

**4. 配置 Proactive**

`config.example.toml` 默认 `proactive.enabled = true`。填上你的 Telegram chat_id（可以向 bot 发一条消息后从日志里拿到），agent 就会在订阅的信息源有内容时主动推送消息。如果不需要主动推送，设为 `enabled = false`。

```toml
[proactive.target]
channel = "telegram"
chat_id = "123456789"   # 你的 Telegram user id
```

**5. 打开 Drift**

```toml
[proactive.drift]
enabled = true
min_interval_hours = 3  # 每次 drift 最小间隔
```

Drift 打开后，没有可推送内容时，agent 会利用空闲时间自主执行 `drift/skills/` 下定义的任务，偶尔也会主动发一条消息。

---

## 链路说明

三条链路的记录。

---

## 一、被动回复链

用户发来一条消息，走完整条链，产出一条回复。

```
InboundMessage
  → AgentLoop._process()        # runtime 入口壳，管 timeout/processing state
  → CoreRunner.process()        # 分流：spawn completion / 普通消息
  → AgentCore.process()         # 主编排 facade
      ├─ ContextStore.prepare() # 读 session history + retrieval + skill mentions → ContextBundle
      ├─ build_system_prompt()
      ├─ tools.set_context()
      ├─ Reasoner.run_turn()     # 完整被动执行入口（retry / trim / preflight）
      │    └─ Reasoner.run()    # 底层 tool loop（llm call → tool exec → repeat guard → fallback）
      └─ ContextStore.commit()  # session append + observe + post_turn + meme + dispatch
  → OutboundMessage
```

五块大抽象各管一段：

| 块 | 职责 |
|----|------|
| `AgentLoop` | runtime 入口，不管业务细节 |
| `CoreRunner` | 分流，外层不需要知道内部事件和普通消息分别怎么跑 |
| `AgentCore` | 串 prepare / execute / commit，不吸收任何实现细节 |
| `ContextStore` | `prepare()` 管本轮输入，`commit()` 管本轮提交 |
| `Reasoner` | `run_turn()` 是完整执行入口，`run()` 是底层 tool loop 原语 |

`ContextBundle`（prepare 产出）：`history` / `skill_mentions` / `retrieved_memory_block` / `retrieval_trace_raw`

`TurnRunResult`（run_turn 产出）：`reply` / `tools_used` / `tool_chain` / `thinking` / `context_retry`

---

## 二、Proactive 信息源处理

主动推送链路在每个 tick 里，先于 agent loop 并行预取所有数据源。

```
AgentTick.tick()
  └─ Pre-gate（冷却 / 用户在线 / busy 检查）
       └─ DataGateway.run()          # 三路并行预取
            ├─ _fetch_alerts()       # 实时告警（完整内容，直接传给 agent）
            ├─ _fetch_context()      # 上下文条目（直接传给 agent）
            └─ _fetch_content()      # feed 内容（并行 web_fetch，存入 content_store）
                                     # agent 通过 get_content 工具按需取正文
       └─ _run_loop(ctx)             # agent loop，max 20 步
            工具：recall_memory / get_content / web_fetch
                  mark_interesting / mark_not_interesting / send_message
```

Gateway 的设计原则是：agent 启动前所有数据已经就位，形成一份本 tick 的静态输入快照。单源失败不影响其他源。

**ACK / 去重**：

| 场景 | TTL |
|------|-----|
| 已引用内容/告警 | 168h |
| interesting 未引用 | 24h |
| delivery/message 去重命中 | 24h |
| mark_not_interesting | 720h |

去重 key 优先按稳定 URL，其次 source+title，最后退化为 event_id；没有内容引用时用消息文本 hash。

---

## 三、Drift 链路

Proactive gateway 没有可推送内容时，进入 Drift 模式——agent 用一段空闲时间自主做一件有意义的事。

```
AgentTick.tick()
  └─ (gateway 没有可发内容，或 agent loop 决定不发)
       └─ DriftRunner.run(ctx, llm_fn)
            1. scan_skills()              # 扫描 workspace/skills/ 下的 SKILL.md 目录
            2. 过滤 requires_mcp 未满足的 skill
            3. 构建 system prompt         # 注入长期记忆 + RECENT_CONTEXT + skill 列表 + 最近运行记录
            4. tool loop（max 20 步）
                 工具：read_file / write_file / edit_file
                       recall_memory / web_fetch / web_search
                       fetch_messages / search_messages / shell
                       send_message（最多一次） / finish_drift
                       mount_server（可挂载 MCP server）
            5. 强制落地机制：
                 step N-3  注入警告提示
                 step N-2  限制 schema 为 write_file/edit_file，强制写文件
                 step N-1  强制调用 finish_drift
```

Drift 的核心约束：

- 每次进入都重新比较所有 skill，不默认继续上次的
- `send_message` 成功后只允许 `write_file` / `edit_file` / `finish_drift` 收尾
- 发出的消息要像自然聊天，不像在汇报内部执行流程
- 执行结束前必须调用 `finish_drift` 保存状态

---

## 其他命令

```bash
python main.py cli      # 连接运行中的 agent（TUI / 纯文本 CLI）

pytest tests/           # 单元测试
akashic_RUN_SCENARIOS=1 pytest -c pytest-scenarios.ini tests_scenarios/  # 场景测试（真实 LLM）
```

## Roadmap

### Memory System (记忆系统演进)
- [x] **Post-Response Invalidation**: `_enqueue_post_memory` 每轮对话后仅做失效检测——调用轻量 LLM 判断用户是否在否定/纠正旧记忆，命中则将旧 procedure/preference 标记 supersede。隐式新记忆抽取已移至 Consolidation 批量阶段，不再每轮调用。
- [x] **Recall Time Range**: `recall_memory` 支持 `grep + time_filter`，可直接列出一段时间内的 event 记忆。
- [x] **Semantic Time Filter**: `recall_memory` 支持 `semantic + time_filter`，可在指定时间窗口内做向量与关键词融合召回。
- [x] **Response Parser Pipeline**: 在 `AgentCore.process` 循环结尾引入统一的文本解析管道。拦截大模型原始输出，抽离出 `§cited:xxx§` 与 `<meme:xxx>` 等标签，将清洗后的干净文本与结构化 Metadata 传入 `ContextStore`，彻底解除底层的正则与协议耦合。
- [ ] **Memory Pluginification**: 提取 `DefaultMemory` 的全部行为，实现记忆系统的完全 SPI 解耦。

### Tools & Integration (工具与集成架构)
- [ ] **Tool Pluginification**: 重构静态的工具注册机制，将其从依赖注入列表解脱，抽象为动态的 Plugin 系统，作为大范围模块化（如 Memory SPI）的前置基建。
- [x] **Managed Subagent Completion**: 长时间任务不走 HTTP Callback URL，而是通过 `spawn` 创建受管 subagent，由 `SubagentManager` 追踪 `job_id`、运行状态和完成事件；任务结束或取消后统一回灌为后台任务完成消息，再由主循环安全合并进当前会话。配套 `spawn_manage` 支持查看和取消运行中的 subagent。
- [x] **Typed Internal Events**: `SpawnCompletionItem` 替代旧的 `InboundMessage` + metadata 编码方式，spawn 子 agent 完成回调以类型化 dataclass 承载 `SpawnCompletionEvent` + `SpawnDecision`，消除字符串 key 分发。

### Core Pipeline & Decoupling (管线架构解耦)
- [x] **Typed EventBus System**: 实现进程内类型化 EventBus（`bus/event_bus.py`），支持四种调度模式——`emit`（顺序拦截链，handler 可修改事件）、`observe`（顺序 fire-and-forget，异常自消化）、`fanout`（`asyncio.gather` 并发广播，汇总失败计数）、`enqueue`（后台队列异步消费，不阻塞主回复路径）。定义 8 种生命周期事件：`TurnStarted`、`TurnCompleted`、`TurnCommitted`、`BeforeReasoning`、`BeforeDispatch`、`ToolCallStarted`、`ToolCallCompleted`、`StreamDeltaReady`。
- [x] **TurnCommitted Post-Turn Unification**: 用单个 `TurnCommitted` fanout 事件统一提交后处理，完全取代 `agent/postturn/` 模块（删除 222 行）。三个 per-session 串行消费者——consolidation 调度、memory ingest（失效检测）、recent-context 刷新——通过 `lifecycle_consumers.py` 集中注册，各自内置 per-session 消费队列 + done callback + 重启逻辑，消除回调织入。核心管线 `commit()` 不再直接持有 consolidation / memory / trace 依赖。
- [x] **Event-Driven Streaming**: 流式输出改为 `StreamDeltaReady` 事件驱动——Reasoner 通过 EventBus 发射增量 delta，TelegramChannel 通过 `event_bus.on()` 自订阅消费。bootstrap 不再需手动调用 `set_stream_sink_factory()` 连接频道与管线。
- [x] **Tool Lifecycle Observability**: Reasoner 在每次工具调用前后发射 `ToolCallStarted` / `ToolCallCompleted` 事件（含 call_id、tool_name、arguments、status），被 blocked 的 deferred tool 调用同样发射，便于外部监控和调试。
- [x] **Empty-Reply + Thinking Retry**: LLM 返回完整 thinking 链但 content 为空时，Reasoner 自动追加一次无工具 retry 要求给出正式回复；retry 仍空则降级为固定占位文本，避免用户收到空白消息。

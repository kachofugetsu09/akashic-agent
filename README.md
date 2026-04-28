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

---

## 一、被动回复链

每条被动 turn 经 6 个生命周期 Phase 按序执行，产出一条回复。

```
InboundMessage
  → AgentLoop → CoreRunner → AgentCore (PassiveTurnPipeline)
      ├─ BeforeTurn        session → context → emit[插件介入] → output
      ├─ BeforeReasoning   tool sync → build ctx → emit → prompt 预热
      ├─ Reasoner.run_turn()   (retry / trim / tool loop)
      │    ├─ BeforeStep    token 估算 → emit → inject hints (每轮)
      │    └─ AfterStep     fanout 通知进度 (每轮)
      ├─ AfterReasoning    parse → emit → persist user/assistant → build outbound
      └─ AfterTurn         TurnCommitted fanout → emit → dispatch
  → OutboundMessage
```

### 生命周期模块链

每个 Phase 内部是一条模块链。核心抽象三件套：

| 抽象 | 说明 |
|------|------|
| `PhaseFrame[I, O]` | 数据盘。`input`（只读）、`slots`（模块间 dict 通道）、`output`（最终产出）。一次 run 一份 |
| `PhaseModule[F]` | 单个步骤。接口：`async (frame) → frame`。可选声明 `requires`/`produces` 供启动校验 |
| `Phase[I, O, F]` | 生产线。`Phase(modules, frame_factory=Frame)`，按序执行 module 列表 |

链式示例（BeforeTurn）：

```
Phase([_AcquireSession, _PrepareContext, _BuildCtx, _Emit(EventBus), _Return], frame_factory=BeforeTurnFrame)

Step1: session 从 manager 取出，写 slots["session:session"]
Step2: 上下文记忆检索，写 slots["session:context_bundle"]
Step3: 拼 BeforeTurnCtx，写 slots["session:ctx"]
Step4: EventBus.emit() 依序执行注册的插件 handler（可修改 ctx）
Step5: ctx → output，Phase 返回
```

内置模块和外部插件在链上是平等节点——都满足 `PhaseModule` 协议，都通过 slots 读写数据。`TurnLifecycle` facade 暴露注册入口（`on_before_turn`、`on_after_reasoning` 等），每个 Phase 在 emit 步骤执行已注册的 handler。每个 phase 提供 `default_xxx_modules()` 工厂函数，插件可以 `insert()` / `replace()` 任意模块。

`PassiveTurnPipeline` 将 6 个 Phase 串联为完整链路。Reasoner 内部的 `BeforeStep` / `AfterStep` 也是相同抽象，每次迭代执行一次。

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

### Done
- [x] **Lifecycle Module Chain**: 6 个生命周期 Phase 统一为模块链架构。`Phase(modules, frame_factory=Frame)` 按序执行 `PhaseModule` 列表，模块通过 `slots` 传递数据，声明 `requires`/`produces` 做启动校验。内置模块和插件模块在链上平等。替代旧的 `GatePhase`/`TapPhase` 两段式 `_setup → emit → _finalize` 模式。
- [x] **Typed EventBus**: 进程内类型化 EventBus，支持 `emit`（顺序拦截） / `fanout`（并发广播） / `enqueue`（后台消费）。TurnCommitted 事件统一提交后处理，替代 `agent/postturn/` 模块。
- [x] **Event-Driven Streaming**: 流式输出通过 EventBus 发射 `StreamDeltaReady`，频道自订阅消费，不再需要 `set_stream_sink_factory()`。
- [x] **Turn Lifecycle Observability**: 每条 turn 发射 `BeforeTurn` → `BeforeReasoning` → `ToolCallStarted/Completed`(per step) → `AfterReasoning` → `AfterTurn` 全链路事件。空回复自动 retry。
- [x] **Managed Subagent (spawn)**: 长时间任务通过 `spawn` 创建受管 subagent，完成/取消后统一回灌为后台任务完成消息，安全合并进当前会话。
- [x] **Memory**: 支持 recall 时间范围过滤、语义+关键词融合检索。post-response 失效检测标记 supersede 旧记忆。Consolidation 批量阶段抽取新记忆。
- [x] **Response Parser Pipeline**: 统一文本解析管道，从 LLM 原始输出抽离 `§cited:xxx§` / `<meme:xxx>` 标签，清洗后传入下游。

### Active
- [ ] **Lifecycle Plugin System**: 基于模块链提供稳定插入点 API——插件按 slot 名声明 `requires`/`produces`，`insert(after="session:session")` 方式注入自定义模块。开放 `default_xxx_modules()` 工厂函数外部使用。
- [ ] **Memory Pluginification**: 提取 `DefaultMemory` 行为为 SPI 接口。
- [ ] **Tool Pluginification**: 重构工具注册为动态 Plugin 系统。

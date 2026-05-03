[![欢迎加入交流群](https://img.shields.io/badge/QQ%E4%BA%A4%E6%B5%81%E7%BE%A4-%E6%AC%A2%E8%BF%8E%E5%8A%A0%E5%85%A5-2ea44f?style=for-the-badge)](./COMMUNICATION.md)

# akashic Agent

---

## Quickstart

**1. 安装依赖**

需要 Python 3.12。推荐用 `uv` 管理虚拟环境：

```bash
git clone <this-repo>
cd akashic-agent
uv venv                              # 创建 .venv
uv pip install -r requirements.txt  # 安装依赖
```

没有 uv？先装：`python -m pip install uv`

**2. 初始化（推荐用交互向导）**

```bash
uv run python main.py setup
```

向导会逐步引导你配置主模型、频道、记忆，自动生成 `config.toml` 并初始化工作区。Telegram 配置完成后会实时获取你的 `chat_id`，proactive 一步到位。

如果是自动化/CI 场景，用非交互模式：

```bash
uv run python main.py init   # 复制 config.example.toml，手动编辑后启动
```

`init` 创建的工作区结构：

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

**3. 填写配置**

编辑 `config.toml`，至少填写主模型 API key 和一个频道（Telegram 或 QQ 选一个）。推荐配置（DeepSeek 主模型 + Qwen 轻量/视觉）：
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

**4. 启动**

```bash
uv run python main.py
```

打开 Telegram，找到你的 bot，发一条消息，就可以开始对话。

**5. 配置 Proactive（可选）**

`config.example.toml` 默认 `proactive.enabled = true`。填上你的 Telegram chat_id（可以向 bot 发一条消息后从日志里拿到），agent 就会在订阅的信息源有内容时主动推送消息。如果不需要主动推送，设为 `enabled = false`。

```toml
[proactive.target]
channel = "telegram"
chat_id = "123456789"   # 你的 Telegram user id
```

**6. 打开 Drift**

```toml
[proactive.drift]
enabled = true
min_interval_hours = 3  # 每次 drift 最小间隔
```

Drift 打开后，没有可推送内容时，agent 会利用空闲时间自主执行 `drift/skills/` 下定义的任务。本轮不打扰用户时用 `finish_drift(message_result="silent")` 静默收尾；如果已经主动发消息，则必须用 `finish_drift(message_result="sent")` 收尾。

**7. 打开 Dashboard（可选）**

Dashboard 用来查看会话、消息、记忆、proactive 记录和插件面板：

```bash
uv run python main.py dashboard
```

默认监听 `0.0.0.0:2236`。如需改地址：

```bash
uv run python main.py dashboard --host 127.0.0.1 --port 2236
```

**8. 配置 MCP servers（可选）**

MCP server 注册表在工作区的 `mcp_servers.json`。也可以在对话里让 agent 调用 `mcp_add` 添加，手动配置格式如下：

```json
{
  "servers": {
    "calendar": {
      "command": ["python", "/path/to/run_server.py"],
      "env": {
        "GOOGLE_CLIENT_ID": "..."
      },
      "cwd": "/path/to"
    }
  }
}
```

启动时会读取这个文件并把 MCP 工具注册进工具列表。

**Troubleshooting**

- `chat_id` 不知道怎么拿：优先跑 `uv run python main.py setup`，向导会在 Telegram 配好后自动获取；手动配置时，先给 bot 发一条消息，再从日志里的 Telegram update 里取 user id。
- `uv` 装不上：先升级 pip，执行 `python -m pip install --upgrade pip`，再执行 `python -m pip install uv`。
- Dashboard 打不开：确认主程序没有占用同一端口，或用 `--port` 换一个端口。
- MCP server 没工具：先确认 `mcp_servers.json` 里的 `command` 能在终端单独启动，且需要的环境变量都在 `env` 里。

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
Phase([_AcquireSession, *early_modules, _PrepareContext, _BuildCtx,
       _Emit(EventBus), *late_modules, _Return], frame_factory=BeforeTurnFrame)

Step1: session 从 manager 取出，写 slots["session:session"]
Step1.5: ★ 插件 early module（如命令拦截 /memory_status，可 abort 跳过后续）
Step2: 上下文记忆检索，写 slots["session:context_bundle"]
Step3: 拼 BeforeTurnCtx，写 slots["session:ctx"]
Step4: EventBus.emit() 依序执行注册的插件 handler（可修改 ctx）
Step4.5: ★ 插件 late module（EventBus emit 后的补充处理）
Step5: ctx → output，Phase 返回
```

内置模块和外部插件在链上是平等节点——都满足 `PhaseModule` 协议，都通过 slots 读写数据。插件可以通过两种方式介入：
- **EventBus 装饰器**（`@on_before_turn`、`@on_after_reasoning` 等）：注入到每个 Phase 的 emit 步骤，GATE 可改写事件，TAP 只读观察
- **PhaseModule 插入点**（`before_turn_modules_early()` / `late()`）：仅在 before-turn 阶段可用，分别位于记忆检索之前和 EventBus emit 之后

每个 Phase 提供 `default_xxx_modules()` 工厂函数供 `PassiveTurnPipeline` 组装内置模块链。

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

Proactive gateway 没有 alert、feed 内容和可用 fallback context 时，进入 Drift 模式——agent 用一段空闲时间自主做一件有意义的事。

```
AgentTick.tick()
  └─ DataGateway.run()
       └─ no alert / no content / no fallback context
            └─ DriftRunner.run(ctx, llm_fn)
                 ├─ scan_skills()
                 │    └─ 读取 drift/skills/ 和内建 skill 的 SKILL.md
                 ├─ filter_skills()
                 │    └─ 跳过 requires_mcp 未满足的 skill
                 ├─ build_context()
                 │    └─ 注入记忆、近期上下文、skill 列表、recent_runs[message_result]
                 └─ tool_loop(max_steps)
                      ├─ read_file / write_file / edit_file
                      ├─ recall_memory / web_fetch / web_search
                      ├─ fetch_messages / search_messages / shell
                      ├─ message_push     # 最多一次
                      ├─ finish_drift     # 必须声明 message_result
                      └─ mount_server     # 可挂载 MCP server
```

Drift 的核心约束：

- 每次进入都重新比较所有 skill，不默认继续上次的
- `message_push` 成功后只允许 `write_file` / `edit_file` / `finish_drift` 收尾
- 发出的消息要像自然聊天，不像在汇报内部执行流程
- 执行结束前必须调用 `finish_drift` 保存状态，并填写 `message_result`
- `message_result="sent"` 要求本轮已经成功 `message_push`
- `message_result="silent"` 要求本轮没有成功 `message_push`
- `drift.json` 的 `recent_runs` 会记录每轮是 `sent` 还是 `silent`
- 到达 `max_steps` 时不再强制写文件或强制 `finish_drift`；如果模型没有主动收尾，本轮保持未完成

---

## 其他命令

```bash
uv run python main.py cli      # 连接运行中的 agent（TUI / 纯文本 CLI）

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
- [x] **Lifecycle Plugin System**: 基于插件目录的声明式 API——`@tool` 注册工具、`@on_tool_pre` 拦截工具调用、`@on_*` 钩入生命周期事件、`before_turn_modules_early/late()` 插入管道模块。插件放在 `plugins/` 下自动发现。见 [`_handbook/plugins-tutorial.md`](_handbook/plugins-tutorial.md)。

### Active
- [ ] **Memory Pluginification**: 提取 `DefaultMemory` 行为为 SPI 接口。
- [ ] **Tool Pluginification**: 重构工具注册为动态 Plugin 系统。

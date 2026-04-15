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
  PROACTIVE_CONTEXT.md # 主动推送规则文件（模板）
  mcp_servers.json     # MCP server 注册表
  schedules.json       # 定时任务列表
  proactive_sources.json  # 信息源列表
  memes/manifest.json  # 表情包清单
  skills/              # 用户自定义 skill 目录
  drift/skills/        # Drift 内建 skill 目录
  sessions.db          # 会话存储
  observe/observe.db   # trace 数据库
  memory/memory2.db    # 语义记忆数据库
  proactive_state.json # proactive 状态
```

**2. 填写配置**

编辑 `config.toml`，至少要改这几项：

```toml
[llm.main]
api_key = "sk-..."          # 你的 LLM API key

[channels.telegram]
token = "123456:ABC..."     # BotFather 给的 bot token
allow_from = ["your_username"]  # 你的 Telegram 用户名（不带 @）
```

**3. 启动并发消息**

```bash
python main.py
```

打开 Telegram，找到你的 bot，发一条消息，就可以开始对话。

**4. 打开 Proactive**

在 `config.toml` 里填上你的 Telegram chat_id（可以向 bot 发一条消息后从日志里拿到）：

```toml
[proactive]
enabled = true

[proactive.target]
channel = "telegram"
chat_id = "123456789"   # 你的 Telegram user id
```

Proactive 打开后，agent 会在你订阅的信息源有内容时主动推送消息。

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

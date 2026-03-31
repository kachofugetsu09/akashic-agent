# Akasic Agent

一个面向个人助理场景的多通道 Agent 项目。

它不只是一个“会聊天的 LLM 外壳”，而是把以下几件事接到了一起：

- 多通道消息接入：Telegram、QQ、CLI
- 被动对话主链路：用户发消息，Agent 检索记忆、决定是否调用工具、生成回复
- 主动触达链路：系统定时轮询外部信息，判断“现在值不值得主动提醒用户”
- 分层记忆系统：短期会话 + 长期语义记忆 + SOP/偏好/画像
- 工具体系：本地工具、MCP 工具、定时任务、消息推送

这个项目整体是典型的 MVP 架构：目标不是做成“超重型通用 Agent 平台”，而是围绕“个人助理”这个明确场景，把关键闭环先跑通。

---

## 1. 我会怎么一句话介绍这个项目

> 这是一个有长期记忆、工具调用能力和主动触达能力的个人助理 Agent。  
> 它把外部消息统一收进 `MessageBus`，由 `AgentLoop` 跑被动对话主链；同时用 `ProactiveLoop` 定期感知外部世界和用户状态，决定是否主动发消息。记忆层用 `memory2` 做语义检索和回复后写回，形成持续演化的助手。

---

## 2. 先看总架构

```text
┌──────────────────────────────────────────────────────────────────────┐
│                              Akasic Agent                            │
└──────────────────────────────────────────────────────────────────────┘

                         ┌──────────────────────┐
                         │      main.py         │
                         │  进程入口 / CLI入口   │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │   bootstrap/app.py   │
                         │   装配整个运行时      │
                         └──────────┬───────────┘
                                    │
            ┌───────────────────────┼────────────────────────┐
            │                       │                        │
            ▼                       ▼                        ▼
┌──────────────────┐    ┌─────────────────────┐   ┌────────────────────┐
│   Channels       │    │    Core Runtime     │   │  Proactive Runtime │
│ Telegram / QQ /  │    │  AgentLoop 主链路   │   │  ProactiveLoop     │
│ CLI / IPC        │    │  ToolRegistry       │   │  AgentTick         │
└────────┬─────────┘    │  SessionManager     │   └─────────┬──────────┘
         │              │  MemoryRuntime      │             │
         │              │  Scheduler          │             │
         │              └──────────┬──────────┘             │
         │                         │                        │
         │                         ▼                        │
         │              ┌─────────────────────┐             │
         │              │      MessageBus     │             │
         │              │ inbound / outbound  │             │
         │              └──────────┬──────────┘             │
         │                         │                        │
         │                         ▼                        ▼
         │              ┌─────────────────────┐   ┌────────────────────┐
         │              │      memory2        │   │   PushTool /       │
         │              │ 检索 / 写回 / 去重   │   │ channel outbound   │
         │              └─────────────────────┘   └────────────────────┘
         │
         ▼
┌──────────────────┐
│ 外部用户 / 群聊   │
└──────────────────┘
```

---

## 3. 核心模块分工

### 3.1 入口与装配

- `main.py`
  - 启动服务：`python main.py`
  - 启动本地 CLI：`python main.py cli`
- `bootstrap/app.py`
  - 这是整个系统的运行时装配中心
  - 负责初始化：
    - `CoreRuntime`
    - channels
    - `ProactiveLoop`
    - `MemoryOptimizerLoop`
    - observe writer

装配关系可以理解成：

```text
┌──────────────────────┐
│   AppRuntime.start   │
└──────────┬───────────┘
           │
           ├─ build_core_runtime()
           │  ├─ MessageBus
           │  ├─ AgentLoop
           │  ├─ ToolRegistry
           │  ├─ SessionManager
           │  ├─ MemoryRuntime
           │  └─ Provider / MCP / Scheduler
           │
           ├─ start_channels()
           │  ├─ IPC
           │  ├─ Telegram
           │  └─ QQ
           │
           ├─ build_proactive_runtime()
           │  └─ ProactiveLoop
           │
           └─ build_memory_optimizer_task()
```

### 3.2 被动对话主链

- `agent/looping/`
  - 负责被动消息驱动的 Agent 主循环
- `agent/turns/`
  - 负责 turn 结果编排、持久化、出站
- `agent/retrieval/`
  - 负责记忆检索流水线
- `agent/tools/`
  - 负责工具定义与工具执行

### 3.3 主动触达链

- `proactive_v2/`
  - 负责主动消息的感知、打分、决策、去重、发送
  - 核心不是“对着用户消息回答”，而是“系统自己决定要不要说一句”

### 3.4 记忆系统

- `memory2/`
  - 长期语义记忆
  - 负责 embedding、召回、记忆注入、回复后写回、去重、画像提取
- `session/`
  - 会话级短期历史
  - 用 SQLite 存消息和 session metadata

### 3.5 基础设施

- `bus/`
  - 进程内异步消息总线
- `infra/channels/`
  - 各聊天渠道适配器
- `infra/providers/`
  - LLM Provider 适配
- `core/observe/`
  - trace 落盘和保留策略

---

## 4. 被动对话链路怎么跑

这是面试里最值得讲清楚的一条链。

### 4.1 数据流

```text
┌──────────────┐
│ 用户发来消息 │
└──────┬───────┘
       │
       ▼
┌──────────────────────┐
│ Channel Adapter      │
│ Telegram / QQ / CLI  │
└──────┬───────────────┘
       │ publish_inbound
       ▼
┌──────────────────────┐
│ MessageBus.inbound   │
└──────┬───────────────┘
       │ consume
       ▼
┌──────────────────────┐
│ AgentLoop.run()      │
└──────┬───────────────┘
       │
       ▼
┌──────────────────────────────┐
│ ConversationTurnHandler      │
│ 1. 取 session                │
│ 2. 跑 retrieval pipeline     │
│ 3. 调用 LLM + tools          │
│ 4. 交给 orchestrator 落盘     │
└──────┬───────────────────────┘
       │
       ▼
┌──────────────────────┐
│ TurnOrchestrator     │
│ - 持久化 session      │
│ - observe trace      │
│ - post-turn 异步任务  │
│ - 产出 OutboundMessage│
└──────┬───────────────┘
       │ publish_outbound
       ▼
┌──────────────────────┐
│ MessageBus.outbound  │
└──────┬───────────────┘
       │ dispatch
       ▼
┌──────────────────────┐
│ Channel send back    │
└──────────────────────┘
```

### 4.2 主链分成四步

#### 第一步：消息进入系统

各 channel 会把外部平台消息统一转换成 `InboundMessage`，再投递进 `MessageBus`。

这里的价值是：

- 渠道差异先被吸收
- 核心 Agent 不直接依赖 Telegram/QQ SDK
- 后续扩渠道时，核心主链基本不用改

#### 第二步：找到 session，组织上下文

`AgentLoop` 先根据 `channel + chat_id` 找到会话，再从 `SessionManager` 取历史消息。

这里的历史不是无限塞给模型，而是做了两层控制：

- 会话窗口控制：只取最近若干轮
- 工具结果裁剪：较老轮次的 `tool_result` 会被清空成占位符，避免上下文爆炸

#### 第三步：先检索记忆，再让模型决定是否调工具

这一层是整个项目和“普通聊天机器人”最不一样的地方。

不是直接把用户消息 + 历史发给模型，而是先走 `DefaultMemoryRetrievalPipeline`：

```text
┌────────────────────────────┐
│ 当前用户消息                │
└──────────┬─────────────────┘
           │
           ▼
┌────────────────────────────┐
│ Gate 阶段                   │
│ - 是否需要查长期记忆         │
│ - 改写 episodic query        │
│ - 判断查哪些 memory type     │
└──────────┬─────────────────┘
           │
           ├─ procedure / preference
           │    规则类记忆
           │
           └─ event / profile
                历史类记忆
           │
           ▼
┌────────────────────────────┐
│ Retriever                  │
│ - embedding 检索            │
│ - 阈值过滤                  │
│ - 注入裁剪                  │
└──────────┬─────────────────┘
           │
           ▼
┌────────────────────────────┐
│ memory block               │
│ 注入到主模型上下文           │
└────────────────────────────┘
```

这一步的目的很明确：

- 让模型先看到“和当前问题最相关的长期记忆”
- 降低纯靠上下文窗口硬撑的成本
- 把“用户偏好 / 执行规程 / 历史事实 / 用户画像”从聊天记录里抽出来，形成可复用知识

#### 第四步：工具调用 + 回复落盘

模型进入 `TurnExecutor` / `SafetyRetryService` 之后，可以按需调工具。

工具执行完成后，不是直接返回，而是交给 `TurnOrchestrator` 统一处理：

- 把 user / assistant 消息写入 session
- 保存工具链 `tool_chain`
- 写 observe trace
- 触发 post-turn 任务
- 再投递到 outbound bus

这个编排层的好处是：**回复生成** 和 **结果落地** 分开了，后续要改 trace、持久化、post-turn，不用碰主推理链。

---

## 5. 记忆系统怎么设计

### 5.1 为什么要拆成两层记忆

这个项目不是只靠 session history。

它实际上有两层：

```text
┌──────────────────────────────┐
│ 短期记忆：session/messages    │
│ - 当前会话上下文              │
│ - 最近几轮工具调用            │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│ 长期记忆：memory2             │
│ - preference 用户偏好         │
│ - procedure 执行规程          │
│ - event 历史事实              │
│ - profile 用户画像            │
└──────────────────────────────┘
```

短期记忆解决“这轮上下文接不接得上”，长期记忆解决“系统过几天还记不记得你是谁、你喜欢什么、应该怎么做”。

### 5.2 memory2 的核心能力

`memory2` 不是单一向量表，而是一条完整链路：

- `Embedder`
  - 负责向 embedding 模型取向量
- `Retriever`
  - 负责按 memory type 检索并做阈值筛选
- `Memorizer`
  - 负责新记忆写入
- `PostResponseMemoryWorker`
  - 负责在回复后异步提取隐式记忆并写回
- `ProfileFactExtractor`
  - 负责从对话中抽用户画像事实
- `DedupDecider`
  - 负责去重和 supersede

### 5.3 memory2 的读取链路

```text
┌──────────────┐
│ 当前用户消息 │
└──────┬───────┘
       │
       ▼
┌──────────────────────────────┐
│ QueryRewriter / Gate         │
│ 决定是否扩写、是否查历史       │
└──────┬───────────────────────┘
       │
       ▼
┌──────────────────────────────┐
│ Retriever                    │
│ 检 procedure/preference/event │
│ /profile                     │
└──────┬───────────────────────┘
       │
       ▼
┌──────────────────────────────┐
│ Injection Planner            │
│ 把命中的记忆整理成 memory block │
└──────┬───────────────────────┘
       │
       ▼
┌──────────────────────────────┐
│ LLM 上下文                   │
└──────────────────────────────┘
```

### 5.4 memory2 的写入链路

这个项目没有把记忆写入塞进主回复同步路径，而是放到 reply 后异步执行。

这样做很实用：

- 用户先收到回复，首响应更快
- 记忆提取失败不会卡死主链
- 记忆写入可以独立演进

写入流程：

```text
┌────────────────────────────┐
│ 本轮 user_msg + reply      │
└──────────┬─────────────────┘
           │
           ▼
┌────────────────────────────┐
│ PostResponseMemoryWorker   │
│ - 提取隐式偏好              │
│ - 识别失效旧规则            │
│ - profile 抽取              │
│ - dedup / supersede         │
└──────────┬─────────────────┘
           │
           ▼
┌────────────────────────────┐
│ MemoryStore2               │
│ SQLite + 向量检索            │
└────────────────────────────┘
```

### 5.5 这套记忆设计的面试亮点

- 不是“把所有聊天记录都塞上下文”
- 记忆分类型，能解释为什么需要 `preference / procedure / event / profile`
- 读写链路分离，主响应和记忆维护解耦
- 有 dedup / supersede，说明记忆不是只增不改的垃圾堆

---

## 6. 主动触达链路怎么跑

被动对话解决“用户问，我回答”。  
主动触达解决“用户没问，但系统判断现在应该说一句”。

### 6.1 核心思路

`ProactiveLoop` 独立运行，不依赖用户即时输入。

它会周期性做这些事：

- 拉内容源和告警源
- 看最近聊天和用户状态
- 评估当前是否适合打扰用户
- 调 `AgentTick` 让模型做一次主动决策
- 决定要不要发消息

### 6.2 主动链路图

```text
┌──────────────────────┐
│ ProactiveLoop.run()  │
└──────────┬───────────┘
           │ tick
           ▼
┌──────────────────────────────┐
│ Sensor / MCP Sources         │
│ - feed                        │
│ - alert                       │
│ - context                     │
│ - recent chat                 │
└──────────┬───────────────────┘
           │
           ▼
┌──────────────────────────────┐
│ AnyActionGate / Energy       │
│ - 冷却                         │
│ - 概率                         │
│ - 忙碌判断                     │
│ - 打扰成本                     │
└──────────┬───────────────────┘
           │
           ▼
┌──────────────────────────────┐
│ AgentTick                    │
│ - 组 prompt                   │
│ - 跑 tool loop                │
│ - 生成主动消息                 │
└──────────┬───────────────────┘
           │
           ▼
┌──────────────────────────────┐
│ TurnOrchestrator             │
│ - persist proactive msg      │
│ - observe                    │
│ - outbound                   │
└──────────┬───────────────────┘
           │
           ▼
┌──────────────────────┐
│ PushTool / Channel   │
└──────────────────────┘
```

### 6.3 这条链的工程价值

很多 Agent 项目只做到“问答”。  
这个项目进一步往前走了一步：让 Agent 能做“主动服务”。

主动链路的难点不是生成文本，而是控制“何时发、值不值得发、会不会重复发、会不会打扰用户”。

所以你面试里可以强调：

- 这不是简单定时任务
- 它有：
  - 用户活跃度感知
  - 忙碌态避让
  - 冷却与去重
  - 外部 feed / alert 汇聚
  - 主动消息持久化

---

## 7. 工具系统怎么设计

### 7.1 为什么单独做 ToolRegistry

工具在这个项目里不是“顺手塞几个函数”，而是显式注册到 `ToolRegistry`。

每个工具都有元信息：

- `tags`
- `risk`
- `always_on`
- `search_hint`

这带来两个直接收益：

- 模型能按需发现工具，而不是每轮暴露全量工具
- 可以按风险、关键词做工具搜索和路由

### 7.2 工具注册结构

```text
┌──────────────────────────┐
│ build_registered_tools() │
└──────────┬───────────────┘
           │
           ├─ meta/common tools
           ├─ memory tools
           ├─ fitbit tools
           ├─ spawn tool
           ├─ scheduler tools
           ├─ MCP tools
           └─ peer agent tools
```

### 7.3 目前工具来源

- 本地内置工具
  - 文件、shell、schedule、message_lookup、message_push 等
- memory 工具
  - `memorize`、SOP 文件写入
- MCP 工具
  - 外部 MCP server 暴露的能力
- peer agent 工具
  - 其他 agent 进程暴露出来的专家能力

这说明它不是封闭单体，而是能继续向外接能力。

---

## 8. 数据与持久化

### 8.1 会话数据

`SessionManager` + `SessionStore` 用 SQLite 管理：

- `sessions`
- `messages`
- `messages_fts`

所以这个项目的 session 不只是内存态，重启后还能恢复。

### 8.2 长期记忆

`memory2` 默认也是 SQLite 路径下存储，配合 embedding 检索使用。

### 8.3 运行时状态

还有几类轻量状态文件：

- `presence.json`
- `proactive_state.json`
- observe traces

整体思路是：

- 核心业务数据进 SQLite
- 轻量状态用 JSON
- 不做过度复杂的基础设施依赖

这很符合 MVP 项目的取舍。

---

## 9. 可观测性与测试

### 9.1 可观测性

项目专门做了 observe trace：

- 对话 turn trace
- memory 写入 trace
- proactive 配置与速率 trace

这意味着出问题时不只能“猜 prompt”，还能看链路数据。

### 9.2 测试策略

测试分成公开和私有两层：

```text
┌──────────────────────────┐
│ tests/                   │
│ 单元 / 模块级测试         │
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│ private_tests/           │
│ 私有高保真 scenario /     │
│ replay baseline          │
└──────────────────────────┘
```

---

## 10. 目录地图

```text
.
├── main.py                 进程入口
├── bootstrap/              运行时装配
├── agent/                  被动对话主链
│   ├── looping/            AgentLoop、handler、executor
│   ├── turns/              turn 编排与出站
│   ├── retrieval/          记忆检索流水线
│   ├── tools/              工具定义与注册
│   ├── mcp/                MCP 工具桥接
│   └── background/         后台子任务 / subagent 管理
├── proactive_v2/           主动触达链路
├── memory2/                长期语义记忆
├── session/                session 持久化
├── bus/                    进程内消息总线
├── infra/                  channel / provider 适配层
├── core/                   observe / http / common
├── tests/                  单元与模块测试
├── private_tests/          私有测试子模块（高保真 scenario / replay）
└── scripts/                运维 / 迁移 / 分析脚本
```

---

## 11. 面试时可以怎么讲

### 11.1 30 秒版本

> 这是一个个人助理型 Agent。系统把 Telegram、QQ、CLI 等渠道消息统一进 `MessageBus`，由 `AgentLoop` 跑对话主链。主链不是直接把聊天记录塞给模型，而是先检索 `memory2` 的长期记忆，再决定是否调用工具，最后通过统一 orchestrator 落盘和回包。除此之外，我还做了一条 `ProactiveLoop`，它可以周期性读外部 feed 和用户状态，决定要不要主动推送消息。

### 11.2 3 分钟版本

> 我把这个系统拆成三块：接入层、对话核心、主动服务。  
> 接入层负责把 Telegram/QQ/CLI 这些外部渠道统一转成内部消息对象，放进 `MessageBus`。  
> 对话核心由 `AgentLoop` 驱动，一轮消息大概会经历四步：先找到对应 session，取最近历史；然后走 `memory2` 检索，把偏好、规程、历史事实、用户画像整理成 memory block 注入上下文；接着模型决定是否调工具；最后 `TurnOrchestrator` 统一做持久化、trace 和回包。  
> 除了被动问答，我还做了 `ProactiveLoop`。它会定期轮询外部内容源、结合用户活跃度和冷却状态做打扰决策，再由 `AgentTick` 生成主动消息。这样这个项目就不是单纯聊天机器人，而是一个能持续服务用户的助理系统。

### 11.3 如果面试官追问“亮点是什么”

- 不是简单 ChatBot，而是被动对话 + 主动触达双链路
- 不是只靠上下文窗口，而是做了长期记忆系统
- 记忆不是只写不管，有去重、失效和 supersede
- 工具体系有 registry、关键词搜索和 MCP 扩展能力
- 测试以单元和模块级验证为主
- 高保真私有场景测试通过 `private_tests/` 子模块维护

### 11.4 如果面试官追问“难点是什么”

- 如何把渠道差异收敛到统一消息模型
- 如何让记忆命中有用，而不是把噪音塞进 prompt
- 如何在不拖慢主响应的前提下做回复后记忆写回
- 如何控制主动触达的打扰成本，避免乱推

### 11.5 如果面试官追问“还有哪些不足”

这部分建议诚实说：

- 当前是 MVP，重点在闭环，不是高可用分布式架构
- 主动链和被动链虽然已经比较清楚，但还没有完全统一成同一套 turn runtime
- 部分能力依赖外部模型质量，稳定性仍需要靠更多真实场景测试继续压
- 配置项已经不少，后续还可以继续收敛默认值和运行模式

---

## 12. 我个人认为这个项目最值得讲的点

如果时间不多，优先讲这四个：

1. `MessageBus + AgentLoop` 让多渠道输入收敛成统一对话主链
2. `memory2` 让系统具备长期记忆，而不是只靠上下文窗口
3. `TurnOrchestrator` 把“生成回复”和“结果落地”拆开
4. `ProactiveLoop` 让系统从“等用户问”进化到“主动服务”

---

## 13. 运行方式

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements-dev.txt
cp config.example.json config.json
python main.py
```

本地 CLI：

```bash
python main.py cli
```

单元测试：

```bash
pytest tests/
```

私有 scenario / replay：

```bash
AKASIC_RUN_SCENARIOS=1 pytest -c private_tests/pytest-scenarios.ini private_tests/tests_scenarios/
```

## 14. 结论

从工程角度看，这个项目最核心的价值，不是“接了几个模型 API”，而是把一个个人助理 Agent 真正拆成了可运行的系统：

- 有接入层
- 有主对话链
- 有长期记忆
- 有工具体系
- 有主动触达
- 有持久化和可观测性

如果你在面试里把这几个层次和它们之间的数据流讲清楚，这个项目的说服力会比“我做了一个聊天机器人”强很多。

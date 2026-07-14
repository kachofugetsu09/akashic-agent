# 程序化控制面目标架构

> 调研基线：`6b8f438d`（2026-07-14）  
> 参考实现：`/mnt/data/source-code/codex` commit `c7a4a7e136`，重点参考
> `codex-rs/app-server/README.md`、`codex-rs/app-server-protocol/`、
> `codex-rs/exec/` 与 `sdk/python/`。本方案借鉴分层和生命周期，不追求 wire
> compatibility，也不复制 Codex 与代码执行相关的 sandbox/approval API。

## 结论

当前 `IPCServerChannel` 不能继续演化成正式程序化 API。它是一个本地聊天渠道：
连接对象的 `id(writer)` 被当作路由 ID，客户端发送无 request id 的 JSON 行，服务端
最终只返回一条 assistant 文本。它不拥有 thread、turn、事件、错误、取消完成、背压
或版本协商语义。继续在这里追加字段会把协议、渠道路由、运行时状态和 UI 逻辑永久
绑在一起。

目标是把程序化控制面提升为运行时的一等入口：

```text
                              ┌──────────────────────────────┐
 stdio / Unix socket ────────▶│ JSON-RPC App Server          │
                              │ handshake / routing / queues  │
                              └──────────────┬───────────────┘
                                             │ typed requests
 Python SDK / akashic exec ──────────────────┤
                                             ▼
                              ┌──────────────────────────────┐
 Telegram / QQ / Web ────────▶│ ConversationRuntime          │
          via channel adapter │ thread / turn / cancellation │
                              │ event stream / persistence    │
                              └──────────────┬───────────────┘
                                             ▼
                              ┌──────────────────────────────┐
                              │ PassiveTurnPipeline           │
                              │ memory / plugins / reasoner   │
                              └──────────────────────────────┘
```

TUI、纯文本交互式 CLI 和旧 IPC channel 全部删除。`MessageBus` 只保留为现有消息渠道的
transport adapter，不再是程序化调用的主 API；程序化调用不能伪造一个 `cli` channel
再等待 outbound 消息。

## 当前状态与根因

### 1. IPC 是聊天路由，不是请求协议

- `infra/channels/ipc_server.py:164-223` 为每条连接生成 `cli-{id(writer)}`，直接把 JSON
  中的 `content` 包装成 `InboundMessage`。
- `infra/channels/ipc_server.py:269-284` 只能按 `chat_id` 找 writer，写回最终
  `assistant` 文本；请求和响应没有稳定相关 ID。
- `infra/channels/ipc_server.py:225-267` 的管理命令是字符串 handler 表，只有
  `ok/message`，无法表达 typed result 或稳定错误码。

### 2. turn 生命周期没有稳定领域 ID

- `bus/events_lifecycle.py:35-50` 的 `TurnStarted` 和 `StreamDeltaReady` 只有
  `session_key/channel/chat_id`。
- `bus/events_lifecycle.py:115-137` 的 tool call 虽有 `call_id`，仍没有 `turn_id`。
- `agent/core/passive_turn.py:449-466` 只生成诊断日志用的 turn id，没有将它作为领域
  对象持久化或回传。

### 3. 执行、调度、取消和 channel dispatch 混在 AgentLoop

- `agent/looping/core.py:376-428` 同时消费 MessageBus、创建 task、保存 active state、
  捕获错误并发布 outbound。
- `agent/looping/core.py:507-543` 的中断 API 以 `session_key` 为目标，无法防止客户端
  中断同一 thread 上已经更换的 turn。
- `agent/looping/core.py:728-765` 的 `process_direct()` 只返回 `str`，丢失状态、usage、
  items 和结构化错误。

### 4. 核心错误被用户文案覆盖

- `agent/core/passive_turn.py:580-603` 将 provider/reasoner 异常转成普通
  `OutboundMessage("处理消息时出错，请稍后再试。")`。
- channel 需要安全的用户文案，但程序化调用方必须拿到明确失败状态、稳定错误类型和
  request/turn id。该转换应该发生在 channel adapter，不应该发生在领域执行层。

### 5. 持久层只有 session/message，无法恢复 turn 状态

- `session/store.py:228-248` 只有 `sessions` 和 `messages` 表。
- `session/manager.py:516-520` 能列 session，但不能读取 turn 的 queued/running/
  completed/interrupted/failed 状态，也无法可靠重建 usage 和错误。

## 从 Codex 借鉴什么

Codex 的可复用设计不是 TUI，而是以下契约：

1. `Thread → Turn → Item` 是稳定领域模型（参考
   `codex-rs/app-server/README.md:64-72`）。
2. 连接先 `initialize`，协商版本和 capability；未初始化请求 fail-loud（同文件
   `74-87`）。
3. `turn/start` 立即返回 turn handle，进度通过 notification 流式发送，最终以
   `turn/completed` 收束（同文件 `76-81`）。
4. request/response 与 notification 分离；连接和 turn 都有独立路由。
5. 有界队列和明确 overload 错误，而不是无限制堆积（同文件 `49-53`）。
6. SDK 提供 `Thread.run()` 的易用路径，也提供 `TurnHandle.stream()/interrupt()` 的
   完整控制路径（`sdk/python/docs/api-reference.md:147-232`）。

本项目不应复制 Codex 的 sandbox、approval、命令执行、review、MCP 表单等代码代理
专属能力。Akashic v1 要完整表达自身已有能力，并为将来 capability 扩展保留边界。

## 目标领域模型

### Thread

- `thread.id` 直接使用现有 `session_key`，它已经是持久层主键和记忆归属键。
- `thread/start` 生成 `programmatic:<uuid7>`；渠道会话继续使用
  `<channel>:<chat_id>`，不做破坏性 ID 搬迁。
- `thread/resume/read/list` 可访问所有 session，但返回 `source` 标明
  `programmatic|channel|internal`。
- thread 不等于 transport connection；连接断开不能删除 thread。

### Turn

- 每次输入先生成不可变 `turn_id`，再排队执行。
- 状态机唯一合法路径：

```text
queued ──▶ in_progress ──▶ completed
   │              ├──────▶ interrupted
   │              └──────▶ failed
   └─────────────────────▶ cancelled
```

- 同一 thread 同时最多一个 active turn。不同 thread 在 v1 仍受当前全局
  `_passive_runtime_lock` 串行约束；协议显式发 `turn/queued`，不能假装并发。
- `turn/interrupt(threadId, turnId)` 必须同时匹配，避免旧请求误杀新 turn。
- v1 不提供 `turn/steer`。当前运行时没有中途安全注入 input 的契约；用“中断后续跑”
  冒充 steer 会破坏语义。以后实现时通过 capability 新增。

### Item 与事件

v1 item 类型保持最小但完整：

- `userMessage`
- `assistantMessage`（started/delta/completed）
- `reasoning`（仅在配置允许暴露时）
- `toolCall`（started/completed，含 call id、name、arguments、status、result preview）
- `error`

所有 turn/item notification 必须带 `threadId`、`turnId`，item 事件还带 `item.id`。
SDK 按 `turnId` 分流，不能按到达顺序猜测归属。

### Result 与错误

`TurnResult` 固定包含：

- `id`, `threadId`, `status`
- `startedAt`, `completedAt`, `durationMs`
- `finalResponse`, `items`
- `usage`（已有 `react_stats.model_usage` 的字段；缺失时为 `null`，不造假）
- `error: {type, message, retryable, data?} | null`

领域层抛 typed exceptions；app-server 映射为 JSON-RPC error 或 failed turn；channel
adapter 再决定用户可见中文文案。日志保留 traceback，但协议不外泄内部堆栈。

## v1 JSON-RPC 协议

wire 使用标准 JSON-RPC 2.0 NDJSON，每条消息都带 `"jsonrpc":"2.0"`。不与 Codex
省略 header 的私有变体兼容。

### Transport

- `stdio`：父进程托管，适合 SDK 集成测试和独立进程模式；日志只能写 stderr。
- Unix socket：gateway 默认控制面，文件权限 `0600`。
- Windows loopback TCP：只允许 loopback，并在 `initialize` 中校验 workspace token。
- v1 不开放公网 TCP/WebSocket/HTTP。远程能力应另做带认证和威胁模型的 transport。

### Handshake

首个请求必须是：

```json
{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"1.0","clientInfo":{"name":"example","version":"0.1"},"capabilities":{"reasoningEvents":false}}}
```

服务端返回协商后的版本、server info、workspace、capabilities；客户端随后发送
`initialized` notification。重复 initialize、未初始化调用和版本不兼容都明确失败。

### v1 methods

- `server/status`
- `thread/start`
- `thread/resume`
- `thread/list`（cursor + limit）
- `thread/read`（可选 includeTurns）
- `thread/delete`
- `turn/start`
- `turn/interrupt`
- `turn/read`
- `thread/consolidate/start`（映射已有 memory consolidation，异步返回 operation id）

### v1 notifications

- `thread/started`, `thread/deleted`
- `turn/queued`, `turn/started`, `turn/completed`
- `item/started`, `item/assistantMessage/delta`, `item/reasoning/delta`,
  `item/completed`
- `operation/completed`

### 错误码

- 标准：`-32700`, `-32600`, `-32601`, `-32602`, `-32603`
- `-32001` server overloaded（retryable）
- `-32002` not initialized
- `-32003` incompatible protocol version
- `-32010` thread not found
- `-32011` thread busy / turn conflict
- `-32012` turn not found
- `-32013` operation not supported by current capability set

## 运行模式与命令

### 持久 gateway

`python main.py gateway` 启动现有完整 runtime、渠道和 Unix socket app-server。默认
SDK 连接这个 endpoint，不再启动 TUI。

### 父进程托管 app-server

`python main.py app-server --stdio` 启动完整 runtime 但不启用 Telegram/QQ/Web/
Dashboard/proactive transport。它必须持有 workspace instance lock；如果同 workspace
已有 gateway，fail-loud，不共享 SQLite 和插件 watcher。

### one-shot exec

```bash
python main.py exec --new --json "总结今天的上下文"
python main.py exec --thread telegram:123 --json - < prompt.txt
```

- 默认连接 workspace gateway；`--endpoint` 可覆盖。
- stdout 只写 JSONL protocol event；人类日志写 stderr。
- `--final-only` 只打印最终文本。
- exit code：0 completed，1 failed，2 参数/连接/协议错误，130 interrupted。

## Python SDK

SDK 放在 `sdk/python/`，包名 `akashic-agent-sdk`，模块名 `akashic_sdk`。提供 async
主实现与同步 facade：

```python
async with AsyncAkashic.connect() as client:
    thread = await client.thread_start()
    handle = await thread.turn("整理最近三轮对话")
    async for event in handle.stream():
        ...
    result = await handle.result()
```

常用路径为 `thread.run()`；高级路径为 `turn()/stream()/interrupt()`。协议 schema 由
server 的单一 Python model source 生成并提交 JSON Schema，再生成 SDK models，避免
手写两份漂移。

## 配置迁移

删除 `[channels].socket`、`[channels.cli]` 和 `cli_session_key`，新增：

```toml
[app_server]
enabled = true
listen = ""       # 空值按 workspace 派生 Unix socket / Windows loopback
max_connections = 32
ingress_queue_size = 128
outbound_queue_size = 512
```

项目仍是 `0.1.0`，这次改造采用明确 breaking migration，不在 runtime 长期保留双配置
解析。启动时遇到旧字段应报带迁移示例的配置错误，而不是静默忽略。

## 安全与可靠性不变量

1. UDS 必须 `0600`；Windows token 文件也必须只对当前用户可读。
2. TCP 非 loopback 地址在配置边界直接拒绝。
3. 每条输入、每连接 pending request、每连接 outbound notification 都有上限。
4. 慢客户端只断开自身，不阻塞 turn 和其他客户端。
5. 客户端断开不自动取消 turn；只有显式 `turn/interrupt` 改变运行时状态。
6. 请求参数在 JSON-RPC 边界一次性严格校验；边界后信任 typed model。
7. 内部契约错误 fail-loud；只在 app-server 边界转换为稳定协议错误。
8. reasoning 默认不对程序化客户端暴露，必须在 initialize capability 中显式请求。
9. 运行时关闭时先停止 ingress，再收束 active turn/notification writer，最后关闭
   SessionStore 和 socket。

## 不做的事

- 不追求与 Codex app-server wire compatible。
- 不在 v1 增加 HTTP REST、WebSocket 或公网监听。
- 不把 dashboard API 机械搬进 JSON-RPC。
- 不实现当前内核没有真实语义的 `turn/steer`、approval 或 output schema。
- 不为了 SDK 复制一套 agent runtime；SDK 只通过协议调用服务端。
- 不保留 TUI fallback 或旧 `{content: ...}` IPC fallback。

## 真实运行验收架构

验收不能停在 schema、单元测试或把 `ConversationRuntime` 直接 import 进测试进程。新控制面
必须在 `docker/debug` 中以生产入口启动，并通过真实 provider HTTP wiring 调用一个确定性
模型 sidecar：

```text
┌──────────────────── host gate controller ────────────────────┐
│ JSON-RPC client  │ fault/barrier control │ evidence collector │
└──────────┬──────────────────┬──────────────────────▲───────────┘
           │ UDS / stdio      │ HTTP control         │ reports
           ▼                  ▼                      │
┌────────────────────┐   ┌─────────────────────┐     │
│ real gateway /     │──▶│ deterministic model │─────┘
│ app-server process │   │ OpenAI-compatible   │
│ real DB/plugins    │   │ stream/error/barrier│
└──────────┬─────────┘   └─────────────────────┘
           ▼
   isolated /sandbox
```

- `/app` 只读，只有本次运行的 `/sandbox` 和 `/tmp` 可写；结束后校验源码 digest 未变。
- readiness 必须完成 `initialize → initialized → server/status`，不能只判断 socket 文件存在。
- 调度和竞态使用模型 sidecar barrier 精确停在 provider 请求处；异步后台工作采用数据库状态
  收敛和 quiet window。禁止用固定 sleep 猜完成时机。
- 每次 gate 生成机器可读 `gate.json`、完整协议 `events.jsonl`、模型请求记录、数据库终态快照
  和容器 stderr/log；任一断言失败均非零退出。
- 外部真实模型只做 nightly/manual canary，不作为 PR 核心门禁；PR gate 必须离线、确定、可重放。

详细场景、阈值和产物见 `006-docker-runtime-acceptance.md`。

## 验收总览

完成七个阶段后必须同时满足：

1. `rg "cli_tui|CLITextualApp|IPCServerChannel|python main.py cli"` 对源代码和文档无命中
   （迁移说明可单独白名单）。
2. app-server contract tests 覆盖 handshake、thread、turn、items、interrupt、错误、
   背压、慢客户端和 shutdown。
3. SDK contract tests 由真实 app-server 子进程驱动，不 mock 协议核心。
4. `exec --json` 输出可被逐行 JSON 解析，stderr/stdout 不串线。
5. Telegram/QQ/Web 现有行为走同一个 `ConversationRuntime`，不再有独立执行路径。
6. `pytest tests/`、`pyright --level error`、前端 typecheck/lint 全部通过。
7. Docker gate 的必选场景全部通过：真实进程 handshake、事件相关性、并发隔离、interrupt、
   断线恢复、慢客户端、provider 故障、SIGTERM、重启与持久化；运行后无残留资源。
8. 所有等待由 protocol event、sidecar barrier 或状态收敛驱动；必选场景不存在固定 sleep。

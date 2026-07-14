# Plan 003: 建立版本化 JSON-RPC app-server

> **Executor instructions**：执行前确认 Plans 001–002 已 DONE。协议边界必须严格；不要
> 为了兼容旧 IPC 接受无 schema 的 dict。完成后更新状态。
>
> **Drift check**：
> `git diff --stat 6b8f438d..HEAD -- agent/control bootstrap infra agent/config.py agent/config_models.py tests`

## Status

- **Priority**: P1
- **Effort**: L
- **Risk**: HIGH（新 public protocol 与进程生命周期）
- **Depends on**: `advisor-plans/002-conversation-runtime.md`
- **Category**: direction / architecture
- **Planned at**: commit `6b8f438d`, 2026-07-14

## Why this matters

ConversationRuntime 解决进程内调用，但外部程序还需要版本化、可并发、可取消的协议。
本计划交付 app-server 本身；它不是 channel，也不包含 UI。

## Current state

- `infra/channels/ipc_server.py:164-218` 接受任意 object dict，无 request id 或初始化。
- `infra/channels/ipc_server.py:225-267` command response 只有 `ok/message`。
- `bootstrap/channels.py:32-41` 无条件把 IPC 当成 channel 启动。
- `agent/config_models.py:39-45` 将 socket 和 CLI session 配在 `ChannelsConfig`。
- `agent/config.py:63-74` 已有按 workspace 派生本地 endpoint 的逻辑，可迁移但应改名。

参考 `codex-rs/app-server/README.md:20-62` 的 transport/backpressure/schema 分层，以及
`74-87` 的 initialize lifecycle。使用标准 JSON-RPC 2.0 header，不复制 Codex wire 变体。

## Commands you will need

| Purpose | Command | Expected |
|---|---|---|
| Protocol | `/mnt/data/coding/akasic-agent/.venv/bin/pytest -q tests/control/test_protocol.py tests/control/test_router.py` | all pass |
| Transport | `/mnt/data/coding/akasic-agent/.venv/bin/pytest -q tests/control/test_stdio_transport.py tests/control/test_socket_transport.py` | all pass |
| Integration | `/mnt/data/coding/akasic-agent/.venv/bin/pytest -q tests/control/test_app_server.py tests/test_runtime_smoke.py` | all pass |
| Typecheck | `/mnt/data/coding/akasic-agent/.venv/bin/pyright agent/control bootstrap agent/config.py agent/config_models.py` | exit 0 |

## Scope

**In scope**:

- `agent/control/protocol/`（models、errors、router、schema export）
- `infra/control/stdio.py`、`infra/control/socket.py`
- `bootstrap/app_server.py`
- `bootstrap/app.py`
- `agent/config_models.py`、`agent/config.py`
- `main.py` 的 `app-server` 命令
- `tests/control/`
- committed JSON Schema artifact（如 `schema/app-server-v1.json`）

**Out of scope**:

- Python SDK 和 `exec`
- TUI/旧 IPC 删除（Plan 005）
- HTTP/WebSocket/公网 TCP
- `turn/steer`、approval、sandbox、output schema

## Git workflow

- Commit: `feat(control): add versioned json-rpc app server`
- 不 push，不开 PR。

## Steps

### Step 1: 定义单一协议 schema source

以严格 typed models 定义 architecture 文档列出的 request/response/notification。生成并
提交 JSON Schema；schema export 必须 deterministic，CI 可重新生成并 diff。

边界拒绝 unknown required-shape violations；可选 forward-compatible `capabilities` 字段
只在其拥有层处理，不在业务方法重复校验。

**Verify**：golden schema test 两次生成字节一致，所有 method 都有 params/result schema。

### Step 2: 实现 initialize 与 JSON-RPC router

每连接维护 `new → initialized → closed` 状态。只允许 `initialize`、随后
`initialized`。实现标准 parse/invalid/method/params/internal errors 和 architecture 中的
application codes。request id 支持 string/int，notification 不得产生 response。

handler 只接收已校验 models，调用 ConversationRuntime/SessionManager ports；不要传 raw
dict。internal traceback 仅日志记录，response 返回稳定 error envelope 和 correlation id。

**Verify**：乱序 handshake、重复 initialize、未知 method、错误 params、handler 异常、
并发 request id 全有精确 tests。

### Step 3: 实现 connection session 与事件路由

`turn/start` 立即返回 queued/in-progress TurnRecord，并把该连接订阅到对应 turn。
notification writer 使用有界队列。连接关闭只清理订阅，不取消 turn；`turn/read` 可恢复
最终状态。terminal notification 必须在该 turn 最后一个 item notification 之后排队。

**Verify**：两个并发 turn 的 event 按 turn id 正确路由；断开重连可 read terminal。

### Step 4: 实现 stdio 和本地 socket transport

两种 transport 只负责 bytes/framing/connection lifecycle，复用同一个 router：

- stdio：stdin/stdout NDJSON，日志仅 stderr；EOF 正常关闭。
- POSIX socket：workspace 派生路径、启动前处理 stale socket、`0600`。
- Windows：loopback TCP + workspace token；拒绝非 loopback bind。

解析必须有单行/消息大小上限。ingress、pending requests、outbound queue 都有配置上限；
overload 返回 `-32001` 或断开无法安全回复的恶意连接。

**Verify**：partial reads、超长行、无效 UTF-8、慢 reader、queue saturation、stale socket、
chmod failure、shutdown tests。

### Step 5: 接入 AppRuntime 和 CLI dispatch

新增 `[app_server]` typed config。gateway 启动本地 socket app-server；
`python main.py app-server --stdio` 启动无 channels/dashboard/proactive transport 的 runtime。
后者必须持有 workspace instance lock，冲突时非零退出并打印明确错误。

保留旧 IPC 直到 Plan 005，但新旧不能绑定同一 endpoint；默认 endpoint 改为新 app-server。

**Verify**：subprocess integration 完成 initialize → thread/start → turn/start → completed，
stdout 每行可解析 JSON，stderr 不混入协议。

## Test plan

- protocol/router 纯 unit tests。
- socket tests 使用 tmp_path，不使用用户 workspace。
- 至少一个真实 subprocess stdio contract test；provider 使用现有正式 wiring 支持的确定性
  test backend，不 monkeypatch `AppRuntime.start()`。
- 测试慢客户端、过载和 shutdown，而不只 happy path。

## Done criteria

- [ ] schema 单一来源且 deterministic
- [ ] handshake/version/capability 可协商
- [ ] stdio 与 socket 共用 router
- [ ] request、response、notification 可并发相关
- [ ] 有界队列与 overload 行为已测试
- [ ] 连接断开不取消 turn
- [ ] target、integration、pyright 全通过

## STOP conditions

- 实现需要 app-server 直接访问 AgentLoop 私有字段。
- stdio 模式无法阻止 logger/print 写 stdout。
- Windows endpoint 无法在无认证 TCP 和受控 token 之间建立明确安全边界。
- schema 需要从 SDK 反向导入，形成 server 对 client 的依赖。

## Maintenance notes

协议新增方法必须先改 schema 和 contract tests，再接 handler。transport 不得包含 method
分支。任何公网 transport 都是单独安全设计，不得仅放宽 host 校验。


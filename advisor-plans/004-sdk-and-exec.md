# Plan 004: 交付 Python SDK 与 one-shot exec

> **Executor instructions**：执行前确认 Plan 003 已 DONE，且 committed schema 与 server
> 行为一致。SDK 不得 import 运行时内部实现。完成后更新状态。
>
> **Drift check**：
> `git diff --stat 6b8f438d..HEAD -- schema agent/control main.py sdk tests/control`

## Status

- **Priority**: P1
- **Effort**: L
- **Risk**: MED（新的用户调用面，但不改核心执行）
- **Depends on**: `advisor-plans/003-jsonrpc-app-server.md`
- **Category**: direction / dx
- **Planned at**: commit `6b8f438d`, 2026-07-14

## Why this matters

删除 TUI 前必须有真正可用的替代面。裸 JSON-RPC 适合集成底座，不适合作为日常 API；
SDK 应提供 `thread.run()` 的简单路径和 turn handle 的完整事件/中断路径，`exec --json`
则为 shell、CI 和其他语言提供稳定 one-shot 接口。

## Current state

- `infra/channels/cli.py:32-81` 只实现交互 stdin 和最终文本打印。
- `infra/channels/cli.py:84-116` 的管理 request 无 id、timeout 或 typed error。
- `main.py:156-175` 默认导入 TUI，失败后回退纯文本 CLI。
- Codex SDK 的可借鉴 surface 位于
  `/mnt/data/source-code/codex/sdk/python/docs/api-reference.md:147-232`：common-case run 与
  low-level turn handle 分层，事件按 turn id 路由。

## Commands you will need

| Purpose | Command | Expected |
|---|---|---|
| SDK unit | `cd sdk/python && /mnt/data/coding/akasic-agent/.venv/bin/pytest -q tests` | all pass |
| SDK type | `cd sdk/python && /mnt/data/coding/akasic-agent/.venv/bin/pyright src tests` | exit 0 |
| Exec | `/mnt/data/coding/akasic-agent/.venv/bin/pytest -q tests/control/test_exec_cli.py` | all pass |
| Schema drift | `/mnt/data/coding/akasic-agent/.venv/bin/python scripts/generate_control_schema.py --check` | exit 0 |

## Scope

**In scope**:

- `sdk/python/pyproject.toml`
- `sdk/python/src/akashic_sdk/`
- `sdk/python/tests/` 与 examples/docs
- generated protocol models（从 server schema 生成）
- `agent/control/client.py` 或共享的轻量 transport client（仅供主仓 exec）
- `main.py` 的 `exec` 命令
- `tests/control/test_exec_cli.py`

**Out of scope**:

- TypeScript/Go SDK
- SDK 内嵌/复制 runtime
- 自动下载二进制或自动启动同 workspace 第二实例
- TUI 文件删除（Plan 005）
- SDK 暴露 schema 未声明的 raw method escape hatch 作为稳定 API

## Git workflow

- Commit: `feat(sdk): add programmatic client and json exec`
- 不 push，不发布 package，不开 PR。

## Steps

### Step 1: 生成并审计 SDK models

由 committed JSON Schema 生成 request/response/notification models 和 method registry。
生成器进入 `scripts/`，输出 deterministic；生成文件有“不要手改”头。SDK 自己拥有
public-friendly facade types，但底层 wire types 必须可追溯到 schema。

**Verify**：schema drift check，且删掉一个 generated field 会让 check 失败。

### Step 2: 实现 async JSON-RPC client

支持 Unix socket、loopback TCP/token 和已提供的 stdio streams。实现：

- initialize/initialized handshake
- 单 reader task
- request id → future map
- turn id → bounded notification queue map
- global notification queue
- close 时让所有 pending future/iterator 收到明确 `ConnectionClosedError`

reader 只做协议路由；业务 convenience 放上层。错误类型至少区分 protocol、transport、
remote application、timeout、slow consumer。

**Verify**：乱序 response、两个 active turn、server error、EOF、cancelled waiter、close、
unknown forward-compatible notification tests。

### Step 3: 构建 AsyncAkashic 与 sync facade

公开：

- `AsyncAkashic.connect()` / `Akashic.connect()`
- `thread_start/resume/list/read/delete`
- `Thread.run()` / `Thread.turn()`
- `TurnHandle.events()/result()/interrupt()`
- `TurnResult`

async 是主实现。sync facade 使用单一受控 event-loop thread 或独立同步 client；禁止每个
方法 `asyncio.run()`，否则 streaming 和 pending request 会跨 loop 损坏。context manager
必须成对启动/关闭。

**Verify**：sync/async parity contract tests；一个 client 同时消费两个 turn。

### Step 4: 实现 exec 命令

增加非交互 `python main.py exec`：

- prompt 可来自 positional 或 `-` stdin，二者冲突时报参数错误。
- `--new` 或 `--thread ID` 必须二选一。
- `--json` 输出协议级 JSONL event；`--final-only` 仅输出最终文本。
- `--endpoint` 覆盖 workspace 默认。
- SIGINT 调用明确 `turn/interrupt`，等待 terminal notification 后以 130 退出。

stdout 只承载选择的机器输出，诊断写 stderr。不要复用旧 CLI print/banner。

**Verify**：subprocess tests 检查每行 JSON、exit codes、stdin、interrupt、连接失败。

### Step 5: 写最小完整文档与 examples

SDK README 至少包含 quickstart、resume、stream、interrupt、错误处理、连接 gateway、
stdio parent-owned 模式的示例。examples 必须在 contract harness 中执行，不提交不可运行
伪代码。

**Verify**：example smoke tests exit 0。

## Test plan

- 使用真实 Plan 003 app-server subprocess/harness，不 mock wire router。
- unit tests 可使用内存 stream 制造乱序/断线。
- 事件顺序断言：item notifications 全部早于 matching `turn/completed`。
- 不要求真实外部 LLM；使用仓库正式 test provider wiring。

## Done criteria

- [ ] sync/async API 具备 parity
- [ ] SDK 按 turn id 路由并支持并发 active handles
- [ ] exec 的 stdout 可机器解析且 exit code 稳定
- [ ] Ctrl+C 显式 interrupt 并等待 terminal
- [ ] SDK 不 import `agent.*`、`bootstrap.*`、`infra.*`
- [ ] tests、pyright、schema drift check 全通过

## STOP conditions

- SDK 需要 import server runtime 才能构造 public models。
- sync facade 只能通过每调用一次 `asyncio.run()` 实现。
- exec 需要解析日志文本判断 turn 是否完成。
- app-server terminal event 不能保证在所有 item event 之后。

## Maintenance notes

server 协议版本升级时先生成 models，再跑 SDK contract matrix。不要把未协商 capability 的
方法放进 high-level facade。其他语言 SDK 应复用 JSON Schema，而不是翻译 Python SDK。


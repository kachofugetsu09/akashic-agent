# 程序化控制面改造计划

由 `improve` skill 于 2026-07-14 生成，基于 commit `6b8f438d`。这是架构迁移，必须按
依赖顺序执行；不要把 app-server 直接接到现有 `IPCServerChannel` 作为捷径。

先完整阅读 `000-programmatic-control-architecture.md`。每个执行者还必须独立阅读自己
负责的计划；计划包含必要上下文和 STOP conditions。

## 执行顺序与状态

| Plan | 标题 | Priority | Effort | Depends on | Status |
|---|---|---:|---:|---|---|
| 001 | 固化现状并建立 thread/turn 持久契约 | P1 | L | — | DONE |
| 002 | 从 AgentLoop 提取 ConversationRuntime | P1 | L | 001 | DONE |
| 003 | 建立版本化 JSON-RPC app-server | P1 | L | 002 | DONE |
| 004 | 交付 Python SDK 与 one-shot exec | P1 | L | 003 | DONE |
| 005 | 迁移渠道并删除 TUI 与旧 IPC | P1 | L | 003, 004 | DONE |
| 006 | 建立 Docker 真实运行验收门 | P1 | L | 001–005 | DONE |
| 007 | 收束 CI、迁移文档与发布门禁 | P1 | M | 006 | IN PROGRESS（PR CI、nightly soak 与迁移文档已完成；真实模型 G4/release evidence policy 待仓库 secret 与发布工作流落地） |

状态值：TODO | IN PROGRESS | DONE | BLOCKED（附一行原因） | REJECTED（附原因）

## 依赖说明

- 001 先建立稳定 ID、状态机和数据库契约，否则 protocol 会固化当前偶然的数据形状。
- 002 让 app-server 和 channel 共享同一 application service；没有它，003 只能继续把
  IPC 假装成 API。
- 003 是 SDK/exec 的唯一底层协议。
- 004 必须在删除旧 CLI 前交付替代调用面。
- 005 最后切换现有渠道并删除旧实现，保证迁移中始终有可验证入口。
- 006 使用 `docker/debug` 启动真实 runtime，以确定性模型 sidecar 驱动协议、并发、故障、
  重启和资源验收；不能用 in-process mock 或固定 sleep 代替。
- 007 只在 Docker gate 通过后收束 CI、迁移文档和 release 决策。

## 关键架构决策

- `thread.id == session_key`，不搬迁已有会话主键。
- 程序化新 thread 使用 `programmatic:<uuid7>`。
- app-server 是 runtime control plane，不是 Channel。
- v1 明确不支持 `turn/steer`；不使用 interrupt+resume 冒充。
- stdio 和本地 socket 共用同一 protocol/router，transport 不拥有业务逻辑。
- 配置执行 breaking migration，不长期保留旧字段兼容层。
- 当前全局 runtime admission 在 v1 保留并显式投影为 queued；并发优化是后续独立工作。

## 已考虑但拒绝

- **直接扩展 `IPCServerChannel`**：它用连接路由消息，缺少 request/turn 相关性，继续
  扩展会固化错误抽象。
- **只做 REST endpoint**：双向流、取消完成和 tool item 需要事件协议；REST 会立刻再
  引入 SSE/WebSocket，形成第二套生命周期。
- **SDK 直接 import AgentLoop**：会绕过进程所有权、插件热重载、workspace lock 和
  transport 边界，无法调用正在运行的实例。
- **复制 Codex 全部 API**：sandbox/approval/review 是不同产品域；应借鉴协议骨架而非
  制造空方法。
- **保留 TUI fallback**：用户明确要求完全不用 TUI，而且 fallback 会让旧协议无法
  真正删除。

## 调研与验证基线

- 当前 targeted baseline：
  `/mnt/data/coding/akasic-agent/.venv/bin/pytest -q tests/test_io_modules.py tests/test_turn_pipelines.py tests/test_runtime_smoke.py tests/test_channel_clients.py`
- 结果：`111 passed in 4.83s`。
- 新 worktree 不自带依赖，验证时复用主 checkout `.venv`；不要为执行计划伪造通过结果。
- 本机调研时 Docker `29.6.1`、Docker Compose `5.3.1` 可用；这只证明基建可执行，不代表
  新控制面已经通过任何 Docker gate。

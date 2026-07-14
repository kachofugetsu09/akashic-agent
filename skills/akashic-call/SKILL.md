---
name: akashic-call
description: 指导 Codex 或其他外部程序调用已运行的 Akashic，并用明确 thread ID 持续对话。当用户说调用 akashic、程序化调用 Akashic、从 Codex 调用 Akashic、外部自动化调用、复用 Akashic session/thread、继续 Akashic 会话时使用。
when_to_use: 外部调用者需要像 Telegram 或 WebUI 一样进入固定 workspace 的正式 Agent runtime，创建或复用持久会话，并取得结构化执行结果时。
metadata: {"akashic": {"always": false}}
---

# 调用 Akashic

## 定位

把已运行 gateway 中固定模型、固定 workspace 的 Akashic 作为可程序化调用 Agent。`Thread` 是持久
session，`Turn` 是一轮对话；CLI、SDK 和 JSON-RPC 与 Telegram/WebUI 共用正式 Agent runtime、
历史、记忆、工具和插件。

这不是模型/profile 切换能力，也不为每次调用启动新 runtime。

## 硬边界

1. 调用方必须在目标 Akashic runtime 外部；gateway 应已运行。不要为调用再启动同 workspace owner。
2. 当前 Akashic turn 内禁止同步执行同 workspace 的 `python main.py exec`：全局串行 admission 会让
   当前 turn 等待自己释放执行权，形成自死锁。此时只生成外部调用命令或代码，不要代为执行。
3. Akashic 调用另一 Akashic 时，目标必须是不同 workspace、不同 runtime endpoint。
4. 自动化续聊必须保存并传入明确 `threadId`；禁止使用 `--last` 一类依赖隐式最近会话的选择方式。
5. 恢复 `telegram:*` thread 只复用历史。程序化结果返回调用方，不会自动发送到 Telegram。

## 选择入口

- 外部 Codex、shell、cron：优先 CLI；机器消费用 `--json`，只取回答用 `--final-only`。
- Python 服务：优先 `akashic_sdk`，保存 `thread.id` 后用 `thread_resume()` 持续对话。
- 非 Python 集成：使用 JSON-RPC 2.0 NDJSON；先 `initialize`，再发 `initialized`。

先确认调用方掌握 repo 路径、固定 endpoint，以及续聊所需的明确 `threadId`。新建 session 后立即把
返回的 thread ID 持久化，再发后续 turn。连接断开不会取消服务端 turn，可重连后用 `turn/read` 查询。

可复制的 CLI、Python SDK 和原始 JSON-RPC 示例见
[`references/external-caller.md`](references/external-caller.md)。可直接运行的 Unix socket 原始协议客户端见
[`examples/raw_jsonrpc_uds.py`](examples/raw_jsonrpc_uds.py)。


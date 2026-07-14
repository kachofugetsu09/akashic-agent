# 程序化控制面迁移

旧的本地聊天 IPC、纯文本 CLI 和 TUI 已删除。运行中的 gateway 现在通过 workspace 私有
`akashic.sock` 暴露 JSON-RPC 2.0 NDJSON app-server；socket 权限为 `0600`。

## 配置

旧配置：

```toml
[channels]
socket = "/tmp/akashic.sock"
```

新配置：

```toml
[app_server]
enabled = true
listen = "" # 留空时使用 <workspace>/akashic.sock
max_connections = 32
ingress_queue_size = 128
outbound_queue_size = 512
```

旧字段会在配置边界明确失败，不会被静默忽略。

## 调用

```bash
python main.py gateway --config config.toml --workspace ~/.akashic/workspace
python main.py exec --new --json "总结最近上下文"
python main.py exec --thread programmatic:THREAD_ID --final-only - < prompt.txt
python main.py app-server --stdio --config config.toml --workspace /isolated/workspace
```

`exec --json` 的 stdout 只包含 JSONL event；诊断日志写 stderr。completed/failed/参数或连接错误/
interrupted 的退出码依次为 `0/1/2/130`。

连接必须先发送 `initialize` request，再发送 `initialized` notification。`turn/start` 立即返回
turn record，进度通过 notification 发送，最终只有一个 `turn/completed`。客户端断线不会取消
turn；重连后使用 `turn/read`。中断必须同时提供 `threadId` 和 `turnId`。v1 不支持 steer。

`thread/start`、`thread/delete` 分别产生 `thread/started`、`thread/deleted` notification。
`thread/consolidate/start` 立即返回 operation id，最终通过 `operation/completed` 汇报成功或失败。
服务端当前明确协商 `reasoningEvents=false`；tool item 的 started/completed 则按真实执行时序发送。

Windows 和显式 loopback TCP 使用 `<workspace>/.app-server-token`，客户端必须在 initialize 的
`workspaceToken` 中提交；非 loopback bind 会在启动边界拒绝。Linux 默认仍使用权限 `0600`
的 workspace UDS。

Python SDK 位于 `sdk/python/`，提供异步 `AsyncAkashic` 和使用单一 event-loop thread 的同步
`Akashic` facade。SDK 只依赖 wire protocol，不 import 服务端 runtime。

升级前备份 config 和 workspace 数据库。回滚时恢复旧 binary 与旧 config 备份，禁止新旧进程
同时占有同一 workspace/socket。

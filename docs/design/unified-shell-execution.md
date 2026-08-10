# Unified Shell Execution 设计

- 状态：accepted / implementing
- 日期：2026-07-31
- 决策：[0014](../decisions/0014-shell-uses-unified-execution.md)
- 参考实现：Codex `c7a4a7e136d96554e1fc6f66532e6060fd2aaf15`
  的 `codex-rs/core/src/unified_exec/`、`codex-rs/core/src/shell.rs` 和
  `codex-rs/shell-command/src/shell_detect.rs`

## 1. 目标与成功标准

把现有前台/后台分叉改成一套 Codex 风格的可续接执行。长任务不再依赖累计日志和短周期 `task_output` 轮询；在不改变任务、模型、verifier 与容器资源的条件下，既往高轮询案例应使用更少的无信息工具调用完成或暴露真实失败。

## 2. 当前调用链与 owner

当前 `agent/tools/shell.py` 同时拥有进程创建、自动转后台、模块级注册表、日志重读、硬超时和 stop。主 runtime 在 `agent/tools/meta/register.py` 分别创建三个互不显式共享 owner 的工具；scripting/general subagent 只获得 `shell`，不能可靠续接被自动转后台的任务。

目标结构把进程状态集中到一个 manager：

```text
┌────────────────────────────────────────────────────┐
│ Tool boundary                                      │
│ shell / write_stdin / task_stop                    │
└──────────────────────┬─────────────────────────────┘
                       ▼
┌────────────────────────────────────────────────────┐
│ ShellProcessManager                                │
│ owner_session_key + execution table + hard timeout │
└──────────────┬───────────────────┬─────────────────┘
               ▼                   ▼
┌──────────────────────┐  ┌──────────────────────────┐
│ HeadTailBuffer 1 MiB │  │ append-only temp log     │
│ per-call drain       │  │ complete diagnostic data │
└──────────────────────┘  └──────────────────────────┘
```

主 runtime 的 manager 由工具注册层创建，对话 owner 来自 `current_session_key`。每个 subagent profile 创建独立 manager；subagent run 结束必须 shutdown。`CoreRuntime.stop` 关闭主 manager。

## 3. 名称与生命周期

- `execution_id`：manager 分配的 opaque integer handle，只标识一次 shell execution。
- `pid`：操作系统进程 ID，只用于平台执行边界控制，不暴露为工具主键。
- `owner_session_key`：拥有该 execution 的 Akashic 对话 session。
- execution 不写入 SessionDB、不跨 runtime 恢复；这与 Akashic 持久对话 session 完全不同。

## 4. 已确认事实与边界

- 已确认：用户要求尽量 1:1 转译 Codex 的 Rust unified exec，并明确接受旧 Shell 工具协议的 breaking change。
- 已确认：Codex 外部字段 `session_id` 在 Akashic 中改名为 `execution_id`；状态机、等待和输出消费语义不变。
- 已确认：Akashic 仍需要 SH-001 的硬超时、对话 owner 和 runtime shutdown 回收；这些是相对 Codex 的窄扩展。
- 受保护：正式 Akashic workspace、线上 runtime、provider 配置、Benchmark task/verifier、模型与推理强度。
- 允许副作用：独立 Git worktree 的源码和文档；测试临时目录；单题独立 Docker trial 及其 artifact。
- 不做：完成通知、跨 runtime 恢复、持久进程队列、隐藏 verifier 读取、完整 89 题 eval。

## 5. 接口

`shell` 参数为 `command`、`description`、`cwd`、`shell`、`login`、`tty`、
`yield_time_ms`、`max_output_tokens`、`timeout`。默认初始等待 10 秒；等待范围
250 毫秒～30 秒；默认硬超时 4 小时。

未指定 `shell` 时，Unix 从 passwd 读取当前用户 shell，不读取 `$SHELL`；不支持或
不可用时按 Codex 的平台顺序寻找 Bash、Zsh 和 Sh。Windows 依次寻找 PowerShell
和 Cmd。显式 shell 只接受 Codex 已定义 argv 语义的类型，缺失或未知时明确失败。
Unix shell 使用 `-lc` 或 `-c`，PowerShell 使用可选 `-NoProfile` 加 `-Command`，
Cmd 使用 `/c`。manager 通过 `create_subprocess_exec(*argv)` 直接创建进程，不再把
字符串交给 Python 隐式选择 `/bin/sh -c`。

这是 clean break：不保留旧 `/bin/sh` 默认、不增加 legacy config、不注册
`task_output` alias，也不接受 `run_in_background`、`auto_promote` 或 `task_id`。
`shell` 和 `login` 是当前接口本身，不是兼容入口。历史 SessionDB trace 仍按原始
工具名和参数只读重放，但绝不重新解释或执行。

`write_stdin` 参数为 `execution_id`、`chars`、`yield_time_ms`、`max_output_tokens`。空 `chars` 是等待/增量读取，默认 5 秒且最大 300 秒；非空输入默认 250 毫秒且最大 30 秒。

`task_stop` 只接收 `execution_id`。它在确认平台执行边界终止后才返回 stopped；终止失败时保留注册项并明确失败。Unix 边界是创建 shell 时独立出的 process group；Windows 边界是 `taskkill /T` 可见的后代集合。这与 Codex 的 process-group ownership 对齐，不承诺追踪显式 `setsid`、daemonize 或外部服务管理器接管的进程；需要该保证的调用方必须使用拥有 cgroup/Job Object 的受控 runner。

`max_output_tokens` 只控制本次工具结果的近似输出预算，不是 provider 的生成上限。完整输出写入临时日志；发生省略时结果返回日志路径。

## 6. 失败、取消和并发

- spawn 或日志创建失败：工具调用失败，不创建假 execution。
- 输出 pump 失败：记录具体异常、终止进程，防止进程在不可观察状态继续运行。
- 初始或后续等待取消：只取消该次等待；execution 与 pump 继续存在。
- 进程退出与等待截止竞争：最终 drain 后再判断存活，已经退出就返回退出码，不返回 `execution_id`。
- 子孙继承 pipe：主进程退出后只排水 200 毫秒，随后关闭 reader，避免 execution 永久悬挂。
- UTF-8 被 chunk 拆开：按字节缓冲，本次返回时统一 replacement decode；完整日志保留原字节。
- owner 不匹配或未知 ID：在工具边界明确返回错误，不泄露其他对话输出。
- manager 达到 64：按 Codex LRU 规则回收；被回收的 live process 必须确认终止。
- pipeline 沿用所选 shell 的退出语义，不自动注入 `pipefail`；任务依赖上游失败时
  必须在命令中显式启用。
- 当前 query compaction 不得淘汰仍处于 running 状态的 execution 所在批次；收到
  terminal `write_stdin` 或成功 `task_stop` 后才允许压缩该批次。
- 主 ReAct 与 SubAgent 的历史裁切都必须识别真实 tool transport envelope，保留 active
  execution 的原始 shell 结果；当前 query 返回、失败或取消时由 owner 回收剩余 execution。
- 当前 query cleanup 与 turn 终态按 [0015](../decisions/0015-cleanup-does-not-own-turn-or-restart-finality.md)
  解耦。cleanup 未确认时保留 execution 并隔离本 runtime 内同 owner 的新 Shell spawn；
  结构化诊断不得反向把已提交回复改成 failed。

## 7. 验证

先做确定性测试，再运行一题隔离 Benchmark：

1. Head/tail 缓冲、增量 drain、输出预算和 UTF-8 边界单测。
2. 短命令、长命令、取消后续接、退出竞争、PTY 输入、非 PTY 输入、Ctrl-C、硬超时、stop、owner、LRU 与 shutdown 集成测试。
3. 工具注册、subagent cleanup 和 `CoreRuntime.stop` 生命周期测试。
4. 运行 targeted pytest、静态检查和 change-impact Gate。
5. 在独立 Docker 中复跑 `train-fasttext`；冻结 task、verifier、provider、模型、effort 和资源，比较旧基线与新协议的 shell/write 次数、无新增输出的轮询数、token、耗时、最终状态和 verifier 结果。

单题结果只判断该假设，不外推为 89 题通过率。若失败来自模型、任务本身或 verifier，按真实层级记录，不为分数修改题目或 oracle。

### 7.1 Codex 测试映射

| Codex 机制 | Akashic 验证 | 处理 |
|---|---|---|
| `head_tail_buffer_tests.rs` 7 个边界 | `tests/test_unified_exec.py` | 按字节预算逐项转译 |
| repeated drain / omission | `tests/test_unified_exec.py` | 转译 collector 有界性和 omission 传播 |
| 多执行、PTY state、timeout、completed reuse | `tests/test_shell_tool.py`、`tests/test_unified_exec.py` | 按 `execution_id` 命名转译 |
| initial wait / stdin poll 与 terminate 竞态 | `tests/test_unified_exec.py` | 转译并断言不残留句柄 |
| exited-first / LRU / recent-8 prune | `tests/test_unified_exec.py` | 逐项转译 |
| shell detection 和 `derive_exec_args` | `tests/test_shell_tool.py` | 转译默认、显式、login 和 argv |
| sandbox、approval、remote exec server、UI events | 无 | Akashic 没有对应 owner，不伪造等价测试 |
| Windows ConPTY | 无 | 当前 stdlib 实现明确拒绝 `tty=true` |

Akashic 另外验证对话 owner 隔离、同 owner 跨 ReAct 调用续接、四小时硬超时、
subagent/runtime shutdown、旧 trace 只读重放、主 runtime 的 active execution compaction pin，
以及 subagent 的内存态 context compaction。

### 7.2 `train-fasttext` 定向结果

冻结 trial
`akasic-bench-v4flash-smoke-train-fasttext-20260731-155617-339973` 在独立容器中运行
满 3600 秒后未收口，未进入 verifier。失败层是模型任务策略：它在多轮训练结果仍
不足后继续启动 900 秒和 1800 秒 autotune，并在 deadline 前继续创建新预处理，
不是 execution 丢失、容器资源或 verifier 故障。

统一协议从旧 run 的 `45 shell + 107 task_output + 1 task_stop` 变为本次
`27 shell + 17 write_stdin`；总工具调用从 154 降为 59。最长 execution 由同一
`execution_id` 连续等待约 15 分钟，三次 300 秒空等待均保持可续接。两次模型 run
不具确定性，因此这个结果只接受“协议消除了累计日志高频轮询”的机制证据，不接受
“提高 case 通过率”的结论。

## 8. 回滚

代码回滚到 `backup/unified-shell-pre-change-20260731` 或任务基线 `e49f2a737c4432088ef7b864878823427784eb0f`。测试容器和临时日志可以在证据收集后删除；正式 workspace 没有 write set。

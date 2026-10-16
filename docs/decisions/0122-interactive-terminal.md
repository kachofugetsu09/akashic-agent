# 0122 · 工作台终端是用户自己的交互 shell

- 状态：proposed / implemented for review
- 日期：2026-10-16
- 修订：0121（PR #1420，工作台标签）中“终端 = Agent shell 命令回放”的切片
- 关联：WEBUI-008、PLG-016、GOV-004

维护者要求工作台的终端用起来像 kitty：用户在侧栏里自己敲命令，不是查看 Agent 跑过的命令。
因此终端改为执行主机上的长期 PTY 会话；Agent 的命令仍由对话里的工具卡展示，不再单独记录一份。

```text
┌───────────────┐  WebSocket   ┌──────────────┐  host.terminals.v1  ┌──────────────────┐
│ 终端标签       │─────────────▶│ terminal 插件 │────────────────────▶│ host_execution    │
│ xterm.js       │ 输入/尺寸/输出│ 会话 → 终端   │ open/write/resize/  │ PTY + 环形缓冲     │
└───────────────┘              └──────────────┘ read/close/list     │ 本机或 Host Bridge │
                                                                     └──────────────────┘
```

## 决定

- **执行主机拥有 PTY。** `host.terminals.v1` 与 `host.processes.v1` 共用同一个 manager、关闭排空和
  owner 隔离；终端运行在 Agent 命令所在的同一台机器上（本机模式或 Host Bridge 宿主）。
- **shell 由执行主机选。** 用用户自己的登录 shell（`$SHELL` → passwd → Agent shell 默认），
  `TERM=xterm-256color`、`COLORTERM=truecolor`；子进程经 `login_tty` 取得控制终端，作业控制和
  Ctrl+C 与普通终端一致。调用方只给工作目录和尺寸，不传环境变量。
- **输出按偏移读，读了不删。** 每个终端保留最近 256 KiB；读取在没有新输出时等待，长轮询即可实时回显，
  Host Bridge 不需要流式传输。断线重连从缓冲开头回放；偏移大于请求值表示更早的输出已滚出。
- **不设硬超时。** 终端与 shell 同生同死；关闭等同关掉窗口：释放 PTY 主端让内核挂断，宽限 2 秒后
  结束会话里剩下的进程（Linux 按会话查找，覆盖转入后台进程组的作业）。owner 清理和 generation
  关闭会关闭其全部终端。每个 owner 最多 8 个，全局 32 个，已退出未关闭的也计数。
- **Agent 不读用户终端。** 不提供读取终端的模型工具；终端是用户的，不进入上下文。

## 信任边界

交互终端等同于让已登录的 Web 用户在执行主机上执行任意命令。这与 Agent 已有的 shell 能力处于同一信任级别，
但入口变成了浏览器。因此终端只经已登录的 Dashboard 路由暴露，沿用宿主的浏览器请求约束；停用 terminal
插件即整体关闭这一入口，host_execution 本身不暴露 HTTP。

## 已知限制

- 应用重启或 host_execution 换代后终端不保留（与 Agent 后台进程相同）。
- 截断回放可能从全屏程序（vim、htop）中途开始，首屏可能错位；按一次重绘键即可恢复。
- Windows 执行主机暂不支持（需要 ConPTY）。

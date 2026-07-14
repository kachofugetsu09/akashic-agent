# Plan 005: 迁移渠道并删除 TUI 与旧 IPC

> **Executor instructions**：执行前确认 Plans 003–004 已 DONE，SDK/exec 已是可用替代面。
> 本计划是 breaking cutover，不保留静默 fallback。完成后更新状态。
>
> **Drift check**：
> `git diff --stat 6b8f438d..HEAD -- infra/channels bootstrap main.py agent/config.py agent/config_models.py requirements.txt README.md tests`

## Status

- **Priority**: P1
- **Effort**: L
- **Risk**: HIGH（删除入口、配置和 channel wiring）
- **Depends on**: Plans 003 and 004
- **Category**: migration / tech-debt
- **Planned at**: commit `6b8f438d`, 2026-07-14

## Why this matters

用户明确要求完全不用 TUI。只新增 app-server 而保留旧 IPC/TUI 会留下两套调用链和两套
配置，后续 bug 会持续出现“程序化入口与真实渠道行为不一致”。本计划完成 cutover：所有
输入共享 ConversationRuntime，旧 CLI transport 和 UI 彻底删除。

## Current state

- `main.py:156-175` TUI-first、纯文本 fallback。
- `infra/channels/cli_tui.py:13-23` 动态导入 Textual/Rich 并转换 ImportError。
- `infra/channels/ipc_server.py:42-316` 同时承担 socket server、聊天 channel、命令表。
- `bootstrap/channels.py:32-41` 把 IPC 无条件作为 channel 启动。
- `agent/config_models.py:39-45` 把 socket/CLI session 混在 ChannelsConfig。
- `requirements.txt:42` 包含 `textual`。

## Commands you will need

| Purpose | Command | Expected |
|---|---|---|
| Channel regression | `/mnt/data/coding/akasic-agent/.venv/bin/pytest -q tests/test_channel_clients.py tests/test_io_modules.py tests/test_runtime_smoke.py` | all pass after obsolete tests removal/replacement |
| Full search | `rg -n "cli_tui|CLITextualApp|IPCServerChannel|main.py cli|channels\.cli|cli_session_key" --glob '!advisor-plans/**' .` | no matches |
| Typecheck | `/mnt/data/coding/akasic-agent/.venv/bin/pyright --level error` | exit 0 |
| Dependency | `rg -n "textual" requirements.txt pyproject.toml uv.lock` | no matches |

## Scope

**In scope**:

- 删除 `infra/channels/cli_tui.py`
- 删除 `infra/channels/cli.py`
- 删除 `infra/channels/ipc_server.py`
- `bootstrap/channels.py`、`bootstrap/app.py`
- channel adapters 与 PassiveMessageWorker wiring
- `main.py`
- `agent/config_models.py`、`agent/config.py`、example config
- `requirements.txt`、lockfile
- README/handbook/AGENTS commands
- obsolete tests 删除并由 app-server/channel contract tests 替代

**Out of scope**:

- 改 Telegram/QQ/Web 产品行为或 UI
- 新增远程 transport
- 保留旧 JSON payload 自动识别
- 顺手重构其他 CLI admin commands

## Git workflow

- Commit: `refactor(control): remove tui and legacy ipc channel`
- 不 push，不开 PR。

## Steps

### Step 1: 让所有现有渠道使用 ConversationRuntime

确认 Telegram/QQ/Web 的 inbound 都经 PassiveMessageWorker/ConversationRuntime，failure
文案由 channel adapter 映射，stream/tool events 来自同一个 turn event source。不得保留
调用 AgentLoop private/process_direct 的渠道路径。

**Verify**：同一 fake turn 对 app-server 与各 channel projector 产生一致 final/status/tool
语义；只允许展示格式不同。

### Step 2: 迁移 runtime 管理命令

当前 `plugin-disable-and-drain` 不能随 IPCServerChannel 删除而丢失。将其变为明确的
application/admin service，并通过合适的 app-server method 或现有本地 admin CLI 调用。
若纳入 app-server，先扩 schema 和 SDK low-level model；不要临时塞 raw command method。

**Verify**：plugin disable/drain 的原有 tests 迁移后继续覆盖等待 generation lease 的语义。

### Step 3: 切换配置所有权

删除 ChannelsConfig 的 `socket`、`cli_session_key` 和旧 parser。只保留 `[app_server]`
配置。发现旧字段时在 config 加载边界抛明确迁移错误和新字段示例；不要静默兼容。

更新 setup/init/example config。按项目要求，执行者在改真实用户配置前必须单独备份；本
计划只改仓库模板和 loader，不自动写用户 workspace。

**Verify**：fresh config、new config、old config rejection tests。

### Step 4: 删除 TUI、纯文本 CLI 和 IPC channel

删除三个文件及所有 imports/wiring/tests。删除 `cli` command 和帮助文案，README 改为
`exec` 与 SDK。移除 `textual`；若 `rich` 没有其他运行时使用，再单独证明确实无引用后
移除，不能仅因 TUI 使用过就猜测。

**Verify**：full search 无命中，依赖 lock 重新生成且不含 textual。

### Step 5: 收紧启动与 shutdown

`AppRuntime` 只启动 app-server control plane + real channels，不再返回 `ipc` object。
shutdown cleanup step 名称、顺序和 tests 更新为 app-server。启动失败应保留原始异常，
socket 清理失败作为 cause，不得吞错。

**Verify**：runtime smoke 覆盖 app-server start failure、channel start failure、shutdown
cancellation 和 socket cleanup failure。

## Test plan

- 迁移 `tests/test_channel_clients.py` 中真正属于 Telegram/Web 的事件测试。
- 删除只测试 TUI CSS/widgets 的 tests，不用 mock 新 UI 替代。
- `tests/test_io_modules.py` 中 IPC tests 由 `tests/control/test_socket_transport.py` 替代。
- 运行 plugin disable/drain、fresh config、runtime smoke 回归。

## Done criteria

- [ ] 仓库无 TUI/legacy IPC source 或 runtime dependency
- [ ] `main.py cli` 不存在，`exec`/SDK 已替代
- [ ] 所有渠道和 app-server 共用 ConversationRuntime
- [ ] 旧 config 明确报迁移错误
- [ ] plugin disable/drain 能力未丢失
- [ ] targeted tests 与 pyright 通过

## STOP conditions

- Plan 004 的 exec/SDK 仍不能完成 start→stream→interrupt→result。
- 任一生产渠道仍依赖 IPC chat id 才能路由最终回复。
- plugin 管理存在外部脚本依赖旧 raw command 且没有已确认迁移策略。
- 删除 textual 会连带删除仍被非 TUI 代码使用的依赖。

## Maintenance notes

后续不要重新引入“本地 channel”作为自动化 API。需要新客户端时，优先基于 SDK；需要新
语言时基于 schema。README 中任何交互示例都应明确是 `exec` 或 Web/Telegram，而非 TUI。


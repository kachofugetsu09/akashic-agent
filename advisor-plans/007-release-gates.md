# Plan 007: 收束 CI、迁移文档与发布门禁

> **Executor instructions**：仅在 Plan 006 DONE 后执行。本计划不再改变控制面架构或降低
> Docker gate；发现缺口退回所属计划修复。
>
> **Drift check**：`git diff --stat 6b8f438d..HEAD`

## Status

- **Priority**: P1
- **Effort**: M
- **Risk**: MED（CI 时间、breaking migration 与发布可恢复性）
- **Depends on**: Plan 006
- **Category**: ci / docs / release
- **Planned at**: commit `6b8f438d`, 2026-07-14

## Why this matters

Docker gate 证明实现能真实运行；本计划把它变成不能被绕过的合并/发布规则，并完成旧 TUI/
IPC 到 app-server/SDK/exec 的用户迁移。两者必须分开：不能用文档完成掩盖 runtime gate，也
不能只有本地脚本而没有 CI 门禁。

## Commands you will need

| Purpose | Command | Expected |
|---|---|---|
| Python type | `/mnt/data/coding/akasic-agent/.venv/bin/pyright --level error` | exit 0 |
| Test type | `/mnt/data/coding/akasic-agent/.venv/bin/pyright --project pyrightconfig.tests.json --level error` | exit 0 |
| Full tests | `/mnt/data/coding/akasic-agent/.venv/bin/pytest -q -W error tests/` | all pass |
| Frontend | `npm run typecheck && npm run lint` | exit 0 |
| Schema | `/mnt/data/coding/akasic-agent/.venv/bin/python scripts/generate_control_schema.py --check` | exit 0 |
| SDK | `cd sdk/python && /mnt/data/coding/akasic-agent/.venv/bin/pytest -q tests` | all pass |
| Docker smoke | `python docker/debug/programmatic_control_probe.py --gate smoke` | exit 0 |
| Docker faults | `python docker/debug/programmatic_control_probe.py --gate failure-matrix` | exit 0 |

## Steps

### Step 1: 接入 PR CI

在现有 `.github/workflows/ci.yml` 上增加 schema drift、SDK tests、G1、Docker G2/G3。设置 job
timeout，上传失败和成功的精简 gate artifacts；禁止 `continue-on-error` 和自动 rerun。

**Verify**：CI 从干净 checkout 执行，未读取开发者 profile/缓存；任一 check 失败则 PR 红灯。

### Step 2: 接入 nightly/release CI

G4 使用受保护 secret；G5 按当前 commit 产出报告。release job 校验该 commit 存在最近成功的
G4 和 G5，不接受其他 commit 的历史绿灯。

**Verify**：无 secret 的普通 PR 不运行 G4；release 缺少对应成功证据时 fail-loud。

### Step 3: 完成配置和调用迁移文档

写清旧 `main.py cli`、`[channels.cli]`、旧 `{content: ...}` IPC 到 `exec`、SDK、
`[app_server]` 的迁移；包含协议版本、stdout/stderr、exit codes、断线不取消、interrupt、历史
读取、UDS 权限/Windows token、workspace lock 和 v1 不支持 steer。

提供可执行示例，并由 docs smoke 真实运行，不能保留过期 copy-paste 命令。

### Step 4: 全仓残留审计

删除 TUI/IPC 实现、依赖、docker entrypoint 分支和测试。迁移说明中的旧词使用明确 allowlist；
其他源代码与文档不得命中 `cli_tui|CLITextualApp|IPCServerChannel|python main.py cli`。

**Verify**：搜索脚本对 allowlist 外命中 exit 1；Docker entrypoint 只暴露新入口。

### Step 5: release 与回滚

更新版本/changelog，明确 breaking migration。发布前备份配置和 DB；回滚是恢复旧 binary +
旧 config 备份，不能让新旧服务同时占用同一 workspace/socket。

**Verify**：fresh install、旧配置明确 rejection、升级、回滚演练均有机器可读结果。

## Done criteria

- [ ] PR CI 强制 G0–G3，无 allow-failure/自动重跑
- [ ] nightly G4/G5 与 commit 绑定，release 缺证据即失败
- [ ] Python、SDK、frontend、schema、Docker gates 全通过
- [ ] 文档示例由 smoke test 执行
- [ ] allowlist 外无旧 TUI/IPC 符号和命令
- [ ] migration/release/rollback 文档和演练完整

## STOP conditions

- CI 环境只能访问真实外部模型才能验证核心控制面。
- Docker gate 被拆成只启动容器、不执行 PC 场景的形式。
- 因 flaky 采用 retry 或放宽 deadline，而没有修复真实同步条件。
- 全量 suite 暴露非本次改造既有失败；记录真实失败并请求范围决定，不得跳过。

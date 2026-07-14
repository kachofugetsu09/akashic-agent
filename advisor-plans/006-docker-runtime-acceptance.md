# Plan 006: 建立 Docker 真实运行验收门

> **Executor instructions**：仅在 Plans 001–005 全部 DONE 后执行。测试必须启动真实
> gateway/app-server、真实 SessionStore、真实 plugin/runtime lifecycle，并经正式 provider
> HTTP 配置访问确定性模型 sidecar。禁止 monkeypatch runtime 入口或用 in-process fake
> 代替本计划的 Docker gate。
>
> **Drift check**：`git diff --stat 6b8f438d..HEAD`

## Status

- **Priority**: P1
- **Effort**: L
- **Risk**: HIGH（跨进程协议、并发、持久化与容器生命周期）
- **Depends on**: Plans 001–005
- **Category**: docker / integration / reliability
- **Planned at**: commit `6b8f438d`, 2026-07-14

## Why this matters

本次改造替换的是进程入口和执行所有权。静态检查只能证明类型和局部函数，不能证明 JSON-RPC
framing、真实进程启动、provider 流式返回、并发排队、断线不取消、背压隔离、SIGTERM 收束、
SQLite 终态和重启恢复。合并门必须验证“像使用者一样从进程外调用”这一整条链。

## 已有基建与需要修正的缺口

### 可直接复用

- `docker/debug/docker-compose.plugin-gate.yml` 已有 `/app:ro`、read-only rootfs、`/tmp`
  tmpfs、独立 `/sandbox` 的安全形状。
- `plugin_hot_reload_probe.py` 已有唯一 compose project、repo digest 审计、容器证据采集、
  `down --remove-orphans` 和结构化 gate result。
- `full_replay_runner.py::wait_until_stable()` 已有“业务状态收敛 + quiet window”范式。
- `runtime_race_probe.py` 已覆盖当前 ChatLane/AgentLoop 的阻塞、串行和取消竞态，可迁移为
  `ConversationRuntime` 的进程内快速门。

### 不能沿用为验收标准

- `docker-compose.yml` 将仓库以可写方式挂到 `/app`，不能作为合并 gate。
- `context_probe.py` 只等待 socket path 出现，并发送旧 `{content: ...}` NDJSON；这既不证明
  server ready，也不验证 request/turn correlation。
- `context_probe.py` 在 consolidate/final 后使用固定 sleep；必须改成 operation notification、
  sidecar barrier 或 DB convergence。
- `plugin_hot_reload_probe.py::_ipc_ready()` 只建立 socket 连接；新 gate 必须完成协议握手和
  `server/status`。
- `entrypoint.sh` 的 `cli` 仍走 `python main.py cli`；Plan 005 后必须只保留 `gateway`、
  `app-server --stdio` 和 `exec`。

## 验收拓扑

新增独立 `docker/debug/docker-compose.control-gate.yml`，不要把控制面场景塞进 plugin gate：

```text
                         compose private network
┌──────────────────┐ UDS/stdio ┌─────────────────────┐ HTTP ┌──────────────────┐
│ control_probe.py │──────────▶│ akashic-control-gate│─────▶│ model-gate       │
│ host controller  │◀──────────│ real PID 1/runtime  │◀─────│ deterministic API│
└────────┬─────────┘ events    └──────────┬──────────┘      └────────┬─────────┘
         │                               │                           │
         ▼                               ▼                           ▼
 reports/<run-id>/                /sandbox DB/socket          scripted requests
```

### `akashic-control-gate`

- 从当前 commit 构建，`/app:ro`，read-only rootfs，`/tmp` 为 tmpfs。
- `/sandbox` 使用 controller 创建的唯一临时目录，不复用开发者 profile。
- UDS gateway 场景启动完整 runtime，但关闭 Telegram/QQ/Web/Dashboard/proactive transport，
  保留真实配置加载、插件加载、数据库、Provider、ConversationRuntime 和 app-server。
- stdio 场景由 controller 用 `docker compose run --rm -T ... app-server --stdio` 托管，
  stdout 只允许 JSON-RPC，日志只能写 stderr。
- 不发布公网端口；model-gate 仅在 compose 私网可见。

### `model-gate`

新增 `docker/debug/model_gate.py`，提供项目 Provider 所使用的 OpenAI-compatible endpoint，
以及只在 gate 网络内开放的控制面：

- `GET /readyz`：进程 ready。
- `PUT /control/script`：加载下一次请求的完整、流式、tool call、429、500、超时或截断脚本。
- `PUT /control/barriers/{name}`：让指定模型请求到达后阻塞。
- `POST /control/barriers/{name}/release`：精确释放，不用 sleep 制造竞态。
- `GET /control/requests`：返回收到的请求、时间、thread/turn correlation metadata。

它不是 fake AgentLoop：请求仍必须穿过真实 Provider、序列化、流解析、usage 汇总和异常映射。
sidecar 只替代外部网络模型，保证场景离线且可重放。

### `programmatic_control_probe.py`

controller 负责创建 sandbox、渲染配置、启动/停止 compose、完成协议调用、注入 sidecar
脚本、查询 DB 终态和写证据。任何阶段异常都必须进入 `finally`：采集日志、停止容器、删除
orphan、检查 socket/PID/container 残留，并审计 repo digest。

## 等待与时序规则

必选 gate 禁止裸 `sleep()`。只允许三种等待：

1. **协议 readiness**：连接后完成 `initialize`、`initialized`、`server/status`，总 deadline
   30 秒；socket path 只能是连接前置条件，不能判定 ready。
2. **确定性 barrier**：并发/interrupt/断线场景等待 model-gate 报告请求已到达，再触发下一
   动作；barrier 释放必须有控制响应。
3. **状态收敛**：后台持久化使用明确 predicate，连续 2 秒无变化才算 stable，总 deadline
   15 秒。输出最后一次未收敛状态，不能只报 timeout。

性能采样可用固定间隔，但不能用它判断业务成功。

## Gate 分层

| Gate | 运行时机 | 环境 | 是否阻断 PR |
|---|---|---|---|
| G0 静态/单元 | 每次 PR | host venv | 是 |
| G1 进程内竞态 | 每次 PR | host venv | 是 |
| G2 Docker 协议 smoke | 每次 PR | local Docker + model-gate | 是 |
| G3 Docker 故障/并发 | 每次 PR | local Docker + model-gate | 是 |
| G4 真实模型 canary | nightly/manual | secret + external network | 否 |
| G5 资源 soak | nightly，发布前必跑 | local Docker + model-gate | 否（发布阻断） |

G4 不得成为合并所必需的“绿色灯”；外部服务不稳定不能掩盖核心协议是否正确。发布候选必须
同时满足最近一次 G4 成功和当前 commit 的 G5 成功。

## 必选场景矩阵

每一行都是独立 check id，失败必须在 `gate.json` 中包含请求、事件、DB 终态和日志位置。

| ID | 场景 | 驱动方式 | 机器可判定的通过条件 |
|---|---|---|---|
| PC-01 | gateway readiness | UDS | 30s 内完成 handshake/status；socket `0600` |
| PC-02 | stdio framing | stdio | stdout 每行均为 JSON-RPC；stderr 无协议帧；正常退出 0 |
| PC-03 | thread/turn 基本流 | UDS | start/read/list 一致；turn 恰有一个 terminal event |
| PC-04 | streaming/tool/usage | scripted stream | item 顺序合法；delta 拼接等于 final；tool call id/usage 持久化一致 |
| PC-05 | 两连接两 thread | two clients + barriers | request/turn/event 无串线；全局 admission 明确表现为 queued |
| PC-06 | 同 thread 冲突 | held first turn | 第二次 start 返回 `-32011`；首 turn 不受影响 |
| PC-07 | 精确 interrupt | held model request | 只终止匹配 `threadId+turnId`；旧 turn id 不影响新 turn |
| PC-08 | 断线恢复 | disconnect mid-stream | turn 继续；重连 `turn/read` 与 DB terminal result 一致 |
| PC-09 | 慢客户端背压 | stop reading notifications | 只关闭溢出连接；另一客户端 turn 正常完成 |
| PC-10 | provider 故障 | 429/500/timeout/truncated stream | failed/error type/retryable 符合契约；无普通成功文案冒充失败 |
| PC-11 | 非法协议输入 | invalid JSON/version/params/oversize | 稳定 JSON-RPC error；连接策略符合协议；server 仍可服务新连接 |
| PC-12 | SIGTERM | queued + running turns | 先停 ingress，再收束 turn；15s 内容器 exit 0，无非终态 DB 记录 |
| PC-13 | stale socket/restart | kill then restart | stale socket 被安全处理；持久 turn 可读；无重复 terminal event |
| PC-14 | workspace lock | start second owner | 第二实例 fail-loud 非零退出；第一实例与 DB 不受影响 |
| PC-15 | source/sandbox hygiene | whole run | repo digest 未变；写入仅发生于 sandbox/tmp；无残留容器/socket/PID |
| PC-16 | channel parity | same deterministic fixture | programmatic 与 channel adapter 的领域状态/items/error 分类一致 |

### 事件不变量

对所有成功或失败 turn 统一检查：

- `turn/queued` 最多一次，`turn/started` 恰一次，terminal event 恰一次。
- terminal 只能是 `completed|failed|interrupted|cancelled`，terminal 之后不得再有 item/turn event。
- 每个 item 先 started 后 completed；assistant delta 的顺序号严格递增且无缺口。
- response id 必须对应发起请求；notification 必带正确 `threadId`、`turnId`，item 再带
  `item.id`。两连接场景不允许出现其他 thread 的私有 notification。
- DB 的 turn status、items、final response、usage/error 与 terminal notification 完全一致。

## 故障注入细节

- 429：第一次 retryable，sidecar 记录重试次数；若 Provider 策略允许重试，最终只能出现
  一个领域 terminal event，不能为每次 HTTP attempt 建 turn。
- 500：按现有 Provider 分类返回稳定 `provider_error`；不要在测试中硬编码用户中文文案。
- timeout：barrier 保持到 Provider deadline，断言取消传播到 HTTP request，runtime 仍可接
  下一 turn。
- truncated stream：至少产生过 delta 后断流，最终必须 failed；不允许把部分文本标为
  completed。
- SIGTERM：用容器 stop 发送真实信号，不直接调用 Python shutdown 函数。
- crash/restart：使用 SIGKILL 只测试恢复路径，明确与优雅 shutdown 的期望不同。

## 量化验收门槛

这些是 gate 的硬断言，不是人工观察项：

- readiness：本地构建完成后，进程启动至完整协议 ready ≤ 30 秒。
- controller 单场景 deadline：普通场景 15 秒；故障 timeout 场景为 Provider timeout + 5 秒。
- interrupt：model-gate 已确认请求阻塞后，发出 interrupt 到 terminal event ≤ 2 秒。
- graceful shutdown：`docker compose stop -t 15` 后容器 exit code 0，≤ 15 秒，无 queued/
  in_progress turn 残留。
- isolation：PC-05/09 中未受影响客户端必须在释放自身 barrier 后 5 秒内完成。
- cleanup：controller 返回前 compose project 容器数为 0；sandbox 外新增/变化文件为 0。
- repeatability：G2+G3 连续运行 3 次全部通过；任一 flaky run 视为失败，不用自动重跑漂绿。

如果当前 CI 主机无法稳定满足时间阈值，应先记录测量并调整资源规格；不能把 deadline 删除或
改成无限等待。

## Nightly / release soak

G5 在 warm-up 10 turns 后运行 100 turns，包含 10 次重连、10 次 interrupt、10 次注入失败，
每 10 turns 采集容器 RSS、open fd、线程/async task、DB 非终态数：

- 100 turns 后全部 turn 有且仅有一个 terminal 状态。
- warm-up 后 RSS 增量 ≤ 64 MiB。
- open fd 增量 ≤ 8。
- idle 收敛后 async task 数 ≤ warm-up baseline + 3，无额外 child process。
- DB 中 queued/in_progress 数为 0，重复 terminal/event id 为 0。
- 整体 deadline 10 分钟；超时直接失败并保留最后一次资源快照。

阈值若经基线测量需要改变，必须在 PR 中附三次测量和原因，不能在失败后临时放宽。

## 证据产物

每次运行写入 `docker/debug/reports/programmatic-control/<run-id>/`（目录 gitignored）：

- `gate.json`：commit、镜像 digest、环境、每个 check 的 pass/fail/duration/evidence path。
- `events.jsonl`：原始 JSON-RPC request/response/notification，敏感字段脱敏。
- `model-requests.jsonl`：sidecar 收到的请求、attempt、barrier 和返回脚本。
- `db-snapshot.json`：thread/turn/item/operation 的终态投影。
- `server.stderr.log`、`compose.log`、`resource.jsonl`。
- `repo-digest.before.json`、`repo-digest.after.json`。

controller 只在全部必选 check 通过、cleanup 成功、digest 相同后 exit 0。即使业务断言通过，
清理或证据写入失败也必须非零退出。

## 实施步骤

### Step 1: 迁移快速竞态 probe

把 `runtime_race_probe.py` 从 ChatLane/AgentLoop 偶然结构迁到 `ConversationRuntime`，保留阻塞
reasoner 作为 G1。新增：全局 admission 排队、同 thread 冲突、旧 turn interrupt、新旧 task
代际、shutdown 时 queued/running 收束。

**Verify**：输出结构化 JSON，所有场景 exit 0；任何异常 exit 1，无 sleep。

### Step 2: 建立 model-gate 与 compose

实现 sidecar、只读 compose 和独立 sandbox。Provider 只通过正式 config `base_url/model/key`
连接 sidecar，不得增加测试专用 runtime branch。

**Verify**：sidecar 收到真实 Provider payload；repo digest 相同；容器内 `/app` 写入失败。

### Step 3: 实现 controller 与 G2

先完成 PC-01–04、15，覆盖 UDS、stdio、基本 turn、stream/tool/usage 和 cleanup。将旧
`context_probe.py` 的程序化调用迁到正式 SDK/protocol client；不要保留旧 payload fallback。

**Verify**：G2 连续三次通过，reports 完整，任一次失败可只凭产物定位到 request/turn。

### Step 4: 实现 G3

用 barrier 实现 PC-05–14、16。每个场景创建独立 thread；会破坏进程的场景独立启动 compose
project，避免前一场景污染预期。

**Verify**：G3 连续三次通过；无固定 sleep、自动重跑或 expected traceback 白噪声。

### Step 5: 建立 nightly canary 与 soak

真实模型 canary 使用专用 profile/secret，只发一个最小 turn；不做主观文本匹配，只断言协议
terminal、非通用错误、usage/持久化和日志脱敏。G5 实现上述资源采样和阈值。

**Verify**：无 secret 进入 report；无 secret 时 G4 明确 skipped 且不影响 PR，发布检查则因
缺少最近成功记录 fail-loud。

## CI 命令接口

最终提供稳定入口，具体脚本名可按项目约定调整，但语义不可拆散：

```bash
python docker/debug/programmatic_control_probe.py --gate smoke
python docker/debug/programmatic_control_probe.py --gate failure-matrix
python docker/debug/programmatic_control_probe.py --gate soak
```

默认 smoke/failure-matrix 均负责 build、up、readiness、run、evidence、down 和 digest audit。
CI 不应在外面复制一套 compose lifecycle。

## Done criteria

- [ ] G1 迁移到 ConversationRuntime，竞态场景无 sleep
- [ ] control-gate compose 使用只读源码和唯一 sandbox
- [ ] model-gate 经真实 Provider wiring，支持 stream/tool/error/barrier
- [ ] PC-01–16 全部机器判定通过，G2+G3 连续三次绿色
- [ ] 事件、DB、退出码、清理和 repo digest 不变量全部纳入 gate
- [ ] reports 足以定位 request/thread/turn/model attempt
- [ ] G4 无 secret 时安全 skip，有 secret 时不泄漏
- [ ] G5 满足资源阈值，发布前可按 commit 查到结果

## STOP conditions

- 为了通过 gate 给 runtime 加测试专用成功路径、fake result 或静默 fallback。
- 必选场景依赖外网、真实模型、开发者现有 profile 或固定 sleep。
- `/app` 必须可写，或 controller 无法证明 sandbox 外未被修改。
- 失败只能靠人工读完整 compose log，无法关联 request/thread/turn。
- cleanup 后仍有本次 compose project 容器、socket、PID 或非终态 DB 记录。

## Maintenance notes

协议新增 method/event/error 时必须增加对应 Docker 场景或在 `gate.json` 明确声明不适用。时间和
资源阈值属于公开验收契约；修改时附基线证据。不要把 G4 外部模型波动混入本地确定性 gate。

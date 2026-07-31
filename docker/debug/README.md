# Docker 调试沙盒

## 统一变更影响 Gate

实现者只需运行一个公开入口：

```bash
python docker/debug/gate.py run --base origin/main
```

Gate 先用 `tests_scenarios/contracts/impact.toml` 解释 Git diff，再运行所选公开语义场景。每个场景都使用新的 `/tmp/akashic-change-gate-*` sandbox，容器只读挂载候选源码，只允许写本次 `/sandbox` 与 tmpfs `/tmp`。`workspace`、`plugin-home`、`HOME` 和 config 都从空目录建立；Gate 不接收正式运行路径。

```text
Git diff
   │
   ▼
公开 capability/state/scenario catalog
   │
   ├── G1：公开 Docker 场景（所有贡献者可运行）
   └── plan.json：group + digest，不含 provider 身份
                         │
                         ▼
               G2：private runtime 维护者验证
```

常用维护命令：

```bash
python docker/debug/gate.py audit
python docker/debug/gate.py plan --base origin/main
```

如果同一 diff 同时包含生产 source set 与 protected contract/policy paths，`plan` 和 `run` 都会报告 `status=protected_contract_mixed`、分别列出两组路径并以非零退出；仅合同/Gate，或生产源码加普通测试，仍按正常 Gate 处理。`migrations/**` 本身不在该 protected 集合内，继续由 append-only/repair Gate 管理。

`init` 只用于仓库第一次建立 coverage baseline。baseline 已存在时再次执行会失败，不能覆盖人工合同。新增未映射可执行文件会先运行全量公开语义场景，最终仍以 `unmapped_change` 失败。报告位于 `docker/debug/reports/change-gate/<run-id>/`。

公开 Gate 不安装也不枚举私有插件。`privateGateRequired=true` 表示公开部分已经给出受影响能力组，维护者还需用 private companion 消费同一 `planDigest`；普通贡献者不需要私有仓库、插件或凭据。

## 程序化控制面验收门

`programmatic_control_probe.py` 拥有独立 Compose project、隔离 sandbox、证据收集、
源码 digest 审计和强制 cleanup。`/app` 与 model-gate 源码均只读挂载，运行时只允许写
`/sandbox` 和 tmpfs `/tmp`。

```bash
python docker/debug/programmatic_control_probe.py --gate smoke
python docker/debug/programmatic_control_probe.py --gate failure-matrix
python docker/debug/programmatic_control_probe.py --gate soak
```

当前基建实现 `smoke`、PR 必选的 `failure-matrix` 和 nightly/release `soak`。`smoke`
覆盖 UDS/stdio、基本 turn，以及 streaming/tool/usage 的事件与 DB 一致性；
`failure-matrix` 覆盖双连接隔离、同 thread 冲突、精确中断、断线恢复、慢客户端背压、
provider 分类、非法协议、Web channel parity、workspace lock、SIGTERM 和 crash/restart。
`soak` 执行 10 次预热与 100 次混合 turn，包含 10 次 reconnect、interrupt 和 provider
failure，并检查 RSS、fd、线程与 DB 非终态阈值。
每次运行的证据位于
`docker/debug/reports/programmatic-control/<run-id>/`。

## Akasha V2 在线与重放等价 Gate

Akasha 专用 Gate 复用同一个只读 runtime 容器，但开启 `memory.engine = "akasha"`。
scripted model-gate 只控制模型回复和 `recall_memory` 工具选择；embedding 使用显式
`--source-config` 中的真实 provider。配置及凭据只进入权限为 `0600` 的唯一 `/tmp`
sandbox，运行结束后删除，不写日志和报告。

```bash
python docker/debug/akasha_v2_runtime_probe.py \
  --source-config /path/to/debug-config.toml \
  --formal-workspace /path/to/formal-workspace
```

Gate 完成两个真实 turn，检查第二轮 provider payload 已收到自动 Akasha 上下文，
在 final response barrier 处证明 `recall_memory` 前后逻辑状态不变，再证明第二轮提交
会改变状态。停止在线 gateway 后，它从同一隔离 `sessions.db` 严格重放，要求 online
与 replay canonical logical hash 相同。最后核对正式 `sessions.db`、正式 `akasha.db`
和仓库摘要未改变，Compose 无残留。证据位于
`docker/debug/reports/akasha-v2-runtime/<run-id>/`。

## 一次性迁移验收门

迁移专用 Gate 在与 runtime control Gate 相同的只读容器边界内运行完整 case matrix：

```bash
python docker/debug/migration_probe.py
```

它覆盖 fresh/legacy 分类、固定 baseline、快速路径、顺序执行、merge、纯代码提交、
blocked/apply/verify/cursor 写入失败与安全重试、分支分叉、shallow history、并发锁、
两类旧 provider 配置、显式恢复、Akasha 状态零读写和 append-only policy。每个 pytest
case 及源码不变性、Compose cleanup 结果记录在
`docker/debug/reports/migrations/<run-id>/gate.json`。新增 migration bundle 前先按
[迁移维护手册](../../docs/design/git-migration-authoring.md)补齐来源 lineage 和相应 case。

## Runtime 扩展生命周期验收门

workspace MCP 和 agent restart 共用 `docker-compose.control-gate.yml`，但每次运行都会创建
独立 Compose project 与 sandbox。Gate 复制当前 tracked 和 non-ignored untracked 源码，记录
Git HEAD、工作树状态、源码摘要、容器 `/app` 摘要与 image ID；运行结束后按 Compose project
label 检查容器、网络和卷均无残留。

```text
┌─ workspace MCP Gate
│  ├─ v1 → v2 → watched-content reload
│  ├─ 旧 turn lease 与新 generation 隔离
│  ├─ 坏声明整批 rollback、修复后自动恢复
│  └─ 删除声明后工具与真实 MCP 进程全部退出
└─ restart + MCP 组合 Gate
   ├─ tool_search → agent_restart → terminal delivery
   ├─ Supervisor 固定、每 boot 一个 Guardian、Gateway/boot ID 更换
   ├─ MCP 热更后跨进程重启恢复，旧 MCP PID 退出
   ├─ 裸 exit 75、stale readiness、断线、TERM 与 owner SIGKILL 矩阵
   └─ 20 轮 FD、线程、zombie 与非终态 turn 资源门禁
```

本地合并前运行：

```bash
python docker/debug/workspace_mcp_reload_probe.py
python docker/debug/restart_probe.py --soak
```

验收以两个 host `gate.json` 的 `status=passed` 为准，不能只使用容器内
`restart-gate.json`。两份报告必须来自同一 HEAD、相同 source digest；CI 还要求工作树为空。
owner 故障矩阵分别杀死 Supervisor 和 Guardian，要求剩余 owner 清空 Gateway、MCP 与
double-fork 后代；未知 PID 和不属于当前 boot 的端口不构成 kill 授权。
20 轮 soak 的阈值为 supervisor FD 增量不超过 2、线程增量为 0，child FD 增量不超过 4、
线程增量不超过 2；supervisor 与新 child 的 sampled RSS 增量及内核记录的 HWM 增量分别
不超过 64 MiB，并要求无 zombie、无 `queued/in_progress` turn。

确定性模型 sidecar 的控制协议：

- `PUT /control/script`：装载一个脚本对象或脚本数组。`mode` 支持 `complete`、
  `stream`、`error`、`timeout`、`truncate`；可提供 `content`、`deltas`、`tool_calls`、`usage`、
  `status` 和 `body`。
- `PUT /control/barriers/{name}`：创建 barrier。将 `"barrier":"{name}"` 放入脚本后，
  对应模型请求到达 provider sidecar 时会精确阻塞。
- `GET /control/barriers/{name}/wait?timeout=30`：服务端长等待请求到达，不靠 controller
  固定 sleep 猜竞态。
- `POST /control/barriers/{name}/release`：释放已到达的模型请求。
- `GET /control/requests`：读取完整 payload、关联 header、脚本和请求状态证据。

示例脚本：

```json
{
  "mode": "stream",
  "barrier": "turn-entered-provider",
  "deltas": ["hello ", "world"],
  "usage": {"prompt_tokens": 7, "completion_tokens": 2, "total_tokens": 9}
}
```

这个目录用于临时调试真实入口，例如 Telegram 图片、多模态链路、独立 bot 配置。调试容器基于 Arch Linux，沙盒不会挂载宿主机 `HOME`，也不会挂载正式 `~/.akashic/workspace`。

```
host
  |
  +-- akashic-agent
      |
      +-- docker/debug
          |
          +-- Dockerfile
          +-- docker-compose.yml
          +-- entrypoint.sh
          +-- profiles
              |
              +-- default
                  |
                  +-- config.toml
                  +-- workspace
                  +-- home
                  +-- akashic.sock

container
  |
  +-- /app                 -> 当前代码
  +-- /sandbox/config.toml -> 调试 bot 配置
  +-- /sandbox/workspace   -> 调试 workspace
  +-- /sandbox/home        -> 容器 HOME
```

## 安全边界

- 默认调试配置只在 `docker/debug/profiles/default/config.toml`。
- 默认调试 workspace 只在 `docker/debug/profiles/default/workspace`。
- 容器内 `HOME` 是 `/sandbox/home`，不是宿主机 HOME。
- 启动脚本会拒绝 `/sandbox` 外的 config/workspace 路径。
- `profiles/` 已加入 `.gitignore`，不要提交调试 bot token 和测试记忆。

## 插件变更 Gate

`akashic-plugin-gate` 用于在真实 Runtime 中验证插件系统改动。它不会复用普通调试容器的可写源码挂载或宿主插件缓存。

```text
┌─ 宿主
│  ├─ akasic-agent             只读挂载到 /app
│  └─ akashic-plugin/*         只读挂载到 /fixtures/plugins
├─ 容器
│  ├─ root filesystem          只读
│  ├─ /tmp                     tmpfs
│  └─ /sandbox                 唯一持久可写目录
│     ├─ home/.akashic-plugin/cache
│     ├─ workspace
│     └─ reports
└─ Compose project
   └─ akashic-plugin-reload-gate
```

从宿主运行完整性 Gate：

```bash
python docker/debug/plugin_hot_reload_probe.py --scenario sandbox-integrity
```

宿主控制器会在 `/tmp` 创建唯一 sandbox，拒绝仓库内路径，再通过独立的 `docker-compose.plugin-gate.yml` 启动容器。普通 `docker-compose.yml` 不受 Gate 环境变量影响。控制器会审计各 Git 仓库状态，并在隔离插件缓存中完成一次写入与更新；Runtime 需要的 `static/` 也覆盖到外部 sandbox，不会写 `/app`。

插件资源作用域场景会安装一次性测试插件，并在真实 `main.py` 的启动和关闭过程中验证订阅、任务与清理回调：

```bash
python docker/debug/plugin_hot_reload_probe.py \
  --scenario full-runtime --phase scope
```

原子热重载场景会构造失败源码、有效新代和回切旧代，确认校验期间旧代仍服务，提交后
request、skill、tool、job、event、MCP 与 service 同时换代，旧代 writer 被 fence，journal
最终完成：

```bash
python docker/debug/plugin_hot_reload_probe.py \
  --scenario full-runtime --phase atomic-reload
```

### Plugin API v2 发布组合 Gate

`plugin-api-v2.lock.json` 固定合同检查器与 21 个外部插件的完整 commit SHA。Gate 只从公开
GitHub HTTPS 地址获取这些对象，不读取宿主插件 cache、正式 workspace 或正式配置。

```text
┌─ 静态合同
│  ├─ 拒绝 API v1 / initialize
│  ├─ prepare 不得取得 data_dir 或启动 task
│  └─ lifecycle task 必须归 generation scope
└─ Docker Debug
   ├─ atomic-reload   失败保旧、成功原子切换、WAL 完成
   ├─ all-plugins     19 个可运行插件逐个热重载和禁用
   └─ fitbit          monitor 单实例、重载、停机、用户数据不变
```

本地运行：

```bash
python docker/debug/plugin_api_v2_gate.py
```

CI 使用 `--require-clean-core`，并上传 `docker/debug/reports/plugin-api-v2/`。任何锁内仓库缺失、
SHA 不可获取、静态合同失败、容器退出异常、源码挂载被修改或业务 oracle 不成立都会返回非零
退出码。

### 移动插件发布组合 Gate

`mobile-plugin-release.lock.json` 固定核心仓库这次发布实际配套的公开插件提交。协作者不需要
安装插件，也不会读取宿主的 `~/.akashic-plugin`：Gate 在 `/tmp` 创建全新 Git checkout，
从锁内的 GitHub HTTPS 地址只取精确 SHA。

```text
┌─ 移动端异步 query
│  └─ 核心 PluginMobileUiProvider
│     └─ ThreadPoolExecutor
│        └─ 插件同步 mobile_ui_query
├─ 核心拥有的 JS ABI runner
│  ├─ default export / dashboard / slots / mount
│  └─ catalog navigation 与插件导出一致
└─ 插件仓库自己的 UI 行为测试
   └─ 固定方法、交互、可访问性和错误状态
```

本地运行：

```bash
python docker/debug/mobile_plugin_contract_gate.py
```

CI 额外使用 `--require-clean-core`，防止报告对应的不是可复核源码。Gate 会验证核心仍是
“异步 provider 在线程池调度同步插件 handler”，并检查每个插件的 Python 签名、方法集合、
资源预算、JS 导出形态与仓库自带测试。Fitbit 的移动查询是插件同步调用其托管 monitor 的
本地 HTTP 投影；Fitbit MCP 与 skills 仍由插件安装，不会被错误地解释成 UI query 本身。

远端默认分支更新不会自动否定旧报告；只有主动修改发布锁，或修改核心合同后重新运行，
才产生新的发布组合。这保证一次通过对应一组不可变源码，同时避免协作者被本地插件状态影响。

## 第一次配置

```bash
docker compose -f docker/debug/docker-compose.yml run --rm akashic-debug setup
```

这里填写专用 Telegram bot、模型 key 和多模态配置。不要填正式 bot。

## 启动调试 Agent

```bash
docker compose -f docker/debug/docker-compose.yml up akashic-debug
```

此时向调试 Telegram bot 发消息或图片，所有会话和记忆都会进入 `docker/debug/profiles/default/workspace`。

调试容器通过固定 Supervisor 启动每个 boot 唯一的 Guardian，再由 Guardian 启动 Gateway。
Supervisor 只会在当前 boot 已通过私有事件 ready、`agent_restart` 的最终回复已经实际送达、
Gateway 提交一次匹配证据、以 75 退出且 Guardian 证明旧 boot 已空时拉起下一代。普通退出、
崩溃、伪造 75、断线和送达超时都不会触发重启。

本机若仍由忽略版本控制的 `start.sh` 启动，应让它调用正式默认入口：

```bash
python main.py --config /absolute/config.toml --workspace /absolute/workspace
```

`supervise` 子命令在 Linux 保留为兼容别名。只有需要让调试器直接附着未托管 child 时才显式使用
`python main.py gateway`；该入口不注册 `agent_restart`。

不要在外层脚本再做 `while`、`pgrep` 或“任意非零退出就重启”；进程唯一性、信号转发、
重启授权和 boot 代际由 Supervisor 持有，Guardian 只拥有当前 boot 的进程清理。

## 多套调试配置

不同功能可以用不同 profile 保存配置和 workspace：

```bash
AKASHIC_DEBUG_PROFILE=multimodal docker compose -f docker/debug/docker-compose.yml run --rm akashic-debug setup
AKASHIC_DEBUG_PROFILE=multimodal docker compose -f docker/debug/docker-compose.yml up akashic-debug
```

对应目录是 `docker/debug/profiles/multimodal/`。

## 调用调试实例

```bash
docker compose -f docker/debug/docker-compose.yml run --rm akashic-debug exec --new "测试消息"
```

app-server socket 固定为 `/sandbox/akashic.sock`，不会连接正式实例。

## 打开调试 Dashboard

```bash
docker compose -f docker/debug/docker-compose.yml run --rm --service-ports akashic-debug dashboard
```

宿主机访问 `http://127.0.0.1:2237`。

## 停止调试环境

```bash
docker compose -f docker/debug/docker-compose.yml down
```

这只会停止容器，不会删除当前 profile 目录。

## 清空调试 workspace

```bash
docker compose -f docker/debug/docker-compose.yml run --rm akashic-debug reset-workspace
```

这个命令只删除并重建当前 profile 下的 `workspace`，会保留当前 profile 下的 `config.toml`。

## 上下文连续性探针

`context_probe.py` 用于复现一段固定纯聊天场景，自动记录用户输入、LLM 回复、工具调用、`RECENT_CONTEXT.md` 和 `memory2.db` 写入结果。

```
context probe
  |
  +-- profile
  |     |
  |     +-- config.toml
  |     +-- workspace
  |
  +-- phase1 chat
  |
  +-- manual consolidate
  |
  +-- phase2 chat
  |
  +-- final question
        |
        +-- markdown report
        +-- json report
```

从已启动的沙盒运行：

```bash
python docker/debug/context_probe.py \
  --profile default \
  --messages docker/debug/scenarios/sleepy_study_plan.json
```

自动重置、启动、运行并停止：

```bash
python docker/debug/context_probe.py \
  --profile v4flash-memory-window \
  --messages docker/debug/scenarios/sleepy_study_plan.json \
  --reset-workspace \
  --start-agent \
  --stop-agent \
  --quiet-agent \
  --disable-qq
```

`--disable-qq` 会在运行期间临时给当前 profile 的 `[channels.qq]` 加 `enabled = false`，结束后恢复原配置，适合只测 CLI 但该 profile 配了 QQ 的情况。

默认报告写到：

```text
docker/debug/profiles/<profile>/workspace/context-probe-<profile>.md
docker/debug/profiles/<profile>/workspace/context-probe-<profile>.json
```

自定义场景 JSON 格式：

```json
{
  "name": "sleepy_study_plan",
  "turns": [
    {
      "role": "user",
      "content": "前置闲聊"
    },
    {
      "action": "consolidate",
      "label": "after_signal",
      "force": false,
      "archive_all": false
    },
    {
      "role": "user",
      "content": "consolidate 后的杂音"
    },
    {
      "role": "user",
      "content": "最后问题",
      "final": true
    }
  ]
}
```

场景 JSON 只描述输入和流程，不写语义结果要求。探针遇到主流程的通用失败回复时会立即失败，正常回复则只记录 observe 结果，不主观判断内容质量。

内置样例在：

```text
docker/debug/scenarios/sleepy_study_plan.json
```

公开场景和 schema 都放在：

```text
docker/debug/scenarios/
```

这里的文件是稳定输入，可以提交；`docker/debug/profiles/<profile>/workspace/` 里的报告 JSON / Markdown 是运行产物，默认不提交。

兼容旧格式：

```json
{
  "phase1": ["第一段闲聊"],
  "phase2": ["consolidate 后的杂音"],
  "final_question": "最后问题"
}
```

## Runtime 竞态探针

`runtime_race_probe.py` 用于在 Docker 沙盒里制造 passive / scheduler / proactive / drift 的可见发送竞态。它复用真实 `MessageBus`、`ChatLane`、`BusOutboundPort`、`PushToolOutboundPort` 和 `message_push`，但 channel sender 和 LLM 都是 fake，所以不需要调试 bot 或模型 key。

```text
┌─────────────────────────────────────────────────────────────┐
│ runtime_race_probe.py                                       │
└──────────────┬──────────────────────────────────────────────┘
               │ fake user inbound
               v
┌─────────────────────────────────────────────────────────────┐
│ MessageBus + ChatLane                                       │
└──────┬──────────────────────────────────────────────┬───────┘
       │ passive reply                                │ non-passive send
       v                                              v
┌──────────────────────┐                     ┌──────────────────────┐
│ BusOutboundPort      │                     │ PushToolOutboundPort │
└──────────┬───────────┘                     └──────────┬───────────┘
           │                                            │
           v                                            v
┌─────────────────────────────────────────────────────────────┐
│ fake sender records start/end order                         │
└─────────────────────────────────────────────────────────────┘
```

运行全部场景：

```bash
docker compose -f docker/debug/docker-compose.yml run --rm akashic-debug \
  python docker/debug/runtime_race_probe.py --scenario all
```

运行单个场景：

```bash
docker compose -f docker/debug/docker-compose.yml run --rm akashic-debug \
  python docker/debug/runtime_race_probe.py --scenario a1-drift-before-push
```

可用控制开关：

```text
AKASHIC_RACE_SCENARIO  选择单个场景，默认 all
AKASHIC_RACE_TIMEOUT   每个等待点的超时秒数，默认 2
AKASHIC_RACE_TRACE     写出 JSON 结果的路径
AKASHIC_RACE_CONFIG    指定 config.toml；不指定时生成无外部 channel 的最小配置
AKASHIC_RACE_WORKSPACE 指定临时 workspace；不指定时使用临时目录
```

## 主动链路操作沙盒

`proactive_sandbox.py` 使用隔离 workspace、真实插件加载器、Slot Lifecycle、Feed MCP 子进程和消息发送编排器。模型使用可预测驱动器，测试失败时可以排除模型随机性。

同一脚本通过 `--lifecycle default` 和 `--lifecycle wake` 分别验证两套主动生命周期。Wake 验证会走真实正文抓取、消息提交与 Feed ACK，不把消费事件误记为兴趣反馈。

```text
┌─ operator
│  ├─ inject-content ──> feed_mcp.sqlite3
│  ├─ clear-content ───> empty gateway
│  ├─ tick-content
│  ├─ tick-drift
│  └─ status
│
└─ Docker sandbox
   ├─ PluginManager ──> runtime/module/lifecycle factories
   ├─ ToolRegistry  ──> Feed MCP stdio process
   ├─ ProactiveLoop ──> compiled Slot Graph
   └─ state
      ├─ proactive.db
      ├─ sessions.db
      ├─ plugin-data/feed-github/feed_mcp.sqlite3
      └─ drift/drift.db
```

完整验证：

```bash
docker compose -f docker/debug/docker-compose.yml build akashic-debug
docker compose -f docker/debug/docker-compose.yml run --rm akashic-debug \
  python docker/debug/proactive_sandbox.py run-all
```

Wake package 已启用的 profile 使用：

```bash
AKASHIC_DEBUG_PROFILE=wake-profile \
  docker compose -f docker/debug/docker-compose.yml run --rm akashic-debug \
  python docker/debug/proactive_sandbox.py run-all --lifecycle wake
```

使用当前 profile 的真实模型配置：

```bash
AKASHIC_DEBUG_PROFILE=dev_verify \
docker compose -f docker/debug/docker-compose.yml run --rm akashic-debug \
  python docker/debug/proactive_sandbox.py run-all --config /sandbox/config.toml
```

手动控制：

```bash
docker compose -f docker/debug/docker-compose.yml run --rm akashic-debug \
  python docker/debug/proactive_sandbox.py reset
docker compose -f docker/debug/docker-compose.yml run --rm akashic-debug \
  python docker/debug/proactive_sandbox.py inject-content
docker compose -f docker/debug/docker-compose.yml run --rm akashic-debug \
  python docker/debug/proactive_sandbox.py tick-content
docker compose -f docker/debug/docker-compose.yml run --rm akashic-debug \
  python docker/debug/proactive_sandbox.py status
```

验证 paused skill 能从已有计划的停点继续，而不是重新执行说明书前置步骤：

```bash
AKASHIC_DEBUG_PROFILE=drift-current-runtime \
docker compose -f docker/debug/docker-compose.yml run --rm akashic-debug \
  python docker/debug/proactive_sandbox.py verify-paused-resume \
  --config /sandbox/config.toml
```

探针会预置已读取需求、已生成 `plan.json`、执行阶段遇到临时 502 的状态。验证要求模型直接使用已有计划写出结果；若重新读取需求或重写计划则失败。

## 真实 Runtime 时间回放基础

`replay_controller.py` 只维护隔离 profile 下的模拟时钟、历史事件和捕获消息，不读取或挂载正式 workspace。`docker-compose.yml` 会让真实 `main.py` 加载调试插件目录；`replay_debug` 插件注册 `replay` 渠道，把 outbound 原样写入 profile。

```text
┌─ replay_controller
│  ├─ clock.json             模拟当前时间
│  ├─ events.jsonl           历史事件输入
│  └─ outbox.jsonl           捕获的 outbound
│
└─ Docker profile
   ├─ python main.py         正式启动入口
   ├─ SystemClock            线上默认时钟
   ├─ ReplayClock            调试文件时钟
   └─ CaptureChannel         channel = replay
```

初始化独立 profile 的回放状态：

```bash
python docker/debug/replay_controller.py \
  --profile wake-replay init \
  --start-at 2026-05-01T00:00:00+08:00
```

该 profile 仍需要自己的 `config.toml`。可以运行 `setup`，或复制另一份专用调试配置。启动前应关闭 Telegram、QQ 等外部渠道，并将待测发送目标设为 `channel = "replay"`。

```bash
AKASHIC_DEBUG_PROFILE=wake-replay \
docker compose -f docker/debug/docker-compose.yml up akashic-debug
```

注入单条历史事件：

```bash
python docker/debug/replay_controller.py --profile wake-replay inject \
  --event-id feed-001 \
  --kind content \
  --source-id rss-example \
  --title "历史候选标题" \
  --content "历史候选摘要" \
  --published-at 2026-05-01T08:30:00+08:00
```

批量输入支持 JSON 数组、`{"events": [...]}` 或 JSONL：

```bash
python docker/debug/replay_controller.py \
  --profile wake-replay import-events /path/to/history.jsonl
```

推进时间并查看当前可见事件和捕获结果：

```bash
python docker/debug/replay_controller.py \
  --profile wake-replay advance --seconds 3600
python docker/debug/replay_controller.py \
  --profile wake-replay status
```

`events.jsonl` 是供后续 `plugins/wake_proactive` 消费的稳定输入面；当前旧 proactive 不读取它，也不会因为推进时钟自动触发。

`agent-loop-runtime` 场景会启动真实 `AgentLoop.run()`，读取 `config.toml`，但不启动 Telegram / QQ / CLI server。它用 fake reasoner 卡住 passive turn，再并发触发 drift 发送和 scheduler soft 的 `process_direct`，验证 runtime lock 与 ChatLane 的联动。

```text
┌─────────────────────────────────────────────────────────────┐
│ config.toml without external channel                         │
└──────────────┬──────────────────────────────────────────────┘
               v
┌─────────────────────────────────────────────────────────────┐
│ real AgentLoop.run                                           │
│ real CoreRunner + AgentCore + MessageBus + ChatLane           │
└──────────────┬──────────────────────────────────────────────┘
               v
┌─────────────────────────────────────────────────────────────┐
│ assert passive reply -> drift send -> scheduler send          │
│ assert scheduler soft waits passive runtime lock              │
└─────────────────────────────────────────────────────────────┘
```

`config-runtime-llm` 场景会读取真实 `config.toml` 并调用其中配置的 LLM。它通过 `build_core_runtime()` 构建真实 runtime，加载真实 provider、memory、tool、plugin、scheduler 接线，但不启动 Telegram / QQ / CLI server；外部 channel sender 用 fake 记录发送顺序，proactive / drift 生成也用 fake 直接提交到 `message_push(_commit_role="non_passive")`。

```bash
docker compose -f docker/debug/docker-compose.yml run --rm akashic-debug \
  python docker/debug/runtime_race_probe.py \
    --scenario config-runtime-llm \
    --config config.toml \
    --timeout 120
```

```text
┌─────────────────────────────────────────────────────────────┐
│ real config.toml + real LLM                                  │
└──────────────┬──────────────────────────────────────────────┘
               v
┌─────────────────────────────────────────────────────────────┐
│ build_core_runtime                                           │
│ provider + memory + tools + plugins + scheduler              │
└──────────────┬──────────────────────────────────────────────┘
               v
┌─────────────────────────────────────────────────────────────┐
│ real AgentLoop.run + real process_direct                     │
│ fake proactive/drift generation -> real message_push          │
│ fake channel sender records order                            │
└─────────────────────────────────────────────────────────────┘
```

## 完全清理

```bash
docker compose -f docker/debug/docker-compose.yml down --remove-orphans
rm -rf docker/debug/profiles/default
```

完全清理后，下次需要重新运行 `setup`。

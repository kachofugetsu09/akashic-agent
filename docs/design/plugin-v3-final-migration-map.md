# 插件 v3 最终迁移地图

本文记录 Issue [#394](https://github.com/kachofugetsu09/akashic-agent/issues/394)
这一轮 Cordis 风格插件改造的目标结构、当前实现栈、剩余迁移范围和 v2 物理删除顺序。
它是 2026-08-16 的实施接手点，不替代
[Cordis 插件迁移能力等价验收](cordis-plugin-capability-parity.md)中的长期能力合同。
逐项执行状态、生产替代门槛和集中 E2E 批次由
[插件 v3 生产替代清单](plugin-v3-production-readiness-checklist.md)唯一记录；本文不承担进度账本。

## 1. 结论

最终目标不是让 v2/v3 长期共存，也不是把 DeepSeek Harness 原样搬进 Python。Akashic 保留
candidate 自验证、promotion、stable snapshot lease 与旧 generation drain；插件装配收敛为
Cordis 风格的 `Context / Service / Fiber / Effect / typed event`。Core 提供窄接入点和能力
provider，插件自己决定内部 client、任务、重试、数据格式与组件拆分。

本文只保留目标架构、最初 PR DAG 与删除顺序。当前目标 fleet、实现状态、exact heads 和
最终 Gate 统一由[生产替代清单](plugin-v3-production-readiness-checklist.md)记录；不得从本文的
2026-08-16 历史盘点推导当前完成比例或上线状态。

## 2. 最终效果图

### 2.1 被动链路

```text
┌─────────────────────────────────────────────────────────────────────────┐
│ Inbound                                                                 │
│ WebUI / Channel adapter ──► Message ──► stable RuntimeSnapshot lease     │
└──────────────────────────────────┬──────────────────────────────────────┘
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ React                                                                   │
│                                                                         │
│  before turn                                                            │
│    ├─ turn.context_prepared ── serial listeners                         │
│    └─ turn.prompt_render ────── serial listeners                         │
│                         │                                               │
│                         ▼                                               │
│  reason / tools                                                         │
│    ├─ tool.input.prepare ───── transform chain                          │
│    ├─ tool.execution.authorize ─ serial allow/deny                       │
│    ├─ invoker ──────────────── execute exactly once                      │
│    └─ tool.result ───────────── final observers                          │
│                         │                                               │
│                         ▼                                               │
│  after reasoning                                                        │
│    ├─ turn.after_reasoning.preprocess ─ serial listeners                 │
│    ├─ Core seals plugin assistant metadata                              │
│    └─ SessionStore commits user + assistant + metadata atomically        │
│                         │                                               │
│                         ▼                                               │
│  after turn ── Core TurnCommitted fanout / lifecycle observers          │
└──────────────────────────────────┬──────────────────────────────────────┘
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ Outbound                                                                │
│ committed Message ──► WebUI / selected Channel                          │
└─────────────────────────────────────────────────────────────────────────┘
```

生命周期仍由 Core 的 Turn 管线拥有。插件不接管管线，也不向 `PluginManager` 追加新的固定
`before_*()/after_*()/jobs()/channels()` 方法；插件只在 typed event 或 service seam 上注册
listener/effect。事件的依赖与 listener 顺序进入 topology，运行时任务状态进入 Health，历史
失败进入 Incident，三者不再混成一个永久 `ready=false`。

### 2.2 组合平面与发布平面

```text
                              publication plane
 source/config change
          │
          ▼
 isolated candidate Root ──► settle ──► validate/health/incidents
          │                                      │
          │ fail                                 │ pass
          ▼                                      ▼
 discard + zero residue                   promotion commit
                                                 │
                                                 ▼
                                      stable RuntimeSnapshot
                                                 │ lease
                                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ composition plane: one stable CompositionRoot                          │
│                                                                         │
│ Core providers                                                          │
│  ├─ plugin data/workspace roots  ├─ memory runtime                      │
│  ├─ tool executor                └─ future narrow MCP/process/channel   │
│                                                                         │
│ plugin Fibers                                                           │
│  ├─ Citation ── provides citation.protocol                              │
│  ├─ Meme ────── requires citation.protocol                              │
│  ├─ Shell Restore ── tool.input.prepare                                 │
│  ├─ Shell Safety ─── tool.execution.authorize                           │
│  ├─ Tool Loop Guard ─ tool.execution.authorize                          │
│  └─ Default Memory ─ context/result listeners + Dashboard               │
│                                                                         │
│ Fiber listener/service/task registrations are Effects.                  │
│ Dashboard/Skill projections follow generation scope and lease drain.    │
└─────────────────────────────────────────────────────────────────────────┘
```

发布平面继续体现 Akashic 相对 DSH 的优势：candidate 不能复用 stable Root，验证只读取
candidate 自己的 Health/Incident/Topology，promotion 后才取得正式 data/workspace。v2-only
过渡 candidate 也使用隔离 Root；stable-to-stable 复用必须证明 v3 composition input 未变。

## 3. v3 插件与 Core 的职责

一个最终 v3 插件只有一个模块级入口：

```python
api_version = 3
name = "example"
version = "1.0.0"
inject = (SOME_SERVICE,)
skill_roots = ("skills",)
workspace_roots = ("assets",)
dashboard_module = "dashboard.py"


async def apply(ctx: Context, config: Config) -> None:
    service = ctx.require(SOME_SERVICE)
    _ = await ctx.on(SOME_EVENT, handle_event)
    _ = await ctx.spawn(run_worker(service, config), name="worker")
```

`apply()` 可以同步或异步。插件可以 mount 子 Fiber、保存自己的 client、选择 retry/backoff、
声明 Health、报告 Incident，并通过 Effect 管理资源。Core 不规定插件一定使用哪一种 job、HTTP
client 或数据库实现。边界如下：

| Core 提供 | 插件自己实现 |
|---|---|
| stable/candidate Root、generation、promotion、lease/drain | client、领域 DTO、缓存、重试与业务状态 |
| typed event 的 serial/parallel/transform/observe dispatch | 选择监听事件、依赖 Service 与 listener 顺序 |
| `ctx.data_root`、声明式 `workspace_roots`、candidate 隔离 | plugin-data 内部 schema 与文件布局 |
| Service/Fiber/Effect、Health/Incident、任务回收 | capability provider、子 Fiber 和后台 worker |
| Skill/Drift skill/Dashboard 的 generation 投影 | Skill 内容、Dashboard route/panel 与 closeable |
| 外部能力的窄 provider seam | 具体 GitHub/MCP/channel/process 行为 |

Core 是受支持 API 的 owner，不是 Python 安全沙箱。同 UID 的恶意插件仍能绕过普通对象边界；
真正的安全隔离需要独立进程或权限域，不在本迁移中伪装实现。

## 4. 当前 external plugin source 索引

以下路径是 2026-09-01 按 hua-home stable artifact、安装 manifest 和工作站 canonical checkout
重新对账的源码入口。正式运行身份仍由 immutable artifact commit 与 stable pointer 证明；本表只拥有
“去哪里审查和修改源码”，不把本地 checkout 或浮动分支当作部署证据。

| 插件 | canonical repository | 工作站 checkout |
|---|---|---|
| Calendar | [akashic-plugins/calendar-mcp](https://github.com/akashic-plugins/calendar-mcp) | `/mnt/data/coding/akashic-plugin/calendar-mcp` |
| Citation | [akashic-plugins/citation](https://github.com/akashic-plugins/citation) | `/mnt/data/coding/akashic-plugin/citation` |
| Emotion | [akashic-plugins/emotion](https://github.com/akashic-plugins/emotion) | `/mnt/data/coding/akashic-plugin/emotion` |
| Feed | [akashic-plugins/feed-mcp](https://github.com/akashic-plugins/feed-mcp) | `/mnt/data/coding/akashic-plugin/feed-mcp` |
| Fitbit | [akashic-plugins/fitbit-mcp](https://github.com/akashic-plugins/fitbit-mcp) | `/mnt/data/coding/akashic-plugin/fitbit-mcp` |
| GitHub Watch | [kachofugetsu09/github-watch](https://github.com/kachofugetsu09/github-watch) | `/mnt/data/coding/akashic-plugin/github-watch` |
| Huayue Skills | [akashic-plugins/huayue-skills](https://github.com/akashic-plugins/huayue-skills) | `/mnt/data/coding/akashic-plugin/huayue-skills` |
| Meme | [akashic-plugins/meme](https://github.com/akashic-plugins/meme) | `/mnt/data/coding/akashic-plugin/meme` |
| Observe | [akashic-plugins/observe](https://github.com/akashic-plugins/observe) | `/mnt/data/coding/akashic-plugin/observe` |
| Plugin Undo | [akashic-plugins/plugin_undo](https://github.com/akashic-plugins/plugin_undo) | `/mnt/data/coding/akashic-plugin/plugin_undo` |
| Proactive Feedback | [akashic-plugins/proactive_feedback](https://github.com/akashic-plugins/proactive_feedback) | `/mnt/data/coding/akashic-plugin/proactive_feedback` |
| Setup Helper | [akashic-plugins/setup_helper](https://github.com/akashic-plugins/setup_helper) | `/mnt/data/coding/akashic-plugin/setup_helper` |
| Shell Restore | [akashic-plugins/shell_restore](https://github.com/akashic-plugins/shell_restore) | `/mnt/data/coding/akashic-plugin/shell_restore` |
| Shell Safety | [akashic-plugins/shell_safety](https://github.com/akashic-plugins/shell_safety) | `/mnt/data/coding/akashic-plugin/shell_safety` |
| Status Commands | [akashic-plugins/status_commands](https://github.com/akashic-plugins/status_commands) | `/mnt/data/coding/akashic-plugin/status_commands` |
| Steam | [akashic-plugins/steam-mcp](https://github.com/akashic-plugins/steam-mcp) | `/mnt/data/coding/akashic-plugin/steam-mcp` |

Feishu 与 QQBot 不在上述 hua-home enabled manifest 中，但它们是 PLG-016 指定的纯 v3 channel
consumer，分别由 [akashic-plugins/feishu](https://github.com/akashic-plugins/feishu) 和
[akashic-plugins/qqbot](https://github.com/akashic-plugins/qqbot) 拥有。本轮 Core surface 删除必须同时
运行这两个仓库的固定 commit 合同测试。

## 5. 2026-08-16 试点盘点（历史）

本节只解释最初 Tool/Passive 试点怎样形成后续架构，不是当前进度账本。当前目标已经收敛为
fleet lock 中 20 个 external 插件与 8 个 in-tree 实现；Computer Use Linux、Context Pressure
已退出目标，GitHub Watcher 已进入 fleet。逐插件状态只读生产替代清单。

仓库当时跟踪 29 个插件实现：21 个 external lock 插件加 8 个 in-tree 插件。当时已有 6 个纯 v3
候选，占 `6/29 = 20.7%`；其余 23 个尚待迁移。数字只表示“存在已评审候选”，不表示 PR 已合并或
运行环境已切换。

| 维度 | 当时状态 |
|---|---|
| Tool/被动回复试点 | 6/6 候选已实现并有行为 Gate |
| 全插件实现 | 6/29 有纯 v3 候选；23 个待迁移 |
| Core 试点底座 | foundation #395/#397～#401 与 remediation #425～#441 已形成可审阅栈；尚未合并 |
| v2 物理删除 | 尚未开始；最后一个真实 consumer 迁走后才删除 |
| 全量 v3-only runtime | 未完成；MCP/process/channel/proactive 等族群仍需迁移 |

已完成候选：

| 插件 | v3 能力 | 证据 |
|---|---|---|
| Citation | prompt protocol、assistant metadata | [Citation #3](https://github.com/akashic-plugins/citation/pull/3) |
| Meme | required `citation.protocol`、prompt/media、Skill、Dashboard | [Meme #3](https://github.com/akashic-plugins/meme/pull/3) |
| Shell Restore | `tool.input.prepare` 串行变换 | [Shell Restore #3](https://github.com/akashic-plugins/shell_restore/pull/3) |
| Shell Safety | `tool.execution.authorize` | [Shell Safety #2](https://github.com/akashic-plugins/shell_safety/pull/2) |
| Tool Loop Guard | typed authorization 与 per-generation state | [Tool Loop Guard #2](https://github.com/akashic-plugins/tool_loop_guard/pull/2) |
| Default Memory | static activation、Memory capability、result observer、Dashboard | [Core #437](https://github.com/kachofugetsu09/akashic-agent/pull/437) |

### 5.1 Core 实现栈

本轮 remediation 并非直接从 `main` 起步。它依赖已经单独评审的组合 foundation：

```text
#395 → #397 → #398 → #399 → #400 → #401
```

#396 是另一条独立修复，不在这条 parent chain；#402～#424 也从 #401 分出，是既有 capability
lane，不是 #425 的父链。下面只列本轮 remediation 与试点 Gate：

| PR | 目的 |
|---|---|
| [#425](https://github.com/kachofugetsu09/akashic-agent/pull/425) | stable 批量原子组装 |
| [#426](https://github.com/kachofugetsu09/akashic-agent/pull/426) | candidate Root、workspace 与 data 隔离 |
| [#427](https://github.com/kachofugetsu09/akashic-agent/pull/427) | immutable topology identity 与 composition revision |
| [#428](https://github.com/kachofugetsu09/akashic-agent/pull/428) | Validation / Health / Incident 分离 |
| [#429](https://github.com/kachofugetsu09/akashic-agent/pull/429) | transform / observe dispatch |
| [#430](https://github.com/kachofugetsu09/akashic-agent/pull/430) | generation-scoped data root |
| [#431](https://github.com/kachofugetsu09/akashic-agent/pull/431) | typed Tool 六段执行链与 v2 删除标记 |
| [#432](https://github.com/kachofugetsu09/akashic-agent/pull/432) | Shell 插件跨仓 exact-commit Gate |
| [#433](https://github.com/kachofugetsu09/akashic-agent/pull/433) | 包级 Skill/Drift skill/Dashboard 声明 |
| [#434](https://github.com/kachofugetsu09/akashic-agent/pull/434) | prepared context 与 Memory capability |
| [#435](https://github.com/kachofugetsu09/akashic-agent/pull/435) | 窄 DashboardContext 与 candidate binding |
| [#436](https://github.com/kachofugetsu09/akashic-agent/pull/436) | static projection 与 exact Root runtime |
| [#437](https://github.com/kachofugetsu09/akashic-agent/pull/437) | Default Memory v3 迁移 |
| [#438](https://github.com/kachofugetsu09/akashic-agent/pull/438) | 被动回复 seam 与 metadata 原子提交 |
| [#439](https://github.com/kachofugetsu09/akashic-agent/pull/439) | Citation/Meme 组合 Gate 与完整 CI |
| [#440](https://github.com/kachofugetsu09/akashic-agent/pull/440) | immutable Dashboard artifact 派生缓存 |
| [#441](https://github.com/kachofugetsu09/akashic-agent/pull/441) | WebUI-only 真实 Docker E2E |

真实依赖从 foundation #401 接入，并在 #431 后再次分叉，不是 #425～#441 的单链：

```text
#395 → #397 → #398 → #399 → #400 → #401
                                      │
                                      ▼
#425 → #426 → #427 → #428 → #429 → #430 → #431
                                                ├─► #432  Tool 跨仓 Gate
                                                └─► #433 → #434 → #435 → #436
                                                             → #437 → #438
                                                             → #439 → #440 → #441
```

PR 只按图中的依赖边合并。#432 在 #431 后可以独立 review/merge，不是 #433 或被动回复分支的
前置；栈顶 Gate 通过也不能解释成底层 PR 可以乱序 cherry-pick。

## 6. 2026-08-16 原始迁移盘点（历史）

本节保留当时的 consumer 调查，不能作为当前待办或 target fleet。现行迁移账本、排除项与
private proactive 边界以生产替代清单为准。

### 6.1 External lock 插件：16 个

| 族群 | 插件 | 迁移时需要的首要 v3 seam |
|---|---|---|
| lifecycle | `context_pressure` | 现有 typed lifecycle；缺 seam 时由第一个真实 consumer 建立 |
| proactive/job/mobile | `daynight_gate`、`emotion` | timer/proactive source、generation job/LLM、mobile query；不复制旧固定方法 |
| command/lifecycle/mobile | `plugin_undo`、`setup_helper`、`status_commands` | committed channel command catalog + typed lifecycle；一次声明后由 channel 投影 |
| Dashboard/mobile/event | `observe`、`proactive_feedback` | committed event observer、Dashboard 和窄 mobile query capability |
| MCP/process/proactive | `calendar-mcp`、`feed-mcp`、`fitbit-mcp`、`steam-mcp` | scoped MCP/process provider、readiness、Effect cleanup、proactive source |
| Skill/MCP | `computer-use-linux` | 包级 Skill 与 scoped MCP provider |
| channel | `feishu`、`qqbot` | inbound/outbound channel capability 与发送提交边界 |
| Skill | `huayue-skills` | 只迁移包级 `skill_roots`，不引入不存在的 MCP/channel seam |

### 6.2 In-tree 插件：7 个

`akasha`、`default_proactive`、`drift_flow`、`proactive_flow`、`wake_drift_flow`、
`wake_proactive`、`wake_proactive_flow` 当时尚未迁移。它们涉及 memory/proactive/wake 生命周期，
应在 external lifecycle 与 process seam 稳定后迁移，避免为旧领域方法复制一套 v3 Manager。

### 6.3 GitHub Watcher

当时公开 lock、`akashic-plugins` 组织和本轮可访问的 canonical source 中没有可锁定的
GitHub Watcher。因此它不计入 29 个实现，也不能声称已经迁移。后续必须先定位 canonical
repository、确认凭证边界、公开性与真实 installed artifact，再加入 exact-commit Gate。
插件继续拥有自己的 GitHub client；Core 只应提供接入 loop、data root、Health/Incident 和
可选 process/timer capability，不预建 GitHub 领域 Service。

## 6. 2026-08-16 原始底座缺口（历史）

下列条目记录第一批 consumer 如何驱动 Core seam；现行 C01～C25 能力状态与验收证据只在
生产替代清单维护。

下面的 seam 只在迁移第一个真实 consumer 时建立，不提前复制 v2 固定目录：

1. **MCP capability**：插件用 scoped provider 声明 server/runtime，Core 拥有进程启动、
   readiness、candidate 隔离与 generation drain；删除 `Plugin.mcp_servers()` 固定收集。
2. **Managed process capability**：用 Effect 管理启动、停止、取消和日志，不再由
   `PluginManager` 识别 `managed_services()` 领域对象。
3. **Channel command catalog**：插件一次注册 canonical name、aliases、description 与对应
   lifecycle handler；Core 拥有 collision、snapshot identity 和 committed publication，candidate
   catalog 不向 channel 暴露。Telegram/mobile/WebUI host 只消费 stable catalog。它与 agent
   ToolExecutor 的 tool catalog 是两项能力。完成后删除 `PluginManager`、`bootstrap/app.py` 与
   `bootstrap/channels.py` 的 `telegram_bot_commands()/mobile_bot_commands()` 聚合路径。
4. **Channel capability**：Core 提供 inbound Message 与 committed outbound 发送边界；插件
   注册 channel adapter，不把 channel 业务字段加入通用 `PluginContext`。
5. **Timer/proactive capability**：Core 提供 timer/clock/turn enqueue seam，插件自己实现
   调度逻辑；不把 `jobs()`、`proactive_*()` 原样翻译成 v3 namespace 方法。
6. **Mobile UI/query capability**：把移动投影建成窄 typed capability，与 Dashboard 类似；
   不保留 `mobile_ui()/mobile_query()` 的 Manager 特判。
7. **Generation job/LLM capability**：Core 拥有 committed trigger/interval catalog、执行期模型
   generation lease 和取消/drain；插件只实现 job handler，不取得整个 legacy `PluginContext`。
8. **v3 generation metadata**：把 `ComposablePlugin` 暂借的 `PluginContext` 数据迁入
   Core-private v3 generation record，才可删除整个 v2 context DTO。
9. **Full-fleet inspection/Gate**：Runtime Inspection 展示 Fiber/Health/Incident；全量 Gate
   覆盖 cold boot、reload、candidate discard/promote、restart、WebUI/channel、MCP/process、
   proactive 与 cleanup。

## 7. v2 物理删除清单

源码中的 `V2_REMOVAL(...)` 是过渡标记，不是永久兼容承诺。删除必须按“最后一个 consumer
先迁走，再删 owner”执行。

| 删除批次 | 当前位置/对象 | 删除前置条件 |
|---|---|---|
| A | `plugins/default_memory/plugin.py` 的旧 recall JSONL 名字接续 | 所有可回滚 runtime 已只读写 generation `data_root` |
| B | `agent/lifecycle/phases/after_reasoning.py` legacy assistant metadata slots | Citation/Observe/Emotion 等 metadata writer 全部使用 v3 seam |
| C | `agent/plugins/dashboard_host.py` legacy register/context ABI | 所有 Dashboard 插件只接收 v3 `DashboardContext` |
| D | `agent/tool_hooks/base.py`、`types.py`、`executor.py` 与 snapshot/manager tool hook catalog、trace injection | 所有 ToolHook consumer 迁入 typed Tool events，完整 Tool Gate 接管 |
| E | `agent/plugins/snapshot.py`、`manager.py` 的 static-active 与 stable-health exemption | 不再存在 v2-only candidate 或 v2 static contribution |
| F | `agent/plugins/context.py` 的 `PluginContext` | v3 generation metadata 已迁到 Core-private record，所有 v2插件已迁移 |
| G | `agent/plugins/doctor.py` v2 declaration/class discovery | installer/doctor 只接受 `api_version=3` namespace |
| H | `agent/plugins/base.py`、registry 与 Manager 的 v2 lifecycle/contribution/command 调用 | command/MCP/process/channel/proactive/mobile/phase 全部有 v3 consumer 与 Gate |
| I | `RuntimeSnapshot` 中 phase/jobs/channels/MCP/managed-service 等 v2 固定字段 | snapshot 只保存 generation、CompositionRoot/topology 与派生 capability catalogs |
| J | `docker/debug/plugin-api-v2.lock.json` 与 v2 Gate | 29 个实现加后续纳管插件都有 pure-v3 full-fleet Gate |

最终删除 PR 不保留 deprecated alias、空 `Plugin` 壳或 v2/v3 自动探测。旧配置/manifest 若需要
离线迁移，应由一次性、可审阅的安装迁移 owner 完成，不能让 runtime 永久双读。

## 8. 2026-08-16 原始实施顺序（历史）

下列顺序只用于审阅最初 stacked PR，不再是当前收尾命令。当前收尾固定为同一 clean head 的
static fleet、Mobile、WebUI、E1～E4，再由生产替代清单关闭 W6～W9。

```text
1. 先按 parent chain review/merge #395 → #397 → #398 → #399 → #400 → #401
2. Review/merge #425 → ... → #431，并合入 exact plugin contract
3. Tool lane：合入 Shell Restore/Safety/Loop Guard pure-v3 source，再 merge #432
4. Passive lane：merge #433 → ... → #438 与 Default Memory；合入 Citation/Meme pure-v3 source
5. Merge #439 → #440 → #441，并在 exact heads 重跑 composition 与 WebUI E2E
6. 按 lifecycle/command → MCP/process → channel → proactive/mobile 迁移剩余 23 个
7. 每个 seam 的最后 consumer 迁走后，提交对应 A～I 小型删除 PR
8. 建立 pure-v3 full-fleet Gate，冷启动/热重载/晋升/回滚/停止全部通过
9. 最后执行 J：删除 v2 lock/Gate 与 runtime 双路径，doctor 只接受 api_version=3
```

每张 PR 都必须可以单独 review：一个能力 seam、一个迁移族群或一个已无 consumer 的删除批次。
跨仓插件先提交 canonical source，再由 Core exact-commit lock/Gate 固定组合；不得直接修改安装
cache。合并前重新核对相邻 diff 与栈顶累计行为。

## 9. 最终 v3-only 验收

只有以下事实同时成立，才可以宣布 v2 已删除：

- `rg 'V2_REMOVAL|api_version = 2|class .*\(Plugin\)|ToolHook|PluginContext'` 不再命中
  production compatibility owner；测试 fixture 的历史格式另行标注；
- 运行时只装载 `api_version=3` namespace，unknown/legacy declaration fail-loud；
- `PluginManager` 不再逐项调用 phase/command/jobs/channels/MCP/proactive/mobile 固定插件方法，
  `telegram_bot_commands()/mobile_bot_commands()` 的 Manager/bootstrap 聚合路径已删除；
- candidate Root 的 data/workspace/external effects 与 stable 隔离，失败/取消零残留；
- stable Root 继续受 promotion、snapshot lease 与 generation drain 保护；
- 29 个已跟踪实现和后续纳管的 GitHub Watcher 全部有 exact-commit 行为证据；
- 一次性 WebUI-only runtime 与至少一个 channel、MCP/process、proactive 场景真实通过；
- plugin-data、SessionDB、workspace assets、immutable artifacts 与外部效果的 write set 符合合同；
- 所有 PR、CI、Gate 和文档指向同一组 exact heads，旧 v2 lock/Gate 已物理删除。

回滚按 PR 栈反向进行。任何阶段只要 v3 Gate 与 v2 baseline 不等价，就保留该 seam 的 v2
owner，修复 v3 consumer 或合同；不能用放宽 oracle、跳过场景或静默 fallback 获得全绿。

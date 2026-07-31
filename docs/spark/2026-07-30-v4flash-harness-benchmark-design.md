# V4 Flash 完整 Runtime Harness Benchmark 设计

日期：2026-07-30

状态：第一阶段已实施；5-case diagnostic 和 H4 pairwise ablation 已完成

目标分支：`feature/v4flash-harness-benchmark`

基线：`origin/main@b3c125f7e9006f8d990aefe36c426ac4b1b36379`

## 1. 问题和用户意图

当前项目已经具备显式 workspace、完整被动 turn 生命周期、程序化控制接口和
Akasha 记忆引擎，但还没有一套能够长期回答以下问题的实验系统：

> 在不修改 benchmark、隐藏 verifier、模型资源和评分口径的前提下，研究者能否从
> DeepSeek V4 Flash 的真实 trial 证据中定位 Akasic harness 缺陷，实施通用改进，
> 并通过消融和重复实验证明改进真实存在？

用户确认的目标结构是：

- Benchmark Controller 独立于被测 Akasic runtime。
- 一个 benchmark problem 的一次 attempt 对应一个独立 Docker 容器。
- 每个容器拥有独立 workspace、HOME、配置、插件目录和运行状态。
- 容器启动完整 Akasic runtime，通过公开程序化接口提交任务。
- runtime 使用 Akasha；不加载外部业务插件、MCP、渠道、Wake、Proactive、Drift
  和 scheduler。
- 任务结束后先正常排空，再停止但不删除容器和 volume，保留现场用于运行时调查。
- V4 Flash 只在 scoring trial 中完成 benchmark task，不读取跨任务 trace，不修改
  Akasic harness，也不承担自优化职责。
- 用户和 Codex 在容器外观察已完成 trial，维护实验 ledger，并在独立 Git worktree
  修改 harness。
- scoring trial 的 Akasha 不跨问题或 attempt 共享。
- 开发主轨固定使用 DeepSeek V4 Flash `reasoning_effort=high`。
- `max` 仅作为独立对照轨道，不能和 `high` 的结果拼接。

第一阶段不是立即宣称一个完整基准成绩，而是先建立隔离边界，运行一个真实
V4 Flash High smoke trial，确认调用链、现场保留和证据采集都成立。只有冻结协议下
所有问题的完整运行才可以命名为 Akasic baseline。

## 2. 参考实验及口径修正

流程参考：

- [Terminal-Bench 2.1 Kimi K3 Recursive Self-Improvement traces](https://gist.github.com/arafatkatze/8ef2e3d452703fc2978715b40dff97fe)
- [Recursive harness improvement prompt](https://gist.github.com/arafatkatze/fe7d3743315c80d5e3e8ab1bdef39903)
- [Cline PR #12465](https://github.com/cline/cline/pull/12465)

参考实验固定 dataset、model route、effort、sampling、资源、超时、重试、任务、
verifier 和并发；每轮先登记可证伪假设，再经过 synthetic reproduction、
预注册 diagnostic slice、完整运行和三次确认。中断运行不会拼接进成绩，失败现场与
成功现场同样保留。

本项目的主要外部比较锚点采用
[Artificial Analysis Terminal-Bench v2.1](https://artificialanalysis.ai/evaluations/terminalbench-v2-1)。
该页面说明其协议使用 89 个 Terminal-Bench v2.1 任务、Terminus 2 agent harness、
e2b sandbox，并报告每题三次重复的平均 pass@1。页面内嵌原始数据为：

| 模式 | Terminal-Bench v2.1 外部参考 |
|---|---:|
| V4 Flash High | 56.9288% |
| V4 Flash Max | 61.7978% |

因此，用户给出的 `56.9%` 是正确的 V4 Flash High Terminal-Bench v2.1
`external_reference`。它仍然不是本项目的 baseline，因为 Artificial Analysis 使用
Terminus 2，而本项目要测的是 Akasic Core + Akasha。Akasic V4 Flash High baseline
必须在本设计冻结的协议下实际运行后生成。

[DeepSeek V4 Flash 官方模型卡](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash)
另列出了 Terminal-Bench 2.0 的 High 56.6% 和 Max 56.9%。这组 2.0 数据保留为历史
参考，不与 Artificial Analysis 的 2.1 数据或 Akasic 实验结果合并。

## 3. 当前真实调用链和状态 owner

程序化调用必须经过公开 SDK/control 协议，不直接调用 reasoner、memory runtime
或内部测试入口：

```text
┌──────────────────────────────────────────────────────────────┐
│ Benchmark Controller                                         │
│ SDK thread/start → turn/start → result/events                │
└────────────────────────────┬─────────────────────────────────┘
                             ▼
┌──────────────────────────────────────────────────────────────┐
│ ConversationRuntime                                          │
│ queued → in_progress → terminal；持久化 control turn         │
└────────────────────────────┬─────────────────────────────────┘
                             ▼
┌──────────────────────────────────────────────────────────────┐
│ execute_control_turn                                         │
│ 收集 tool lifecycle、stream delta、TurnCommitted 和 usage    │
└────────────────────────────┬─────────────────────────────────┘
                             ▼
┌──────────────────────────────────────────────────────────────┐
│ AgentLoop.process_direct_message                             │
│ stateless=false，dispatch_outbound=false                     │
└────────────────────────────┬─────────────────────────────────┘
                             ▼
┌──────────────────────────────────────────────────────────────┐
│ PassiveTurnPipeline                                          │
│ BeforeTurn → BeforeReasoning/Akasha → Reasoner/tool loop     │
│ → AfterReasoning/session commit → AfterTurn/Akasha commit    │
└──────────────────────────────────────────────────────────────┘
```

当前调用链证据：

- [`Thread.run`](../../sdk/python/src/akashic_sdk/client.py)；
- [`ConversationRuntime`](../../agent/control/runtime.py)；
- [`execute_control_turn`](../../bootstrap/control_execution.py)；
- [`AgentLoop.process_direct_message`](../../agent/looping/core.py)；
- [`PassiveTurnPipeline`](../../agent/core/passive_turn.py)；
- memory provider 解析入口 [`bootstrap/wiring.py`](../../bootstrap/wiring.py)；
- Akasha runtime [`plugins/akasha/`](../../plugins/akasha/)。

当前 owner：

| 能力或状态 | 权威 owner |
|---|---|
| trial 编排、资源和保留策略 | Benchmark Controller |
| container 生命周期 | Docker backend |
| task 内容和公开资源 | 固定 dataset artifact |
| 隐藏 verifier 和 reward | 容器外 evaluator |
| control turn 状态 | `ConversationRuntime` / trial `sessions.db` |
| 正式会话消息 | trial workspace 内 `sessions.db/messages` |
| Akasha 派生状态 | trial workspace 内 Akasha runtime |
| candidate 源码身份 | Git commit、tree 和 image digest |
| experiment 判断与 incumbent | 不可变 run records + experiment ledger |

Akasha 当前通过 memory plugin contract 接入。本文的“无插件”准确含义是“无外部或
可安装业务插件”；允许列表只包含内置 Akasha memory provider。若配置为真正
`plugin_count=0` 并同时移除 memory provider，实验将不再测试用户要求的系统。

### 3.1 实施合同

| 字段 | 约定 |
|---|---|
| `capability_owner` | Benchmark Controller；Akasic runtime 只执行一个 trial |
| `consumer_scope` | 本地 Docker campaign，未来可替换为云实例 backend |
| `runtime_patch` | 首个 smoke 不修改线上 runtime，也不注入运行时 patch |
| `runtime_patch_reason` | 不适用；后续 candidate 必须通过新 commit 和 image 构建 |
| `authoritative_state_owner` | controller manifest/ledger 与各 trial 独立 workspace |
| `client_only_alternative` | 不适用；程序化 SDK 是调用方，不拥有 runtime 语义 |
| `change_type` | 新增 benchmark orchestration 和验证能力 |
| `semantic_delta` | 增加可重复、可取证、默认保留的独立 trial；不改变线上消息语义 |
| 允许副作用 | 新 worktree 文件、测试 image/container/volume、模型 API 调用和实验 artifact |
| 受保护状态 | 正式 workspace、线上进程、当前用户 checkout、外部插件、隐藏 verifier 和凭据 |
| 停止条件 | 任一受保护状态发生变化、隔离证明失败、secret 泄漏或证据链不完整 |
| 回滚点 | 基线 commit、上一 image digest、retained trial 和 append-only ledger |

## 4. 已确认事实、推断和未知边界

### 4.1 已确认事实

- SDK 的 `Thread.run()` 最终进入 `ConversationRuntime` 和正式被动 turn。
- `process_direct_message` 默认 `stateless=false`，会持久化 user 和 assistant
  消息；control 调用只关闭外部 outbound dispatch。
- Akasha 能从 trial 自己的 `sessions.db` 建立和发布派生状态。
- 项目 DeepSeek provider 支持 `reasoning_effort=high`；内部 `xhigh` 会映射为
  DeepSeek `max`。
- 正式 workspace 与 Git worktree 是不同的状态根，benchmark 不需要也不得挂载
  正式 Akashic workspace。

### 4.2 设计推断

- 一个 problem 一个实例可以消除跨任务 session、Akasha、进程、临时文件和插件状态
  污染。
- 研究记录保存在容器外的 Git 文档和 append-only ledger；scoring Akasha 只保存本
  problem 当前 attempt 的运行状态，可以避免模型记住其他已评分问题。
- 先停止而不删除实例，可以冻结任务完成时的现场；需要观察长期行为的 soak trial
  使用单独类型，不改变普通 scoring trial。

### 4.3 实施时已验证

- Harbor pin 为 `v0.16.1@137c27874df6163309c6c0cb218a56a7b0e00e79`，
  Terminal-Bench 2.1 artifact 含 89 tasks。
- candidate source 通过 Git bundle 进入容器，保留 migration baseline 和 dirty
  overlay；不 bind 宿主源码。
- verifier 由 Harbor lifecycle owner 在 agent terminal 后单独执行。
- smoke 通过后并发硬上限原为 3；资源画像确认宿主仍有余量且维护者批准后，上限提高
  到 6。semaphore 继续保证 slot 释放后才启动下一实例。
- DeepSeek V4 Flash High 的真实 wire trace 包含 `reasoning_content` 和结构化 tool
  calls。H4 证明旧的 8,192 显式输出上限会截断该 slice 的 `regex-log`，当前
  benchmark 改为 provider-default output policy。

完整资源画像和 89 题重复运行仍未执行；第一阶段结果不能命名为完整 baseline。

## 5. 目标架构和权限边界

```text
┌──────────────────────────────────────────────────────────────┐
│ Host / future cloud control plane                            │
│                                                              │
│  Campaign Spec ── Scheduler ── Docker Backend                │
│       │                              │                       │
│       │                              ├─ trial-001-attempt-01 │
│       │                              ├─ trial-002-attempt-01 │
│       │                              └─ trial-002-attempt-02 │
│       │                                                      │
│       ├─ external verifier                                   │
│       ├─ artifact collector                                  │
│       └─ append-only experiment ledger                       │
└──────────────────────────────────────────────────────────────┘
```

Controller 只暴露窄生命周期接口：

```text
create_trial(spec)
start_trial(trial_id)
invoke_trial(trial_id, task)
quiesce_trial(trial_id)
capture_trial(trial_id)
stop_trial(trial_id)
inspect_trial(trial_id)
destroy_trial(trial_id, authorization)
```

普通实验路径不调用 `destroy_trial`。物理删除只能由明确的 campaign cleanup 或人工
授权触发。

### 5.1 Trial 隔离

每个 attempt 创建以下独立对象：

- container name / ID；
- writable task workspace volume；
- Akasic runtime workspace volume；
- HOME volume；
- config volume；
- external plugin home 和 cache volume；
- logs/artifacts volume；
- 独立 control socket 或 loopback port；
- 独立网络 namespace；
- 独立 CPU、memory、PID、file descriptor 和磁盘 quota。

不得挂载：

- 正式 `~/.akashic/workspace`；
- 用户当前 checkout；
- 其他 trial volume；
- Docker socket；
- Harbor hidden verifier；
- controller experiment ledger；
- 研究者的 Git worktree 和宿主 artifact 根。

candidate 源码以只读 image layer 进入 scoring trial。任务允许改动的代码仓库由
benchmark task 自己提供，并位于 task workspace，不得反向修改 harness image。

### 5.2 网络和凭据

- 默认阻断任意外网，只允许到固定模型代理或 DeepSeek API egress。
- trial 不获得宿主云凭据、GitHub 凭据和 Docker 控制权限。
- 模型凭据通过 runtime secret 或只读临时文件注入，不写入 config artifact、日志或
  container inspect 可见环境变量。
- verifier 运行时不复用 trial 的模型凭据。
- 采集器在归档前执行 secret scanner；发现疑似 secret 时本次 capture 失败并隔离
  artifact，不静默删字段后宣称成功。

### 5.3 Runtime profile

冻结 profile：

```text
model                  deepseek-v4-flash
provider route         DeepSeek direct API
reasoning effort       high
memory                 enabled, engine=akasha
programmatic control   enabled
stateless              false
external dispatch      disabled
external plugins       empty
MCP                    disabled
channels               disabled
Wake/Proactive/Drift   disabled
scheduler              disabled
```

具体 sampling、context、output reserve、timeout、retry 和资源值在 preflight 后写入
带 digest 的 Campaign Spec；确定后在同一 campaign 内不得变化。

## 6. Trial 生命周期和保留协议

```text
DECLARED
   │ create
   ▼
CREATED ──start──▶ RUNNING ──turn terminal──▶ QUIESCING
                                                │
                                   runtime shutdown + flush
                                                ▼
                                            CAPTURING
                                          ┌─────┴─────┐
                                      success       failure
                                          │            │
                                          ▼            ▼
                                      STOPPED      CAPTURE_FAILED
                                          │            │
                                          └─────┬──────┘
                                                ▼
                                             RETAINED
                                                │ explicit authorization
                                                ▼
                                             DESTROYED
```

`QUIESCING` 必须等待：

- 当前 turn 进入唯一 terminal 状态；
- runtime 停止接收新 turn；
- lifecycle cleanup 完成；
- 数据库连接关闭；
- 日志 writer flush；
- 子进程被 runtime owner 正常收束。

采集完成后执行 `docker stop`，不执行 `docker rm` 或 volume 删除。

若 quiesce 超时，controller 记录 `failed_quiesce`，采集仍然存在的 container、
volume、进程和日志现场，再停止容器。不得把强制终止后的 trial 标成普通模型失败。

### 6.1 持久对象的增加、更新、失效和删除

| 对象 | 正常增加 | 允许原位更新 | 逻辑失效 | 物理删除 |
|---|---|---|---|---|
| trial manifest | controller 创建一条 | 只允许状态机字段和结束时间按序推进 | `invalidated_reason` | campaign cleanup 明确授权后 |
| session messages | runtime 正常 turn 追加 | 按项目既有不变量，不因 benchmark 改写正文 | session/trial 标记无效，不删消息 | 仅随授权销毁整个 trial volume |
| Akasha 派生库 | runtime 从本 trial sessions 构建/发布 | 按 Akasha 原子发布协议 | trial invalidated 后不参与评分 | 仅随授权销毁整个 trial volume |
| logs/traces | runtime 和 collector 追加 | 不覆盖原始文件；派生索引单独生成 | manifest 标注 incomplete/corrupt | 归档保留期结束并获授权后 |
| result/ledger | controller 追加 | 决策使用新记录勘误，不覆盖历史 | superseded/invalidated | 当前不得自动减少 |
| container/volume | Docker backend 创建 | container 状态允许 start/stop | `retained` 后不再进入评分 | `destroy_trial` 明确授权后 |

恢复证据包括 manifest、container ID、volume ID、image digest、源码 tree、配置
digest、artifact checksums 和原始 terminal event。

## 7. 证据包

每个 trial 生成一个不可变 manifest 和 checksummed artifact bundle：

```text
trial/
├── manifest.json
├── campaign-spec.json
├── invocation.json
├── control-events.jsonl
├── provider-usage.json
├── runtime.log
├── docker-inspect.json
├── resource-samples.jsonl
├── process-snapshot.json
├── filesystem-diff.json
├── verifier-result.json
├── workspace/
│   ├── sessions.db
│   ├── sessions.db-wal
│   ├── sessions.db-shm
│   └── akasha-derived-state
└── SHA256SUMS
```

数据库证据从已经停止写入的 trial volume 复制。若 runtime 未正常关闭，则同时保留
DB、WAL 和 SHM，不对原件执行修复或 checkpoint。任何恢复、查询或 rebuild 都在
只读副本或 forensic clone 上进行。

manifest 至少记录：

- campaign、problem、attempt 和 trial ID；
- baseline/candidate/smoke/diagnostic/confirmation 类型；
- task artifact 和 verifier digest；
- source commit、tree、dirty patch digest 和 image digest；
- model ID、provider route、effort、sampling 和上下文参数；
- CPU、RAM、磁盘、timeout、retry 和 concurrency；
- workspace、config、Akasha seed 和 external plugin catalog digest；
- container、volume、network 和 control endpoint 身份；
- turn/session ID、terminal 状态、usage、成本和时间；
- verifier 原始结果；
- quiesce、capture、stop 和 retention 状态；
- invalidation 原因；
- artifact bundle checksum。

## 8. Benchmark 与基线取得

### 8.1 主基准

第一阶段使用 Terminal-Bench 2.1，原因是它直接覆盖 terminal、文件编辑、长进程、
工具调用、完成提交和 provider 稳定性，适合发现 harness 故障。它不充分覆盖 Akasha
长期记忆，因此后续必须增加独立的多 turn memory campaign，不能用 Terminal-Bench
成绩替代 Akasha 质量结论。

### 8.2 首个 smoke trial

首跑只选择一个公开、耗时短、不会依赖特殊硬件的任务。其目标是验证：

- trial 未挂载正式 workspace、用户 checkout 和外部插件；
- V4 Flash High 请求真实发出；
- SDK 调用经过完整 lifecycle；
- user/assistant/tool/usage 和 Akasha 状态按预期写入；
- verifier 在容器外运行；
- runtime 能正常 quiesce；
- container 和 volume 在 stop 后仍可 inspect；
- 现场可从 forensic clone 读取；
- 主 checkout、正式 workspace 和线上进程前后 digest/状态不变。

单任务成功或失败均不得命名为 baseline，也不得直接套用外部 `56.9288%`。

### 8.3 正式 baseline

smoke 通过后冻结 Campaign Spec。每个 problem/attempt 创建一个独立实例，调度器可按
固定并发逐批运行，但实例和状态不得复用。

初始开发 baseline 允许一次完整运行，用于生成 failure census。第一次候选准备晋级
前，再补足相同 baseline artifact 的重复运行，以便比较方差。最终 claim 至少报告：

- 完成问题数和总数；
- pass rate；
- exception rate 和类型；
- timeout、setup failure、provider failure；
- task-level pass/fail；
- token、缓存、成本和时长；
- 三次 baseline 与三次 candidate 的均值、中位数、范围和 paired flips。

## 9. 证据驱动的 Harness 优化梯级

```text
Baseline failure census
        │
        ▼
一个可证伪假设
        │
        ▼
Stage A：synthetic reproduction + unit/integration test
        │ pass
        ▼
Stage B：预注册 diagnostic failures + stable pass controls
        │ 无 control regression
        ▼
immutable candidate image + digest
        │
        ▼
Stage C：完整任务集
        │ 达到晋级门槛
        ▼
Stage C-A：组件消融
        │ 单改动、组合改动和 baseline 使用同一协议
        ▼
Stage D：同一 artifact 三次 confirmation
        │
        ▼
Stage E：独立五次 publishable protocol
```

每轮 experiment ledger 必须记录：

- 假设；
- trace 证据；
- 最小通用改动；
- synthetic test；
- commit、tree、image 和 checksum；
- 精确配置；
- diagnostic/full job；
- gains、losses、异常、token、成本和时长；
- 单组件和组合组件的消融结果；
- keep/revise/revert；
- 下一假设。

候选晋级条件：

- 完整任务集全部产生可判定终态；
- 相比当前 incumbent 至少提升三个问题，或具有强 paired evidence 并通过重复确认；
- 提升不能由 provider outage、资源、超时、重试、sampling 或 verifier 变化解释；
- 不包含 task 名称判断、benchmark 检测、隐藏测试推导或固定解法；
- 通用测试和项目 Gate 通过；
- 没有严重通用 runtime 回归。

消融规则：

- 每个可独立成立的机制使用独立 commit 和独立 image。
- 一个候选同时包含机制 A 和 B 时，至少比较 baseline、A-only、B-only 和 A+B；
  若成本不足以执行全部完整运行，先在预注册 diagnostic slice 做四格消融，晋级后再
  对仍有交互不确定性的组合执行完整运行。
- 组件确实不可分时，ledger 必须写明共享不变量和无法拆分的触发路径，不能只以
  “一起改更方便”为理由跳过消融。
- 后续候选以当前 incumbent 为底时，比较的是 `incumbent`、`incumbent + Hn` 和
  `incumbent + Hn` 的回退构建；不得把多个未测假设连续堆叠。
- 消融发现某组件无增益或产生稳定回归时，该组件必须回退；不能因为组合总分上涨就
  一并保留。

中断、controller 故障或缺少问题的运行整体 invalidated。保留所有实例和证据，但不将
成功问题拼接成一个新成绩。

## 10. 研究者与 Scoring Runtime 分离

```text
┌────────────────────────────────────┐
│ 用户 + Codex 研究流程              │
│ retained traces / results / ledger │
│ 独立 Git worktree                  │
└──────────────────┬─────────────────┘
                   │ 假设、通用修复、测试、commit
                   ▼
┌────────────────────────────────────┐
│ immutable candidate harness image  │
└──────────────────┬─────────────────┘
                   │ fresh instance per attempt
                   ▼
┌────────────────────────────────────┐
│ V4 Flash Scoring Runtime           │
│ 只完成当前 benchmark task          │
│ task-local Akasha                  │
│ 结束后停止并保留                   │
└────────────────────────────────────┘
```

V4 Flash scoring runtime 不接收：

- 其他 task 的 trace、reward、session 或 Akasha；
- experiment ledger 和研究假设；
- harness 源码的 writable mount；
- controller、verifier、Git 或 Docker 控制工具。

研究者可以读取：

- 已结束 trial 的完整普通轨迹、工具调用、时间、资源和 provider 错误；
- verifier 的 pass/fail、公开结果和错误分类；
- baseline 与 candidate 的 task-level paired flips；
- 本项目代码、测试和公开 benchmark 协议。

研究者不能读取隐藏 verifier 实现来设计 task-specific 行为，也不能把隐藏 reward
结构写入生产 prompt。实验知识由 Git commit、设计文档和 append-only ledger 保存，
不引入一个跨轮持久 Akasha 的 optimizer runtime。

研究改动只进入新 candidate commit 和 image。scoring runtime 启动后，harness image
保持只读；模型只能修改 benchmark 提供的 task workspace。

## 11. 失败、取消和并发

- 同一 trial 同时只允许一个 active turn。
- 一个 problem 的不同 attempt 使用不同实例。
- concurrency 是 Campaign Spec 的固定字段；资源紧张时排队，不临时降低单实例资源。
- provider 429、timeout、连接错误与模型错误分开统计。
- controller retry 只处理创建或连接前的幂等基础设施操作。已进入模型调用的 turn 不由
  controller 静默重试。
- runtime、controller 或 verifier 取消分别产生不同终态。
- capture 失败不能把 task verifier 成绩升级为可信结果。
- retained container 重启后只能进入 forensic 模式，不能继续原评分 attempt。
- soak trial 使用独立 `trial_kind=soak`，不计入普通 baseline。

## 12. Anti-reward-hacking 约束

禁止：

- 修改 Harbor task、test、verifier、reward 或 expected output；
- 把隐藏 verifier 挂载进 trial；
- 添加 task-name、dataset 或 benchmark 检测；
- 在生产 prompt 中加入 Terminal-Bench 专用文本；
- 增加资源、timeout、retry、attempt 或上下文制造更高成绩；
- 排除失败、选择性重跑后拼分；
- 只公布最好一次而隐藏确认回归；
- 让 scoring runtime 修改 harness、controller、ledger 或 verifier；
- 为提高成绩把研究性 controller/verifier 改动混入 candidate harness；
- 把 High 和 Max、不同 provider route 或不同 dataset 的结果放进同一统计分布。

所有生产改动必须能够在不提 Terminal-Bench 的情况下解释其通用价值。若发现
controller 或 verifier 基础设施缺陷，应建立独立非评分修复，冻结新协议并重新取得
baseline，不能把基础设施修复前后的分数解释为 harness 提升。

## 13. 分阶段实施

### Phase 1：合同和 controller 骨架

- 定义 CampaignSpec、TrialSpec、TrialManifest 和状态机。
- 定义 Docker backend 窄接口。
- 建立 append-only ledger 和 artifact checksum。
- 添加拒绝正式 workspace、Docker socket、非空 external plugin catalog 的边界校验。
- 使用 fake backend 测试状态机，不调用真实模型。

### Phase 2：完整 runtime Docker backend

- 构建不可变 Akasic image。
- 创建一次性 workspace/HOME/config/plugin/log volume。
- 注入 V4 Flash High runtime profile 和临时 secret。
- 启动完整 runtime/control service。
- 通过 SDK 执行 turn。
- 实现 quiesce、capture、stop 和 retained inspect。

### Phase 3：单任务真实 smoke

- 在执行前记录线上进程、端口、正式 workspace 和当前 checkout 的只读指纹。
- 创建一个独立 trial。
- 运行一个 V4 Flash High 公开问题。
- 容器外验证。
- 排空、采集、停止并保留。
- 再次核对线上和正式 workspace 指纹不变。
- 从 forensic clone 检查 runtime 日志和数据库。

### Phase 4：Harbor task/verifier bridge

- 固定 Terminal-Bench 2.1 artifact 和 verifier digest。
- 将每个 problem 映射为独立 TrialSpec。
- 固定 concurrency、timeout、资源和 retry。
- 先执行预注册小集合，确认无隔离和评分泄漏。

### Phase 5：完整 V4 Flash High baseline

- 运行完整任务集。
- 保留全部实例。
- 生成 failure census、task matrix、异常分类和成本报告。
- 补足 baseline 重复运行后，才允许建立统计比较。

### Phase 6：研究者驱动的 Harness 优化

- 用户和 Codex 从已完成 trial 提出一个可证伪假设。
- 每轮只实施一个最小通用 harness 改动，并添加 synthetic test。
- 按 diagnostic、full、ablation、confirmation 梯级推进。
- 每个单组件和组合构建使用独立 commit、tree、image 和 checksum。
- V4 Flash 在所有 scoring trial 中只完成 benchmark task。

## 14. 验收标准

### 隔离

- 每个 attempt 的 container、workspace、HOME、config、plugin home 和 Akasha identity
  唯一。
- trial 内无法读取正式 workspace、用户 checkout、其他 trial、Docker socket 和隐藏
  verifier。
- smoke 前后线上进程、端口和正式 workspace 证据一致。

### 生命周期

- 程序化调用经过 `ConversationRuntime`、`execute_control_turn`、
  `AgentLoop.process_direct_message` 和 `PassiveTurnPipeline`。
- `stateless=false`，user/assistant 消息和 Akasha 状态来自本 trial。
- task 完成后 container 处于 stopped/retained，volume 仍可 inspect。

### 证据

- manifest 包含所有身份、配置、资源、终态和 checksum。
- 缺少任何必需证据时 trial fail-loud，不进入可信分数。
- forensic clone 能在不修改原 volume 的前提下读取现场。

### 科学口径

- smoke 不被报告为 baseline。
- High、Max 和外部公开参考分开。
- invalidated run 不拼接。
- V4 Flash 不读取跨任务实验记忆，也不修改 harness。
- 组合候选完成所需消融后才允许归因。
- candidate 只有通过完整运行和重复确认后才能成为 incumbent。

### 项目验证

- controller 单元测试和 Docker integration tests 通过。
- 相关静态检查通过。
- `python docker/debug/gate.py run --base origin/main` 生成与当前源码 digest 一致的报告。
- 未运行的私有 Gate 准确标记，不用公开测试代替。

## 15. 回滚和恢复点

- 当前用户 checkout 不参与本任务写入；恢复点是其原有 dirty state。
- 实施 worktree 基线为
  `b3c125f7e9006f8d990aefe36c426ac4b1b36379`。
- 每个 candidate 由 commit、tree 和 image digest 唯一恢复。
- 普通回滚选择上一 incumbent image，不改写历史 ledger。
- retained trial 默认不删除。需要释放空间时先生成待删除清单、确认 artifact checksums
  和归档状态，再由维护者显式授权 `destroy_trial`。
- 任何线上进程或正式 workspace 指纹变化都会立即终止 smoke，并先进行只读事故调查。

## 16. 首个实施接手点

下一阶段从 Phase 1 开始，不直接运行完整 89 问题：

1. 调查 Harbor 当前 Python API、dataset artifact 和 verifier 边界。
2. 调查现有 Docker debug/runtime image 能否复用。
3. 写最小任务合同和 schemas。
4. 实现一个 fake trial，再实现一个真实 V4 Flash High smoke trial。
5. 在 smoke 通过且线上隔离证据成立后，冻结完整 baseline Campaign Spec。

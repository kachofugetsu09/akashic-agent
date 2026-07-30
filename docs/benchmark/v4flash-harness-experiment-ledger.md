# V4 Flash Harness 实验 Ledger

日期：2026-07-30

状态：89/89 手工诊断遍历已收口；未启动优化后 fresh final eval

设计合同：
[V4 Flash 完整 Runtime Harness Benchmark 设计](../spark/2026-07-30-v4flash-harness-benchmark-design.md)

需要维护者最终定夺的行为 treatment 单独记录在
[V4 Flash Benchmark 待定语义改变](v4flash-semantic-changes-pending.md)；遍历期间不实施。
每个已触达 case 的终态、初步归因和核心证据维护在
[V4 Flash 逐题结果](v4flash-case-results.tsv)。

## 冻结变量

| 变量 | 当前值 |
|---|---|
| Dataset | `terminal-bench/terminal-bench-2-1`，89 tasks |
| Smoke task | 当前 source-bound Gate 使用 `cancel-async-tasks`；历史 S0–S4 使用 `openssl-selfsigned-cert` |
| Harbor | `v0.16.1@137c27874df6163309c6c0cb218a56a7b0e00e79` |
| Model | `deepseek-v4-flash` |
| Effort | `high` |
| Context / max output | `1,000,000 / provider default`；H4 前 baseline 为 `8,192` |
| Runtime | 完整 Akasic gateway，Akasha enabled |
| 调用入口 | public Python SDK / control socket |
| Agent / verifier timeout | `900s / task official value` |
| Retention | 核心 artifact 留证；完成诊断后删除可重复创建的 container/network，明确需要运行时取证时例外 |
| Concurrency | smoke 为 `1`；Gate 打开后全局硬上限 `6` |

## Official Harbor infra 对账

Harbor 的 lifecycle owner 仍负责 environment start、agent setup、agent run、外部
verifier 和 stop。自定义 agent 只补 Akasic runtime 自身安装与 SDK 调用。

| 层 | 必需依赖 | Preflight / 失败语义 |
|---|---|---|
| Host control | Docker Engine、Compose、Harbor pin、dataset artifact、uv binary | 缺失即在建实例前失败 |
| Image OS | Linux/POSIX shell；setup 阶段可使用 root | Windows 或无 root setup 不进入当前 campaign |
| Git / CA | 内容寻址 Docker volume，只读挂载；trial 内不再运行 `apk`、`apt-get` 或 `yum` | cache、manifest、版本或挂载权限不匹配时 setup 失败 |
| Runtime fetch | DNS、HTTPS、CA；可访问 Python/PyPI、DeepSeek、Dashscope | 不静默使用宿主依赖或假数据 |
| Filesystem | `/app` 为 task 工作区；`/tmp` 和 `/logs` 可写 | source 复制入容器后设为只读；不 bind 宿主源码 |
| Git history | source bundle 必须包含固定 migration baseline 与 candidate HEAD | 宿主和容器内分别用 Git 校验 |
| Python | uv 安装 CPython 3.13 和 frozen requirements | wheel/build 失败归为 image compatibility |
| Control | 独占 Unix socket、完整 gateway、public SDK | 无 terminal 或持久化不一致时 fail-loud |
| Verifier | Harbor 官方 task tests，写 `/logs/verifier/reward.txt` | agent 正常收束后由 Harbor 执行 |
| Retention | 每 attempt 独立 compose project | stop 后保留容器；禁止 Docker socket和 host port |

Harbor 官方 Codex installed-agent 同样会先探测容器包管理器，以 root 安装系统包，再以
agent 用户安装 runtime。这说明题目镜像不保证自带 agent 完整工具链；兼容层属于
agent setup，而不是 benchmark task 或 verifier。

## Smoke 尝试

### S0 — 缺少 Git

- Trial：`akasic-bench-v4flash-smoke-openssl-20260730-033645`
- 结果：infra failed；没有模型请求。
- 证据：task image 不含 `git`，启动 migration 在 `git` 子进程创建前失败。
- 处理：setup 按 Harbor installed-agent 模式显式补 Git。
- 边界：容器已停止保留；线上 PID 和源码 digest 未变；3 并发 Gate 关闭。

### S1 — 快照缺少 migration baseline

- Trial：`akasic-bench-v4flash-smoke-openssl-20260730-034117`
- 结果：infra failed；没有模型请求。
- 证据：容器内伪造的单提交 snapshot 无法解析固定 baseline
  `012e37c8b51df045353972bb551d8e868ab52455`。
- 处理：删除伪造 commit；宿主生成并校验 Git bundle，容器恢复真实历史后用
  `git reset --mixed candidate HEAD` 保留当前 worktree overlay。
- Synthetic：临时两提交仓库验证 baseline、HEAD 和 dirty overlay 同时保留。
- 边界：容器已停止保留；线上 PID 和源码 digest 未变；3 并发 Gate 关闭。

### S2 — terminal delivery gap

- Trial：`akasic-bench-v4flash-smoke-openssl-20260730-034813`
- 结果：模型完成 task，但 harness failed；官方 verifier 未运行。
- 模型行为：11 次请求、15 个工具调用，持久化 turn 为 `completed`，17 items，
  final response 560 chars。
- 证据：`sessions.db/turns` 已在 `2026-07-30T03:49:23.792520Z` 提交 completed；
  同时 router 报 `SlowConsumerError`，SDK trace 缺少 `turn/completed`，driver 等待。
- 现场：容器、原 trace、runtime stderr、Akasha workspace 和 11 份 LLM request
  snapshot 均保留。
- 边界：该 attempt 不计分；3 并发 Gate 关闭。

### S3 — H1 命中，Harbor context projection 失败

- Trial：`akasic-bench-v4flash-smoke-openssl-20260730-035600`
- 结果：H1 treatment 成功收束 terminal delivery gap；Harbor verifier 前置适配失败。
- 模型行为：12 次请求、11 个工具调用，持久化 turn 为 `completed`，13 items；
  trace 含 `terminal_recovered` 和 `turn_persisted`。
- H1 观察：第二次真实模型运行复现 `SlowConsumerError`；5 秒 grace 后 public
  `turn/read` 恢复成功，gateway 正常关闭。
- 新失败：Harbor `AgentContext.metadata` 的 schema 默认值为 `None`，custom agent
  直接 `.update()` 触发 `AttributeError`，因此 verifier 未启动。
- 处理：按 Harbor 内置 agent 约定赋值
  `context.metadata = {**(context.metadata or {}), ...}`，并把精确 control usage
  投影到 Harbor token 字段。
- Synthetic：`metadata=None` 的 context 投影测试已覆盖。
- 边界：该 attempt 不计分；容器停止保留；线上 PID 和源码 digest 未变；3 并发
  Gate 关闭。

## 当前可证伪假设 H1

> 如果模型任务已经形成权威 terminal turn，但 control notification 在传输层丢失，
> 那么让 benchmark driver 并行观察 event stream 与 public `turn/read`，在 5 秒
> grace 后以显式 `terminal_recovered` 证据收束，可以让 Harbor verifier 正常运行，
> 且不改变模型请求、task 文件、runtime 持久化语义或 verifier。

最小 treatment 只修改 benchmark driver：

1. 事件活跃时仍逐帧记录，不额外轮询。
2. 空闲时通过 public `turn/read` 检查权威状态。
3. terminal event 正常到达时保持原路径。
4. terminal 已持久化但通知超过 5 秒未到达时，记录
   `delivery_gap=true` 和 `terminal_source=turn/read_recovery`。
5. `turn/read` 与最终结果不一致、非 completed 或超时仍然失败。

Synthetic reproduction 已覆盖“收到 started、terminal notification 永不抵达、
turn/read 已 completed”的场景。真实 treatment 必须在同一 task、模型、effort、
资源和 verifier 下重新运行；只有 trace、turn result、官方 verifier、容器保留和线上
不变量全部通过，smoke Gate 才能打开。

## 五题 diagnostic slice

预注册 slice：

1. `openssl-selfsigned-cert`：shell 与多文件 artifact。
2. `cancel-async-tasks`：Python 并发和取消语义。
3. `db-wal-recovery`：长工具循环与数据库恢复。
4. `regex-log`：精确文本约束。
5. `fix-code-vulnerability`：代码阅读、安全修复和测试。

第一批 campaign：
`akasic-bench-v4flash-campaign-20260730-040813`，硬上限三并发。

| Case | Lifecycle | Reward | 观察 |
|---|---|---:|---|
| `openssl-selfsigned-cert` | completed | 0 | 5/6；脚本未输出 `YYYY-MM-DD` |
| `cancel-async-tasks` | completed | 1 | 官方 verifier 通过 |
| `db-wal-recovery` | failed after model terminal | 未评分 | SDK 无法读取约 315KB terminal record |
| `regex-log` | failed before model | 未评分 | image 缺少 IANA timezone data |
| `fix-code-vulnerability` | failed after model terminal | 未评分 | SDK 无法读取约 228KB terminal record |

campaign 同时最多存在三个 running compose project；第四题只在一个 slot 释放后启动。
批次结束时源码 digest 不变、正式 workspace owner PID 不变，所有已创建容器均停止保留。
未进入 verifier 的三题不计入 5-case score，必须修复 infra/control transport 后重跑同一
task。

## 当前可证伪假设 H2

> Python SDK reader 沿用 asyncio 默认约 64KiB line limit，而 control 协议现有
> `max_message_bytes` 是 2MiB。若 terminal notification 或 `turn/read` response
> 大于 64KiB，则模型已经完成的复杂任务会在 SDK transport 层失败；把 SDK reader
> limit 对齐到 2MiB，可以让同一类 terminal 进入 Harbor verifier，而不改变模型、
> prompt、工具、task 文件或 verifier。

Baseline 证据：

- `db-wal-recovery`：权威 turn `completed`，49 items，`items_json=314720` bytes；
- `fix-code-vulnerability`：权威 turn `completed`，33 items，
  `items_json=227619` bytes；
- 两者 driver 都报
  `ValueError: Separator is not found, and chunk exceed the limit`。

Treatment：

- `AsyncAkashic.connect` 与同步 `Akashic.connect` 新增可选
  `max_message_bytes`，默认沿用协议的 2MiB；
- TCP 与 Unix socket 都把 `asyncio` reader limit 设置为
  `max_message_bytes + 1`；
- 非正数在 SDK public boundary 直接拒绝。

消融：

- control：64KiB `StreamReader` 对 128KiB NDJSON frame 稳定复现 `ValueError`；
- treatment：默认 SDK 通过真实 Unix socket 读取 128KiB final response 并拿到
  `completed`；
- 真实确认：重跑 `db-wal-recovery` 和 `fix-code-vulnerability`，必须到达外部
  verifier。reward 只记录，不作为 H2 transport 假设的验收条件。

`regex-log` 的 timezone 缺失是正交 infra 问题：benchmark runtime venv 显式安装
`tzdata`，不修改 task、模型或 verifier，也不计入 H2。

真实确认：

- `db-wal-recovery` treatment 到达外部 verifier；reward `0`；
- `fix-code-vulnerability` treatment 到达外部 verifier；reward `1`；
- 两个超过 asyncio 默认 reader limit 的 terminal record 都由 public SDK 完整读取；
- H2 只证明 transport 修复成立，不把 reward 差异归因于 transport。

## 五题停止点

用户要求在 5 个有效 case 后停止。这里的“有效”要求模型运行完成、官方 verifier
执行、隔离检查通过且容器停止保留；infra failure 不进入分母。

| Case | 最终有效 Trial | Reward | 主要观察 |
|---|---|---:|---|
| `openssl-selfsigned-cert` | `akasic-bench-v4flash-smoke-openssl-20260730-035855` | 0 | 5/6；日期格式错误 |
| `cancel-async-tasks` | `akasic-bench-v4flash-diagnostic-cancel-async-tasks-20260730-040813-024970` | 1 | verifier 通过 |
| `db-wal-recovery` | `akasic-bench-v4flash-diagnostic-db-wal-recovery-20260730-042053-226551` | 0 | 40 轮、42 items，出现长工具循环 |
| `regex-log` | `akasic-bench-v4flash-diagnostic-regex-log-20260730-042053-153022` | 0 | 8,192 reasoning 截断后 retry 输出 literal DSML |
| `fix-code-vulnerability` | `akasic-bench-v4flash-diagnostic-fix-code-vulnerability-20260730-042053-683463` | 1 | verifier 通过 |

原始冻结配置成绩为 `2/5 = 40%`。这是 diagnostic slice，不是 89 题 Akasic
baseline，也不能和外部 56.9288% 直接比较。

## H3 — empty-thinking retry 保留工具协议

### 假设

> 如果 DeepSeek 首次只返回 reasoning、没有 content 或 tool call，retry 不应移除
> tool schema；保留相同结构化工具并加入纠正消息，才能让后续结果继续走受控工具协议。

Baseline 在 `regex-log` 复现：

1. 首次 response 消耗完整 `8,192` output budget，只有 reasoning；
2. empty-thinking retry 发送 `tools=[]`；
3. 模型随后输出形似 `<｜DSML｜function_calls>` 的普通文本；
4. 文本没有被执行，task 未修改，reward `0`。

对照 Codex 当前实现后，拒绝了“解析 DSML 文本并执行”的候选方案。Codex 只执行
结构化 `function_call` item；畸形 arguments 返回模型错误，不把任意 assistant
文本升级为可执行调用。因此 treatment 只冻结并复用原 tool schemas，把 retry
放在 tool-call 分支之前。

Synthetic 覆盖证明 retry 仍携带工具并能执行结构化 tool call。真实 8,192 treatment：

- Trial：`akasic-bench-v4flash-smoke-regex-log-20260730-043652-473330`
- 两次请求都保留 18 个工具；
- literal DSML 消失，但两次各耗尽 8,192 reasoning，仍没有 action；
- reward `0`。

结论：H3 修复了协议安全和可执行路径，但没有改善该 case 的分数。不得把它描述成
性能提升。

## H4 — 关闭 Akasic 的 8,192 输出上限

### 假设

> 如果 V4 Flash High 在形成首个工具调用前需要超过 8,192 output tokens，那么
> Akasic 显式发送 `max_tokens=8192` 会在 reasoning 阶段截断；不发送
> `max_tokens`、由 provider 模型边界负责，可以让同一 task 进入原生工具循环。

单变量 treatment：

- `max_output_tokens = 0` 明确定义为 provider default；
- Chat Completions request 在值为 `0` 时不发送 `max_tokens`；
- 负值仍在配置边界 fail-fast；
- summary retry 仍使用独立的 `2,048` 上限；
- model、effort、prompt、tools、task、verifier、timeout 和容器隔离均不变。

真实 treatment：

- Trial：`akasic-bench-v4flash-smoke-regex-log-20260730-044609-483351`
- 首个 request 没有 `max_tokens`，仍为 high、thinking enabled、18 tools；
- 约 124 秒后产生首个原生 `write_file`；
- 共 9 requests、8 个原生 tool calls，没有 literal DSML；
- usage：input `248,357`、cached input `240,256`、output `23,274`、
  reasoning `21,422`；
- verifier `1/1`，reward `1`；
- terminal 由 public `turn/read` recovery 收束，20 events；
- 容器停止保留，候选源码、正式 workspace 和线上 owner 均未变化。

`21,422` reasoning 明确超过旧的 `8,192`，与 baseline 截断症状一致。H4 在同一个
`regex-log` pair 上把 reward 从 `0` 提升为 `1`，因此接受为当前 incumbent。

仅把这个预注册 pair 替换进五题 slice 后为 `3/5 = 60%`；这是 pairwise
counterfactual，不是完整五题重跑，更不是完整 Terminal-Bench 成绩。

## 后续假设，不在本阶段执行

### H5 — task shell PATH 暴露 runtime Python

H4 的 `regex-log` 中，shell PATH 没有 `/opt/akashic/venv/bin`。模型先尝试
`python3`，随后搜索解释器并安装 `curl/uv`，产生额外工具调用。后续可预注册
“仅给 task shell 增加 runtime venv bin”消融；在此之前不把它当作已验证缺陷。

### H6 — control terminal backpressure

H4 仍触发 terminal notification delivery gap，并通过 public `turn/read` recovery
完成。benchmark recovery 保证了评分链路，但 `SlowConsumerError` 的核心
backpressure 仍需独立调查，不能用恢复路径掩盖。

## 第二阶段手工诊断波次

本波次由主调度者手工启动三个独立容器；没有新增 scheduler 或状态机。每题只在模型
terminal、官方 verifier、隔离检查、artifact seal 和停止保留全部完成后计为有效。

| Case | 有效 Trial | Reward | 归因与停止决定 |
|---|---|---:|---|
| `adaptive-rejection-sampler` | `akasic-bench-v4flash-diagnostic-adaptive-rejection-sampler-20260730-062118-021442` | 1 | 官方 verifier 9/9；未发现 Agent 缺陷，停止 |
| `bn-fit-modify` | `akasic-bench-v4flash-diagnostic-bn-fit-modify-20260730-062128-586452` | 0 | 有向边 tuple 按 `to,from` 表头直写，语义反转；属于模型/Agent 结构化产物校验失败，单例不足以改生产行为 |
| `break-filter-js-from-html` | `akasic-bench-v4flash-diagnostic-break-filter-js-from-html-20260730-062139-894279` | 0 | 公开检查命令空输出且退出 0，模型随后手工探索至 40 轮仍未解决；task 可观测性缺陷与停止策略信号混杂，不做定向修复 |

三题都保持候选源码 digest、正式 workspace owner 和线上进程不变；容器停止但保留，
便于继续检查运行时。该波次的 `1/3` 只描述三个诊断样本，不外推为 baseline。

`bn-fit-modify` 暂存一个跨任务可证伪假设：对带方向或坐标语义的结构化产物，在提交前
做一次从最终文件反向读取的语义核对。只有后续不同任务出现同类“内部推理正确、序列化
方向错误”时才设计通用 treatment。

`break-filter-js-from-html` 暂存两个正交假设：

1. 文档声明为 validator 的命令若没有产生任何断言或可观察结果，退出 0 不能自动视为
   强验证证据；
2. 连续多轮没有缩小失败空间时，Agent 需要显式总结已排除条件并判断是否停止。

当前 case 的 helper 本身缺少可执行入口，无法把失败单独归因给 Agent，因此两项都等待
跨 case 复现，不修改 task、不注入答案，也不为该题增加特殊规则。

## H7 — manifest 不能记录自身的稳定 digest

### 假设

> `campaign-manifest.json` 先对自身求 hash、再写入包含该 hash 的最终内容，会使记录值
> 必然失效；从 artifact digest 集合排除 manifest 自身，可以恢复证据完整性，同时
> 不改变模型、任务、verifier 或 Agent 行为。

Baseline 在本波次及前置 smoke 共四个真实 trial 复现：manifest 记录的自身 SHA-256
与最终磁盘文件均不一致，其他 artifact 未出现同类问题。

Treatment 只从 `artifacts.digests` 排除 `campaign-manifest.json`。targeted tests
覆盖“manifest 不自引用、其他 artifact 仍记录 digest”，相关 benchmark/runtime
测试共 124 项通过。真实 treatment：

- Trial：`akasic-bench-v4flash-smoke-regex-log-20260730-064531-898385`；
- reward `1`，preflight、source isolation 和 online invariant 全部通过；
- manifest 记录的其余 18 个 artifact digest 与磁盘 SHA-256 全部一致；
- 容器停止保留。

H7 是 benchmark 证据链的功能性修复，不改变 Agent 语义，也不能解释或改善 task
reward。

## 第三批手工诊断波次

按数据集目录的固定顺序选择三个未跑 case，仍由主调度者手工启动三个独立容器：

| Case | 有效 Trial | Reward | 归因与停止决定 |
|---|---|---:|---|
| `build-cython-ext` | `akasic-bench-v4flash-diagnostic-build-cython-ext-20260730-065406-145240` | 0 | verifier 9/11；40 轮上限以未完成进度总结收尾 |
| `build-pmars` | `akasic-bench-v4flash-diagnostic-build-pmars-20260730-065406-441568` | 1 | verifier 通过；停止 |
| `build-pov-ray` | `akasic-bench-v4flash-diagnostic-build-pov-ray-20260730-065406-683196` | 1 | verifier 通过；停止 |

三题 lifecycle、source isolation、online invariant、artifact seal 和停止保留均通过。
该波次的 `2/3` 只描述三个诊断样本，不外推为 baseline。

`build-cython-ext` 的直接失败证据：

1. Agent 已读取 `ccomplexity.pyx`，其中存在 NumPy 2.x 不支持的 `np.int`；
2. 后续搜索和批量替换却只覆盖 `*.py`；
3. Agent 没有核对“搜索范围与修改范围”的差集，转去修复 planarity 相关 repo test；
4. 第 40 轮仍未收束，runtime 生成进度总结；官方 verifier 的
   `test_ccomplexity` 和一个 repository test 失败。

它与 `break-filter-js-from-html` 共同证明“固定 40 轮可以替代真实完成条件”是可达的，
但不能单独证明取消上限会修复 Agent 的验收与优先级问题。

## H8 — benchmark 取消固定 iteration 上限

### 假设

> benchmark 专用 `max_iterations=40` 截断了仍在产生有效进展的任务；使用已有
> `0 = unlimited` 配置语义，同时保留 840/900 秒外层 deadline，可以让同一冻结 case
> 自主完成。

单变量 treatment 只把 `benchmark/harbor_v4flash/config.toml` 的
`max_iterations` 从 `40` 改为 `0`。生产默认、正式 profile、模型、prompt、工具、
task、verifier 和外层 deadline 均不变。当前源码 smoke：

- Trial：`akasic-bench-v4flash-smoke-regex-log-20260730-070217-141758`；
- 7 requests，reward `1`；
- source/online isolation、artifact seal 和停止保留全部通过。

`build-cython-ext` treatment：

- Trial：`akasic-bench-v4flash-diagnostic-build-cython-ext-20260730-070526-127111`；
- 从 40 requests 继续到 46 requests，并自主返回 final response；
- verifier 从 9/11 改善到 10/11，但 reward 仍为 `0`；
- 第 40 轮之后完成 planarity 修复和 repo tests，却仍未修改
  `ccomplexity.pyx` 的 `np.int`；
- final response 声称所有 Cython extensions 和 18 项 core tests 全部通过，与官方
  verifier 的直接失败证据冲突。

结论：H8 证明固定上限影响执行轨迹，但否定“取消上限足以解决本题”的强假设。
`max_iterations=0` 保留为隔离 benchmark 的实验配置；它改变停止、时延和成本语义，
不能描述成生产 Agent 的无语义鲁棒性修复。该 case 到此停止，不继续靠增加轮数重试。

新的跨 case 候选是假设“Agent 缺少任务要求、搜索范围、修改范围与直接验收证据之间的
闭环”。后续 treatment 必须是通用、可证伪的行为实验，并包含一个正常 control；
不得为 Cython 文件扩展名或 hidden verifier 写定向规则。

## 参考实现对账

- Codex reference commit：`c7a4a7e136d96554e1fc6f66532e6060fd2aaf15`；
- ordinary Responses API agent request 不设置对应的固定 output 上限；
- tool execution 只接受结构化 `FunctionCall`，畸形 arguments 会显式返回模型错误；
- framing 使用 `max_message_len + 1`，超过协议边界时 fail-loud。

以上只用于选择安全的 treatment，不把 Codex 的实现本身当作本项目实验结果。

## H9 — task deadline 与 retained network pool

### Task deadline

旧 controller 把所有 Agent turn 固定为 `900s`，没有读取 Terminal-Bench task
`[agent].timeout_sec`。这会让官方允许的长任务在 task 自身 deadline 前被 harness
截断。treatment：

- 从 Harbor task schema 读取 `agent.timeout_sec`；
- SDK turn 使用同一 deadline；
- Harbor 外层 Agent timeout 只增加 `120s` cleanup reserve；
- 非正数和损坏 task 配置在创建 trial 前 fail-loud。

该改动只对齐 benchmark lifecycle，不改变生产 Agent 默认值、task、模型或 verifier。
实现为 draft PR `#251`。

### Retained network pool

保留容器时同时保留 Compose 默认 network，Docker 自动地址池为每个 trial 分配 `/20`；
连续诊断后地址池耗尽，新 trial 在容器创建前失败。treatment 使用专用
`10.240.0.0/16`，每 trial 预留一个带 benchmark owner labels 的 `/28` external
network。空 endpoint network 可以在保存 inspect 证据后释放，container 和 volume
不得随之删除。

- 释放前证据：
  `benchmark-runs/_infra/network-release-20260730-081843.json`；
- 只删除 14 个同时满足 benchmark owner、旧 `/20`、`Containers={}` 的 network；
- 没有删除 container 或 volume，30 个 stopped-retained benchmark container 保留；
- treatment smoke：
  `akasic-bench-v4flash-smoke-regex-log-20260730-074701-402789`，
  reward `1`，新 network 为 `10.240.104.160/28`；
- public Gate 8/8：
  `docker/debug/reports/change-gate/20260730-162533-0cb71d94`。

实现为 draft PR `#252`。这是 retained benchmark infra 的容量与失败封口修复，不改变
Agent 行为。

## Caffe 诊断停止点

- Trial：
  `akasic-bench-v4flash-diagnostic-caffe-cifar-10-20260730-075112-133971`；
- task/turn deadline `3600s`，Harbor 外层 `3720s`，`max_iterations=0`；
- 冷 setup 中 `uv venv` 约 `370s`，完整 runtime setup 约 `391s`；
- 官方 CIFAR 下载约 50～60KB/s，预计 47～53 分钟，已经超过本次 task 剩余预算。

同一个 turn 还暴露两个独立 Agent/tool 问题：

1. Agent 三次从同一官方 URL 重试，每次都删除已有 partial tarball，从零开始；
2. Agent 明确请求 `run_in_background=true, timeout=3600`，shell 工具却返回
   `timeout_s=600`。

在三次相同来源、相同清零策略后，主调度者通过 public control interrupt 停止该
attempt。turn 终态为 `interrupted`，持续 `1,604,816ms`；Harbor 仍完成 evidence
seal，source unchanged、online passed、container stopped-retained、17 个 artifact
digest。缺少 `verifier/reward.txt` 是中断后的预期结果，因此该 attempt 不计 reward，
不继续原样重跑。

`3600 → 600` 已归因为 shell tool contract 的功能性问题：普通前台命令仍保留
`600s`，显式后台 timeout 在既有 4 小时 lifecycle 内按请求保留，超过上限直接拒绝，
不再静默夹断。36 个 shell 定向测试和 public Gate 8/8 通过；实现为独立 draft PR
`#253`。这项兼容行为修复不能掩盖 Agent 删除 partial download 的决策问题。

## H10 — 复用不可变 benchmark runtime

### 假设

> 如果跨 trial 只复用 Python、venv 和冻结依赖，并让每个 task container 只读挂载，
> 就能消除重复的 runtime 网络安装，同时保持 task、HOME、workspace、trace、源码、
> secret 和 verifier 初态独立。

treatment 使用显式 builder 创建按 recipe digest 命名的 Docker volume：

`akasic-bench-runtime-v1-79ea7f8bd2cbcb92b44062c0`

recipe 冻结 requirements、uv binary/version、Python `3.13.7`、平台、builder image
ID 和带 distribution hashes 的 resolved lock。trial 必须显式传入该 volume；
cache miss、manifest/lock 不匹配、额外 volume 或 `RW=true` 都 fail-loud，不自动
回退冷安装。CIFAR、Caffe source 和其他 task 下载物明确不进入共享 cache。

真实兼容性探针在 `regex-log` 和 `caffe-cifar-10` 两个不同 task image 中只读挂载
同一 volume，并成功导入 `cryptography`、`numpy`、`scipy`、`sklearn` 和
`sqlite_vec`。

第一次完整 smoke：

- Trial：
  `akasic-bench-v4flash-smoke-regex-log-20260730-084754-347608`；
- 在模型请求前 fail-loud；
- Docker 真实 `Mounts[].Source` 是宿主
  `/var/lib/docker/volumes/.../_data`，volume 名称位于 `Mounts[].Name`；
- mock 把 source 当 name，未覆盖真实结构；
- container stopped-retained、source unchanged、online passed，不计 reward。

修复 projection 后的完整 smoke：

- Trial：
  `akasic-bench-v4flash-smoke-regex-log-20260730-084951-336991`；
- reward `1`，总计 `179.233s`；
- environment setup `0.900s`，Agent setup `34.577s`，Agent execution
  `109.777s`，verifier `23.253s`；
- 同题旧冷路径 Agent setup `40.820s`，本次减少 `6.243s`，约 `15.3%`；
- source unchanged、online passed、container stopped-retained；
- 19 个 artifact digest，`missing=[]`；
- isolation preflight 记录一个 allowlist volume 且 `RW=false`。

`regex-log` 的直接 pair 只证明 setup 降低约 15.3%；Caffe 的 `391s` 冷 setup
说明慢网络下潜在收益更大，但尚未用 Caffe 做完整 cached pair，因此不能把两者直接
当作严格消融结果。实现为 draft PR `#254`；最终 public Gate 8/8：
`docker/debug/reports/change-gate/20260730-165409-a577680e`，private Gate 仍为
`pending_maintainer`。

## 全量诊断遍历检查点

更新时间：2026-07-30 23:13（Asia/Shanghai）

这轮是混合 treatment 下的诊断遍历，用来发现 Akashic Agent 与 harness 的通用问题，
不是最终分数。每个 task 使用独立 Docker environment 和 `/app` workspace；维护者
已批准把并发硬上限从 3 提高到 6，当前按全局最多 6 个真实 task container 手动调度。
共享项只有不可变、只读的 runtime volume：

`akasic-bench-runtime-v1-79ea7f8bd2cbcb92b44062c0`

核心 artifact 根目录为：

`/mnt/data/coding/akasic-agent-worktrees/benchmark-runs`

有效结果必须同时满足：Harbor lifecycle 正常结束、外部 verifier 实际运行并写出
reward、trace/turn-result/manifest 已封存、source 与 online 隔离检查通过。超时、
人为中断、provider 错误和仍在运行的 trial 只算“已触达”，不进入通过率。

### 当前计数

| 指标 | 数量 |
|---|---:|
| Dataset 总数 | 89 |
| 已触达 | 55 |
| 有效结果 | 44 |
| 有效通过 | 28 |
| 有效失败 | 16 |
| 暂不计分 | 11 |
| 尚未触达 | 34 |

`filter-js-from-html` 的首次 attempt 因 verifier 下载 `uv` 断流而无效；重跑
`akasic-bench-v4flash-smoke-filter-js-from-html-20260730-124859-743829`
已正常执行 verifier，两个测试失败，因此计入有效 `reward=0`。

### 任务索引

这里记录的是每题最新、可用于诊断的核心结果。trial 名可以直接在 artifact 根目录
下查找，避免重新扫描所有历史 attempt。

| Task | 状态 | Reward | 初步归因 / 处理 | 最新 trial |
|---|---|---:|---|---|
| `adaptive-rejection-sampler` | 有效 | 1 | 通过；停止 | `...062118-021442` |
| `bn-fit-modify` | 有效 | 0 | Agent：输出边方向与表头语义相反 | `...062128-586452` |
| `break-filter-js-from-html` | 有效 | 0 | 模型：安全绕过能力；Agent 测试解释次要 | `...062139-894279` |
| `build-cython-ext` | 有效 | 0 | Agent：搜索范围漏掉 `.pyx` 中的 `np.int` | `...070526-127111` |
| `build-pmars` | 有效 | 1 | 通过；停止 | `...065406-441568` |
| `build-pov-ray` | 有效 | 1 | 通过；停止 | `...065406-683196` |
| `caffe-cifar-10` | 暂不计分 | — | 手动中断；下载重试删除 partial，另暴露 shell timeout 问题 | `...075112-133971` |
| `cancel-async-tasks` | 有效 | 1 | 通过；停止 | `...040813-024970` |
| `chess-best-move` | 有效 | 1 | 通过；停止 | `...071317-060896` |
| `circuit-fibsqrt` | 有效 | 1 | 通过；停止 | `...071317-245614` |
| `cobol-modernization` | 有效 | 1 | 通过；停止 | `...093103-982111` |
| `code-from-image` | 有效 | 1 | 通过；停止 | `...095611-321488` |
| `compile-compcert` | 有效 | 1 | 通过；停止 | `...093105-796149` |
| `configure-git-webserver` | 有效 | 1 | 通过；停止 | `...101854-109233` |
| `constraints-scheduling` | 有效 | 1 | 通过；停止 | `...101855-618691` |
| `count-dataset-tokens` | 有效 | 1 | 通过；停止 | `...101856-809053` |
| `crack-7z-hash` | 有效 | 1 | 通过；停止 | `...103027-310152` |
| `custom-memory-heap-crash` | 有效 | 1 | 通过；停止 | `...103028-883213` |
| `db-wal-recovery` | 有效 | 0 | Agent：未生成目标 `/app/recovered.json` 就结束 | `...042053-226551` |
| `distribution-search` | 有效 | 1 | 通过；停止 | `...103029-250691` |
| `dna-assembly` | 有效 | 0 | 模型：Golden Gate 领域约束未满足 | `...110025-535688` |
| `dna-insert` | 有效 | 0 | Agent：搜索候选与最终写入候选不一致 | `...113818-729922` |
| `extract-elf` | 有效 | 0 | Benchmark：oracle 的 signed/unsigned 口径有歧义 | `...110027-158216` |
| `extract-moves-from-video` | 暂不计分 | — | 1805s task timeout；保留 trace 与中间产物 | `...115742-599841` |
| `feal-differential-cryptanalysis` | 有效 | 1 | 通过；停止 | `...115743-791764` |
| `feal-linear-cryptanalysis` | 暂不计分 | — | 1805s task timeout；保留 trace 与中间源码 | `...115744-046447` |
| `filter-js-from-html` | 有效 | 0 | Agent/模型：XSS 漏检且改写 5/12 个 clean HTML | `...124859-743829` |
| `financial-document-processor` | 有效 | 1 | 通过；停止 | `...123141-258087` |
| `fix-code-vulnerability` | 有效 | 1 | 通过；停止 | `...042053-683463` |
| `fix-git` | 有效 | 0 | Agent：冲突后停在 `UU` 并请求用户介入 | `...123142-146878` |
| `fix-ocaml-gc` | 暂不计分 | — | Agent 完成；官方 verifier 在 3600s 超时 | `...131750-790941` |
| `gcode-to-text` | 有效 | 0 | 模型：读取 metadata shortcut，未从几何恢复文字 | `...123734-863702` |
| `git-leak-recovery` | 有效 | 1 | 通过；停止 | `...123735-799649` |
| `git-multibranch` | 有效 | 1 | 通过；停止 | `...124814-185022` |
| `gpt2-codegolf` | 暂不计分 | — | 905s timeout；反复重写不完整实现，未进入 verifier | `...124815-372510` |
| `headless-terminal` | 有效 | 1 | 通过；停止 | `...125343-755608` |
| `hf-model-inference` | 有效 | 1 | 通过；停止 | `...125344-124726` |
| `install-windows-3.11` | 有效 | 0 | verifier 3/4；键盘操作未产生要求的视觉变化，待 trace 归因 | `...131751-264234` |
| `kv-store-grpc` | 有效 | 1 | 网络恢复后重跑通过；停止 | `...131753-510273` |
| `large-scale-text-editing` | 有效 | 1 | 网络恢复后重跑通过；停止 | `...132920-184342` |
| `largest-eigenval` | 暂不计分 | — | Docker Hub TLS EOF，镜像未拉取且未创建容器 | `...130550-692405` |
| `llm-inference-batching-scheduler` | 有效 | 0 | 待读 trace 归因；不先归到模型或 Agent | `...140807-464965` |
| `log-summary-date-ranges` | 有效 | 1 | 通过；停止 | `...140808-885933` |
| `mailman` | 有效 | 1 | 通过；停止 | `...140810-558360` |
| `make-doom-for-mips` | 暂不计分 | — | 905s task deadline；未进入有效 verifier | `...141114-239624` |
| `make-mips-interpreter` | 有效 | 0 | 待读 trace 归因；不先归到模型或 Agent | `...143013-416980` |
| `mcmc-sampling-stan` | 暂不计分 | — | 旧路径在 task image manifest 请求处 Docker Hub EOF | `...141552-139415` |
| `merge-diff-arc-agi-task` | 有效 | 1 | 通过；停止 | `...141703-042043` |
| `model-extraction-relu-logits` | 有效 | 1 | 通过；停止 | `...143015-324716` |
| `modernize-scientific-stack` | 暂不计分 | — | Agent 完成；官方 verifier 在 600s 超时 | `...143016-641648` |
| `mteb-leaderboard` | 暂不计分 | — | 旧路径并发拉取大 image，environment start 在 600s 超时 | `...143017-409280` |
| `mteb-retrieve` | 有效 | 0 | 待读 trace 归因；不先归到模型或 Agent | `...143018-042684` |
| `multi-source-data-merger` | 暂不计分 | — | 旧路径 environment start 在 600s 超时 | `...143020-079675` |
| `openssl-selfsigned-cert` | 有效 | 0 | Agent：输出日期格式违反原始合同仍宣告完成 | `...035855` |
| `regex-log` | 有效 | 1 | 通过；停止 | `...115050-533926` |

### 有效失败的初步分层

当前 12 个有效失败中，已有 11 个完成初步分层：

1. Agent/Akashic 执行策略 7 个：`bn-fit-modify`、`build-cython-ext`、
   `db-wal-recovery`、`dna-insert`、`filter-js-from-html`、`fix-git`、
   `openssl-selfsigned-cert`；
2. 模型能力 3 个：`break-filter-js-from-html`、`dna-assembly`、
   `gcode-to-text`；
3. benchmark/oracle 歧义 1 个：`extract-elf`。

新增的 `install-windows-3.11` 已证明 verifier 有效执行，失败点是键盘输入没有造成
要求的视觉变化；在读取 Agent trace 前保持“待归因”，不先归到模型或 Agent。

这不是“只有 7 个值得改 Agent”。模型主因 case 也可能暴露过程控制问题，例如错误
测试入口被当成成功、没有在最终回答前核对失败证据。当前隔离 workspace 中 Akasha
记忆为空，没有证据表明这些失败由记忆召回或污染造成。

### H11 — 非交互式完成 Gate（预注册，尚未实施）

> 在 Agent 给出最终回答前，用同一个模型对“原始任务合同、真实工具结果、目标产物、
> 未解决错误”做一次轻量完成检查，若发现可行动缺口则继续当前 turn；能够减少
> “已经找到正确方向但没有交付有效产物”的失败，而不依赖隐藏 verifier 或
> benchmark 专用规则。

首轮消融候选冻结为 `bn-fit-modify`、`build-cython-ext`、`db-wal-recovery`、
`dna-insert`、`fix-git`、`openssl-selfsigned-cert`。control 与 treatment 保持同一
task、模型、effort、工具、workspace 初态和 timeout，只改变最终回答前是否执行完成
Gate。它会改变 Agent 在非交互任务中的结束语义，不是纯 infra 鲁棒性修复；必须先
实现通用机制并通过最小语义 Gate，再运行 pair，不能依据 verifier 细节写规则。

截至本检查点，已落地的纯 harness/infra 修复包括：

- `#255`：并发 Gate 校验真实 Docker bind source；
- `#256`：凭据使用模板，secret 值不进入持久化配置；
- `#257`：Agent workspace 与官方 task workspace 对齐到 `/app`；
- Harbor 本地 patch `59e76ec`：secret 值不进入 Docker Compose argv；22:08 检查
  发现 benchmark venv 默认仍从未补丁 checkout 导入，之后的 controller 显式使用
  `PYTHONPATH=.../harbor-secret-env/src`，并以宿主 `/proc/*/cmdline` 负向扫描验证。

这些修复不改变 task 解法或 Agent 推理语义；H11 仍处于预注册状态。

### 21:06 运行时检查点

`gpt2-codegolf` 在 905 秒外层 deadline 到达时仍在反复重写不完整的 `/app/gpt2.c`，
没有生成 `turn-result.json`，也没有进入 verifier。trace 已封存；最终 5,908 字节的
`gpt2.c` 另存为：

`akasic-bench-v4flash-diagnostic-gpt2-codegolf-20260730-124815-372510/diagnostic/gpt2.c`

这个 attempt 暂不计 reward，但它支持“缺少 deadline 感知、checkpoint 和策略切换”
这一 Agent 侧诊断，不支持简单增加 timeout：最后一版源码仍包含明确未完成的内存
布局，并未处于接近可验证完成的状态。

宿主经 Mihomo 路由访问 `registry-1.docker.io`、`pypi.org` 和 `astral.sh` 时出现 TLS
握手 EOF；DeepSeek API 仍可访问。影响边界为：

- `install-windows-3.11` 的 Agent lifecycle 完成，但 verifier 因无法安装 `uv/uvx`
  而失败，不能把 reward `0` 归因给 Agent；
- `kv-store-grpc` 在 Docker image pull 阶段失败，没有创建 task container；
- 尚未触达的 48 个 task image 当前均未缓存，因此在该网络恢复前不继续制造同类
  无效 attempt。

已删除本波次 6 个 stopped、可重复创建的 task container 及对应空 network；
`gpt2.c`、`filter.py`、trace、verifier 输出、manifest 和 digest 均已留存，task
images 与不可变 runtime volume 保留。线上 gateway PID、start ticks 和命令行在
各 manifest 的前后快照中一致。

### 21:38 网络恢复与续跑

维护者切换节点后，宿主探针恢复为 Docker Hub `401`、GitHub `200`、PyPI `200`、
DeepSeek `401`。未同步的 ledger commit 随后成功推送，线上 gateway PID 和启动时间
未变化。

先重跑三个纯 infra 无效题：

- `kv-store-grpc`：7/7 verifier tests 通过，reward `1`；
- `fix-ocaml-gc`：Agent 完成后进入官方 verifier，仍在下载 oracle repo；
- `install-windows-3.11`：Agent 完成后进入官方 verifier，正在运行 `uvx` 测试。

`kv-store-grpc` 释放的槽位已由 `large-scale-text-editing` 接续，当前全局仍为 3 个
独立 task container。controller 使用受管长连接 session；一次 `nohup` 启动被宿主
回收并留下两个 `prepared` container，已在确认 Agent 未启动后删除对应孤儿 container
和空 network。该事件不计 task attempt，也不改变 harness 源码。

execution-budget treatment 的 `regex-log` probe 同样发生在旧网络故障窗口：原有效
control 仍为 reward `1`，本次 verifier 因无法下载 `uvx` 得到的 `0` 不进入消融比较，
必须在网络恢复后重新跑 pair。

### 22:12 六槽调度与有效终态

`large-scale-text-editing` 重跑通过，`install-windows-3.11` 的官方 verifier 完整执行后
得到 `reward=0`；两者的 source、online PID 和容器停止检查均通过。唯一有效结果口径
更新为 37 题，其中 25 通过、12 失败。

并发上限变更已提交 draft PR `#258`（commit `cf17ab59`），19 个 targeted tests、
public change-impact Gate 和远端 `contract`/`locked-runtime` checks 通过；PR 不等待
合并，不阻塞 discovery。新源码的 `regex-log` smoke 因官方 verifier 冷下载超过
900 秒得到 `VerifierTimeoutError`，只记 invalid infra，Agent trace 保留。另一个
`large-scale-text-editing` source-bound smoke 正在运行，用于打开 concurrency=6 Gate。

当前首次 discovery 为 `llm-inference-batching-scheduler`、
`log-summary-date-ranges`、`mailman` 和 `make-doom-for-mips`；`fix-ocaml-gc` 仍在前一
官方 verifier。调度只按全局 active container 数补槽，不让多个 controller 各自的
局部 semaphore 叠加越过 6。

### 23:13 task image / Git 基础设施复用

旧六槽路径在 trial 内各自承担 task image 拉取和 Git/CA 安装，真实出现两个
`EnvironmentStartTimeoutError`；`modernize-scientific-stack` 另在官方 verifier
阶段超时。该批只有三个有效结果：

- `model-extraction-relu-logits=1`；
- `make-mips-interpreter=0`；
- `mteb-retrieve=0`。

H12 treatment 不修改 task、verifier、instruction、模型或 Agent 策略：

1. campaign 创建 trial 前按唯一 image reference 预拉取，registry 并发最多为 2；
2. 本地完整 image 直接复用，trial compose 固定不可变 image ID 并设置
   `pull_policy=never`；
3. Git/CA 一次安装进内容寻址 Docker volume，记录 builder、包版本、内容摘要和
   manifest digest，之后只读挂载；
4. Git cache 再次执行 build 时直接命中本地校验结果，不重新运行 apt；
5. setup 仍从固定 source bundle 恢复真实历史，所有 Git 操作保持 fail-loud。

Git volume：

`akasic-bench-git-v1-e4706df27562c97921ac2a6c`

真实兼容性探针在 Debian 11/12/13 和 Ubuntu 24.04 的官方 task image 中完成
`git init + bundle fetch + rev-parse`。`cancel-async-tasks` source-bound smoke：

- Trial：`akasic-bench-v4flash-smoke-cancel-async-tasks-20260730-150438-090296`；
- reward `1`；
- environment setup `0.892s`，Agent setup `1.544s`；
- task image `cache_hit=true`、`pull_attempts=0`；
- runtime/Git 两个 volume 都是 `RW=false`；
- setup artifact 中没有 `apt-get`、`apk add` 或 `yum install`；
- source unchanged、online PID unchanged、container/network 在 artifact seal 后删除。

实现为 stacked draft PR `#259`（commit `3ac8634d`）。35 个 targeted tests 通过，
Pyright 为 0 errors；相邻 base public Gate 通过：
`docker/debug/reports/change-gate/20260730-230948-607bbb55`，private Gate 为
`pending_maintainer`。对 `origin/main` 的累计 Gate 因前序 stacked PR 的受保护合同
变化报 `protected_contract_mixed`，不冒充本提交通过。

运行公开 Gate 后，本地生成的忽略目录仍属于 `upload_dir` 的输入，因此 source digest
从 smoke 时的 `sha256:6c4d...` 变为 `sha256:2e8a...`。下一批六题在创建 trial 前被
source-bound Gate 正确拒绝；没有制造无效 container。必须先对新摘要重跑单题 smoke，
再打开六槽，不手工绕过 Gate。

### 23:48 逐题记录与 runtime 状态隔离

当前已触达 65 个唯一 task；逐题 TSV 检查点为 33 个 `reward_pass`、17 个
`reward_fail`、4 个 `agent_deadline`、7 个 `infra_invalid`、3 个
`oracle_blocked` 和 1 个仍在运行的 `active_prepared`。pass/fail 只在同时存在
官方 CTRF 和 reward 时成立，无 CTRF 的 `reward=0` 不冒充有效失败。

本波新有效结果包括：

- `nginx-request-logging=1`、`polyglot-c-py=1`、`polyglot-rust-c=1`、
  `prove-plus-comm=1`；
- `password-recovery=0`；
- `path-tracing=0` 与 `path-tracing-reverse=0` 均只失败 image/oracle 相关断言，
  trace 分别暴露自测度量不一致和产物副作用不一致；
- `portfolio-optimization=0` 只失败性能断言，且 wall-clock oracle 受同机并发负载
  影响，暂记 mixed；
- `overfull-hbox` 的 Agent 已生成 PDF 并把日志中的 Overfull 清到 0，但官方
  verifier 360 秒内未产出 CTRF/reward，记 oracle blocked，不记模型失败。

另发现 Akasic 的 durable workspace 与官方 task root 同为 `/app`，会把
`sessions.db`、`memory/`、`plugin-data/`、socket 和 lock 写入题目树。该状态不跨题，
但会污染全目录扫描类 oracle。纯 infra treatment 保持 Agent 工具 cwd 为 `/app`，
仅把 durable workspace/socket 移到 `/opt/akashic-workspace`；与 driver 失败后总是
shutdown 的生命周期修复完成累计 49 个 benchmark tests 和相邻 public Gate。driver
修复为 stacked draft PR `#260`；workspace 分离需等旧 source 的最后一个任务排空后
跑 source-bound smoke，再决定是否发布 stacked draft PR。

线上 gateway 仍为 PID `162463`、启动时间 `2026-07-30 09:53:27 +0800`，命令行和
workspace 未变化。

### 02:31 89/89 诊断遍历检查点

89 个 Terminal-Bench 2.1 task 均已至少触达一次，逐题 TSV 当前共 89 条。计入最后两个
已到达、等待 TSV owner 回填的终态后，本检查点分布为：

- 46 个 `reward_pass`；
- 26 个 `reward_fail`；
- 9 个 `agent_deadline`；
- 5 个 `infra_invalid`；
- 3 个 `oracle_blocked`；
- 0 个仍在运行的 task。

按主要问题层级归因：

- 46 个通过，`primary_layer=none`；
- 20 个以 Agent 的规划、工具执行、验证或停止闭环为主；
- 5 个以 LLM 的领域推理或解法合成为主；
- 9 个为 Agent、LLM、外部依赖或 oracle 共同作用的 mixed；
- 3 个以 harness/runtime infra 为主；
- 6 个以 benchmark oracle 或官方 verifier 为主。

这里的 89/89 表示诊断遍历已经覆盖全部题目，不表示最终官方评测完成。记录混合了
不同 wave 的有效结果和被明确隔离的 infra/oracle 无效 attempt；
优化后的 fresh 89-task official eval 尚未运行，因此当前记录不能计算或冒充最终
Terminal-Bench 分数，也不能与公开的 56.9% 基准直接比较。

wave 10–12 的主要结果进一步收窄了归因边界：

- portable runtime control 修复了旧 task image 中的 GLIBC ABI 不兼容，
  `qemu-alpine-ssh` 与 `qemu-startup` 重跑均通过；
- `caffe-cifar-10`、`multi-source-data-merger` 在运行环境修复后通过，证明旧失败不应
  留在 Agent 侧；
- `largest-eigenval`、`mteb-leaderboard` 暴露 mixed 问题，
  `mcmc-sampling-stan`、`video-processing` 暴露 Agent 执行闭环或证据覆盖问题；
- `make-mips-interpreter`、`train-fasttext`、`tune-mjcf` 和
  `write-compressor` 在有效运行中耗尽 deadline；这些记录支持后续通用策略消融，
  但不支持为具体题目增加提示或规则；
- Torch 两题的 Agent turn 已完成，官方 verifier 仍停在大体积冷下载并超时，没有
  CTRF/reward，继续按 infra invalid 隔离。

### 02:31 隔离控制与事件消费根因

本轮纯 infra 链路保持 task 指令、verifier、模型和 Agent 策略不变：

1. task image 与 Git 内容寻址 volume 复用由 `#259` 承担；Git 仍从固定 source
   bundle 恢复，volume 只读挂载，不把宿主 checkout 暴露给 task；
2. `#260` 保证 driver 任意终态后都关闭 gateway；
3. `#261` 保持 Agent 工具 cwd 为 `/app`，只把 durable workspace、socket、
   `sessions.db`、memory 和 plugin-data 移到 `/opt/akashic-workspace`，避免污染
   task root；
4. `#262` 在 control buffered delta replay 中协作让出调度，但不合并、不丢弃、
   不重排 delta；
5. `#263` 让共享 runtime 按 manifest、lock、Python 版本和 ABI 做可移植校验，
   避免复用宿主 ABI 偶合的产物。

`winning-avg-corewars` 把事件拥塞拆成了两个独立层次。第一次修复只覆盖 control
replay；第二次仍出现 SDK notification queue overflow，说明问题不在 delta 数量本身。
`#264` 让 SDK reader 在连续路由已缓冲通知时协作让出调度。随后 trace 又证明
runtime driver 读取一个 event 后等待 `turn/read`，会暂停 stream 消费；
`#265` 因此改为持续 drain event stream，并与 `turn/read` 共同观察终态。两层修复都
保留原 event、sequence、terminal 和 `SlowConsumerError` 语义，不用扩大队列或合并
provider delta 掩盖背压。

最终 continuous-drain 重跑
`akasic-bench-v4flash-smoke-winning-avg-corewars-20260730-181015-097845`
的 source `unchanged=true`、online check 通过、resource classification 为 `none`，
并从 event stream 收到 1,259 个事件后正常完成 terminal。官方 CTRF 为 2/3、
reward `0`，唯一失败是相对 stone/snake 的性能阈值。transport treatment 因此通过；
trace 中模型共使用 65 次请求和 80 次工具调用，最终回答还明确知道 62% 低于 75%、
12% 低于 33%，却没有合成满足阈值的交付，主因转为 LLM/domain synthesis，而不是
SDK、control 或 driver 传输失败。

### 02:31 资源与凭据证据

`sam-cell-seg` 的旧 attempt 在 4 GiB memory cgroup 中把任务进程推近上限，gateway
与任务进程处于同一 cgroup，OOM 后 driver 表面只得到 `ConnectionClosedError`。
这既有 Agent 资源行为的触发因素，也有 harness 丢失真实失败原因的问题，不能直接算
模型失败。`#266` 在 driver 成功、失败或启动失败的生命周期边界读取固定 cgroup v2
白名单并保存 `memory.current`、`memory.peak` 和 `memory.events`，同时保持原始异常为
主失败。

第一轮 resource-evidence 重跑
`akasic-bench-v4flash-smoke-sam-cell-seg-20260730-174923-561855` 的官方 verifier
实际为 9/9、reward `1.0`，cgroup 记录为 4 GiB limit、约 2.68 GB peak、
`oom_kill=0`。但是运行期间宿主候选源码从
`sha256:bd16...` 变为 `sha256:45b1...`，manifest 的 source Gate 得到
`unchanged=false`；因此该结果只证明本次容器没有 OOM，不能进入逐题有效结果。固定在
稳定 source bundle 后的重跑
`akasic-bench-v4flash-smoke-sam-cell-seg-20260730-182419-562086`
最终 source `unchanged=true`、online check 通过、resource classification 为
`none`；memory peak 为 429,051,904 bytes、`oom_kill=0`，event terminal 共 550 个
事件，官方 CTRF 9/9、reward `1.0`。这次稳定重跑完成了 source Gate 与资源归因闭环。

凭据模板先前已经避免把 secret value 写入持久化 config，但真实运行检查发现
`docker exec --env NAME=value` 仍会把值短暂暴露在宿主进程 argv。`#267` 将真实值只放
进当前 Docker client 的子进程环境，exec argv 只保留 `--env NAME`，并以宿主
`/proc/*/cmdline` 负向扫描作为验收；这修复 secret transport，不改变 Agent 可见的
环境变量和值。

### 02:31 draft PR 与归因边界

本轮后半段形成的 draft PR 均未合并，当前远端 `contract` 与 `locked-runtime`
checks 为 green：

| PR | 职责 | base |
| --- | --- | --- |
| `#260` | driver 退出后总是关闭 runtime | `perf/benchmark-portable-git` |
| `#261` | durable workspace 与 task root 分离 | `fix/benchmark-driver-finally-v2` |
| `#262` | control delta replay 协作让出调度 | `fix/benchmark-workspace-separation-v2` |
| `#263` | shared runtime ABI 可移植校验 | `fix/control-delta-cooperative-replay-v2` |
| `#264` | SDK notification reader 协作让出调度 | `fix/benchmark-portable-runtime-abi` |
| `#265` | driver 持续消费 terminal event stream | `fix/sdk-reader-cooperative-drain-v2` |
| `#266` | 采集 cgroup resource-limit 证据 | `fix/benchmark-portable-runtime-abi` |
| `#267` | secret value 不进入宿主 argv | `fix/benchmark-resource-evidence-v2` |

每题都使用独立 Docker 实例和新的 `/opt/akashic-workspace`，Akasha memory 不跨题，
且当前 harness 只向 Akasic 发起一个 programmatic turn。因此这些 trace 可以归因
single-turn 的规划、工具使用、验证、停止行为，以及 SDK/control/harness/runtime
问题；不能据此声称测到了跨 turn 的 Akasha 记忆形成、巩固、遗忘或长期召回能力。
相反，fresh workspace 也排除了历史记忆污染作为本轮失败原因。最终 89 题 eval 只有
在待定语义改变得到维护者裁决、纯 infra PR 固定并从统一 source 重跑后才能开始。
截至本检查点，所有 task container 和对应 network 已清理；线上 gateway 仍为 PID
`162463`、start ticks `45439946`，命令行与正式 workspace 未变化。

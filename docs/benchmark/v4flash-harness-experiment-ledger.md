# V4 Flash Harness 实验 Ledger

首次日期：2026-07-30；最近更新：2026-08-05

状态：第一阶段、89 题 discovery 和 H2/H3/H4/H7 定向实验已完成

设计合同：
[V4 Flash 完整 Runtime Harness Benchmark 设计](../spark/2026-07-30-v4flash-harness-benchmark-design.md)

逐题证据：
[V4 Flash Terminal-Bench 2.1 逐题诊断](v4flash-terminalbench-89-case-diagnostics.md)

## 冻结变量

| 变量 | 当前值 |
|---|---|
| Dataset | `terminal-bench/terminal-bench-2-1`，89 tasks |
| Smoke task | `openssl-selfsigned-cert` |
| Harbor | `v0.16.1@137c27874df6163309c6c0cb218a56a7b0e00e79` |
| Model | `deepseek-v4-flash` |
| Effort | `high` |
| Context / max output | `1,000,000 / provider default`；H4 前 baseline 为 `8,192` |
| Runtime | 完整 Akasic gateway，Akasha enabled |
| 调用入口 | public Python SDK / control socket |
| Agent / verifier timeout | `900s / task official value` |
| Retention | Docker stop，不 delete；artifact 和 workspace 复制留证 |
| Concurrency | smoke 为 `1`；Gate 打开后硬上限 `3` |

## Official Harbor infra 对账

Harbor 的 lifecycle owner 仍负责 environment start、agent setup、agent run、外部
verifier 和 stop。自定义 agent 只补 Akasic runtime 自身安装与 SDK 调用。

| 层 | 必需依赖 | Preflight / 失败语义 |
|---|---|---|
| Host control | Docker Engine、Compose、Harbor pin、dataset artifact、uv binary | 缺失即在建实例前失败 |
| Image OS | Linux/POSIX shell；setup 阶段可使用 root | Windows 或无 root setup 不进入当前 campaign |
| Package manager | 缺少 Git 时支持 `apk`、`apt-get` 或 `yum` | 无受支持 manager 时 agent setup 失败 |
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

## H7 — Shell 统一 execution 消除累计日志轮询

### 假设

> 把前台、自动后台和累计 `task_output` 改为 Codex 风格的统一 execution，并让
> `write_stdin` 只返回增量输出，可以减少长任务的无信息轮询和重复上下文；它不应
> 改变 task、模型、verifier 或容器资源，也不保证模型会选择正确任务策略。

对照是 89 题 discovery 中的 `train-fasttext`：`45 shell + 107 task_output +
1 task_stop`，共 154 次工具调用，最终 3600 秒 TIMEOUT。Treatment trial 为
`akasic-bench-v4flash-smoke-train-fasttext-20260731-155617-339973`，源码来自独立
bundle，使用独立 Docker runtime/workspace，模型、effort、task、verifier 和资源
不变。

Treatment 结果：

- `27 shell + 17 write_stdin`，连同文件工具共 59 次工具调用；
- 同一 `execution_id` 连续存活约 15 分钟，300 秒空等待后仍能续接；
- turn 在 3600 秒超时，未进入 verifier；没有 resource failure；
- 模型在已有训练精度不足后继续尝试 900 秒与 1800 秒 autotune，并在 deadline
  前开启新的预处理，故未收口属于任务策略，而非 shell execution 丢失；
- treatment 还暴露两个真实 shell 语义问题：Python 隐式 `/bin/sh -c` 与 Codex
  用户 shell argv 不一致，以及 pipeline 上游失败可能被末端过滤命令掩盖。

接受“统一增量协议显著减少累计日志轮询”假设；拒绝“因此该 case 会通过”的外推。
随后按 Codex reference commit
`c7a4a7e136d96554e1fc6f66532e6060fd2aaf15` 把 shell detection、login argv 和直接
process spawn 纳入同一候选，并迁移适用的核心测试。pipeline 不自动注入
`pipefail`，需要上游失败语义的命令必须显式声明。

## 参考实现对账

- Codex reference commit：`c7a4a7e136d96554e1fc6f66532e6060fd2aaf15`；
- ordinary Responses API agent request 不设置对应的固定 output 上限；
- tool execution 只接受结构化 `FunctionCall`，畸形 arguments 会显式返回模型错误；
- framing 使用 `max_message_len + 1`，超过协议边界时 fail-loud。

以上只用于选择安全的 treatment，不把 Codex 的实现本身当作本项目实验结果。

## 2026-08-04 · OpenCode Go Max 自动化轨道

维护者明确要求把当前本机 OpenCode Go 订阅用于独立的新轨道，并以自动完成、官方
时限、官方 verifier 和有界 Docker 占用为优先级。它不改写上文 High 轨道的历史
冻结变量，也不与历史 attempt 拼分。

```text
┌──────────────┐   原始 task.toml 时限   ┌────────────────┐
│ Campaign WAL │ ───────────────────────▶ │ Akashic turn   │
└──────┬───────┘                          └───────┬────────┘
       │ 逐题 fsync                                │ 任意终态
       ▼                                           ▼
┌──────────────┐                          ┌────────────────┐
│ 可恢复进度   │ ◀──── official reward ── │ Harbor verifier│
└──────────────┘                          └───────┬────────┘
                                                ▼
                                      冷证据后定向 Docker cleanup
```

冻结值与失败语义：

- provider/model 为 `opencode-go/deepseek-v4-flash`，`reasoning_effort=max`，
  `max_output_tokens=0`；正式 `config.toml` 不改写，只从其中读取当前 route 密钥。
- Harbor setup 继续使用独立 900 秒上限。`Agent.run` 从入口开始消费 task
  `[agent].timeout_sec`；gateway readiness 与 turn 共用该预算。Harbor 外层只增加
  120 秒收束保留，不给模型增加解题时间。verifier 使用 task 原始
  `[verifier].timeout_sec`。
- `timed_out` 和已经开始后的 `agent_failed` 仍停止 runtime、采集证据并进入 Harbor
  官方 verifier；gateway readiness、配置或 controller 故障标为 infra，不进入有效
  attempt。failed turn 的 control usage 合法为 `null`，此时 Harbor token 字段保持未知，
  但 metadata、轨迹和 verifier 流程不得因此失败。
- campaign 的默认和硬上限都是 4 并发。该值来自 OpenCode Go 当前端点的受控探针
  （短请求 6 并发均成功但未返回公开 rate-limit header）后的保守选择，不宣称是官方
  硬限制。明确的 provider 429 记为
  `provider_rate_limited` 基础设施无效 admission，不把其 verifier `reward=0` 混入
  pass@1；最多三次 admission，按 30 秒起步指数退避并加入逐题稳定抖动。等待退避时
  释放 semaphore slot，使其他题可以继续。明确的 provider 500/502/503/504 同样按
  transient infra 处理；`GoUsageLimitError` 则停止该题并等待额度 reset 或维护者启用
 余额，不做短间隔 admission 重试。
- 新 campaign 只按每题 `task.toml [agent].timeout_sec` 生成 Longest Processing Time
  first 贪心队列，官方预算更长的任务先进入四个 slot；历史 harness、provider 表现和
  旧耗时完全不参与顺序。冻结后的顺序与官方时限写入 manifest，resume 原样恢复。
- campaign 使用 append-only `events.jsonl`，每条事件写入后执行 `fsync`；
  `accepted-results.json` 是从 WAL 原子再生的当前有效数据投影，最终 manifest 继续保存
  全部 accepted outcomes 与汇总分数。进程若在两者之间中断，续跑时以 WAL 为权威重建，
  不重复接受 task。
  `--resume-campaign-dir` 只在源码、task 集合、并发和 dataset 身份完全相同时续跑。
- 每题创建前同时检查 artifact 文件系统、`/tmp` 和 Docker Root，默认低水位分别为
  20 GiB、2 GiB 和 20 GiB。默认 `retention=none` 在证据与 verifier 落盘后，只按
  当前 project 的容器 ID 和 managed network ID 删除终态现场；不运行全局 prune，
  不自动删除 image 或共享 cache volume。需要复盘文件层时显式使用
  `retention=failures` 或 `all`。
- campaign 取消时，各 trial 使用精确 project 容器 ID 执行 10 秒有界 stop，再复用同一
  managed network 身份检查删除现场；campaign manifest/WAL 记录 `interrupted`，不把
  中断误写成完成。
- `--dataset-dir` 稳定发现直接子目录的 89 个 task，并冻结逐题 digest 与有序集合
  digest。没有独立核对外部 revision 时，manifest 必须写
  `provenance=unverified_local_copy`，不能把本地目录名冒充官方来源证明。

Maka PR #1719 的公开最终账本采用相同的证据原则：task-native deadline ×1、官方
verifier、accepted dataset 与 attempts WAL 分离、只替换 infrastructure-invalid
admission、轨迹保留而容器可删除。本轨道没有照搬其最高 4 task-pair / 8 cell 并发，
而是按 OpenCode Go 订阅容量冻结为 4 个 Akashic cell。

真实 smoke：

- Trial：`akasic-bench-v4flash-smoke-prove-plus-comm-20260804-015952-307817`；
- task agent/verifier 时限：`900s / 900s`；实际 route 为 OpenCode Go Max；
- Akashic 经 public SDK 完成 10 轮、10 次工具调用并持久化 terminal turn；
- Harbor 官方 `reward.txt=1`，manifest artifact 缺失为 0，源码和线上 owner 不变；
- 冷证据落盘后，唯一停止容器和 managed network 已按不可变 ID 删除，残留计数为 0；
- 运行前记录 `/mnt/data` 约 81 GiB、`/tmp` 约 14 GiB 可用。本次补充核对 Docker
  Root 所在文件系统约 42.9 GiB 可用。

首个无效预跑同时发现新主分支要求空 workspace 先初始化 `memory/VEDA.md`。harness
现通过仓库 `veda-reset` 正式入口初始化每题独立 workspace；readiness timeout 与题目
deadline 使用不同终态，避免 infra 故障伪装成有效超时并打开 concurrency Gate。

已知边界：当前凭据仍由 gateway 进程环境注入，题内进程可观察同容器进程环境；旧
trace 也可能包含模型自行打印的环境内容。本轨道暂不把宿主受限 provider proxy 作为
运行前置，但发布或共享 artifact 前必须先执行 secret scan，并建议轮换历史暴露凭据。

### 2026-08-04 · verifier 依赖准备与官方评分计时分离

本地全量运行暴露出 `mteb-retrieve` 和 `torch-tensor-parallelism` 的官方
`tests/test.sh` 在 pytest 前通过 apt、curl 和 uvx 下载依赖；网络下载耗尽 900 秒后，
测试正文尚未开始。这类结果不能与真实断言失败混为一类。

- Agent turn 结束并冻结候选后，harness 从原始 `test.sh` 提取 pytest 前的安装段；
  uvx 使用完全相同的 Python 与 `-w` 依赖集合，但以 `python -c 'pass'` 只完成解析、
  下载和环境缓存，不运行测试正文。
- 依赖准备拥有独立 14,400 秒基础设施上限，最多两路并发；它发生在 Harbor
  `VERIFICATION_START` 之前。随后仍由 Harbor 执行未改写的官方 `test.sh`，并使用
  task 原始 `[verifier].timeout_sec`。因此这是“官方 verifier 与官方正文时限、下载
  不计时”的本地诊断口径，不冒充 Terminal-Bench 官方端到端 verifier 计时口径。
- 准备前后分别计算当前 task workdir 的确定性摘要；任何候选变化都 fail-loud，禁止
  进入评分。`agent/verifier-bootstrap.json` 记录准备时长、独立上限和前后摘要。
- 补验只选择 Agent 已正常完成、首次 verifier 在 pytest 前被依赖下载耗尽的两题。
  已有真实 pytest 断言失败、真实 Agent timeout、无候选输出或 provider failure 的题
  不因这次计时修正重跑。

补验启动后，OpenCode Go 两路请求同时发生 `RemoteProtocolError`，最终由 runtime
fallback 生成“模型未返回可用回复，请重试。”，没有有效 delta、工具调用或 usage。
该状态现在归为 `provider_transient`：跳过 verifier 及其大依赖下载，保留 trace 和候选
摘要后按 campaign admission 策略重试，不能再伪装成 completed reward 0。维护者随后
明确把 `path-tracing-reverse` 纳入补验；其原 attempt 的 provider 断流持续占满 Agent
预算，因此本次与前两题一起作为新的三题 campaign 运行，不改写旧记录。

### 2026-08-04 · 三题补验切换 DeepSeek 官方 API

OpenCode Go smoke 的首个模型流持续无有效事件直至 task agent deadline。进一步核对
发现旧 credential loader 无条件读取正式配置的 `llm.main`；当前该引用为
`deepseek_main`，却把得到的 DeepSeek 官方 key 命名为 `OPENCODE_GO_API_KEY` 并注入
OpenCode Go endpoint。该次 smoke 因 provider、endpoint 与凭据身份不一致而无效，
保留冷 trace 后停止，不能进入三题补验或计入分数。

维护者明确接受把三题补验切换到 DeepSeek 官方 API。候选轨道因此冻结为
`deepseek/deepseek-v4-flash`、`https://api.deepseek.com/v1`、
`reasoning_effort=max`；credential loader 按明确的 `deepseek_main` runtime 读取并
注入 `DEEPSEEK_API_KEY`，不再根据可变的 `llm.main` 猜测 route。新 smoke 与后续每道
题都创建独立 trial，并从 task 官方 agent deadline 重新计时；旧 OpenCode Go 结果不与
该 provider 阶段拼成同一 pass@1。

### 2026-08-05 · 补验终态与逐题审计

Max 全量主 campaign 经 `025734 → 030932 → 052229 → 104726 → 114103` 恢复，最新投影
为 `88/89 accepted`、59 个 reward `1`。逐项 trace 审计后，三个 runtime fallback 被旧
harness 错收为 reward `0`，`pytorch-model-recovery` 又在 verifier 下载约 2.6 GiB
PyTorch/CUDA 依赖时耗尽 900 秒且没有 accepted outcome。`torch-tensor-parallelism` 的
v7 已获得完整 Agent 官方时限并超时，按维护者确认的规则记有效失败，不再补模型时限。
固定 89 题集合最终如实报告为 59 pass、30 not-pass，即 `59/89 = 66.3%`。其中三个
provider fallback 和一个 verifier 下载失败继续保留具体基础设施归因，但不改变分母，
本 PR 不再补跑任务。

三题补验中只有 `mteb-retrieve` 形成有效模型失败。`path-tracing-reverse` 的 v7 在 Agent
deadline 后发生 gateway 与 verifier 生命周期重叠，未替换主运行的真实 Agent timeout；
`torch-tensor-parallelism` 的 v7 Agent 超时后残留 `apt-get` 持有 dpkg lock，导致
verifier 依赖准备 exit `100`。该题仍按 Agent timeout 记失败；v7 campaign 自身终态为
`failed`、`accepted=0/2`，不形成独立分数。

完整口径、三题证据和 89 题机器可读索引见
[2026-08-05 运行审计](terminalbench-2.1-run-audit-2026-08-05.md)与
[逐题 CSV](terminalbench-2.1-case-results-2026-08-05.csv)。

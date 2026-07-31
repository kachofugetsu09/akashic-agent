# V4 Flash Harness 实验 Ledger

首次日期：2026-07-30；最近更新：2026-08-01

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

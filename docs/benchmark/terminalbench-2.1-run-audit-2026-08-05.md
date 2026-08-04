# Terminal-Bench 2.1 运行审计

日期：2026-08-05

逐题机器可读数据：
[terminalbench-2.1-case-results-2026-08-05.csv](terminalbench-2.1-case-results-2026-08-05.csv)

## 1. 结论

本报告保留两条互不拼分的轨道：

| 轨道 | Provider / effort | 有效结果 | 结论 |
|---|---|---:|---|
| 2026-07-31 历史全量诊断 | DeepSeek V4 Flash / `high` | 89/89 | `63/89 = 70.8%` |
| 2026-08-04 定向补验 | DeepSeek 官方 API / `max` | 1/3 | 只有 `mteb-retrieve` 得到可计分结果，reward `0`；另外两题基础设施无效 |

历史全量结果仍是当前唯一完整的 89 题成绩。定向补验不是一次新的全量运行，不能把
`mteb-retrieve` 的新结果替换进历史 High 轨道，也不能把三个 nominal reward `0`
写成 Max 轨道 `0/3`。

## 2. 历史 89 题

历史逐题证据来自
[V4 Flash Terminal-Bench 2.1 逐题诊断](v4flash-terminalbench-89-case-diagnostics.md)：

| 分类 | 数量 |
|---|---:|
| `PASS`，含 `PASS + RESOURCE` | 63 |
| `ASSERT` | 13 |
| `TIMEOUT` | 12 |
| `TIMEOUT + RESOURCE` | 1 |
| 合计 | 89 |

CSV 的每一行都保留历史状态、reward、失败摘要和 trial 名。`historical_*` 列只描述
2026-07-31 High 轨道；`revalidation_*` 列只描述 2026-08-04 Max 定向补验。

## 3. Max 定向补验

### `mteb-retrieve`：有效模型失败

- Trial：`akasic-bench-v4flash-diagnostic-mteb-retrieve-20260804-144354-124543`
- Agent 正常完成，回答的是 `HumanEval: Benchmarking Python code generation via
  functional examples`；期望答案为 `MTEB: Massive Text Embedding Benchmark`。
- 官方 verifier 为 1 pass、1 fail，reward `0`。
- verifier 依赖准备耗时约 1066 秒，发生在评分计时前；准备前后候选摘要一致。
- 分类：`valid_model_failure`。这次失败可以用于分析模型策略，不需要因基础设施原因替换。

### `path-tracing-reverse`：provider 与生命周期污染

- 首个 Max trial：`akasic-bench-v4flash-diagnostic-path-tracing-reverse-20260804-144356-248789`。
  模型流中断后 runtime 生成 fallback 文本，Agent 没有创建 `/app/mystery.c`；该 nominal
  reward `0` 不应被 campaign 接受为真实模型结果。
- 再次运行：`akasic-bench-v4flash-diagnostic-path-tracing-reverse-20260804-152820-788304`。
  Agent 达到原题 1800 秒 deadline 后，gateway 仍继续执行并与 verifier 阶段重叠。
  verifier 最终写出 reward `0`，但候选和进程生命周期已经不再满足独立评分前提。
- 分类：`invalid_provider_and_lifecycle`。这两个结果都不进入 Max 分母。
- 下一步：deadline 必须先终止 gateway 及其子进程，确认 task workdir 冻结后才能启动
  verifier；随后使用新 trial 重跑。

### `torch-tensor-parallelism`：verifier 基础设施无效

- 首个 Max trial：`akasic-bench-v4flash-diagnostic-torch-tensor-parallelism-20260804-144358-056485`。
  模型流中断后只得到 fallback，未形成候选；nominal reward `0` 无效。
- 再次运行：`akasic-bench-v4flash-diagnostic-torch-tensor-parallelism-20260804-152822-805189`。
  Agent 在原题 900 秒 deadline 前生成了 `/app/parallel_linear.py`，但其残留 `apt-get`
  继续持有 `/var/lib/dpkg/lock-frontend`。verifier 依赖准备以 exit `100` 失败，未执行
  官方评分，因此没有 reward。
- 分类：`invalid_verifier_infrastructure`。该结果不进入 Max 分母。
- 下一步：修复 Agent 超时后的进程组清理，再从独立 trial 重跑；若只补验冻结候选，也必须
  先证明候选摘要不变且没有 Agent 进程存活。

## 4. Harness 审计发现

```text
┌──────────────┐   deadline   ┌────────────────────┐
│ Akashic turn │ ───────────▶ │ 停止 gateway/子进程 │
└──────────────┘              └─────────┬──────────┘
                                       │ workdir 冻结
                                       ▼
                              ┌────────────────────┐
                              │ verifier 依赖准备   │
                              └─────────┬──────────┘
                                       ▼
                              ┌────────────────────┐
                              │ 官方 verifier 计时  │
                              └────────────────────┘
```

当前实现已经覆盖 WAL 恢复、最长官方时限优先、四并发上限、provider transient
重试、磁盘低水位、依赖准备与 verifier 计时分离，以及按精确 Docker project 清理。
这轮运行仍暴露两个不能靠结果投影修补的问题：

1. fallback 文本会把 provider 断流伪装成正常 completed turn，随后得到 nominal
   reward `0`；provider 失败必须在 accepted-results 之前分类。
2. Agent deadline 只终止等待者，不保证 gateway 及其题内进程已经停止；verifier
   必须以“进程已停且候选已冻结”为硬前置条件。

因此 v7 campaign 的终态是 `failed`、`accepted=0/2`。它正确地没有形成新分数，
但仍说明 lifecycle cleanup 需要继续修复。

## 5. 证据入口

运行根目录：
`/mnt/data/coding/akasic-agent-worktrees/benchmark-runs/official-automation-max-20260804`

| 证据 | 相对运行根目录的位置 |
|---|---|
| 三题 Max campaign | `_campaigns/akasic-bench-v4flash-campaign-20260804-144353/` |
| 两题 v7 campaign | `_campaigns/akasic-bench-v4flash-campaign-20260804-152819/` |
| Campaign WAL | `<campaign>/events.jsonl` |
| Accepted 投影 | `<campaign>/accepted-results.json` |
| Agent trace | `<trial>/agent/trace.jsonl` |
| Agent deadline 终态 | `<trial>/agent/driver-outcome.json` |
| Verifier 输出 | `<trial>/verifier/test-stdout.txt` |
| 完整结果与 artifact digest | `<trial>/campaign-manifest.json` |

运行 artifact 不提交进 Git；报告和 CSV 只记录可审阅索引与结论。容器终态由每个
trial manifest 的 `containers_stopped` 记录，清理策略不运行全局 Docker prune。

# Terminal-Bench 2.1 Max 全量运行审计

日期：2026-08-05

逐题机器可读数据：
[terminalbench-2.1-case-results-2026-08-05.csv](terminalbench-2.1-case-results-2026-08-05.csv)

## 1. 结论

2026-08-04 运行的是 DeepSeek V4 Flash、`reasoning_effort=max` 的 89 题全量轨道。
最新主 campaign 投影为 `88/89 accepted`、`59` 题 reward `1`。trace 审计发现三个
provider fallback 被错收成 reward `0`，另有一题在 verifier 下载阶段耗尽时限；
`torch-tensor-parallelism` 的补跑则按真实 Agent timeout 记失败。

固定 89 题集合的最终报告为：**通过 59 题，`59/89 = 66.3%`**。

逐题失败原因如下：

| 分类 | 数量 |
|---|---:|
| 官方 verifier reward `1` | 59 |
| 有效 reward `0`，含真实 Agent timeout | 26 |
| Provider fallback | 3 |
| Verifier 依赖下载未进入测试正文 | 1 |
| 合计 | 89 |

本报告不因失败原因改变分母，也不再使用 `59/85`。Provider、verifier、harness 和
Agent timeout 只用于解释失败及后续改进；当前 PR 不继续重跑任务。

历史 High 全量 `63/89 = 70.8%` 只保留在 CSV 的 `historical_high_*` 对照列，不是
本报告的主结果，也不能与 Max attempt 拼接。

## 2. 全量 campaign 恢复链

运行根目录：
`/mnt/data/coding/akasic-agent-worktrees/benchmark-runs/official-automation-max-20260804`

| Campaign | Task set | Accepted | Reward 1 | 终态 |
|---|---:|---:|---:|---|
| `20260804-025734` | 89 | 0 | 0 | 为修 harness 主动中断 |
| `20260804-030932` | 89 | 13 | 11 | 中断，WAL 保留 |
| `20260804-052229` | 89 | 85 | 59 | failed，缺 4 题 |
| `20260804-104726` | 89 | 85 | 59 | failed，继续恢复 |
| `20260804-114103` | 89 | 88 | 59 | failed，缺 `pytorch-model-recovery` |

这些 campaign 是同一 Max 全量轨道的恢复过程，不是五次独立 benchmark，也不能把
重复 attempt 相加。逐题 CSV 以 `114103` 的 88 个 accepted outcome 为主投影，再应用
trace 审计和有效补验。

## 3. 四个基础设施失败结果

### Provider fallback：3 题

| Case | 主运行现象 | 当前结论 |
|---|---|---|
| `code-from-image` | 做过 OCR 工具工作，最后只生成“模型未返回可用回复，请重试。” | provider fallback，旧 harness 错收 reward `0` |
| `modernize-scientific-stack` | 只生成同一 fallback | 基础设施无效 |
| `overfull-hbox` | 只生成同一 fallback | 基础设施无效 |

这三题没有可接受的模型终态，在固定 89 题报告中按未通过记录。修正后的 harness
应避免再次产生同类误分类，但当前 PR 不重跑这些任务。

### Verifier 依赖下载：1 题

`pytorch-model-recovery` 三次都在 verifier 内下载约 2.6 GiB PyTorch/CUDA 依赖时耗尽
900 秒，官方测试正文没有开始，campaign 因此始终没有 accepted outcome。该题在固定
89 题报告中按未通过记录，失败原因归为 verifier 基础设施。

## 4. 已补验和额外诊断

### `mteb-retrieve`：有效失败

- 替代 Trial：`akasic-bench-v4flash-diagnostic-mteb-retrieve-20260804-144354-124543`；
- verifier 依赖准备发生在评分计时前，准备前后候选摘要一致；
- 官方 verifier 1/2，Agent 回答 HumanEval，期望 MTEB；
- reward `0` 是有效模型失败，已经计入固定 89 题结果。

### `path-tracing-reverse`：保留原始 Agent timeout

- 主运行 Trial：`akasic-bench-v4flash-diagnostic-path-tracing-reverse-20260804-075349-176596`；
- 原始结果是达到题目 1800 秒 Agent deadline，reward `0`，当前按真实 timeout 计入；
- 额外 v7 Trial 在 deadline 后发生 gateway 与 verifier 重叠，属于无效诊断，没有替换
  原始结果。

### `torch-tensor-parallelism`：按真实 Agent timeout 记失败

- v7 Trial：`akasic-bench-v4flash-diagnostic-torch-tensor-parallelism-20260804-152822-805189`；
- Agent 获得完整 900 秒官方时限后超时，按本轨道规则直接记有效失败，不再给模型补时；
- Agent 残留 `apt-get` 让后续 verifier 无法取得 dpkg lock。该 lifecycle 缺陷仍要修，
  但不会触发本题再次运行 Agent。

## 5. Harness 审计发现

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

当前候选已经覆盖 WAL 恢复、最长官方时限优先、四并发上限、provider transient 重试、
磁盘低水位、verifier 依赖准备与评分计时分离，以及按精确 Docker project 清理。这轮
运行仍证明两个前置条件尚未成立：

1. runtime fallback 不能代表 completed model turn；accepted projection 必须先读取
   provider 终态并把 transient admission 排除；
2. Agent deadline 后必须证明 gateway 和题内子进程已经停止、候选 workdir 已冻结，
   才能开始 verifier。

v7 campaign 的 `failed`、`accepted=0/2` 是额外诊断终态，不是 Max 全量得分。

## 6. 证据入口

| 证据 | 相对运行根目录的位置 |
|---|---|
| Max 主投影 | `_campaigns/akasic-bench-v4flash-campaign-20260804-114103/` |
| 三题补验 | `_campaigns/akasic-bench-v4flash-campaign-20260804-144353/` |
| 两题 v7 诊断 | `_campaigns/akasic-bench-v4flash-campaign-20260804-152819/` |
| Campaign WAL | `<campaign>/events.jsonl` |
| Accepted 投影 | `<campaign>/accepted-results.json` |
| Agent trace | `<trial>/agent/trace.jsonl` |
| Agent deadline 终态 | `<trial>/agent/driver-outcome.json` |
| Verifier 输出 | `<trial>/verifier/test-stdout.txt` |
| 完整结果与 artifact digest | `<trial>/campaign-manifest.json` |

运行 artifact 不提交进 Git；报告和 CSV 只记录可审阅索引与结论。逐题 CSV 以 Max
状态开头，历史 High 只作为末尾对照列。

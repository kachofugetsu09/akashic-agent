# V4 Flash Benchmark 待定语义改变

日期：2026-07-30

状态：89 题诊断遍历完成前只记录，不实施；最终由维护者统一定夺

关联：
[实验 Ledger](v4flash-harness-experiment-ledger.md) ·
[诊断循环设计](../spark/2026-07-30-agent-benchmark-diagnostic-loop-design.md) ·
[0010](../decisions/0010-provider-default-output-and-benchmark-diagnostics.md)

## 记录边界

本文件只保存会改变 Agent 通用停止、规划、工具能力或跨轮状态的候选 treatment。
task image 预拉取、协议 framing、证据封存、隔离和 timeout owner 对齐等不改变任务
解法的 infra 修复继续直接实施、验证和重跑，不进入这里。

任何候选都不得读取 hidden verifier、写 task 特例或根据单题答案设计规则。遍历结束后，
每项必须同时给出失败 case、正常 control、允许变化、受保护行为、成本和可回滚实现；
证据不足的候选直接关闭，不为了提高 benchmark 分数实施。

## 当前候选

| ID | 候选 | 已观察证据 | 会改变的语义 | 当前决定 |
|---|---|---|---|---|
| SEM-001 | 非交互式完成 Gate | `bn-fit-modify`、`build-cython-ext`、`db-wal-recovery`、`dna-insert`、`fix-git`、`openssl-selfsigned-cert` 出现“宣告完成但产物或直接验收仍失败” | 模型准备结束后，harness/Agent 可能继续同一 turn；增加请求、时延和成本，也可能推翻模型主动停止 | 待定；89 题结束后才选跨领域 treatment/control |
| SEM-002 | deadline-aware checkpoint 与策略切换 | `gpt2-codegolf`、`caffe-cifar-10`、`make-doom-for-mips` 在 deadline 前仍重复同类尝试或没有形成可恢复交付 | Agent 会感知剩余时间、保存中间结果、切换方案或提前明确失败；改变规划与停止行为 | 待定；先继续收集不同任务的超时轨迹 |
| SEM-003 | 验证证据强度 Gate | `break-filter-js-from-html` 的 validator 空输出且 exit 0；`build-cython-ext` 的搜索/修改范围与最终声称不一致 | Agent 不再把“命令退出 0”一律视为强证据，可能要求观察断言数、输出或重新读取产物 | 待定；与 SEM-001 高度重叠，先判断能否合并为一个通用合同 |
| SEM-004 | 向 task shell 暴露共享 runtime Python | `regex-log` 曾因 task PATH 没有 runtime Python 而自行安装工具 | task 获得原镜像未声明的 Python/依赖，直接扩大可用能力 | 暂不实施；不是纯 infra 等价 |
| SEM-005 | 共享 Git 的版本归一化 | 旧路径按 task distro 安装 Git；PR `#259` 当前共享 Git 为 `2.30.2`，而 Debian 12/Ubuntu 24.04 可安装更高版本 | task 和 Agent 看到的 Git feature set 可能减少或变化 | draft PR 不合并；遍历完成前记录真实兼容性，最终选择保持 distro 版本或提供不减能力的 portable 版本 |
| SEM-006 | 延长官方 verifier 总 timeout | `modernize-scientific-stack` 的 Agent 30 秒完成且真实输出正确，但 verifier 在 apt/uv 下载依赖时用尽 600 秒；`fix-ocaml-gc` 的 clean rebuild 用尽 3600 秒 | 改变官方 oracle 的资源预算，可能把官方设置的失败变成有效结果，也会显著增加机器占用 | 不实施；先各自无并发重跑一次，复现后记为 benchmark/oracle blocked |
| SEM-007 | verifier 共享 apt/uv 下载 cache | 多题在 `test.sh` 的依赖下载阶段超时，尚未进入 pytest | verifier 不再从官方初始环境启动，cache 命中会改变下载时序、可见包和外部网络依赖 | 不实施；先保留官方 verifier 原样，区分“Agent 已完成”和“oracle 未运行” |
| SEM-008 | agent deadline 终态的统计口径 | `gpt2-codegolf`、`make-doom-for-mips` 未在官方 Agent timeout 内完成，因而没有 verifier 结果 | 将该类终态记 `score=0` 或 `invalid` 会改变聚合分母和可比口径，但不改变 Agent 行为 | 诊断表保留独立状态 `agent_deadline_diagnostic_closed`；正式统计口径最终统一定夺 |

## 明确不作为候选

- 不跨 task 共享 Akasha 学习状态。每题独立 workspace、fresh memory 和单 turn 是
  benchmark 隔离的一部分；当前没有历史候选注入首轮模型，跨题学习会造成顺序依赖
  和信息泄漏，普通代码错误也不能归因于 Akasha 召回质量。
- 不解析模型输出的伪 function-call 文本并执行。只有结构化 tool call 可以越过执行
  边界。
- 不增加 task 特定 Prompt、答案模板、隐藏断言或 verifier 适配。
- 不为达到 100% 而无限重试同一模型策略；同类失败五次后必须先形成新证据或停止。

## 最终审阅输出

89 题全部形成有效结果后，本文件逐项更新为：

1. `accept`：通用收益有跨任务证据，正常 control 不回归，语义与成本已获维护者确认；
2. `reject`：只改善特定题、改变能力边界过大或反例否定；
3. `defer`：证据仍不足，保留明确的后续实验而不进入生产。

最终 eval 只在维护者确认接受项并完成独立 Gate 后运行；当前诊断遍历不把 treatment
混合成绩包装成正式 Terminal-Bench 分数。

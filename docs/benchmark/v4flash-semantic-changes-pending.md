# V4 Flash Benchmark 待定语义改变

日期：2026-07-30

状态：89/89 诊断遍历已收口；未实施的语义 treatment 由维护者统一定夺

关联：
[实验 Ledger](v4flash-harness-experiment-ledger.md) ·
[诊断循环设计](../spark/2026-07-30-agent-benchmark-diagnostic-loop-design.md) ·
[0010](../decisions/0010-provider-default-output-and-benchmark-diagnostics.md)

## 记录边界

本文件只保存会改变 Agent 通用停止、规划、工具能力或跨轮状态的候选 treatment。
task image 预拉取、协议 framing、证据封存、隔离和 timeout owner 对齐等不改变任务
解法的 infra 修复继续直接实施、验证和重跑，不进入这里。

这里的“不进入”不等于 `semantic_delta: none`。若修复改变可达错误分类、让原本误失败
的路径恢复成功、增加持久诊断 artifact 或改变 secret transport，应按项目合同标为
`compatible`；只有所有外部行为和持久结果都不变时才可标 `none`。

任何候选都不得读取 hidden verifier、写 task 特例或根据单题答案设计规则。遍历结束后，
每项必须同时给出失败 case、正常 control、允许变化、受保护行为、成本和可回滚实现；
证据不足的候选直接关闭，不为了提高 benchmark 分数实施。

## 当前候选

| ID | 候选 | 已观察证据 | 会改变的语义 | 当前决定 |
|---|---|---|---|---|
| SEM-001 | 非交互式 completion / evidence review | `bn-fit-modify`、`build-cython-ext`、`db-wal-recovery`、`dna-insert`、`fix-git`、`openssl-selfsigned-cert` 出现“宣告完成但产物或直接验收仍失败”；`regex-chess` 在结束前已看到明确红测 | 模型准备结束后，Agent 可能再执行一次同模型证据复核并继续同一 turn；增加请求、时延和成本，也可能推翻模型主动停止 | 待定；只考虑一次通用复核消融。拒绝按字符串、退出码或 benchmark 特例实现硬 Gate |
| SEM-002 | deadline-aware checkpoint 与策略切换 | `gpt2-codegolf`、`make-doom-for-mips`、`make-mips-interpreter`、`train-fasttext`、`write-compressor` 在 deadline 前仍重复同类尝试或没有形成可恢复交付；`caffe-cifar-10` 在 infra 修复后通过，不再作为 Agent 证据 | Agent 会感知剩余时间、保存中间结果、切换方案或提前明确失败；改变 prompt、规划与停止行为 | 待定；若批准，只做一次通用提醒的跨领域 control/treatment，不直接延长 deadline |
| SEM-003 | 任务合同 checklist | `build-cython-ext` 的搜索范围漏掉 `.pyx`；`dna-insert` 的搜索候选与最终产物不一致；`protein-assembly` 以占位猜测填补未证实结构 | 在 prompt 或完成阶段强制逐项核对原任务合同，改变注意力分配、token 和结束行为 | 合并进入 SEM-001 的同一语义家族，不另造功能；仅事后写诊断 checklist 可直接实施 |
| SEM-004 | 向 task shell 暴露共享 runtime Python | `regex-log` 曾因 task PATH 没有 runtime Python 而自行安装工具 | task 获得原镜像未声明的 Python/依赖，直接扩大可用能力 | 暂不实施；不是纯 infra 等价 |
| SEM-005 | 共享 Git 的版本归一化 | 旧路径按 task distro 安装 Git；PR `#259` 当前共享 Git 为 `2.30.2`，而 Debian 12/Ubuntu 24.04 可安装更高版本 | task 和 Agent 看到的 Git feature set 可能减少或变化 | draft PR 不合并；遍历完成前记录真实兼容性，最终选择保持 distro 版本或提供不减能力的 portable 版本 |
| SEM-006 | 延长官方 verifier 总 timeout | `modernize-scientific-stack`、`fix-ocaml-gc` 和多道 PyTorch 题在重跑中仍停在官方 verifier 下载、编译或冷依赖阶段 | 改变官方 oracle 的资源预算，可能把官方设置的失败变成有效结果，也会显著增加机器占用 | 拒绝用于当前诊断或正式成绩；保留 `oracle_blocked`，不把无 CTRF 的结果归到 Agent |
| SEM-007 | verifier cache / sidecar | 多题在 `test.sh` 的依赖下载阶段超时，尚未进入 pytest；当前 Harbor verifier 与 Agent 共用 task container | 挂载 cache 或预装依赖会改变初始文件系统与 Agent 能力；字节保持的代理 cache 仍会改变 oracle timing；纯观测 sidecar 只增加证据 | 共享挂载和预装：拒绝。下载观测 sidecar：可直接做。代理 cache：只允许独立 diagnostic profile，需维护者确认且不得混入 official score |
| SEM-008 | agent deadline 终态的统计口径 | `gpt2-codegolf`、`make-doom-for-mips` 未在官方 Agent timeout 内完成，因而没有 verifier 结果 | 将该类终态记 `score=0` 或 `invalid` 会改变聚合分母和可比口径，但不改变 Agent 行为 | 诊断表保留独立状态 `agent_deadline_diagnostic_closed`；正式统计口径最终统一定夺 |
| SEM-009 | exact invocation / side-effect 自检 | `path-tracing-reverse` 用 stdout 重定向完成自测，但官方合同要求原命令自身创建 `image.ppm`，最终只过 2/3 | 自动重放真实命令可能重复发送、删除、部署或其他外部副作用；只做提醒也会改变 Agent 完成行为 | 待定；最多评估 permission-aware 的模型自检提醒。拒绝 harness 自动重放任意命令 |
| SEM-010 | provider delta 合并或扩大事件队列 | `winning-avg-corewars` 的两次失败发生在 driver 等待 `turn/read`、停止持续消费时；事件可观测序列本身有效 | 合并 delta 会改变 chunk、时序和客户端观察；扩大队列会改变 `SlowConsumerError` 阈值与内存语义 | 拒绝；采用逐帧保序的 cooperative yield + continuous drain 功能性修复 |

## 明确不作为候选

- 不跨 task 共享 Akasha 学习状态。每题独立 workspace、fresh memory 和单 turn 是
  benchmark 隔离的一部分；当前没有历史候选注入首轮模型，跨题学习会造成顺序依赖
  和信息泄漏，普通代码错误也不能归因于 Akasha 召回质量。
- 不解析模型输出的伪 function-call 文本并执行。只有结构化 tool call 可以越过执行
  边界。
- 不增加 task 特定 Prompt、答案模板、隐藏断言或 verifier 适配。
- 不为达到 100% 而无限重试同一模型策略；同类失败五次后必须先形成新证据或停止。
- 不把共享 verifier cache 挂入 Agent 可见容器，也不把 verifier 依赖预装成 Agent
  起始能力。只观测下载阶段不改变语义；任何代理 cache 都必须使用独立 diagnostic
  profile 标注为不可与 official score 混合。
- 不自动重放任意“原始命令”做完成校验。外部发送、删除、部署和不可逆命令没有通用
  的安全重放语义。

## Akasha V2 归因边界

本轮每题使用 fresh、互不共享的 Akashic workspace，任务以单个 user turn 交给
Akasic public SDK。这个设计验证完整 runtime、显式记忆初始化、路径隔离和
read-before-write 接线，但没有提供跨 turn 的历史候选，因此普通单题编码失败不能
归因于 Akasha V2 的长期情景召回。

[Akasha V2 技术报告](https://github.com/kachofugetsu09/akasha-v2-engine/blob/main/docs/papers/akasha-v2-online-explicit-memory-system-2026-07-30.md)
将完整 user–assistant turn 作为原子节点，并在下一轮输入读取旧图后再写入本轮；要评估
记忆系统本身，应另做多轮、可控历史、read-before-write 的召回与污染消融，不能把
Terminal-Bench 单轮 reward 当成记忆质量指标。

## 维护者审阅输出

本轮 89 题已经形成可审计终态。维护者后续逐项更新为：

1. `accept`：通用收益有跨任务证据，正常 control 不回归，语义与成本已获维护者确认；
2. `reject`：只改善特定题、改变能力边界过大或反例否定；
3. `defer`：证据仍不足，保留明确的后续实验而不进入生产。

最终 eval 只在维护者确认接受项并完成独立 Gate 后运行；当前诊断遍历不把 treatment
混合成绩包装成正式 Terminal-Bench 分数。

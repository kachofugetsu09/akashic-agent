# Akasic Agent Benchmark 诊断与持续改进循环设计

日期：2026-07-30

状态：用户已逐节批准，实施中

关联设计：

- [V4 Flash 完整 Runtime Harness Benchmark 设计](2026-07-30-v4flash-harness-benchmark-design.md)
- [V4 Flash Harness 实验 Ledger](../benchmark/v4flash-harness-experiment-ledger.md)

## 1. 目标

本系统使用 Terminal-Bench 2.1 的 89 个 case 发现 Akasic Agent 与 harness 的通用
工程问题，持续完成以下循环：

1. 在独立环境中运行真实 Akasic + Akasha。
2. 封存 trace、runtime、verifier、usage 和隔离证据。
3. 让只读分析 agent 提出可证伪假设。
4. 由 Root Coordinator 审核归因、实验和停止条件。
5. 实施最小通用候选。
6. 通过 synthetic、case 消融、现实 controls 和项目 Gate 验证。
7. 对成立的功能性修复或鲁棒性优化提交 draft PR。
8. 对失败方向、模型边界、task 问题和语义待定项同等留痕。

Terminal-Bench 是工程诊断探针，不是产品目标。优化阶段不追求、计算或发布一个由不同
candidate、不同 attempt 最好结果拼成的总分。

最终完整 89 题 eval 不属于自动循环。只有用户明确授权后，才冻结最终 artifact，
从零运行一次完整协议。

## 2. 非目标与禁止行为

本设计明确不做：

- 为提高 Terminal-Bench 分数添加 task name、题面内容或 expected output 检测。
- 把 task 解法、文件名、命令、flag、verifier 细节写进生产代码或 prompt。
- 读取隐藏 verifier 来推导生产行为。
- 修改 dataset、task、verifier、reward、资源或 timeout 来制造提升。
- 选择性拼接多个 attempt 的最好结果。
- 把 invalid infra attempt 解释为模型或 Agent 能力失败。
- 为难以解决的 case 无限重复相同尝试。
- 把单题翻转直接宣传为通用 Agent 改进。
- 让 scoring runtime 读取其他 case 的 trace、分析结论或共享 Akasha。
- 在用户授权前执行优化后的正式完整 89 题 eval。
- 自动合并 PR、修改正式 workspace 或切换线上 runtime。

所有生产改动都必须能够在不提 Terminal-Bench 的情况下解释其现实价值。

## 3. 已批准的长期配置语义

`max_output_tokens` 的默认值改为 `0`：

- `0` 表示请求不发送 provider output-token 参数，由 provider/model 自身边界负责。
- `0` 不表示模型拥有无限输出。
- 正整数继续表示显式输出上限。
- 负数在配置边界 fail-fast。
- 缺少该字段的新配置、setup wizard、settings UI 和 runtime config 默认使用 `0`。
- 已经显式配置正整数的存量配置不自动改写。
- summary、标题生成或其他内部小任务可以继续拥有独立局部上限；这些局部上限不能被
  主 runtime 的 `0` 意外取消。

这是行为语义改变，不伪装成普通重构。用户已批准该默认变化。实施时同时更新
`projectneed` 的增量默认规则和存量配置解释。

## 4. 系统架构与角色

```text
┌──────────────────────────────────────────────────┐
│ Root Coordinator / LLM Gate                      │
│ 冻结协议、调度任务、审核归因、批准实验、决定停止 │
└───────────────┬──────────────────────────────────┘
                │ 最多并发 6
       ┌────────┼────────┐
       ▼        ▼        ▼
┌──────────┐ ┌──────────┐ ┌──────────┐
│ Case A   │ │ Case B   │ │ Case C   │
│ 独立容器 │ │ 独立容器 │ │ 独立容器 │
│ 独立Akasha│ │ 独立Akasha│ │ 独立Akasha│
└────┬─────┘ └────┬─────┘ └────┬─────┘
     │ terminal 后封存，不实时干预
     └─────────────┼─────────────┘
                   ▼
┌──────────────────────────────────────────────────┐
│ Evidence Store                                   │
│ trace、payload、usage、runtime、verifier、identity│
└───────────────┬──────────────────────────────────┘
                ▼
┌──────────────────────────────────────────────────┐
│ Terra High Analysis Pool                         │
│ 每个副手只读一个 case 或一个故障簇               │
└───────────────┬──────────────────────────────────┘
                ▼
┌──────────────────────────────────────────────────┐
│ Root LLM Gate                                    │
│ 复核结论，合并共同根因，拒绝 benchmark hack      │
└───────────────┬──────────────────────────────────┘
                ▼
┌──────────────────────────────────────────────────┐
│ 独立实现 Worktree                                │
│ 一个通用机制、一个 writer、可审阅 commit         │
└───────────────┬──────────────────────────────────┘
                ▼
┌──────────────────────────────────────────────────┐
│ Synthetic + Ablation + Controls + CI Gate        │
└───────────────┬──────────────────────────────────┘
                ▼
        Draft PR 或失败实验归档
```

### 4.1 Root Coordinator / LLM Gate

Root 是唯一实验调度者和最终归因 owner，职责包括：

- 冻结每个 attempt 的 candidate、配置、环境和 task identity。
- 控制六个 Docker slot。
- 指派只读 trace 分析。
- 核对分析 agent 的证据、反例和代码 owner。
- 判断问题分类、实验是否可证伪、是否继续尝试。
- 阻止 benchmark-specific 修改。
- 审核语义 delta、现实泛化证据和 PR 范围。
- 维护跨 case failure cluster。

分析 agent 的输出只是建议，不能自动触发代码修改。

### 4.2 Case Runtime

每个 attempt 使用独立 Docker 容器、workspace、HOME、配置、Akasha 和 runtime：

- 只接收标准 task instruction。
- 不接收其他 case 的 trace 或分析结论。
- 不共享 scoring Akasha。
- 不挂载正式 workspace 或 Docker socket。
- 不发布 host port。
- 任务进行期间不接受候选代码更新。
- 完成后先停止但不删除。

### 4.3 Analysis Pool

默认使用 Terra High 分析已封存 trace：

- 一个副手只读一个 case 或一个明确故障簇。
- 只输出证据、分类、假设、反例、最小复现和停止建议。
- 不修改 scoring 容器、verifier、task 或候选 worktree。
- 不在 attempt 结束前读取实时 pass/fail 并指导当前模型。
- 不把 task 解法写入生产建议。

父 agent 必须复核真实 trace 与代码，不能以多数投票替代判断。

### 4.4 实现与评审

- 每个候选使用独立 Git worktree。
- 同一语义合同、PR 和权威文档同时只有一个 writer。
- 一个 candidate 只测试一个通用机制。
- 独立只读 reviewer 检查 diff、权限、写集合、错误语义和 benchmark 特化。

## 5. Attempt 有效性 Gate

任何失败归因前，先证明 attempt 有效：

1. dataset、task、image、verifier 和资源身份正确。
2. candidate commit/tree、source digest 和 runtime config 已记录。
3. 实际 model route、effort 和请求策略已记录。
4. Akasic turn、tool lifecycle 和 terminal 状态可读取。
5. trace、usage、runtime log、verifier 和 result artifact 齐全。
6. 容器停止，正式 workspace、线上 PID 和源码摘要未变化。

失败处理：

- runner、镜像、网络、provider 或 setup 失败标记为 `invalid_infra`。
- turn 已持久化但 terminal notification 丢失时，记录 delivery gap，并用权威
  `turn/read` 收束；backpressure 作为独立问题保留。
- 中断 attempt 永不与后来重跑结果拼接。
- invalid attempt 不完成该 case 的首次扫描，修复后需要重新运行。

## 6. 证据模型

每个有效 attempt 至少记录：

- case、attempt、generation 和 run identity。
- candidate commit、tree、bundle/image/container digest。
- task/image/verifier/dataset identity。
- 实际模型 route、effort、payload policy 和响应元数据。
- terminal、exception、verifier、duration 和 usage。
- 工具调用序列与 task 文件实际变化。
- 最后一次有效进展、重复动作与停止原因。
- trace、runtime log、workspace snapshot 和 checksum。
- analyzer 结论、反例与置信度。
- Root Gate 最终分类。
- 关联 hypothesis、candidate、PR 或停止记录。

不同 candidate/generation 的 evidence 可以共同用于诊断，但不能拼成一个分数。

## 7. 故障分类

失败归因与候选语义变化分开记录。

| 分类 | 证据门槛 | 自动处理 |
|---|---|---|
| Harness 功能性 bug | 违反已有合同；无模型或受控模型可稳定复现 | 修复、测试、消融、draft PR |
| 鲁棒性优化 | 存在通用可恢复路径；不掩盖永久错误 | 最小候选、controls、现实泛化 |
| 行为语义改变 | 改变权限、重试/费用、停止、上下文、记忆、持久化或用户结果 | 已确认项可实施；不确定项延后 |
| LLM 能力不足 | runtime、工具、提交链正常；失败来自规划、推理或执行 | 记录，不为提分改 harness |
| Task/环境问题 | 题面、镜像、资源或 verifier 无法一致满足 | 留证，不算 Akasic 修复 |
| 证据不足 | trace 不完整或多个根因不可区分 | 改善观测或设计新 reproduction |

### 7.1 功能性 bug

必须指出：

- 被违反的现有合同。
- 真实可达的触发路径。
- 当前错误如何导致 case 失败。
- 无 Terminal-Bench 解法的最小复现。
- 修复前后错误、持久化和外部副作用。

### 7.2 鲁棒性优化

如果新实现保留所有旧的合法输入、权限、失败暴露、持久化和取消语义，只扩大成功处理
范围，则可以称为鲁棒性优化。

如果扩大成功范围需要更多请求、费用、等待、权限或不同终态，则仍有行为语义 delta，
不能仅因更容易成功而称为纯优化。

### 7.3 行为语义改变

- 已由用户确认的变化可以进入实现、`projectneed` 和 PR。
- Root 能明确从已有合同推出的修复按对应 bug/优化流程处理。
- Root 仍拿不准的变化可以在隔离实验分支验证，但不进入 incumbent、不提生产 PR、
  不更新长期需求。
- 所有不确定项进入 `semantic-pending`，89 题扫描完成后统一与用户讨论。

## 8. 调度策略

系统维护两条共享六个 Docker slot 的队列：

```text
Discovery Queue                 Validation Queue
尚未扫描的 89 个 case           已有明确假设的 case + controls
        │                               │
        └──────────┬────────────────────┘
                   ▼
           Scheduler，max=6
```

默认调度：

- `5 discovery + 1 validation`。
- 没有成熟假设时，六个 slot 都扫描未见 case。
- 出现会让大量 attempt 无效的基础设施问题时，暂停新任务并先恢复有效性。
- trace 分析不占 Docker slot。
- 已失败 case 的重复尝试不能长期挤占尚未扫描 case，除非它阻塞所有任务。

每个 attempt 使用启动时冻结的 candidate。incumbent 更新只影响之后创建的实例。
优化阶段允许不同 case 属于不同 generation，但必须显式记录，且不得汇总为统一成绩。

## 9. 单 case 实验循环

1. 完成第一次有效 discovery run。
2. 封存 evidence。
3. Terra 分析 trace。
4. Root Gate 归因并建立一个 falsifiable hypothesis。
5. 没有明确 harness hypothesis 时不盲目重跑。
6. 先做 synthetic 或 fault-injection reproduction。
7. 在独立 worktree 实施单一最小 candidate。
8. 重跑受影响 case，并加入至少一个稳定 control。
9. 比较机制证据、task 行为、usage、异常和现实影响。
10. 决定保留、修改、回滚、待定或关闭。

### 9.1 尝试次数

单个 case 不设机械 attempt 上限。

从第六次 attempt 开始，每次继续前必须有 escalation record：

- 前五次分别排除了什么。
- 新增了什么 trace、机制、反例或最小复现。
- 为什么仍属于 harness，而不是 LLM、task 或语义待定。
- 这次实验与前一次的唯一因果差异。
- 继续尝试的现实泛化价值。

以下情况关闭当前方向：

- 最近两次尝试没有增加信息。
- 无法建立稳定的最小复现。
- 唯一剩余方案是 task-specific patch。
- controls 出现受保护语义回归。
- 证据支持 LLM 能力不足或 task 问题。
- 需要用户决定的语义改变。

关闭不是永久删除。后续跨 case 的新证据可以重新打开。

## 10. 故障簇与优先级

多个题面不同的 case 如果共享同一机制，合并成一个 failure cluster。

优先级：

1. 会让 trial 无效、trace 丢失或产生假成功的基础设施/harness bug。
2. 同时影响多个 case 的功能性缺陷。
3. 可能造成现实任务数据损坏、错误执行、无限循环或静默失败的问题。
4. 通用鲁棒性、成本和效率问题。
5. 单 case、低置信度问题。
6. LLM 能力不足和 task-specific 问题只记录。

PR 的单位是通用机制，不是 case。

## 11. 消融与现实泛化

每个候选按以下顺序晋级：

```text
机制 reproduction
  → candidate vs candidate-without-feature
  → 受影响 case
  → 稳定 controls
  → 无 TB 解法的现实 shadow scenario
  → 项目 semantic Gate
```

现实 shadow scenario 应覆盖问题机制，而不是复制 task：

- 长工具输出。
- 慢任务或后台进程。
- provider 短暂失败。
- 断线或迟到 terminal event。
- context/compaction 后继续执行。
- 工具调用格式错误。
- 取消、重启或恢复。
- 持久状态和外部副作用。

没有机制证据的单题翻转只算相关性，不足以保留候选。

## 12. CI、Review 与 PR Gate

每个保留候选至少通过：

- 直接相关单元测试和集成测试。
- Pyright/typecheck 和仓库要求的 lint。
- Change-impact public Gate。
- 所需 private Gate 的准确状态。
- 独立只读 cumulative diff review。
- 正式 workspace、线上 PID、源码和外部插件不变检查。
- task ID、expected output、verifier path、Terminal-Bench prompt 和条件分支搜索。

PR 要求：

- 一个 draft PR 只承载一个通用机制。
- case 名只能出现在证据和实验记录，不得成为生产条件。
- 描述问题、失败路径、修改、消融、现实泛化和语义 delta。
- 功能性 bug 和鲁棒性优化必须明确区分。
- 不确定的行为语义改变不提生产 PR。
- 不自动 merge。

## 13. 文档与状态 owner

| 文档或对象 | 负责内容 |
|---|---|
| `experiment-ledger` | 每个 attempt、失败、消融、usage 和决策 |
| `failure-catalog` | 跨 case 故障簇、影响面、owner 和状态 |
| `hypothesis-registry` | 假设、反例、实验次数、继续或停止理由 |
| `semantic-pending` | 最终需要用户确认的语义变化 |
| PR | 已保留机制的可审阅实现与验证 |
| `projectneed` | 已确认的长期产品语义 |
| artifact store | 大 trace、workspace、payload 和容器证据 |

实验文档不能把分数或临时假设升级成产品需求。

实施阶段确认更新 `projectneed`：

1. `max_output_tokens=0` 的 provider-default 长期语义及存量兼容规则。
2. Benchmark 用于发现通用 Agent/harness 缺陷，不得驱动 task-specific 生产行为；
   行为语义改变与纯鲁棒性优化必须区分。

## 14. 状态机与恢复

```text
prepared
   → running
   → terminal_sealed
   → analysis_pending
   → classified
   → experiment_active
   → closed_fixed / closed_model / closed_task
                  / closed_no_evidence / semantic_pending
```

- analyzer 失败只重启分析，不重跑 case。
- candidate 测试失败保留 diff 和 evidence，不修改 oracle 获取全绿。
- 正式 workspace、线上 PID 或源码摘要变化时停止调度，进入只读事故调查。
- Docker 地址池、磁盘或 provider 可用性危及证据时暂停发新任务，不中止正常收束中的
  attempt。
- 所有重启、恢复和 aggregate repair 都必须显式记录，不能伪装成原 run 自然完成。

## 15. Evidence 保留策略

采用用户批准的策略 A。

### Hot evidence

- 停止的 container。
- volume 和独立 workspace。
- runtime stdout/stderr 和完整现场。

### Cold evidence

- trace、payload 摘要、usage 和 verifier。
- manifest、源码/image/container identity。
- 文件 checksum、归因和实验记录。

规则：

- attempt 完成后 stop-retain。
- 问题关闭后完成 cold archive，但 container 仍保留。
- 89 题初始扫描完成前不自动删除 container 或 volume。
- 空且无 endpoint 的 benchmark network 可以在记录后删除，以释放 Docker 地址池。
- 全部扫描结束后汇总磁盘占用、待定问题和保留价值，再由用户批准是否删除。

## 16. 89 题完成条件

自动诊断阶段只有同时满足以下条件才结束：

- 89 个 case 都至少完成一次有效 discovery attempt。
- 所有 invalid case 已重跑或有明确外部 blocker。
- 所有 failure cluster 已处于 fixed、model、task、no-evidence 或
  semantic-pending。
- 没有仍在运行、未封存或无人归因的 attempt。
- 所有保留修复都有 synthetic、消融、controls、现实泛化和 CI evidence。
- 所有失败方向都有停止理由。
- draft PR、failure catalog、experiment ledger 和 semantic-pending 已对账。
- 正式 workspace 和线上 runtime 未受影响。

随后向用户提交：

- 已修复的通用问题及 PR。
- 失败实验和回滚。
- LLM 能力边界。
- Task/环境问题。
- 不确定的行为语义改变。
- 仍未解决的问题及继续成本。
- 容器、volume 和 artifact 保留清单。

Root 不自动启动优化后完整 89 题 eval。只有用户审阅上述结果并明确授权后，才建立
独立 final-eval 合同。

## 17. 验收标准

本设计的成功不以 Terminal-Bench 分数定义，而以流程和产品证据定义：

- 所有 89 个 case 被用作独立故障探针。
- 每个生产修改都对应可复现的通用机制。
- 没有 task-specific hack、选择性拼分或隐藏无效 run。
- 功能性 bug、鲁棒性优化和行为语义改变被准确区分。
- 超过五次的 case 没有机械重试。
- LLM 和 task 边界被诚实关闭，而不是强行改 harness。
- 每个 PR 都能说明现实场景价值并通过项目 Gate。
- 不确定语义集中留待用户最终决策。
- 正式完整 eval 仅在用户明确授权后执行。

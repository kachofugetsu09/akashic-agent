# 0039 · React 原子能力留在 Core，来源保持非特权

- 状态：accepted
- 日期：2026-08-22
- 关联条款：RUN-001～RUN-003、RUN-007～RUN-009、OUT-001～OUT-004、PLG-003、PLG-006、PLG-014、SCH-001～SCH-003、PRO-001、CTRL-003、SEC-005、SEC-007、TST-001～TST-006
- supersedes：无
- superseded by：无

## 背景

被动消息、Scheduler SOFT、Subagent 和 Proactive 都可能运行模型与工具，但当前实现分别从 `AgentLoop.process_direct()`、独立 `SubAgent` 循环和 proactive runtime 进入。若为每种来源复制一套 ReAct、生命周期、Prompt、记忆、取消和快照逻辑，语义会逐渐漂移；若把它们全部塞进一个带来源分支的总调度器，Core 又会理解 Scheduler、Subagent 和 Proactive 的业务状态。

现有合同已经确定：`Turn` 是用户可理解的逻辑工作单元；插件 publication、generation 和 lease 仍由 Core 拥有；Scheduler 是用户持久化调度，不是 generation-scoped background job。需要选择的不是“是否做一个万能 React 类”，而是哪些能力必须由 Core 只实现一次，哪些来源只负责组合。

## 决定

Core 只拥有执行一个 Turn 所必需、且跨来源必须一致的原子能力：

1. 接受一个 `Message` 并进入既有 `react` 控制流；
2. 为 Turn 建立唯一 owner、父子 lineage、取消和 terminal/cleanup；
3. 冻结模型与 `RuntimeSnapshot` lease；
4. 按 exact scope 组合生命周期、Prompt 和 Tool grant；
5. 提供合法 `Enter` / `Skip` 准入结果；
6. 提供不含业务语义的一次性 Timer 与只读结构化 trace。

Core 不出现 Scheduler、Subagent、Proactive、cron、misfire、spawn profile、hazard、delivery target 或 memory policy 的业务分支。`react` 仍是“输入 Message，产生输出 Message”的动词和控制流，不新增与 `Loop`、`Turn` 平行的第二套执行模型。

Scheduler 与 Subagent 首先迁成仓库内置的 v3 插件。它们通过公开 Service 和 exact generation scope 使用上述能力，与外部插件拥有相同权限：不能 import `PluginManager`、私有 provider、全局 ToolRegistry、任意 Session repository 或任意 SQL，也不能要求 bootstrap/Core 为插件 ID 添加特判。仓库内置只是发行方式，不是特权等级。

Scheduler 插件继续唯一拥有 `schedules.json`、cron/interval、misfire、P90 提前量、`run_count`、禁用/取消和投递提交。它只用 Core Timer 等待一次，再用 scoped Turn port 执行一次 SOFT Turn；下一次何时等待仍由插件决定。

Subagent 插件继续唯一拥有 spawn 准入、profile、task directory、完成回传和 `spawn_trace.jsonl`。子任务在一个 ephemeral Session 中递归使用同一条 `react`，父子关系和 Tool grant 显式绑定；首版不把子任务提升为 durable Session，也不学习长期记忆。

Proactive 本阶段不迁移。Wake 风格链路只作为结构验算：`Timer → observation/gate → Skip|Enter → react → delivery/state settle` 必须能由同一组原子能力表达，且 passive-only lifecycle 不会因此生效。真实 proactive/Wake/Drift 状态机、数据库和 hook 迁移需后续独立合同与批准。

```text
                         Core
┌─────────────────────────────────────────────────────┐
│ Message → Turn admission → react → terminal/cleanup │
│ snapshot lease · scoped lifecycle · Tool grant      │
│ one-shot Timer · trace                               │
└───────────────┬───────────────────┬─────────────────┘
                │ public Service    │ public Service
                ▼                   ▼
      ┌──────────────────┐  ┌──────────────────┐
      │ Scheduler plugin │  │ Subagent plugin  │
      │ schedule + fire  │  │ spawn + complete │
      └──────────────────┘  └──────────────────┘

未来结构验算：Wake plugin = Timer + gate + react + settle
```

## 理由

Turn owner、generation lease、取消、工具执行权限和 terminal 是跨来源的一致性事实，放在 Core 可以避免多套实现。调度规则、spawn profile、主动 gate 和领域持久状态各有独立 owner，留在插件可以避免 Core 形成来源枚举与权限后门。

一次性 Timer 与递归 Turn 是正交积木：Timer 只回答“什么时候唤醒”，Turn 只回答“这次工作怎样执行”。生命周期和记忆由 scoped capability 决定，而不是由 `if source == ...` 或在 `before_turn` 中先执行再跳过决定。

## 影响

- 正面影响：Scheduler、Subagent 和未来 Proactive 可以复用同一 Turn 语义；插件能够单独卸载、替换和差分验收。
- 兼容性：迁移必须保持现有 Prompt、phase/slot 顺序、工具可见性、记忆排除、发送顺序、持久 write set 和错误分类；未批准差异一律失败。
- 数据和迁移：首轮不增加数据库 schema，不迁移正式 workspace；`schedules.json`、`spawn_trace.jsonl` 和 `subagent-runs/` 的 owner 与保留规则不变。
- 失败与回滚：每个迁移批次保留旧路径，只有差分 Gate 证明等价后才删除；Git 回滚不能冒充已发生的消息或外部调用已撤销。

## 验收

- [ ] Core 原子能力的公开接口和测试中没有 Scheduler、Subagent 或 Proactive 业务词与插件 ID 特判。
- [ ] Scheduler 和 Subagent 通过正式 v3 loader、普通 Service、generation lease 与 Effect cleanup 运行；缺失依赖在 admission 前 fail-loud。
- [ ] 固定时钟和 recording adapter 能复现真实 Scheduler/Subagent 代表场景，旧/新差分回执除登记的时间、UUID、PID、端口外无差异。
- [ ] lifecycle scope、Prompt 注入/替换、Tool grant、记忆读写、取消、外部发送和持久 write set 均有独立 oracle 与 mutant。
- [ ] Wake 风格结构验算无需扩展 Core 接口；这项验算不等于 proactive 已迁移。

## 未决问题

- Durable child Session、跨进程 child resume、subagent fork 上下文和 proactive 迁移均不属于首轮；出现真实 consumer 后分别立项。

# 0010 · Provider 默认输出边界与 Benchmark 诊断边界

- 状态：accepted
- 日期：2026-07-30
- 关联条款：RUN-006、TST-009
- supersedes：无
- superseded by：[0011](0011-benchmark-concurrency-six.md)（仅并发上限）

## 背景

Akasic 已经支持以 `max_output_tokens = 0` 省略 provider 输出上限字段，但配置模型、缺省加载、安装向导和设置界面仍使用 `8192`。这一默认值会在长工具任务中提前截断 runtime，掩盖 Agent 是否能够完成任务。

Terminal-Bench 2.1 的 89 个 case 可以暴露 Agent 与 harness 的工程故障，也会带来按公开题目刷分、混合无效 run 和把模型问题误判成 runtime 问题的风险。维护者已批准把这些 case 用作诊断探针，而不是产品目标或自动自优化器。

## 决定

新建或缺省配置使用 `max_output_tokens = 0`，请求端不发送 provider 输出上限字段。显式正整数继续代表用户选择的上限，存量正整数不自动迁移；负数在配置边界失败。内部 summary 等小任务继续拥有独立局部上限。

Benchmark 控制面只负责隔离运行、封存证据、归因、消融和候选 Gate。每个 attempt 使用独立 Docker runtime 与 workspace；分析只读取终态证据。生产修改必须解释不依赖具体 case 的现实价值，优化后的完整 89 题 eval 只在维护者再次明确授权后运行。

## 理由

由 provider 拥有缺省输出边界可以避免 Agent 在不知情时被固定应用层上限截断，同时保留显式成本和延迟控制。局部内部任务继续有界，避免把主 runtime 的选择扩散到无关调用。

把 benchmark 定义成诊断输入而非目标函数，可以保留失败样本的工程价值，又不允许 task-specific 条件、选择性拼分或降低 verifier 约束进入生产代码。独立的 reality controls 和项目 Gate 承担泛化判断。

## 影响

- 正面影响：长工具任务默认不再受应用层 `8192` 输出上限；89 个 case 可以形成可审计的 Agent/harness 故障样本。
- 兼容性：显式正整数配置保持原值；`0` 仍受 provider/model 自身边界约束。
- 数据和迁移：不自动改写任何正式 workspace 或存量配置；benchmark evidence 写入独立 artifact store，不进入长期记忆。
- 失败与回滚：默认值实现可回退到本决定前 commit；每个 benchmark 候选和失败 attempt 独立留痕，不通过修改 oracle 获得全绿。

## 验收

- [ ] 缺省配置、setup wizard 和 settings UI 都生成 `0`，显式正整数保持不变，负数失败。
- [ ] provider 请求在值为 `0` 时省略输出上限字段，正整数时继续发送。
- [ ] summary 等局部内部调用仍保持独立上限。
- [ ] 诊断控制面限制最多三个独立 attempt，封存终态后才允许归因，并保留 invalid 与有效失败的差别。
- [ ] 超过五次的重复尝试要求新的可证伪证据，不能机械重试。
- [ ] 正式完整 89 题 eval 不属于自动调度动作。

## 未决问题

- 全部 89 题初始诊断结束后，停止容器、volume 和冷证据的删除范围由维护者另行批准。

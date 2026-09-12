# 0064 · 并行工具调用重叠执行、按模型顺序提交

- 状态：accepted
- 日期：2026-09-12
- 关联条款：PRM-004、SES-003～SES-005、RUN-003、Tools 回执合同（[0059](0059-abandon-settles-tool-calls.md)、[0063](0063-execution-failures-have-terminal-results.md)）

## 背景与决定

模型可以在一次响应里返回多个独立工具调用，原 ReAct 循环逐个串行结算，浪费了 provider
已经给出的并行度。决定让执行可重叠，但不改变任何持久语义：

- 工具注册新增 `parallel: bool` 字段，默认 `False`。只有显式声明的调用允许与同批
  parallel 调用重叠；旧归档描述缺少该字段时按 `False` 解释，行为与升级前一致。
- ReAct 把连续 parallel 调用编成一组，走有界滚动池（`max_parallel_calls`）；
  exclusive 调用是屏障，前面的组排空后才执行，其后的调用等它结算。
- 组内每个调用仍走 `ToolExecution.execute_call` 的完整回执协议；唯一新增的是
  `commit_after` 提交门——每个调用在 `finish` 前等前驱的结算事件，因此 ToolResult
  与回执的落盘顺序永远是模型顺序，与完成先后无关。
- 前驱无论成功或失败都在 `finally` 放行后继，一次失败不锁死整批；批级取消先放行
  所有提交门再排空已开始调用，未开始的调用留给原有 pending/abandon 协议结算。

## 理由与影响

dispatch 并发、commit 保序与 Codex（`FuturesOrdered` + read/write 门）和
deepseek-harness（prepare/dispatch/finalize 三段式 + 有界池）的主结构一致。
`finish` 把结果消息与回执放在同一事务，这条不变量不拆；保序因此落在提交时机上，
而不是先落盘再排序。

`parallel` 是工具对自身并发安全的声明，不是调度策略：分类只读注册描述，调度只读
分类结果，效果归属、幂等、授权与恢复完全留在 `ToolExecution`。fail-closed——binding
失效、描述缺失或字段非 `True` 一律串行。

并行不减少 provider 请求轮数；轮数只取决于模型在一个响应里放几个调用。本改动减少
的是同一响应内多个独立调用的等待时间。已验证为无共享可变状态的只读工具（web
fetch/search、tool_search、load_skill、list_schedules、recall_memory、read_file/
list_dir）标记 `parallel=True`；其余一律保持 exclusive。

## 持久化与恢复

不新增表、不改 schema、不改写旧消息或旧回执。取消语义不变：`interrupted` 仍不表示
外部效果已撤销；abandon 的"先提交终态再发取消"合同不受影响，晚到的组内提交按
first-committed-wins 读取已存在终态。

## 验收

- 同批 parallel 调用的 `invoke` 可重叠；较快完成的结果不抢先落盘，ToolResult 顺序
  等于模型顺序，provider 投影顺序不变。
- exclusive 调用在前组完全结算前不启动；其后的 parallel 调用在它结算前不启动。
- 池上限生效；任一并发度下日志与串行执行逐字节同构。
- 组内失败停止补发、排空已开始调用；批级取消不死锁。

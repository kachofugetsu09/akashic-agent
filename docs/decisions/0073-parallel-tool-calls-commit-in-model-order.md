# 0073 · 并行工具调用重叠执行、按模型顺序提交

- 状态：accepted
- 日期：2026-09-25
- 关联条款：PRM-004、SES-003～SES-005、RUN-003、STA-001、[0059](0059-abandon-settles-tool-calls.md)、[0063](0063-execution-failures-have-terminal-results.md)

## 背景与决定

模型可以在一次响应里返回多个工具调用。ReAct 原先逐个结算，只读调用的等待时间相加。
执行可以重叠，持久语义保持串行提交：

- `parallel` 是 `ToolCatalog.register` 的注册事实，默认 `False`。只有 `risk="read-only"`
  可以设为 `True`。它不进入工具描述、binding 归档、provider schema 或消息日志。
  旧 binding 的描述比较保持不变。
- 调度只读 `ToolMenu.parallel`。读不到注册或字段不是 `True` 时一律串行。
  连续的 parallel 调用组成一组，走有界池；exclusive 调用是屏障。
  abandon 区仍只走 `settle_abandoned`，不进入这个池。
- 每个调用仍由 `ToolExecution` 拥有回执。`commit_after` 只挡住 `finish`：
  结果消息和 done 回执仍在同一事务里，顺序是模型顺序。
- 前驱成功、取消排空，或错误回执已经落盘后，才放行后继。
  前驱在 `ToolResult` 落盘前失败时，门改为中止，后继不抢先写结果。
  工具执行异常在错误回执落盘后仍按原异常失败，回合不因此继续请求模型。

## 理由与影响

重叠执行、保序提交和 exclusive 屏障是同一结算策略的三面，不新增 Turn、Attempt
或第二套效果 owner。`parallel` 若写进 binding 描述，会让只改调度旗标的工具在升级后
无法恢复进行中的调用，所以它留在目录里。

直接调用 `react` 时上限默认是 1。`run_reply` 默认上限是 4。
已声明重叠的只读工具：`read_file`、`list_dir`、`web_fetch`、`web_search`、
`tool_search`、`load_skill`、`list_schedules`、`recall_memory`。

## 持久化与恢复

不新增表，不改 schema，不改写旧消息或旧回执。`started` 回执仍在真正执行前落盘，
可以早于前驱的 `ToolResult`；done 回执和 `ToolResult` 不提前。
取消仍不表示外部效果已经撤销。

## 验收

- 同组 `invoke` 可以重叠；较快完成的结果不能抢先落盘。
- exclusive 调用在前组完全结算前不启动。
- 池上限生效。
- 前驱在 `ToolResult` 落盘前失败时，日志里还没有后继的 `ToolResult`。
- 前驱的错误回执已经落盘时，后继按模型顺序提交，然后原异常仍失败本回合。
- 取消排空之后，已提交的 `ToolResult` 仍按模型顺序排列。
- `parallel=True` 不改变 binding 描述；非 read-only 工具不能声明重叠。

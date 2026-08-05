# 插件与 Skill 自验证参考

## 目录

1. 前置能力检查
2. 安装到 latest
3. 用 Shell 观察 programmatic child
4. 行为 oracle
5. Promote、Discard 与递归
6. 当前实现缺口

## 1. 前置能力检查

先读取当前 CLI help 和 accepted 设计，不猜接口：

```bash
python main.py --help
```

完整递归验证需要同时存在：

- `exec --runtime stable|latest`
- programmatic 新 session 默认只读语义记忆和显式 `--persist-memory`
- `plugin-status`
- `plugin-promote`
- `plugin-discard`
- attached control disconnect cancellation

任何一项缺失时，不执行下面的假命令，不用 sleep 或第二个 Gateway 替代。若 runtime 已有跨 session 并发，则按本页第 6 节在隔离环境执行 current-snapshot 自验证；否则继续按 [runtime-diagnostics.md](runtime-diagnostics.md) 检查 reload、SessionDB、日志和既有子 turn 内容。

## 2. 安装到 latest

完整合同实现后：

```bash
python main.py plugin-install \
  --source /absolute/path/to/committed-plugin \
  --marketplace local

python main.py plugin-status
```

`plugin-install` 返回成功必须意味着 `latest_ready`，不是“cache 已写、watcher 以后也许会发现”。读取 status，记录 stable/latest identity、candidate phase 和 source provenance。若已经有未决 latest，停止并处理它；不要覆盖。

## 3. 用 Shell 观察 programmatic child

从父 Agent turn 调用统一 Shell：

```bash
python main.py exec --new --runtime latest --json \
  "加载目标 Skill，使用新插件完成一个可独立断言的只读任务。"
```

命令超过初始等待后，Shell 返回 `execution_id`。使用 `write_stdin(execution_id=..., chars="")` 读取新增 JSONL，直至出现唯一 terminal。不要启动第二个 runtime；`exec` 是当前 Gateway 的 control client。

首次返回后立即记录 `execution_id`、`thread_id` 和 `turn_id`。超过一次有界等待仍没有 terminal 时，停止轮询，按 [runtime-diagnostics.md](runtime-diagnostics.md) 读取 `turns.created_at/started_at/completed_at`、`items_json` 和 final response。`queued` 且没有 `started_at` 是调度证据，不是插件行为失败；不要用第二次长超时重复同一调用。

默认调用必须满足：

- 新 programmatic session。
- latest snapshot。
- recall/search allowed。
- semantic memory writes disabled。
- SessionDB thread/messages/tool items 正常持久化。
- attached：父 turn cleanup、task_stop、CLI 退出或 socket 断开会取消子 turn。

只有用户明确要求让验证内容进入长期记忆时，创建 session 时增加 `--persist-memory`。

## 4. 行为 oracle

terminal `status=completed` 和 final response 不是充分条件。至少核对：

```text
┌─ Snapshot
│  └─ child 绑定的 identity == install 后的 latest
├─ Skill（存在时）
│  ├─ catalog source == plugin
│  ├─ 正文/引用资源可加载
│  └─ 真实触发请求遵循关键步骤
├─ Tool
│  ├─ tool item 名称正确
│  ├─ arguments 命中测试输入
│  ├─ status/result 正确
│  └─ 领域 before/after 状态符合预期
├─ Memory
│  ├─ recall 可用
│  └─ semantic write set == 0（默认）
└─ Persistence
   └─ child thread/turn/messages/tool items 可回读
```

读型插件使用固定 fixture 或稳定 API 响应。写型插件只在真实事务/dry-run、隔离环境或用户明确授权下执行。`message_push` 必须核对真实 delivery receipt；发送正文不会出现在父 Prompt或目标 session history，证据位于 child tool trace。

## 5. Promote、Discard 与递归

通过：

```bash
python main.py plugin-promote
python main.py plugin-status
```

重新读取 status，证明 `stable == 已验证 latest` 且没有未决 candidate。pointer 提交原子，因此不必用第二次昂贵 LLM 调用重演相同行为；pointer identity、第一次真实行为 trace 与 promotion journal 共同构成证据。

失败：

```bash
python main.py plugin-discard
python main.py plugin-status
```

证明 stable 未变、latest 回到 stable。根据 terminal/tool/domain error 修复 canonical source、运行 source tests、提交并重新 install。相同错误且没有新改动时不要重复循环。

递归在下面任一条件结束：

- 行为通过并成功 promote。
- 错误不可在当前授权范围修复。
- 需要未授权外部效果、独占 endpoint 或破坏性状态。
- 用户取消、预算耗尽或 runtime 返回明确 fatal/blocked。

## 6. 当前实现与降级路径

当前代码已经实现：

- `ConversationRuntime` 的不同 control thread 可以并发；同一 thread 仍拒绝第二个 active turn。
- `AgentLoop` 以 `session_key` 持有整轮 lane；相同 session 串行，不同 session 并发。
- passive `message_push` 不等待目标 session turn 结束，但实际 channel send 仍经过 ChatLane 串行提交。
- reload 发布 `committed` 后，新 session 能租用 current snapshot；旧 snapshot 可以同时处于 `draining`，直到父 turn 释放 lease。

当前仍缺少：

- validating snapshot `accepting_leases=False`。
- `plugin-install` 成功只覆盖 cache/manifest，不等待 `latest_ready`。
- `exec --new` 能连接当前 Gateway 并输出 JSONL，但没有 runtime selector。
- Shell 终止 CLI 后，control socket 断开尚不拥有服务端 turn cancellation。
- stable/latest pointer、promote/discard 和未验证候选隔离。

因此父 turn 可以证明刚安装插件在 current snapshot 中的真实行为，但只能在隔离 workspace/runtime 中安全使用：

```text
T install → reload event committed → Shell 启动新 session V
          → V 在 T 完成前返回 terminal/oracle → T 决定修复或接受
```

不要等待 `reload_transactions.phase == committed`：提交后该字段会立即前进到 `draining`。应查询同一 `tx_id` 的 `reload_events` 是否出现 `committed`，再用 SessionDB 证明 V 的 `started_at/completed_at` 早于 T 的 `completed_at`。这构成 call-return 自验证证据。

这条路径仍会把插件暴露给同时开始的普通新 turn，失败时也不能原子恢复旧默认指针，所以必须报告 `safe candidate self-validation unavailable`。完整诊断步骤见 [runtime-diagnostics.md](runtime-diagnostics.md)。

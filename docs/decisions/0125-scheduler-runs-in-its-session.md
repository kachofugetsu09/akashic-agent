# 0125 · 定时任务在独立内部会话执行并向目标会话投递结果

- 状态：accepted（2026-11-09 修订：恢复独立 internal Session 执行并自动搬运文件附件）
- 日期：2026-10-18（2026-11-09 修订）
- 关联条款：SCH-003、MEM-001～MEM-002
- 关联决策：0058、0117

## 背景

2026-10-18 初版尝试让定时任务直接在目标会话原位执行，以便显示 `render_visual` 卡片和工具过程。但线上长期运行暴露出严重副作用：

- 每次定时任务触发都会向主会话追加几百字的任务规则 `Input` 以及 20～40 条工具调用与重试 `ToolResult`，严重污染主会话时间线与账本；
- 在主会话原位执行需要争抢 `CONVERSATION_COMPLETE` 锁，与用户正常聊天互相阻塞。
- 与此同时，0117 已将 `render_visual` 产物改为普通自包含 `.html` 文件附件，聊天宿主在任意 `Output` 消息上都能直接渲染 `.html` 附件。

## 决定

1. 每个任务仍记录创建它的目标会话与投递渠道（`job.session_id` / `channel` / `chat_id`）。
2. `run` 每次触发在全新的独立内部会话 `scheduler:<fire_key>` 中执行，通过 `SESSION_ADMISSION` 设为 `SessionAttributes(visibility="internal", learning="excluded")`：
   - Web 会话列表只展示 `visibility="listed"`，因此 `scheduler:<fire_key>` 在 Web 界面上完全不可见；
   - 每次触发都是干净会话，固定使用 `history="own"`，绝不复用旧对话或主会话历史；
   - 内部会话在启动时继承目标会话保存的 `model.selection` 与 `WORKING_DIRECTORY` 快照。
3. 内部会话跑完后，Scheduler 扫描本次触发中成功 `ToolResult` 产出的 `kind == "file"` 附件（如 `render_visual` 正式发布的 `.html` 交互卡片，自动排除 `draft=True` 预检截图 `kind == "image"`），与最终 `Output` 的文本合并。
4. 在 `delivery.wait_idle` 后，向目标会话只追加一条 `scheduler-notification:<fire_key>` 的完整 `Output` 消息并经 `Delivery` 投递。用户的主会话只看到最终卡片与结论，没有任何过程噪音。
5. 模型失败或取消记录保存在内部会话 `scheduler:<fire_key>` 中，并将本次触发结算为 `failed` 或 `cancelled`。

## 记忆

定时任务不会写脏长期记忆，依据都是现有机制：

- akasha 与 markdown_memory 只把配置里列出的来源（默认 `conversation`、`programmatic`、`legacy-unattributed`）当作学习样本，`scheduler:*` 不在其中。markdown_memory 核对摘要覆盖范围时会读完整快照，但只抽取允许来源。
- 兴趣原型只用 `conversation` 的完整轮次。
- 任务里关闭三个写记忆工具；`recall_memory` 只读，保留。
- 用户之后在对话里提到任务结果，是用户真实说过的话，照常学习。

新增记忆插件必须同样按来源白名单学习，默认只含对话来源。本次不在宿主层另建“可学习来源”注册表。

## 已知限制

- `none` 不压缩：单次任务超出上下文时明确报错结束，不截断。
- 用户取消执行中的任务时，Scheduler 等原执行排空后为未结束轮次追加现有 `abandon` 控制记录；未执行的工具不重跑，原消息与工具回执保留，后续对话和压缩可以继续。重启后从持久取消状态补齐同一条记录。
- 普通对话之后会看到会话里的任务过程（与 subagent 回报相同）。若嫌占上下文，再单独设计“其他来源对对话只公开结论”。
- 用户插话时，若任务的工具正在执行，用户的回复要等这次工具结束后才开始（与其他来源排空工具的规则一致）。
- 目标会话不存在时 `run` 直接以失败结算；`notify` 照旧写入。

## 迁移

`schedules.json` 升到 version 3，由 scheduler 自己的 Yoyo 迁移 `20261018_01_schedules_v3` 完成（`plugins/scheduler/migration.catalog.toml`），启动前在 workspace 锁内执行，失败阻止启动；运行时只认 v3，不保留读时换算。迁移先写 `schedules.json.pre-v3-<时间>` 备份再原子替换：`tier/message/prompt` 换成 `action/text`，`context` 记为 `none`，目标会话取原 `channel:chat_id`，触发 key 不变。旧流程未完成的 `run` 触发在内部 Session 开始过，以失败结算（“升级前未完成的旧触发，未续跑”），恢复阶段把仍指向已结算触发的周期任务推进到下一次、one-shot 停用；`notify` 的 pending 触发由新流程补完，已发布的 `scheduler-notification:<key>` 仍被认作最终消息。旧内部 Session 原样保留。未接线的数组迁移脚本 `plugins/scheduler/migration.py` 删除。

回滚：v3 文件旧版本读不了。退回旧 release 前，用迁移留下的 `pre-v3` 备份替换 `schedules.json`，并删除迁移账本里的 `20261018_01_schedules_v3` 回执。

## 验收

`scripts/scheduler_tools_scenario.py`（真实插件栈，本地 HTTP 模型）：

- 任务消息写在目标会话，来源独占；`none` 的请求里没有会话历史，`session` 的请求里有。
- 任务一批两个工具执行中用户发消息：先完成当前工具，回复用户（请求里不含进行中的任务），再继续任务；两边都完成，投递各一次。
- 一批工具执行中取消任务：旧记录保持不变，轮次以 `abandon` 结束，后续用户输入仍正常回复。
- 模型上下文超限：保存真实失败原因并结算；已有失败记录的 pending 触发恢复时不再调用模型。
- 任务里调用 `schedule` 创建原文提醒，提醒写入同一会话并送达。
- 旧流程已开始未发布的 pending 触发以失败结算，不发模型请求、不重复写 Input。
- 晚注册、并发固定目录、卸载、禁用工具拒绝保持原行为。

`scripts/timer_plugin_scenario.py`：`notify` 的取消、重启、换代与送达。

# 插件与 Skill 自验证参考

## 1. 三个动作

父 Agent turn 只使用三个插件管理动作：

```text
plugin-install    安装本 turn 的候选
plugin-uninstall  登记本 turn 结束后的卸载
plugin-revert     撤销本 turn 最近一次未提交操作
```

不要调用 `plugin-status`、`plugin-promote`、`plugin-discard`、`plugin-enable`、`plugin-disable` 或手工 restart。stable/latest、验证绑定、提交、排空、服务切换和恢复由 Core 管理。

## 2. 安装候选

先完成 source test、提交 Git HEAD，再从父 Agent turn 的 Shell 执行：

```bash
python main.py plugin-install \
  --source /absolute/path/to/committed-plugin \
  --marketplace local
```

成功返回必须同时说明：候选准备成功；当前父 turn 仍使用原版本；本 turn 的 attached programmatic child 会自动使用候选；正常结束后系统自动切换且下一 turn 生效。命令失败时按返回的具体阶段和对象修复，不把非零退出伪装成成功。

## 3. 用 Shell 观察 programmatic child

安装成功后直接创建 attached child，不指定 runtime：

```bash
python main.py exec --new --json \
  "加载目标 Skill，使用新插件完成一个可独立断言的只读任务。"
```

Core 根据 Shell 传入的 parent turn lineage 冻结 `plugin_id + generation_id + source_revision`，只让该 attached child 使用本次候选。不得添加 `--runtime latest`，不得使用 `--detach`，也不要启动第二个 Gateway。

命令超过初始等待后，Shell 返回 `execution_id`。使用 `write_stdin(execution_id=..., chars="")` 读取新增 JSONL，直至唯一 terminal。保存 `execution_id`、`thread_id`、`turn_id`、tool items 和 final response；超时或 queued 不推进时按 [runtime-diagnostics.md](runtime-diagnostics.md) 定位，不重复安装。

默认 child：

- 创建新 programmatic session，默认不沉淀语义记忆；
- 自动绑定父 turn 当前候选，其他 turn 保持 stable；
- recall/search 可用，candidate 的非 read-only Tool/MCP 默认禁用；
- SessionDB 正常保存 terminal、messages 和 tool trace；
- 父 turn cleanup、CLI 退出或连接断开会取消 attached child。

## 4. 行为 oracle

`status=completed` 和 final response 不是充分条件。至少核对：

```text
┌─ Identity
│  └─ child 的 generation/source == 本次 install 返回的候选
├─ Skill（存在时）
│  ├─ catalog source == plugin
│  ├─ 正文与引用资源可加载
│  └─ 真实触发请求遵循关键步骤
├─ Tool
│  ├─ tool item、arguments、status/result 正确
│  └─ 领域 before/after 状态符合目标
├─ Memory
│  └─ recall 可用且 semantic write set == 0（默认）
└─ Persistence
   └─ child thread/turn/messages/tool items 可回读
```

领域正确性由 Agent 根据结果和轨迹判断，Core 只证明候选身份、terminal、受保护状态和 endpoint 安全。读型能力可直接验证；写型能力只在事务、dry-run、隔离目标或明确授权下执行。`message_push` 必须核对真实 delivery receipt，不能从成功字符串推断外部效果；目标渠道是否另写自己的 durable event 必须读取其 owner 证据。

## 5. 通过、失败与递归

通过时不再执行任何发布命令。向用户说明“候选验证通过，本轮结束后系统自动切换，下一轮生效”，然后正常结束父 turn。Core 在父 turn terminal 和所有旧 lease 释放后提交；下一次用户 turn 会收到自然语言运行事实。

失败时先撤销：

```bash
python main.py plugin-revert
```

`plugin-revert` 只撤销当前 turn 最近一次未提交 install/uninstall。成功后根据 child terminal、tool trace 和领域错误修改 canonical source，运行 source tests，提交，再次 `plugin-install` 和递归验证。它不能跨 turn 回滚已发布版本。

下面任一条件都不能发布候选：没有 attached child、child 失败/取消/超时、身份不一致、父 turn 非正常结束或已经 revert。不要用 status 查询、sleep 或手工 promote 补救。

## 6. 独占 managed service 与 Channel

改变固定 listener 的插件必须通过通用 Core 合同声明隔离端口：

```python
ManagedServiceSpec(
    id="monitor",
    command=("python", "monitor/server.py"),
    readiness_url="http://127.0.0.1:18765/ready",
    validation_port_env="PLUGIN_MONITOR_PORT",
)
```

服务进程必须真正读取 `validation_port_env` 指向的环境变量；同插件 MCP 也必须读取同名变量。Core 会复制 plugin-data 到隔离验证目录、分配临时 loopback 端口、启动候选服务，并把候选 MCP 路由到该端口。未声明、服务忽略变量、readiness 失败或端口被占用都会 fail-loud，不能绕过 Gate。

正式 Channel ownership 不在 candidate child 中复制。child 只验证不接管入口的能力；父 turn 结束后 Core 按 `old Channel.stop → service switch → new Channel.start` 切换。`stop()` 返回即承诺 ingress 和在途工作已收束并释放 ownership，`start()` 返回即承诺已取得 ownership 且 ready；无法证明时发布失败并恢复旧代。

## 7. 完成判定

只有以下事实同时成立才告诉用户任务完成：

- canonical source 的目标改动已提交，远程安装时对应 commit 已推送；
- source tests 和声明/readiness 检查通过；
- attached child 的真实行为和 tool trace 通过；
- install 返回明确说明 turn 边界，父 turn 正常结束；
- 下一用户 turn 的 Core 运行事实确认已提交，或明确说明恢复/清理失败；
- SessionDB、memory、正式 plugin-data 和未授权外部效果未被候选验证改写。

卸载成功返回只表示本 turn 已登记：当前 turn 可完成，结束后 Core 停止 endpoint、移除能力和已安装代码，保留 plugin-data。要取消就在同一 turn 执行 `plugin-revert`。完整诊断步骤见 [runtime-diagnostics.md](runtime-diagnostics.md)。

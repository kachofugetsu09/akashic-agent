# 插件 v3 Agent Work 能力任务合同

本文只为 GitHub Watcher 的两个现存 consumer 建立最窄 Core seam：在一个 committed
Background Job 内提交 programmatic Turn，以及向同一 RuntimeSnapshot 登记插件工具。它不恢复
旧 `AgentInputService`、`PLUGIN_TOOLS` 或 `TIMER_SERVICE`，也不向插件暴露 AgentLoop、
Session repository、全局 ToolRegistry 或控制面客户端。

## 1. 目标与停止条件

```yaml
change_type: migration
semantic_delta: compatible
capability_owner: core
first_consumer: kachofugetsu09/github-watch
allowed_effects:
  - Core source, tests, docs and disposable workspaces
  - controlled programmatic turns and a dedicated remote test repository in the final Gate
forbidden_effects:
  - hua-home or its formal workspace
  - production GitHub repositories, channels, credentials or messages
  - candidate Session creation, Turn admission, tool execution or remote calls
protected_state:
  - existing sessions.db rows and plugin-data
  - stable RuntimeSnapshot and ToolRegistry while a candidate is validating
  - GitHub Watch durable event ledger and checkout ownership
rollback: backup/plugin-v3-akasha-pre-20260817
```

完成必须同时满足：

1. GitHub Watcher 不再 import 旧三项能力，只声明 `BACKGROUND_JOBS` 与 v3 tool catalog；
2. candidate 只冻结 job/tool metadata，不创建 Session、不提交 Turn、不执行 Tool、不读正式 PEM；
3. formal job 通过 invocation-scoped port 创建/复用插件 Session，并在 durable ledger 已进入
   `turn_submitting` 后提交一次 Turn；
4. Tool 只进入 exact committed snapshot，执行时保持原始 turn provenance 与 generation lease；
5. candidate discard、formal reload、进程内失败和 Core 进程崩溃后，ledger、checkout、Root、
   listener、job、Tool 与 module owner 均可证明收束；
6. controlled client 与专用远端测试仓库 Gate 通过后，清单才可把 GitHub Watcher 标为
   `CANDIDATE`。

## 2. Background Job 的 Turn 提交口

`BackgroundJobDefinition` 以 `programmatic_turns=True` 显式声明需要 Turn 提交能力；未声明的
job 的 `BackgroundJobContext.turns` 必须是 `None`。该 port 只提供：

```python
class ProgrammaticTurnPort(Protocol):
    async def create_session(self, *, metadata: Mapping[str, object]) -> str: ...
    async def submit(self, session_id: str, content: str) -> ProgrammaticTurnReceipt: ...

@dataclass(frozen=True, slots=True)
class ProgrammaticTurnReceipt:
    session_id: str
    turn_id: str
```

- port 由 `BackgroundJobActivityAdapter` 在实际执行 job 时按 exact snapshot lease 构造；插件 `apply()`、
  candidate Root 和普通 listener 均拿不到它；只要整张 snapshot 含任一 validation candidate，
  其中所有 job（包括未变化的 stable job）都只能看到 `turns=None`；
- bootstrap 必须在 `PluginManager.load_all()` 打开 Background Job admission 前绑定唯一
  `ConversationRuntime` owner；声明了该能力但 owner 未绑定时，stable boot 在 job publication 前
  fail-loud，不能等到第一次 interval 才失败；
- `create_session` 只创建 `programmatic:*` Session，不接受插件指定物理 key；metadata 必须是
  JSON object，插件不得覆盖 Core 保留字段；Core 追加插件、job、generation 与外部 event
  identity；
- `submit` 接受同一 invocation 创建的 Session，也允许后续 invocation 复用 durable Session；后者必须由
  Core Session owner 证明 `programmatic=True` 且 `plugin_id/job_name` 与当前 exact binding 相同，插件
  不能查询或复用别人的 Session；
- 调用返回只代表 Turn admission，不等待回复。Core 只把已经证明未取得 Turn handle 的错误转换为
  `ProgrammaticTurnPreAdmissionError`，插件可按自己的 event ledger 重试；已经取得 handle、但 durable
  receipt 未确认时抛 `ProgrammaticTurnUncertainError` 并进入 `manual_reconcile`，禁止自动重复 Turn；
- job 的 snapshot lease 覆盖 create/submit；取消时 Core 完成 admission 临界段后再恢复
  `CancelledError`，不能出现已提交 Turn 但插件误判为未提交。

Core 的 `JobOutcomeLedger` 在调用 `ConversationRuntime` 前写 `submitting`，取得 handle 后写
`admitted + turn_id`。同进程 handler 后续失败或取消、以及进程崩溃后发现任一 admission 状态时，
都不得重跑 handler；记录转为带 `manual reconcile` 原因的失败事实。只有明确未取得 Turn handle
且由 `ConversationRuntime` 证明尚未持久化 Turn 的异常，才能清除 `submitting` 并沿既有 retry
policy 重试。若 Runtime 已写入 Turn、却在发布 handle 前失败，Runtime 先把该 Turn 收束为 durable
failed 并释放 active owner，再向 job port 抛 typed uncertain；ledger 保留 `submitting`，禁止重放。
重启扫描必须先收束所有带 admission state 的记录，再判断旧 generation/job 是否仍存在。

Session 与 Turn 的权威 owner 仍是现有 Control/Session runtime。该 port 不新增第二份消息、
Session 或 Turn 模型，也不允许删除、编辑或任意查询 Session。

## 3. exact Root 工具目录

Core 提供新的 Root-local `TOOL_CATALOG = ServiceKey("core.tool_catalog")`，插件在
`apply(ctx, config)` 中登记不可变定义。该名称有意不复用已删除的 `PLUGIN_TOOLS`：

```python
PluginToolDefinition(
    name="github_watch_post_comment",
    description="...",
    parameters={...},
    handler_export="run_github_watch_post_comment",
    risk="external-side-effect",
    always_on=True,
)
```

`handler_export` 必须解析为精确 async callable：

```python
async def run_github_watch_post_comment(
    context: ToolExecutionContext,
    arguments: Mapping[str, object],
) -> str | ToolResult: ...
```

- definition 只保存 JSON schema、risk、search metadata 与 module export 名，不保存 callable；
- handler 必须精确接受 `context, arguments` 两个无默认 positional 参数；Core 传入当前
  `ToolExecutionContext` 与已经按 schema 校验、复制的参数对象，禁止插件再从全局 registry 查
  current tool 或接收任意 Core repository；
- admission 集中校验名称、严格 JSON Schema、risk 枚举、export 和重复名；失败发生在任何
  data-root 写入、candidate publication 或 ToolRegistry mutation 之前；
- candidate catalog 参与 snapshot identity，但不进入公开 stable ToolRegistry；formal rebuild
  必须生成新的 exact Root binding，不能复用 validation callable/module；
- Tool 执行 adapter 从当前 turn 的 RuntimeSnapshot 取得 exact generation binding，调用导出并
  保留 `ToolExecutionContext`；adapter 必须核对当前 task 绑定的 runtime lease 正是构造它的
  snapshot，Root/lease 已失效时 fail-loud，不回退到当前同名插件；
- Tool 名冲突在整张 snapshot compile 时 fail-loud；candidate discard 和旧 snapshot drain 后，
  adapter、module 与 Root Effect 一同失效。

GitHub 写操作的业务授权仍由插件 ledger 的 `operation_id + origin_session_key` 拥有。Core
只拥有目录、lease、调用和结果事件，不理解 repository、review 或 branch 语义。

## 4. GitHub Watcher 迁移

1. interval polling 改为 `BACKGROUND_JOBS` 的 `IntervalTrigger`；handler 从
   `BackgroundJobContext.turns` 取得 port；
2. `ctx.data_root` 保存 event ledger、mirror 与 checkout；candidate 使用隔离副本且不初始化
   GitHub client；
3. 四个外部写工具和 runtime-info 工具改为 `PluginToolDefinition + handler_export`；handler
   在 exact generation module 中解析当前 ledger owner；
4. `AFTER_TURN_COMMITTED` 只清理同时匹配 `session_key + turn_id` 的 checkout；清理失败仍由
   TTL sweeper 接管，不删除 ledger 事实；
5. PEM 只在 formal credential/runtime owner 内解析。candidate tree、config projection、报告和
  日志不得包含 PEM bytes 或真实 token。

## 5. 最小验证

每层只运行对应单元、Manager 与组合 Gate，不为每项改动启动完整服务：

- registry：malformed/duplicate definition 零 mutation；candidate/formal identity 相同而 Root
  token 不同；旧 lease drain 后 adapter 不可执行；
- job turn port：candidate 无 port；formal admission、pre-admission failure、post-admission cancel、
  uncertain result 与进程崩溃重开 ledger；
- Manager：stable load、candidate inert、discard、promote、old lease drain、tool catalog collision、
  listener/job/module/effect 清零；
- plugin：正常/空 discovery、owner mention、非 owner、draft、重复 event、turn cleanup、四个授权
  Tool 与拒绝路径；
- 最终一次 Gate：controlled GitHub client 为主，专用远端仓库只做一次受控 issue/PR/comment/
  review/branch 行为；记录 exact Core/plugin/repository commits、外部 effects 与 cleanup，不使用
  hua-home 凭据。

崩溃恢复只覆盖维护者指定的两类：同进程失败与 Core 进程崩溃。断电、主机停机和宿主级灾难
恢复不在本合同扩展。

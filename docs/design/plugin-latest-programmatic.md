# latest 普通程序与更新授权

- 状态：已按维护者授权实现，等待主协调者只读 review；未运行测试。
- 控制可见性切片基线：`cf5cc7b39ffba1d3e7de8deeb866611b74ae4891`；分支 `codex/plugin-latest-control-v3`。
- 模型接线切片基线：`b9356e1418120c411d8464d6fc2be8c73f2ea3bc`；分支 `codex/plugin-model-settings-latest`。
- 上游：[0071](../decisions/0071-plugin-composition-and-whole-runtime-updates.md)、[0070](../decisions/0070-plugins-own-persisted-data.md)。
- 本文由 programmatic writer 独立维护；共享 INDEX、NOW 和整体设计由主协调者合栈后统一对账。

## 调用与归属

原 `Validation.run` 已调用普通 `reply.execute.v1`，但额外要求 `passed/reason` JSON，
安装后的 watcher 还会自动运行该程序。现在安装只固定候选和原请求，Agent 显式调用
`plugin_latest`：`run` 启动原请求的 prompt、工具与材料选择，立即返回固定 update、candidate、
session 和真实 Task handle；`status` 只读，`revert` 撤销授权。发起 Agent 可顺序调用这三个动作，
不需要并行工具调用。prepare 从实际 CallSource Message 取得 session，invoke 再核对原安装请求；
外部参数不能自报 session。status/revert 还核对调用记录与原 update/candidate/session 的关联。
没有新的后台裁判、业务兼容性检查或批准 JSON。来源 watcher 只沿原 Message/Delivery owner 报告更新状态。

```text
┌─────────────────────────────────────────┐
│ plugin_install → 固定 update / candidate │
└────────────────────┬────────────────────┘
                     ▼
┌─────────────────────────────────────────┐
│ plugin_latest run → 接纳 Task → 返回句柄│
│ Task → 原 reply.execute / Message       │
│ status 读取过程；revert 撤销提交权        │
└────────────────────┬────────────────────┘
                     ▼
┌─────────────────────────────────────────┐
│ 普通 complete → 关闭隔离资源和 Task Scope│
│ 请求晋升 → 换代 → 同步授权/base 检查    │
└────────────────────┬────────────────────┘
                     ▼
┌─────────────────────────────────────────┐
│ 唯一 stable 提交 → 原状态/通知 owner 报告│
└─────────────────────────────────────────┘
```

来源只把实际普通 `Output.finish=complete` 视为程序正常结束；内容不作为第二套授权协议。
异常、取消、缺少完整结果和重启后的未知都不请求晋升。已有调用只能查询，不能因 query
或通知重跑。程序正常结束后默认请求晋升。调用运行时，发起 Agent 用返回的 update_id
查询或撤销；最终结果保存在原候选 MessageLog。run 的 ToolResult 只报告接纳，不冒充最终回答。
运行中 status 按精确 candidate 读取活动 MessageLog；关闭后沿 journal 的原 validation_id，
只读同一 sessions.db，复用 Message owner 的行解码器。SDK 不接受文件路径或 SQL，且只准读
该 update 的固定调用 session。证据目录和候选必须匹配；缺失或损坏明确报错，不初始化空库。

来源通过 TASKS 的同步准入保存一次不可变的 update/candidate/session/Task handle 关联，
另一个调用 key 只保存恢复指针。没有 running/completed 等持久 Task 镜像；status 中 task 为 null
只表示没有活动 Task，不能据此推断成功。原 Message、更新收据与真实关闭 owner 共同显示结果。
重启后只查询原消息与 journal，绝不重新接纳该调用。

来源在真实 Scope 内取得固定 update/candidate 的窄 publication 回调。
ctx.spawn 登记的 Fiber-owned 完成任务只 join 同一普通 Task，等待其 finally 和 Scope 真正退出后
同步使用该回调；不再调用模型或解释回答，不持有新 RuntimeScope。Task 的异常/取消继续由
实际 owner 报告，只有成功 join 才请求 publication。发布入口仍核对原候选、撤销状态和 base。

底座只拥有固定候选与基线、实际 Root/Scope、隔离消息 owner、有限操作和唯一提交点。
`UpdateStatus` 补充候选身份、既有 journal 阶段和证据路径，不新增可变运行状态表。
普通读取不能取得提交许可；可信更新来源显式调用 `publish` 才请求晋升，operator 的显式入口保留。

## 隔离与数据

候选结构装配和 latest 程序分别创建独立实例，两者与正式新实例使用同一组不可变制品和配置输入。
底座不再备份正式消息库、复制历史 binding/附件、扫描历史排除声明，或按文件头识别并复制业务 SQLite。
各实例仍以自己环境下的数据目录初始化。插件配置来自固定归档；模型设置按下节的用户授权
由 models 接续现有来源。其他业务数据仍由 `apply(ctx)` 和普通工具负责准备，缺少数据就明确失败。
这个改变不表示“空库验证已证明现有业务数据兼容”；需要真实历史数据的检查必须由该插件
通过自己的明确数据准备流程提供，不能恢复 Core 全目录复制或把正式目录链接进候选。

| 对象 | 增加/更新及 owner | 减少与恢复 |
|---|---|---|
| 原安装请求、latest 调用关联/恢复指针 | plugin_update 在自己的 owner_records 追加原参数和固定关联；接纳由普通 ToolResult 保存，实际结果在原候选 MessageLog | 不保存 Task 状态副本，不自动减少；原 MessageLog 备份保留请求和正文 |
| 候选 MessageLog、binding、附件、plugin-data | 只在该次隔离目录由实际消息/插件 owner 增加或按自身协议更新 | Scope 只关闭连接和资源，不删除证据目录；没有自动 GC |
| 完整制品与配置 descriptor | 安装/归档 owner 只增加；隔离副本核对精确内容身份 | 本层不清理归档；基线 archive 是源码恢复点 |
| journal | 既有 owner 保存调用证据与错误，revert 推进原候选 discarding | 诊断不等于物理效果回滚；不删除旧事件 |
| stable | 原 selection owner 同步核对授权和 base 后提交 | 已提交或写入不确定不能伪称 revert 成功；恢复沿原 owner |
| 正式业务数据 | 本层无复制、迁移或回滚权；正式新插件仍处理自己的数据 | 不删除、覆盖或迁移正式状态 |

## 现有模型设置与凭据接续

维护者已明确批准：候选默认使用现有模型设置和凭据，不另配账号、不增加凭据审批。
插件是可信本地代码。该授权替代此前候选完全不读正式模型凭据的限制；底座仍不认识模型库，
不获得业务复制或迁移职责。仅固定插件配置归档不够：空模型库没有 default，普通调用会在输出前失败。

```text
┌──────────────────────────────────────────────┐
│ 发起 Task / 原 Scope：MODEL_SETTINGS.read_source │
│ 只取得 models 设置位置，不携带旧模型服务或 driver│
└──────────────────────┬───────────────────────┘
                       ▼
┌──────────────────────────────────────────────┐
│ exact latest Scope：MODEL_SETTINGS.use_source │
│ 新 models 读设置 → 新 driver → 普通 reply.execute│
└──────────────────────┬───────────────────────┘
                       ▼
┌──────────────────────────────────────────────┐
│ 原 connection owner：凭据读取/refresh/备份/锁  │
│ 候选 models owner：真实调用账及结算            │
└──────────────────────────────────────────────┘
```

`MODEL_SETTINGS` 增加 `read_source/use_source`，来源只含 settings path 和 backup_dir。
两端都核对当前真实 Scope 与服务 owner。接收方必须已封印、模型设置为空且未接续过；
它用自己的 models 实现建立 store，不调用来源 Root 的 `CHAT_MODELS`、ModelsState 或 driver。
接续不初始化来源数据库，不复制数据库文件或历史调用账。缺失、损坏、不兼容 schema、缺 default
或新 driver 不支持现有配置都明确失败，不填默认模型、不重跑、不请求晋升。

每次 execution 按既有模型合同读取当前 committed 设置，并固定其 revision、connection、endpoint、
角色、模型参数和 auth identity；实际执行使用 exact latest 的 driver 注册。
凭据仍通过既有 connection-scoped handle 读取和刷新。候选取消或 revert 不撤销已发生的远程请求、
token refresh 或调用账；刷新沿原 credential owner 的锁、事务和备份协议，不产生第二份 token。
普通连接、模型和角色修改在接续 Root 明确拒绝，需从原设置 owner 执行；正式设置 API 保持原行为。

| 对象 | 正常增加或原位更新及 owner | 减少与恢复 |
|---|---|---|
| 原模型连接、角色、参数与 revision | 候选只读；原 models 设置 owner 继续按已有 CAS 修改 | 本片不删除、不复制、不迁移 |
| 原 connection credential | 新 driver 使用原 models credential handle；refresh 原位更新 payload，不增加 revision | 既有 models 备份保存写前状态；revert 不恢复可能已失效的旧 token |
| 候选 model_calls | 候选 models 在真实 driver I/O 前增加 started，按实际成功、失败或取消结算 | 无自动减少；保留同一证据目录，不混入正式调用账 |
| 设置来源位置 | 只在本次 Root 内保存引用，跟随实际 Scope/关闭资源生命周期 | 不写 update journal，不创建持久状态镜像；关闭不删除来源 |

本片只把这次关联更新的普通程序接上现有模型设置；自动晋升仍要求程序正常完成且未 revert，
并等待 Scope 释放。没有提前给结构候选运行模型，也没有新增后台调用或通用数据准备框架。

## 取消、失败与真实 owner

来源 Task 与调用 Scope 先释放，随后才提交 publication 请求。发布任务还会等待其他正式租约归还，
再关闭候选与进入整体切换。等待来源租约仍属于同一个可取消的 Manager operation；
租约归还后才启动实际换代截止时间，不让正在观察结果的发起者耗尽执行时限。
revert 先同步把原候选推进 discarding，再取消相关程序/发布任务。等待和清理失败会明确报错，
提交权不会恢复；资源仍归原 Scope、ValidationHost 或 publication owner，不能先移除句柄。
调用 scope 没有退出或资源没有真正关闭时禁止发布。清理异常写入原 update error，并保留真实
ValidationHost；status 不会因普通输出已 complete 就隐去关闭失败。

提交前再次检查 update、reload transaction、候选授权和 stable base；检查与同步选择提交之间没有 await。
提交已确认或结果不确定时 revert 拒绝，不恢复旧内存指针、安装记录或数据来冒充取消成功。
revert 在资源释放期间发生时可能留下需要显式恢复的错误；“授权已撤销”与“清理/恢复已完成”不同。

## 静态交付与后续验证

本层只进行静态搜索、阅读与 `git diff --check`。新增/适配测试覆盖普通文本结果、显式调用、
读 latest 不晋升、失败与未知不重跑、运行中 revert、等待调用租约、提交后拒绝假取消、
固定制品、插件自建候选数据、正式业务库锁不阻塞装配、资源清理失败保留 owner。
原 Core 数据复制算法的专属测试随算法删除，生命周期回归保留。
本片增加真实工具顺序 run/status/revert、跨 session 拒绝、关闭及重启后原消息读取、
证据缺失不创建数据库、跨更新授权拒绝，以及请求 publication 时来源租约已释放的断言。
模型接线测试使用真实 models、OpenAI-compatible driver、设置 RPC 和普通 latest 工具，只有 HTTP
传输使用受控响应；分别改变候选 models/driver 代码证明选择生效，检查原 endpoint、角色、参数、
credential refresh 与候选调用账。另覆盖缺 default 不调用/不晋升/不重跑、跨 Root 服务拒绝、
缺失或损坏来源不初始化、重复接续拒绝及接续 Root 不改正式设置。测试已写，未运行。

没有运行测试、Gate、CI、build、lint、AST 或产品命令。没有改 Manager Channel 段或 Snapshot。
主协调者需对累计栈做静态 review、共享文档入口对账，然后按用户授权 push/开 stacked Draft PR。
真实插件数据准备、真实模型/工具调用和发布故障恢复尚无本层运行证据，不能据此部署。

本片恢复点：`/tmp/akasic-latest-control-cf5cc7b3-before.tar`，完整源码基线为上述 commit。
模型接线恢复点：`/tmp/akasic-model-settings-b9356e14-before.tar`；源码基线为 `b9356e14`。
前片恢复点：`/tmp/akasic-programmatic-v2-df9179cf-before.tar`。
正式 workspace、安装 cache 和原 checkout 未被修改；代码回退不代表插件数据已回滚。

后续可独立评审第三份 validation Root 是否能合入现有候选调用 Scope；范围应只涉及
候选实例、真实资源关闭和精确制品归属，先证明隔离与凭据合同。本片不重写 Root，
不新增 EventBus，也不接管 TOML 字段删除或 Manager Channel/Snapshot。

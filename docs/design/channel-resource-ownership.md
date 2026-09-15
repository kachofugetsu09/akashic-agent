# Channel 资源归属迁移

- 状态：已授权实施的 Draft；只做静态阅读与 `git diff --check`，测试未运行。
- 基线：`df9179cf9e145d2cfd661e1b90780e26a2d6b006`。
- 上游：[0071](../decisions/0071-plugin-composition-and-whole-runtime-updates.md)、[持久化状态地图](persistence-state-map.md)。
- writer：独立 channels-v2 worktree；Manager 验证与晋升合同由另一 writer 负责。

## 归属与调用链

```text
┌─────────────────────────────────────┐
│ Manager：关闭接纳 → 排空 → 整 Root 退出 │
└──────────────────┬──────────────────┘
                   ▼
┌─────────────────────────────────────┐
│ channels provider：固定贡献、binding │
│ 贡献 Context Effect：adapter / task  │
└──────────────────┬──────────────────┘
                   ▼
┌─────────────────────────────────────┐
│ Telegram / QQ / Akashic：协议与回执   │
└─────────────────────────────────────┘
```

底座不导入 Channel provider，不寻找 factory export，不编译 Channel catalog，也不保存
ChannelGeneration。`ChannelDefinition.factory` 是贡献插件的实际闭包；配置解释和控制回调归插件。
`channels` 是必须显式选入完整组合的普通 provider，没有自动补装或旧 Core facade。
Snapshot 只删除 Channel 专用字段，不调整其他 writer 的验证字段或提交语义。

`SourceAdmission` 绑定实际 Root、来源关闭/开放回调与宿主 boot identity。换 Root 不换 boot；
新宿主才产生新 boot。Manager 的整体操作关闭来源和新 lease，旧工作继续持有原 binding。
启动事件改为可等待的串行事件，全部 adapter closed-ready 后才提交 stable、开放来源。
排空取消时只有尚未释放的 Root 可以重开；已开始释放时必须按整体选择重建。

注册先把关闭 Effect 登记到实际贡献 Context，再允许 factory/start。监听 task 保存在这个
Effect 持有的 binding 中，adapter.stop 先正常结束监听，随后收回尚未结束的 task。
若另加更晚的 task Effect，LIFO 会先取消监听，再让 adapter 等待已取消的 task，因此不这样注册。
停止失败保留 binding、adapter、task 与预约输入；并发关闭加入同一关闭任务。
没有跨 Root 连接共享，也没有恢复旧内存指针即视为资源恢复的路径。

## 数据与外部效果

| 对象 | 增加与更新 | 减少条件与 owner | 恢复证据 |
|---|---|---|---|
| Message / Control | 来源插件按既有协议追加；/stop 追加 pause | 本改动没有正文减少权限 | 原消息日志 |
| inbound handoff | Bus reserve/prepare/complete/retain | 只由原传输 owner 确认接纳后 complete；停止只 defer | 原 handoff 行及附件引用 |
| channel identity | 原 identity owner 按 receipt 增加/更新 | 失败接纳仅按精确 receipt 回退 | 原 identity 表及版本 fence |
| adapter / listener | 本次 Context 初始化 | Scope 正常退出；失败保留原句柄 | 原 Root 与 binding 关闭回执 |
| 凭据 client | Telegram 在实际启动 scope 申请 | adapter 连接确认关闭后释放 client，再退出 Root 的凭据 factory | 原句柄及失败回执 |
| stable / 制品 | 仍由整体选择 owner 提交 | 无新增 GC 或指针回退协议 | 原 selection / archives |

`InputCustody`、身份及附件端口仅暴露明确方法；不把完整 Bus、任意 SQL 或完整 store 给 provider。
隔离装配只收到显式拒绝 I/O 的端口，不借用正式 workspace。独立验证宿主若需要接纳测试输入，
现在由 `ValidationHost` 将自己的 MessageBus 绑定到 `input_custody`，不能借用主宿主。

Telegram 的 `/stop` 回调通过来源的现行 CHANNEL_INPUT 提交 pause 并等待旧工作；ack 从原
binding 发送。去重在 await 前取得，重复控制不再次 pause 或发送。idle 仍返回明确 ack。
发送报错或取消保留真实失败/未知回执；provider 不重放未知发送。普通 pending 输入在开放后
由 provider Scope 的任务请求 Bus 恢复，关闭期间不借用内部 recovery lease 绕过许可。
恢复失败留下 pending 行和诊断，不撤销已提交 stable，也不伪造成功。

## 静态验收与交接

测试源码迁移覆盖请求 Scope、lease 排空、durable reservation、取消、停止失败、旧 port 失权、
真实 Channel identity、客户端展示及凭据释放。新增 provider 测试检查真实 Context 的资源归属。
这些测试均未执行；没有 Gate、CI、build、lint、AST 或产品运行证据。

恢复点：`/tmp/akasic-channels-v2-df9179cf-before.tar` 与基线 commit。源码备份不代表运行数据备份。
未修改正式 workspace、cache 或旧草稿 worktree。主协调者合并 writer 切片时需加入文档索引，
核对独立验证宿主的 I/O 绑定，并审查累计生命周期与正式安装组合是否显式包含 channels provider。

## 39b0b498 后的验证宿主接线

本切片基线为 `39b0b4988492e1e77be013b13bf9acf448e956c8`，只修改 Manager 的
验证宿主构造及 SourceAdmission 标记、ValidationHost、独立测试和本文。
不修改 plugin_latest 或晋升控制。恢复点为 `/tmp/channel-validation-39b0b498-before.tar`。

```text
┌──────────────────────────────┐
│ 普通验证 program / 验证 Root │
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│ 独立 InputCustody / MessageBus│
│ admissions / handoffs / identity│
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│ 本次验证 workspace/sessions.db │
└──────────────────────────────┘
```

EventBus 仍只处理事件。MessageBus 绑定本次验证库的 SessionAdmissions 和
InboundHandoffStore；ChannelIdentity 与附件同样只绑定验证库。恢复回调只取验证 Root
当前公开 lease，不借主 Bus。没有出站 dispatcher，也不自动运行来源启动事件。
`candidate or self._validation_only` 标记 SourceAdmission，验证 Root 仍执行普通 program，
但不能借 child 的 `candidate=False` 启动正式 listener。

关闭顺序为 Manager 运行资源 → Root → MessageBus → EventBus → 本地连接 → parent lease。
Bus 释放接纳租约后才关闭 stores；失败不移除 host 或关闭其依赖连接。这里没有新建关闭队列，
沿用 ValidationHost 的 lock 和 Manager 的原 cleanup owner。MessageBus 自身已失败的
`_close_task` 可能持续报告原失败，本切片不改它的重试协议，也不承诺所有故障可自动恢复。

验证数据的增改减沿用上表：输入交接增加、明确拒绝才结算行、停止释放接纳租约但保留 pending；
identity 按原 receipt 协议写入，消息只追加。本切片不复制正式业务数据、不删除验证库，
停止后验证库保留输入与 program 证据。源码备份不代表运行数据备份。

新增 `test_channel_validation_custody.py` 使用真实 Manager 验证 Root、MessageBus、SQLite
owner 和普通 program，检查独立交接接纳/结算、pending 保留、关闭失败保留句柄、正式库
完整 SQL 内容不变和关闭后拒绝输入。测试未运行；只做静态搜索、阅读及 diff --check。

### 生产分发与默认组合遗漏（只读交接）

- `scripts/build_plugin_distribution.py:426` 扫描全部 `plugins/**/plugin.py`，会发现 channels；
  静态未发现 provider 打包枚举遗漏，未实际 build。
- `docker/host-runtime/profiles/default.json:151` 选入 akashic_clients，但整个 profile 没有
  channels，且其 depends_on 缺 channels。需由组合 writer 显式补选 provider 和依赖关系。
- `tests/fixtures/formal_plugins.py:26` 的 FULL_RUNTIME_PLUGINS 同样包含 akashic_clients
  却没有 channels，不能作为已完整的生产组合测试证据。
- `bootstrap/tools.py:208` 只读取明确开发目录；正式安装器按 profile 安装，
  不会自动补 channels。旧 installed/stable 选择也需显式完整换代；本切片不改正式状态。
- Telegram/QQ 不在默认本地 profile，属于有意后续选入；扩展组合选入它们时必须同时选 channels。
  未读取任何正式安装 cache 或运行选择，因此不声称已审计部署实例。

## 930b5ede 后：验证宿主不再创建子 Manager

本切片只收缩隔离装配与关闭 owner。父 Manager 继续固定候选归档、持有更新请求与
parent lease，并独占撤销、提交阻断和 stable；没有修改 latest、消息证据读取或安装回退。
上一节 profile/FULL fixture 遗漏已由协调者修复，不是本切片待办。

```text
┌─────────────────────────────┐
│ 父 Manager：exact candidate  │
│ 同一装配方法：实例化/挂载/封存 │
└──────────────┬──────────────┘
               ▼
┌─────────────────────────────┐
│ ValidationHost              │
│ 独立 Root / SnapshotStore   │
│ Task / Process / Bus / stores│
└──────────────┬──────────────┘
               ▼
┌─────────────────────────────┐
│ 排空 lease 后关闭 Root/module│
│ 失败保留原宿主；成功释放 parent│
└─────────────────────────────┘
```

`_prepare_validation` 复用 `_resolve_composition_root`，删除第二份挂载/ready/sealing
流程。物理 Root 仍相互独立，初始化前固定服务；不交换已封存服务，不合并第一候选与验证实例。
ValidationHost 没有安装、selection、reload journal、发布操作或嵌套 Manager。
每个构建 Root 从分配起归真实 owner；已导入模块的卸载回调仍登记在原 Root，
验证实例不进入父 Manager 的 building Roots 或 stable alias 集合。

关闭先关闭来源接纳、Task、Process，再由 SnapshotStore 核对 lease 并关闭 Root；
未发布 Root 由 ValidationHost 直接关闭。Root 内的模块和 Scope 关闭成功后才解除资源归属。
随后关闭本地 frames、输入 Bus、EventBus、存储连接，最后释放 parent lease。
准备取消且关闭失败时，open_validation 的 finally 不立即重复关闭；显式 retry 或 terminate
继续使用原宿主。消息、pending 和 plugin-data 保留，不新增数据减少或兼容性裁决协议。

Mobile UI 的运行实现只读取传入对象的 snapshot_store，本次传真实 ValidationHost，
避免无继承 context 的请求回落到主 Store。其现有构造类型仍写 Manager；因文件 writer
边界未扩大，只在接线处记录类型例外，后续可把该类型收窄到 Store，不应恢复子 Manager。

新增 owner 测试覆盖未释放 lease 保留模块、无 context 的 UI 查询使用验证 Store、
取消构建且关闭失败保留未发布 Root、显式重试后卸载模块。迁移既有真实程序/MCP/关闭
并发测试，输入接纳测试明确拒绝再次构造完整 Manager。所有测试只写未运行。

恢复基线：`930b5edebc240ecc82dd9eda61a1421d1e17c167`；源码备份：
`/tmp/validation-host-930b5ede-before.tar`。验证限静态搜索、阅读与 git diff --check，
没有 Gate、CI、build、lint、AST 或产品运行证据。正式 workspace、cache 与其他 writer
worktree 未改；共享 INDEX、NOW 和整体设计指南未改。

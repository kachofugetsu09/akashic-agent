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
必须由验证 writer 将它自己的 Bus 绑定到 `input_custody`，不能借用主宿主。

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

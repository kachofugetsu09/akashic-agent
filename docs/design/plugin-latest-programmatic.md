# latest 普通程序与更新授权

- 状态：已按维护者授权实现，等待主协调者只读 review；未运行测试。
- 本层基线：`df9179cf9e145d2cfd661e1b90780e26a2d6b006`。
- 上游：[0071](../decisions/0071-plugin-composition-and-whole-runtime-updates.md)、[0070](../decisions/0070-plugins-own-persisted-data.md)。
- 本文由 programmatic writer 独立维护；共享 INDEX、NOW 和整体设计由主协调者合栈后统一对账。

## 调用与归属

原 `Validation.run` 已调用普通 `reply.execute.v1`，但额外要求 `passed/reason` JSON，
安装后的 watcher 还会自动运行该程序。现在安装只固定候选和原请求，Agent 显式调用
`plugin_latest`：`run` 执行原请求的 prompt、工具与材料选择，`status` 只读，`revert` 撤销授权。
没有新的后台裁判、业务兼容性检查或批准 JSON。来源 watcher 只沿原 Message/Delivery owner 报告更新状态。

```text
┌─────────────────────────────────────────┐
│ plugin_install → 固定 update / candidate │
└────────────────────┬────────────────────┘
                     ▼
┌─────────────────────────────────────────┐
│ plugin_latest run → Task → reply.execute │
│ status 读取过程；revert 撤销提交权        │
└────────────────────┬────────────────────┘
                     ▼
┌─────────────────────────────────────────┐
│ 普通 complete → 关闭隔离资源 → 请求晋升 │
│ 释放调用租约 → 换代 → 同步授权/base 检查 │
└────────────────────┬────────────────────┘
                     ▼
┌─────────────────────────────────────────┐
│ 唯一 stable 提交 → 原状态/通知 owner 报告│
└─────────────────────────────────────────┘
```

来源只把实际普通 `Output.finish=complete` 视为程序正常结束；内容不作为第二套授权协议。
异常、取消、缺少完整结果和重启后的未知都不请求晋升。已有调用只能查询，不能因 query
或通知重跑。程序正常结束后默认请求晋升，发起 Agent 仍能在自身正式调用租约释放前
读取普通工具结果并 revert。最终消息保存在候选证据库和原工具结果中；运行中 status
按精确 candidate 读取隔离消息，退出后保留证据目录地址，不重新打开数据库制造执行状态。

底座只拥有固定候选与基线、实际 Root/Scope、隔离消息 owner、有限操作和唯一提交点。
`UpdateStatus` 补充候选身份、既有 journal 阶段和证据路径，不新增可变运行状态表。
普通读取不能取得提交许可；可信更新来源显式调用 `publish` 才请求晋升，operator 的显式入口保留。

## 隔离与数据

候选结构装配和 latest 程序分别创建独立实例，两者与正式新实例使用同一组不可变制品和配置输入。
底座不再备份正式消息库、复制历史 binding/附件、扫描历史排除声明，或按文件头识别并复制业务 SQLite。
各实例只获得自己环境下的数据目录。配置仍来自固定归档，凭据授权仍在原宿主边界；
`apply(ctx)` 和普通工具负责自身数据初始化及所需样本，缺少数据就明确失败。
这个改变不表示“空库验证已证明现有业务数据兼容”；需要真实历史数据的检查必须由该插件
通过自己的明确数据准备流程提供，不能恢复 Core 全目录复制或把正式目录链接进候选。

| 对象 | 增加/更新及 owner | 减少与恢复 |
|---|---|---|
| 原安装请求、latest 调用请求 | plugin_update 在自己的 owner_records 追加原参数；执行结果由普通 ToolResult Message 保存 | 不自动减少；原 MessageLog 备份保留请求和正文 |
| 候选 MessageLog、binding、附件、plugin-data | 只在该次隔离目录由实际消息/插件 owner 增加或按自身协议更新 | Scope 只关闭连接和资源，不删除证据目录；没有自动 GC |
| 完整制品与配置 descriptor | 安装/归档 owner 只增加；隔离副本核对精确内容身份 | 本层不清理归档；基线 archive 是源码恢复点 |
| journal | 既有 owner 保存调用证据与错误，revert 推进原候选 discarding | 诊断不等于物理效果回滚；不删除旧事件 |
| stable | 原 selection owner 同步核对授权和 base 后提交 | 已提交或写入不确定不能伪称 revert 成功；恢复沿原 owner |
| 正式业务数据 | 本层无复制、迁移或回滚权；正式新插件仍处理自己的数据 | 不删除、覆盖或迁移正式状态 |

## 取消、失败与真实 owner

发布任务先等待正式调用租约归还，再关闭候选与进入整体切换，等待仍受原 Manager deadline 限制。
revert 先同步把原候选推进 discarding，再取消相关程序/发布任务。等待和清理失败会明确报错，
提交权不会恢复；资源仍归原 Scope、ValidationHost 或 publication owner，不能先移除句柄。
调用 scope 没有退出或资源没有真正关闭时禁止发布。

提交前再次检查 update、reload transaction、候选授权和 stable base；检查与同步选择提交之间没有 await。
提交已确认或结果不确定时 revert 拒绝，不恢复旧内存指针、安装记录或数据来冒充取消成功。
revert 在资源释放期间发生时可能留下需要显式恢复的错误；“授权已撤销”与“清理/恢复已完成”不同。

## 静态交付与后续验证

本层只进行静态搜索、阅读与 `git diff --check`。新增/适配测试覆盖普通文本结果、显式调用、
读 latest 不晋升、失败与未知不重跑、运行中 revert、等待调用租约、提交后拒绝假取消、
固定制品、插件自建候选数据、正式业务库锁不阻塞装配、资源清理失败保留 owner。
原 Core 数据复制算法的专属测试随算法删除，生命周期回归保留。

没有运行测试、Gate、CI、build、lint、AST 或产品命令。没有改 Manager Channel 段或 Snapshot。
主协调者需对累计栈做静态 review、共享文档入口对账，然后按用户授权 push/开 stacked Draft PR。
真实插件数据准备、真实模型/工具调用和发布故障恢复尚无本层运行证据，不能据此部署。

恢复点：`/tmp/akasic-programmatic-v2-df9179cf-before.tar`，完整源码基线为上述 commit。
正式 workspace、安装 cache 和原 checkout 未被修改；代码回退不代表插件数据已回滚。

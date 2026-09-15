# 通用候选丢弃与安装更新结算

状态：已实现，仅静态复核，运行验收未执行。基线：`324d26e3`。
依据：[0071](../decisions/0071-plugin-composition-and-whole-runtime-updates.md)。

## 问题与归属

安装入口先保存 armed 更新恢复点，再准备 ready candidate。原先
`AppRuntime._discard_plugin → PluginManager.drop_candidate → _drop_ready`
只关闭候选并清空 ready，安装 latest 指针和 armed 更新仍在。
随后 `discard_update` 找不到 ready，无法通过原入口完成安装回退。

```text
drop_candidate
  ├─ reload 关联安装更新 → discard_update(update_id)
  │                         ├─ 撤销提交权、等待调用退出
  │                         └─ operation：关闭候选 → 安装 owner 恢复文件
  └─ 无关联更新 → operation：关闭普通候选
```

关联来自现有 reload journal，不增加第二份状态。更新路由在进入 operation
之前执行，沿用 `discard_update` 的候选匹配、租约、取消与已提交拒绝规则。
成功仍返回 `publication_state=discarded`；失败直接传播，不伪造完成回执。
bootstrap 消费者无需改变，普通候选保留原有运行清理路径。

## 持久化与失败边界

安装增加不可变制品和更新记录。安装 owner 在显式丢弃时恢复该更新保存的
旧安装指针与启用状态，再将 armed 更新记为 rolled_back；首次安装可移除
本次新增的指针文件与清单项。旧记录、制品、正式选择和业务数据不随丢弃删除。
恢复证据仍是原 update 记录，文件状态冲突仍明确拒绝覆盖。

后续切片基于 `1b60cd2e` 补上关闭后结算重试：实际 `discard_latest` 等待租约归还、
Root 与 generation 关闭成功后，既有 aborted 事件增加 `candidate-root-closed`
关闭回执；失败不写该回执。SnapshotStore 仅在关闭成功后移除 snapshot，失败继续
保留原资源 owner。随后取消或文件回退失败即使已经清空 ready，也可用同一 update ID
再次调用 `discard_update`，不重建或重复关闭候选。

重试在现有 operation 内重新读取原 armed 更新，核对 reload/plugin/artifact 关联、
aborted 和关闭回执；候选 snapshot 不得仍被 store 保留，不能有新 ready、pending
候选、building Root 或该候选的验证 owner。保留中的已提交/不确定 publication 拒绝
回退。selection_candidate 必须有完整证据，当前唯一 stable 必须仍是原 base ref。
检查后再核对当前 operation 提交许可，才允许安装 owner 同步恢复文件。
已提交、其他候选、外部新选择及未知历史关闭证据均不能被当成本次重试。

`update_rollback.rollback` 在写任何文件前核对当前指针和目标插件启用状态。
它接受本次安装的新值与已恢复的旧值：写指针前后、写清单前后失败都能重试；
首次安装删掉本次指针文件后同步目录失败，也可从文件缺失状态重试。
文件恢复全部成功后才在原数据库事务中记 rolled_back；数据库提交失败仍可重试。
不属于这组已知值的文件状态拒绝覆盖，清单中其他插件条目按当次读取保留。
没有新增 schema、镜像状态或正式运行数据操作；旧事件没有关闭证明时不补造回执。

`update_rollback.rollback_linked` 全仓只有定义，无导入、调用、字符串注册或
动态模块消费者；实际回退由 `ReloadJournal.rollback_updates → rollback` 完成。
删除死函数，保留历史退役迁移的路径与 hash 证据。

## 验收与交接

新增测试通过真实安装构造 armed/ready，检查丢弃后的 rolled_back、原安装
指针、清单、stable、业务数据与新制品保留，并再次安装及丢弃以观察旧 armed
约束不再阻塞新更新。普通候选沿用现有 hot reload 与 runtime control 覆盖。
后续测试覆盖关闭后取消、指针/清单写入前后失败的精确重试、资源关闭失败保留、
旧更新拒绝丢弃新候选，以及外部指针、启用状态、stable 改写后的拒绝。
本次未执行测试、Gate、CI、lint、build、AST 或产品命令，仅做静态阅读与 diff 检查。

源码恢复点为上述基线，修改前备份位于
`/tmp/akasic-drop-update-324d26e3-backup/source-before.tar`。
后续切片恢复点为 `1b60cd2e`，修改前备份为
`/tmp/akasic-discard-retry-1b60cd2e-backup/source-before.tar`。
文档索引由主协调者统一接入；本切片不修改其他 writer 的指南、latest 工具或验证宿主。

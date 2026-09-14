# 0069 · Binding 跟随调用已选的 runtime scope

- 状态：accepted / implementing
- 日期：2026-09-14
- 部分取代于：[0070](0070-plugins-own-persisted-data.md)（旧数据处理责任，取代第 4 项兼容性准入）
- 关联条款：PLG-003、PLG-004、PLG-009、PLG-013、PLG-018、RUN-008、RUN-009、ERR-001
- 部分取代：[0062](0062-tools-flow-through-provider-views.md) 中“已提交 binding 继续打开原归档实现”的选择

## 背景

旧的 `Bindings.open()` 会为每个 binding 重新装配一个 archive Root，并按持久化的
`root_ref` 恢复历史 generation。这样把“这次调用选择哪一个运行时”和“binding 保存了什么业务事实”
混成了两个 owner：普通任务在插件更新后会继续依赖旧 generation，历史任务和记忆也会被隐式绑定到
旧代码；每次打开还要重新导入组件、装配 Root 和等待资源排空。

本项目只承诺两个 runtime pointer：正式执行使用 `stable`，更新中的候选使用 `latest`；候选晋升前
发生崩溃时，启动恢复到本次更新开始前的 stable。不存在任意历史 generation 的通用复活承诺。

## 决定

1. Binding descriptor、binding ID、消息引用、历史事实和业务 metadata 继续按原合同保存，既有
   `sessions.db` 行不迁移、不重写。`root_ref` 可以作为现有 provenance 保留，但不再是普通执行打开
   binding 的历史代码入口。
2. `Bindings.open()` 使用调用者已经取得的 `RuntimeSnapshotLease`。调用期间始终复用这份 lease；
   不在每次 `await` 后重新读取全局 stable。没有 scope 时，只从该 `Bindings` 所属的 CompositionRoot
   获取一次 scope；已有不相干 Root 的 scope 不得静默改选 stable，而是明确失败。
3. 普通调用的选定 scope 是 stable，显式业务验证的 scope 是 candidate。candidate 仍使用独立数据、
   资源和 provider Root；本决定不把 candidate 变成正式执行 owner。
4. binding 打开的是当前选定 scope 中实际提供的 service。Tool、Delivery、Akasha、Models 和其他
   领域继续由各自 owner 检查 descriptor、receipt、effect key、目标和兼容性。已开始的外部效果不能
   仅凭同名或 schema 猜测重试；无法证明原 provider 合同兼容时，沿现有 terminal interrupted/error
   语义收尾。
5. 进程死亡时仍由 stable/latest pointer 和 reload journal 恢复：晋升前回到更新前 stable，明确提交
   后使用新 stable；不新增跨 generation 恢复队列、历史 Root 管理器或通用恢复框架。

## 理由与影响

调用 runtime 的选择只有一个 owner：`RuntimeSnapshotLease`。Binding 只保存不可变业务事实，避免
每个历史引用拥有一套隐含运行时。这样旧消息和回执仍可读，后续新工作可以使用当前 stable，而不把
旧任务永久锁在每次升级前的 generation 上。

本决定需要后续分层收口各领域对旧 binding 的停止、回执和兼容性处理；第一层只改变 Core binding
打开路径，不能单独宣称完整热更新恢复合同已经交付。

## 验收与恢复

- 同一调用在晋升前后仍观察到它开始时的 snapshot ID；candidate 验证不读取正式 stable service。
- 没有当前 scope 时，binding 只从所属 Root 获取一次 lease；跨 Root 调用明确失败。
- 删除或不可用旧 archive 不影响 binding metadata、消息引用和历史事实读取；普通打开不导入旧
  binding archive closure。
- 取消、lease 排空和 Root dispose 仍沿现有 `RuntimeScope`/`RuntimeSnapshotStore` owner；不修改
  正式 workspace、消息数据库、plugin-data 或 artifact。

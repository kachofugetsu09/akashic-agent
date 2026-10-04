# 0086 · 会话软删除：逻辑失效可恢复，Akasha 学习与会话数据继续参与

- 状态：accepted
- 日期：2026-10-04（2026-10-04 维护者裁定确认：软删会话数据允许继续存在于
  Akasha，只是不再在左侧栏显示——与本决策“软删不等于遗忘、目录排除”一致）
- 关联条款：SES-003、SES-005、STA-003、MEM-009、WEBUI-009、MIG-001、WSP-003
- 补充：[0073](0073-session-scope-routes-akasha-graphs.md)、[持久化状态地图](../design/persistence-state-map.md)

## 背景与决定

Chat 前端要引入会话删除。维护者明确选择软删除：逻辑失效、物理保留、可恢复；
并要求事先想清楚与 Akasha 等消费会话数据的插件如何交互。本层只交付服务端
能力与语义决策，前端交互在后续层实现。

决定：

1. `sessions.db/sessions` 增加可空列 `deleted_at`（NULL = 正常，时间戳 = 已软删），
   由 yoyo 迁移只增加列、不改写任何已有数据；不改 `updated_at`，不改
   `visibility`/`learning` 等接纳时固定属性。
2. 会话生命周期为：正常 → 软删（逻辑失效）→ 恢复；只有未来独立授权的
   “物理删除/遗忘”操作才减少消息，本层不提供任何物理删除路径。
3. Chat API 新增窄接口 `POST /api/chat/sessions/<id>/delete` 与
   `.../undelete`，幂等：重复 delete 保留原 `deleted_at`，对不存在的会话
   返回 404，非本聊天目录（非 `akashic:` 前缀）的会话返回 400。
   列表与导航默认排除已软删会话；直接访问（消息接口）仍返回全部消息并带
   `deleted: true` 标记，不 404 假装不存在，也不阻止只读查看。
4. Pins：置顶引用继续保留在 `navigation:pins` 记录中，但已软删会话不再
   返回 pin 会话行，前端以既有“会话暂不可用”占位呈现；恢复后 pin 原样可用。
   不自动减少引用（沿用“目标失联不自动减少引用”的语义）。
5. Akasha 交互语义（经代码调查确认，见下）：软删**不等于遗忘**。已被 Akasha
   吸收的记忆是独立事实，软删会话不删除已学记忆（书下架不抹掉读者脑中的知识）；
   软删会话在 Akasha 图重建（重放）时**仍参与**——重放与在线学习共用同一个
   `MessageConsumer`，输入是 `snapshot_heads()` 的全部会话加 `learning` 资格，
   软删不改变这两个输入，同一份 canonical 来源必须重建出同一张图。

## 调查证据（Akasha 与会话数据的交互）

- 学习数据源：`Learning.samples()` 从 `MessageCatalog.snapshot_heads()` 取全部
  会话 head，只按 `attributes.learning == "eligible"` 过滤（不按 visibility、
  不按任何目录呈现状态），再读取 `sessions.db/messages` 正文；向量复用
  `message_embeddings` 已存值（[006/0006](0006-akasha-v2-is-the-canonical-explicit-memory-engine.md)、
  F-007）。
- 重放：`rebuild_from_catalog()` 是“空图 + 无切换上界重放同一个
  `MessageConsumer`”，输入同样是全部会话 head + 学习资格。软删会话物理存在，
  重建输入不变，因此重放自然包含软删会话；这正是“同输入可复现图”
  （F-007A）的要求。
- 溯源引用：Akasha 图节点保存 `session_key`（`Turn.session_key`、
  `Applied.session_id`、召回出处 `ContextSource.session_id`）。软删后这些引用
  全部保留；召回渲染时 `catalog.reader(hit.session_id)` 仍读得到原消息
  （消息物理保留），出处不会悬空。若未来做物理删除，出处会在渲染时
  `召回出处消息缺失` fail-loud——所以物理删除必须走独立的 Akasha 协调协议
  （类比 interaction 撤销的 source fence，F-007C），不能借软删实现。
- 结论：本次软删除**不需要** Akasha 任何代码改动；上述语义以本决策为准。

## 理由

软删除是“名称明确的数据管理操作”的温和形态：它满足用户“从列表移除会话”的
意图，同时保留可恢复性与全部审计事实，与 `sessions.db/messages` 只追加合同
（SES-003/INT-002）不冲突。恢复能力是软删除与物理删除的分界：没有恢复，
误删就是数据事故；有了恢复，删除才敢做进默认交互。

把 `deleted_at` 做成独立列而不是写进 `attributes`：attributes 是会话接纳时
固定的身份事实（同 ID 冲突失败），删除是后来的管理状态，混进去会让身份
校验承担生命周期语义。沿用 `_has_metadata` 的先例，运行期容忍缺列旧库
（按未软删读取），迁移由 yoyo 在启动前完成。

列表排除放在 `MessageCatalog.sessions()` 的 SQL 一处：目录是所有列表消费者
（chat、通知、工作台、Wake 设置、控制面）的共同事实源，软删会话在语义上
已退出“live 目录”，逐消费者过滤会造成分页 total/cursor 不一致。

Pins 保留引用而不是过滤：置顶记录只存 typed 引用，目标失联、项目归档、
插件停用、分页缺席都不自动减少引用；软删是一种可恢复的目标失联，语义相同。
若过滤引用，恢复后置顶丢失，反而制造了第二份删除语义。

## 重要替代方案为何不选

- **物理删除会话及其消息**：直接违反 SES-003/INT-002 的“只有用户主动删除
  会话时管理命令才能减少正文”，且需要附件、embedding、compaction、Akasha
  出处等跨 owner 级联协议与备份审计，属于独立的“遗忘”授权范围。
- **用 `visibility=internal` 或新属性值表达删除**：attributes 接纳后不变，
  且 `visibility`/`learning` 被投递、学习等消费者按固定语义消费，复用会
  改变这些消费者的输入语义（内部会话不参与投递展示、可被学习策略排除），
  与“软删不等于遗忘”矛盾。
- **已删会话 404 或拒绝消息读取**：消息物理存在却假装不存在，既不诚实也
  挡住用户找回与导出；恢复前用户应能看到自己要恢复的是什么。
- **Pins 接口过滤已删引用**：见上，会丢失可恢复的置顶，制造第二份删除语义。

## 代价与重议条件

- 代价：已删会话仍占用存储；没有“回收站列表”管理视图（本层不需要）；
  `sessions` 表多一列；旧库必须跑 yoyo 迁移后才能软删（未迁移库直接软删
  明确失败，读取不受影响）。
- 维护者要求“遗忘”（彻底抹除某会话及其已学记忆）时：需新建物理删除决策，
  含 Akasha 协调、附件/向量/compaction 级联、备份审计与恢复演练；本决策
  的软删语义保持不变。
- 需要“回收站/批量管理”视图时：给 `sessions()` 增加显式 `include_deleted`
  管理参数即可，当前刻意不暴露。
- 若 Akasha 未来改为按目录呈现状态过滤学习输入（当前不是），重放一致性
  合同（F-007A）会被破坏，需先修订该合同再动过滤。

## 影响与验收

- 新持久事实：`sessions.deleted_at`；无自动减少协议；恢复依赖原行与原库。
- 真实行为验收（隔离 workspace + 真实 uvicorn + curl）：软删后列表消失、
  undelete 后回来、重复 delete/undelete 幂等、消息行数与正文物理保留、
  pin 引用保留且会话行缺席、直接访问带 `deleted: true`；yoyo 迁移只加列；
  Akasha 在线学习与全量重建均仍学习软删会话的样本。
- 校验：pyright 无新增错误、`plugin_boundary.py check`、
  `check_yoyo_migrations.py --base origin/main` 通过。

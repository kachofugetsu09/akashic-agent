# 0073 · Session scope 宽键路由 Akasha 物化图

- 状态：accepted / implemented（首版）
- 日期：2026-09-25
- 关联条款：SES-010、MEM-009、MEM-013、CTRL-003、PLG-001～PLG-014
- 关联草案：[未来方向草案](../design/akashic-future-roadmap-issue-drafts.md) 第 6～7 节（#369、#370）。本决定只提升“项目归属”和“项目记忆图”两部分，并改用通用 scope 宽键代替草案的 `session_kind/project_id` 专用字段；`working_root`、项目指令、coding context 仍是草案

## 背景

用户需要 Project → Session → Message 的组织方式，并希望记忆可以选择是否与全局共享；以后还会增加
`computer` 等维度。已有 Akasha 从全部可学习 Session 学习，只有 `learning=excluded` 这一个开关。

## 决定

1. **宽键属于 Session，不属于 Message。** `SessionAttributes.scope` 是接纳时固定的
   `维度 → 取值`，Message 经 `session_id` 继承。缺失维度即 `default`，旧 Session 不迁移就属于
   `(default, …)`。这相当于分区表的 DEFAULT 分区：加维度不改旧行。
2. **Core 只存不解释。** 维度由唯一 owner 插件通过 `SESSION_ADMISSION.register_dimension`
   声明取值校验；`projects` 插件拥有 `project` 维度。Web 在项目内的首条消息携带
   `session_dimensions`，conversation 在写入 Input 前完成 create-once 接纳。
3. **图是视图，不是分区。** 类比 Kafka：日志只有一份，每张 Akasha 图像一个 consumer group，
   有自己的成员选择器与进度。不相交分区无法表达“共享给全局”，视图可以。
4. **策略归 Akasha。** 每个 `(维度, 取值)` 一条 `global | isolated | off`，缺失为 global。
   isolated 在多维间传染并按显式偏键（如 `computer=lab&project=p_x`）命名图，不做 hash 取模，
   避免 Kafka 分区数变化导致的重排问题。
5. **策略 set-once。** 只能在该取值尚无 Session 时写入；写入与“尚无 Session”检查在同一个
   写事务内完成，接纳无法插入其间。因此每个 Session 的路由永远确定，首版不需要策略变更重建。
6. **存储。** default 图保持 `memory/akasha.db` 零迁移；独立图放在
   `memory/akasha-graphs/<sha256 前缀>/akasha.db` 与 `manifest.json`；独立图从成员的第一条
   消息开始学习，不设 cutover 上界。embedding 与召回记录仍共享。
7. **记忆槽位。** Akasha 提供 `plugin.claim.embedding_memory`，同一 Root 只允许一个
   embedding 记忆系统；一个 Akasha 实例管理全部图。

## 需要与不需要独立图的场景

| 场景 | 策略 | 是否独立图 |
|---|---|---|
| 日常对话、希望项目知识被全局利用 | global | 否，进入 default 图 |
| 客户项目、敏感资料、不希望串味 | isolated | 是 |
| 实验、临时测试，但想用已有记忆 | off | 否，不写入，只读 default |

## 未做与后续

- 策略变更（例如 isolated → global）需要名称明确的重建协议与恢复点，本决定不提供。
- “isolated 但也读全局”“isolated 且同时贡献全局”这两种读写集合分离的组合没有进入首版。
- 前端宿主侧栏按插件 ID 组合 `projects` 与 `akasha` 的查询；将来可以用专门的
  `project.settings` 插件槽位替代宿主对插件 ID 的认识。

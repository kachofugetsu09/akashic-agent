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
4. **策略归 Akasha。** 每个 `(维度, 取值)` 一条学习策略 `global | isolated | off` 和独立的 `recall` 开关，缺失分别为 global 和 true。
   isolated 在多维间传染。图身份是有序维度元组的无歧义 JSON 编码（前缀 `v2:`），
   不做 hash 取模；维度取值含 `&`、`=` 时仍不能串图。
5. **策略 set-once。** 只能在该取值尚无 Session 时写入；写入与“尚无 Session”检查在同一个
   写事务内完成，接纳无法插入其间。因此每个 Session 的路由永远确定，首版不需要策略变更重建。
6. **存储。** default 图保持 `memory/akasha.db` 零迁移；独立图放在
   `memory/akasha-graphs/<sha256 前缀>/akasha.db` 与 `manifest.json`；独立图从成员的第一条
   消息开始学习，不设 cutover 上界。embedding 与召回记录仍共享。
7. **记忆槽位。** Akasha 提供 `plugin.claim.embedding_memory`，同一 Root 只允许一个
   embedding 记忆系统；一个 Akasha 实例管理全部图。
8. **项目创建。** Web 先以稳定项目 ID 提交 Akasha 的 set-once 策略，再让 Projects
   按该 ID 幂等创建记录。Web 在本地保存未确认请求，刷新和目录读取只展示它；用户显式
   “继续创建”才以原 ID、原策略重放，或“停止尝试”只移除本地请求，不撤销可能已提交
   的策略或项目。没有 Projects 记录前，不开放可接纳 Session 的项目。
   Projects 不解释策略，Akasha 不解释项目名称。

本 PR 内先前预览版本使用分隔符拼接独立图键。新格式改变独立图摘要路径；不自动
移动、删除或回退读取旧预览图。已有预览工作区应先备份，再由维护者按显式逐图重建
协议核对。正式 default 图路径保持不变。

## 逐图故障隔离（2026-09-26）

MEM-013 的不可用状态按图保留。选择逐图停止，是因为每张图有独立的数据与消费进度；
一张图需要重建不应阻断其他项目。公共 embedding 配置仍由公共健康项报告，图错误由
Akasha 按图身份记录并汇总到健康视图；实际召回只报告目标图的错误，不回退到其他图。
状态只属于当前运行实例，重启从原图重新校验，不新增持久状态或自动重建路径。

```text
┌─ 图 A 需要重建 → 报告 A 的原因，保留原图
└─ 图 B 正常     → 继续学习与召回，A 的错误仍可见
```

验收使用真实插件安装链、MessageLog 和图文件：故障图没有被自动重建，健康图完成学习
和召回，且成功访问健康图后仍能观察到故障图状态。

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

## 聊天导航偏好（2026-09-30）

按 WEBUI-009，置顶是现有 `akashic_clients` 的私有导航偏好，不属于 Projects、Session 或 Akasha。
现有 `OWNER_STATE` 的 `navigation:pins` 只保存一份有序 `{kind, id}` 列表；项目 ID 和完整 Session key
来自各自 owner。按目标幂等修改使用既有 OwnerStore 事务，避免两台客户端更新不同目标时相互覆盖。
没有新插件、通用 sidebar framework、用户账户或 Session metadata 字段。

```text
Projects / Session 事实 ──▶ 侧栏投影 ◀── 服务端有序置顶引用
                                  ▲
                       当前浏览器的手动展开状态
```

会话资格由后端实际 scope 检查；Projects 只通过其当前 `project.list` 查询校验新项目置顶，不读取
它的 owner storage。已提交引用的读取与取消置顶不依赖 Projects 在线；旧会话通过原 Message reader
点读首条消息与 scope，不以近期目录首页是否命中决定存在性。界面继续从 Projects 目录读取名称，
目标暂不可用时显示可取消的入口；不复制项目名称或建立成员表。

新增/取消置顶只原位更新这一偏好记录；不减少 Session、Message、Project 或 Akasha 事实，不新增
schema 或迁移。插件停止/卸载不清除该记录；代码回滚保留引用，恢复同版本代码后可重读。浏览器
展开状态独立保存在本地，搜索临时展开不写回。服务端失败保留原偏好并报告；响应丢失可重读或按
同一目标重试，不能把未确认写入当成已取消。

## 项目可关闭 Akasha（2026-10-04）

创建项目可选择“不使用记忆”：Akasha 保存 `learn=off, recall=false`。学习与召回是两个独立变化轴；保留 `off` 的“不学习但可召回”合同，不把新选择加入学习枚举。任一 scope 禁止召回时，该 Session 不产生自动召回材料、查询 embedding 或召回记录，显式召回和记忆反馈工具也返回明确的策略拒绝。学习仍通过同一个成员选择器排除这些 Session，重建沿用该选择器。

新字段不改写旧策略或 Message；旧记录缺少 `recall` 时继续允许召回。两项策略一起 set-once，同值重放仍幂等。该选项只控制 Akasha；会话上下文、聊天记录及其他插件拥有的 Markdown 档案不随之关闭或删除。

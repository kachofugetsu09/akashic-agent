# 0048 · EventMail 统一信封并保持三类生命周期

- 状态：accepted / implemented
- 日期：2026-08-28
- 关联条款：PLG-014～PLG-016、PRO-001～PRO-005、WSP-001～WSP-004
- supersedes：0040 与 0047 中由 Content 或 Wake 分别拥有 Content/Alert/Context ingress 的局部选择
- superseded by：0049 对 Wake Content 检查和衰减池语义的局部选择

## 背景

现有 Content 是没有时钟的 durable inbox，Wake 另行拥有 Alert、Context 和主动决策。外部
Calendar、Fitbit、Steam 为上报事实直接导入 `plugins.wake.contracts`，Wake 也直接导入
Content、Drift 和 Memory 插件源码。代码位置因此变成隐含权限；Wake 无法在不改变行为的前提下
作为普通外部插件安装。Alert、Content、Context 同时又有共同的来源身份、不可变正文、冻结读取
和可恢复 receipt 需求。

## 决定

用一个普通 EventMail 插件取代 Content 邮箱 owner。EventMail 提供 Content、Alert、Context 三个
独立 source capability，并保存共同的不可变 envelope ledger。每类使用自己的 reducer 和合法
transition：Content 可选择与结算，Alert 可选择、确认、跳过与过期，Context 只可 supersede 与
过期。公开 API 不提供 `publish(kind, payload)` 或通用 status writer。

Wake 保留在当前仓库便于共同开发，但只能依赖公开 Plugin API 和本地声明的版本化 ServiceKey/
Protocol；其隔离提取 Gate 必须证明复制为外部 package 后行为等价。来源插件同样不导入 EventMail
或 Wake 源码，主动 adapter 使用可选 child Fiber，不阻止各自 MCP/UI 在消费者缺席时运行。

0047 的 provide-to-Tool 关系保持有效。Memory provider 决定与 `memory.recall.v1` 绑定的 Tool，
Wake 只 inject capability 并从 exact-generation catalog 解析，不复制 Tool schema 或 handler。

Wake 每次实际 Timer fire 先追加 attempt；attempt 引用 EventMail watermark、Turn 和 delivery
receipt，但不进入 EventMail。Dashboard 显示没有进入模型的合法终态和真实失败。

## 理由

- EventMail 只统一一项真实共同边界：不可变来源信封、顺序和冻结读取。
- 三个 source capability 让类型错误在组合或提交边界 fail-loud，不依赖运行时 `kind` 分支猜测。
- Wake 移除后 EventMail 继续接收事实，重新安装或替换 Wake 不需要来源迁移。
- 仓库位置不再授予插件访问兄弟源码或 Core 私有实现的权限。
- attempt 与领域邮件分离，清理诊断状态不会改变用户事实。

## 影响

- Core Yoyo `20260828_01_migrate_eventmail_state` 在 runtime 前备份两个旧 SQLite，把旧 Content 与
  Wake Alert/Context 全部迁入 `eventmail-builtin/eventmail.sqlite3` 并核对状态与 integrity。成功后
  旧 Content 根只保留在 migration backup，Wake 删除 Alert/Context 表；正式运行不保留第二真源。
- EventMail 查询投影允许原位更新，但它不是权威事实，必须有确定性 rebuild/digest Gate。
- Feed、Calendar、Fitbit、Steam 和 Wake 的直接插件 import 必须退出。
- Core 不新增 EventMail、Wake、Content、Alert 或 Context 名词和固定能力表。

## 验收

- [x] 三类信封同 identity/revision/bytes 重放幂等，不同 bytes fail-loud。
- [x] 原始 envelope 和 transition 只 INSERT；Content、Alert、Context 投影都可从 ledger 确定性重建。
- [x] Context 不可选择或投递，Alert 不进入 Content 兴趣初筛。
- [x] 来源的 MCP/UI 不依赖 EventMail 或 Wake 存在；重新装入 Wake 可读取 EventMail 积压。
- [x] Wake 和 EventMail 从任意外部目录加载，不依赖主仓库或兄弟插件源码。
- [x] 每次实际 Wake fire 都有终态，Content 不足也在 Dashboard 可见。
- [x] 本地 E2E 覆盖迁移、source → EventMail → Wake → Tool → delivery → settlement 与恢复。

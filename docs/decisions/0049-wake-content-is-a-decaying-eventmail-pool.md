# 0049 · Wake Content 是 EventMail 中的衰减池

- 状态：accepted / implemented
- 日期：2026-08-28
- 关联条款：PRO-004～PRO-006、PLG-014～PLG-016
- supersedes：0048 中仅按 EventMail due deadline 安排 Wake 检查的局部选择
- superseded by：无

## 背景

EventMail 迁移后，Wake 会在一次 admission 拒绝时记录 revision 已见，但 EventMail 仍保留 pending
Content。若 Timer 只跟随未见 Content 的 deadline，旧 Content 不再衰减、不会淘汰，Dashboard 也
没有固定频率的空检查记录。反过来，若每次 Timer 都让旧 Content 重新抽签，同一批积压会在没有
新事实时反复唤醒。

## 决定

EventMail 的 due pending/deferred Content 本身就是唯一 Content 池，不新增 Wake payload store。
Wake 为稳定 source/item/revision 在首次到期时只计算并保存一次初始质量，同时保存该 revision
是否已经贡献过一次“新到达检查”。初始质量沿用旧 Wake 的
`-ln(1-interest) × publication-confidence × eligibility` 尺度，其中 interest 已合成 preprocess 与
首次会话语义；以后不重新读取会话语义或重算静态特征，只按发布时间或首次观察时间以 36 小时
半衰期衰减，低于 admission floor 后不再贡献。所有仍参与的质量直接相加，pool mass 超过固定 threshold 才进入
Wake Turn，不使用随机抽签或 refractory。拒绝只消费本次检查，Content 留在池中；没有新到达时
不重复启动 Turn。

Wake 用独立于职责 Turn 的 one-shot Timer 循环，始终安排不超过五分钟的下一次维护心跳。这个
循环只记录 attempt、重算衰减质量和淘汰，不重新评分，也不启动第二个 Turn；Alert、Content、Drift
仍由原职责循环串行选择。每次职责检查也先维护 Content 池，再按 Alert 优先级选择 owner，避免
持续到期的 Alert 挡住池维护。revision 驻留至少 24 小时且低于 admission floor 后，Wake 只通过
EventMail 公开的 exact-ref `expire` command 请求逻辑失效。EventMail 以 CAS 拥有状态改变，并
保留不可变 envelope 与 `expired` transition。Core 不新增 Wake、EventMail 或 pool 专属定义。

## 理由

- 池内容只有 EventMail 一个 owner；Wake score ledger 只拥有一次性初始分，seen set 只拥有一次性
  新到达检查，不复制正文或生命周期。
- 新到达负责启动确定性 threshold 检查，旧池只贡献衰减后的分数，避免积压自行反复唤醒。
- 固定心跳让无内容、证据不足、拒绝和淘汰都能在同一个 attempt ledger 中观察。
- 维护 Timer 不等待 provider、delivery 或职责 Turn，因此长 Turn 不会停止五分钟记录。
- 单次维护失败会写入 `failed` attempt 和 error log，再按同一五分钟规则重排；不会静默吞错，
  也不会让一次外部读错永久停止维护。
- Wake 移到仓库外后仍只依赖 Plugin API、EventMail capability 和 Core one-shot Timer。

## 影响

- 一次 durable selection 成功后，冻结 snapshot 中所有当时 due 的新 revision 都视为已经贡献检查；
  初筛分页不再制造第二次检查。
- Wake v8 的 `content_scores` 以 source/item/revision 保存不可变初始质量、语义分和评分时间；v7 升级先用
  SQLite backup API 创建可恢复副本，再移除已失去语义的旧冷却时间并新增空 score ledger。现有 pending revision 在首次维护时
  补算一次，之后与新 revision 使用同一条路径。
- EventMail Content snapshot 暴露原始 `observed_at`，供 Wake 判断最小驻留期。
- `expired` 是逻辑失效，不删除权威 envelope、transition 或 Wake attempt。
- 五分钟是实际运行的最大维护间隔；停机期间不伪造理论 attempt。

## 验收

- [x] 拒绝后的 Content 留在 EventMail pool，下一次没有新到达时不再启动 Turn。
- [x] 每个 revision 的初始质量只计算一次，之后只随时间衰减。
- [x] 低于 floor 的 revision 不参与 pool mass；其余分数之和超过固定 threshold 才准入。
- [x] 低质量 Content 在 24 小时前不淘汰，达到驻留期后由 EventMail CAS 标记 expired。
- [x] 没有 Content 仍按五分钟心跳记录独立 attempt，并在重启后从最近一次 durable fire 续排。
- [x] 持续到期 Alert 和未结束的 scoped Turn 都不能阻塞池淘汰或五分钟 attempt。
- [x] 单次维护失败后仍会重排下一次心跳；旧 deferred Content 只参与池维护，不单独重启 Turn。
- [x] attempt detail 可重建 active、due、expired、scored、new、mass、threshold、below-floor 和 driver。
- [x] Wake/EventMail 外置提取 Gate 不依赖仓库内兄弟源码或 Core 特权定义。

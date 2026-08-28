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
Wake 只保存稳定 source/item/revision 是否已经贡献过一次“新到达推动”。每次有新到达时，用新条目
质量推动一次概率抽签，并用全池随时间衰减的质量放大概率；拒绝只消费推动，Content 留在池中。
没有新到达时不抽签。

Wake 始终安排不超过五分钟的下一次维护心跳。每次实际 fire 先记录 attempt，再重算衰减质量。
revision 驻留至少 24 小时且低于 admission floor 后，Wake 只通过 EventMail 公开的 exact-ref
`expire` command 请求逻辑失效。EventMail 以 CAS 拥有状态改变，并保留不可变 envelope 与
`expired` transition。Core 不新增 Wake、EventMail 或 pool 专属定义。

## 理由

- 池内容只有 EventMail 一个 owner；Wake seen set 只拥有一次性推动，不复制正文或生命周期。
- 新到达负责启动抽签，旧池只放大抽签，避免积压自行反复唤醒。
- 固定心跳让无内容、证据不足、拒绝和淘汰都能在同一个 attempt ledger 中观察。
- Wake 移到仓库外后仍只依赖 Plugin API、EventMail capability 和 Core one-shot Timer。

## 影响

- 一次 durable selection 成功后，冻结 snapshot 中所有当时 due 的新 revision 都视为已经贡献推动；
  初筛分页不再制造第二次推动。
- EventMail Content snapshot 暴露原始 `observed_at`，供 Wake 判断最小驻留期。
- `expired` 是逻辑失效，不删除权威 envelope、transition 或 Wake attempt。
- 五分钟是实际运行的最大维护间隔；停机期间不伪造理论 attempt。

## 验收

- [x] 拒绝后的 Content 留在 EventMail pool，下一次没有新到达时不再抽签。
- [x] 新 Content 可以借旧池的衰减质量提高 admission 概率。
- [x] 低质量 Content 在 24 小时前不淘汰，达到驻留期后由 EventMail CAS 标记 expired。
- [x] 没有 Content 仍按五分钟心跳记录独立 attempt，并在重启后从最近一次 durable fire 续排。
- [x] attempt detail 可重建 active、due、expired、new、mass、概率、draw、refractory 和 driver。
- [x] Wake/EventMail 外置提取 Gate 不依赖仓库内兄弟源码或 Core 特权定义。

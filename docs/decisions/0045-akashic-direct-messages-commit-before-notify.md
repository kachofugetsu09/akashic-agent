# 0045 · Akashic 主动消息先提交 Session 再通知客户端

- 状态：accepted
- 日期：2026-08-27
- 关联条款：AKC-001～AKC-003、MOB-002、MOB-005、MOB-008、OUT-001、OUT-003～OUT-004
- supersedes：[0016](0016-channel-delivery-uses-complete-logical-messages.md) 与 [0044](0044-akashic-channel-uses-web-and-mobile-adapters.md) 中 Akashic 主动消息“adapter 送达后再投影 Session”的部分

## 背景

Wake 已能把成功投递追加到目标 Session，Scheduler 与 `message_push` 却只发送 Web/Mobile
事件。移动端因此需要另一套 proactive 正文身份、持久 inbox 和历史合并规则。客户端既要记
事件 cursor，又要猜这条正文何时成为 Session Message；Scheduler 离线发送还不会推进目标
Session 的 `seq`。

## 决定

1. 面向 `akashic` 的 non-passive 完整消息，在共享 Channel dispatcher 内先幂等追加为目标
   Session 的 assistant Message，再通知 Web 与 Mobile adapter。
2. 该消息由 SessionDB 分配唯一 `message_id` 与连续 `seq`，metadata 固定包含
   `effects.post_commit = "suppress"`。调用来源仍可独立生成内容，但不再拥有第二份正文。
3. Web 收到带 canonical `session_message_id` 的 `message.final` 后回源 Session；Mobile 只收
   `session.updated {message_id, head_seq}`，再比较 Room 中连续最大 `serverSeq` 并拉取缺尾。
4. Mobile 删除 `message.proactive`、`proactive:*` 临时身份与按内容/时间兼容合并。Room 消息行
   本身就是本地同步进度；不新增或持久化 history cursor。Realtime event ACK cursor 仍只拥有
   传输重放，不充当 Session 进度。
5. Session 提交成功就是 Akashic 逻辑投递成功。adapter 通知失败或没有在线设备只影响即时
   可见性；重连后的 Session list/head 比较负责补齐。Telegram、QQ 等外部渠道仍按 provider
   receipt 提交，不受本决定影响。

```text
Wake / Scheduler / message_push
              │
              ▼
     SessionDB INSERT assistant
       message_id + seq + suppress
              │
       ┌──────┴──────┐
       ▼             ▼
 Web message.final   Mobile session.updated
       │             │
       └──── history after local max seq ────┘
```

## 理由

SessionDB 已经拥有 Message identity、顺序和历史恢复。把它作为唯一提交点后，消息来源、客户端
通知和本地缓存成为三条独立变化轴：来源只产出 Message，adapter 只提示，客户端只投影。
无需再造 proactive 正文协议、服务端移动 cursor 或跨两套身份的合并状态机。

## 影响

- 这是 breaking removal。旧 APK 收到新协议会失败；没有兼容窗口、双发或 fallback。发布时
  清除移动端历史并重新配对全量同步。
- Akashic 客户端离线不阻止 Scheduler/Wake/message_push 的 Session 提交和 `seq` 推进。
- 附件先成为 Core artifact，Session Message 保存 artifact identity；Mobile adapter 只建立
  下载投影，历史仍以同一 Message 为准。
- 不创建服务端 per-device history cursor。多个客户端各自用本地缓存与服务端 head 对账。

## 验收

- 三种主动来源都只在目标 `akashic:*` Session 追加一次 assistant Message，`seq` 连续且
  suppress；重复 durable delivery 不产生第二条 Message。
- 无在线 Web/Mobile 时提交仍成功，重连后能从本地最大连续 `seq` 补齐。
- Core schema 与 Android consumer 均不存在 `message.proactive`；新 APK 不生成
  `proactive:*` 或 `ephemeral:*` assistant identity。
- 外部渠道的 provider receipt、失败和 unknown 语义保持不变。

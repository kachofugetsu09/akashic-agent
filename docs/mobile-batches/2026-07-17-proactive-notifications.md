# 2026-07-17 主动消息通知闭环

## 目标

主动消息必须沿 Akashic 已有的 `message_push` 和 Mobile durable inbox 到达手机；Android 只负责持久化、展示和系统通知，不为移动端新增一套主动任务实现。

```text
Akashic / 插件 / scheduler
        │ message_push(channel=mobile)
        ▼
MobileRealtimeChannel ─► durable inbox ─► WSS resume / replay
        ▼                                  ▼
 message.proactive            Room 消息 + cursor + 通知待办
                                            │
                                            ▼
                                       系统通知
                                       ├─ 点击定位消息
                                       └─ 通知快捷回复
```

## 真实问题与修复

服务端和 Room 原本已经完整处理 `message.proactive`，但 `RealtimeSession` 只把 `message.final` 发布到 `finalMessages` 通知流。结果是主动内容已经出现在会话里，应用退到后台后却只显示“实时连接”常驻通知。

- `message.proactive` 现在与最终回答复用同一个 `FinalMessageEvent` 和系统通知链。
- `deliveredAssistantMessageId()` 成为投递消息 identity 的唯一 owner：普通最终回答使用 canonical `message_id` 或 `ephemeral:<frame>`，主动消息使用 `proactive:<frame>`。
- Room 持久化、通知 PendingIntent 和 WebView 导航使用同一个 ID；点击通知能落到真实本地消息，不会只打开会话顶部。
- Room schema v8 新增通知待办；消息、附件关系、cursor 和通知待办在同一个事务提交。系统通知成功或被前台/权限策略明确抑制后才删除待办，进程在两者之间死亡会在下次启动重放。
- 系统通知继续使用稳定 message ID 与 `onlyAlertOnce`。若进程恰好在 `NotificationManager.notify()` 后、删除待办前退出，重放只会覆盖同一条通知，不重复响铃；消费者只在 `serverId` 真正变化时重订阅。
- 移除了 `RealtimeSession` 中容量 64 的内存 Channel；离线积压不再因为瞬时消费速度丢通知。完整通知 payload 会在 cursor 推进前校验，畸形 `message.final` 或 `message.proactive` 会回滚整个事务。
- 没有启用 Mobile Lab 的 proactive scheduler，也没有修改正式 workspace；测试通过运行中 Agent 的真实 `message_push` 工具发出受控消息。

## 自动验证

- `MessageNotificationPolicyTest` 新增主动消息 identity / 内容 / 附件语义测试，通过。
- Room migration v7→v8、最终消息与主动消息的畸形 metadata 回滚、65 条主动通知积压均有 instrumentation 覆盖；Pixel 7 定向 `47/47` 通过。
- Android `testReleaseUnitTest lintRelease assembleRelease`、R8 和 APK v2 签名通过。
- 本轮最终签名 APK SHA-256：`3115f24b1da0284d7d59c2345bc3588eba506019fe11d43ddbb11319059e22b3`。

## Pixel 7 / 隔离 Mobile Lab

设备 `28151FDH200478`，只连接 `/mnt/data/coding/akasic-agent/docker/debug/profiles/mobile-lab`；测试前以 SQLite online backup 备份 `sessions.db` 与 `mobile_realtime.db` 到 `/mnt/data/coding/backups/mobile-lab-proactive-e2e-before-20260717-133519/`。

1. 修复前真实 `message_push` 已写入会话，但通知栏只有“Akashic 实时连接”；截图 `/tmp/pixel7-proactive-background-notification.png`。
2. 修复后应用在后台收到“Akashic 已完成”，正文为“主动消息内测：通知修复已生效”，并提供系统 RemoteInput“回复”；截图 `/tmp/pixel7-proactive-notification-fixed.png`。
3. 点击通知后进入正确 mobile session，并把 `proactive:<event_id>` 对应消息定位到视口；截图 `/tmp/pixel7-proactive-notification-tap-target3.png`。
4. 从通知直接回复 `QUICK_REPLY_OK`。服务端日志真实记录 `Processing message from mobile: QUICK_REPLY_OK`，Agent 回复落回同一会话；截图 `/tmp/pixel7-proactive-quick-reply-field.png`、`/tmp/pixel7-proactive-quick-reply-result.png`。
5. 强停应用后发出离线主动消息：当前设备 cursor 从 `next=10399/sent=10398/ack=10398` 变为 `next=10400/sent=10398/ack=10398`，证明只持久化、没有假在线发送。冷启动立即退后台后收到“这是离线 durable 补发”，最终 cursor 收束为 `sent=10411/ack=10411`；截图 `/tmp/pixel7-proactive-offline-replay-notification.png`。
6. 全程 logcat 无 FATAL、RenderProcessGone 或 event sequence gap。
7. v7 真实用户数据库无损迁移到 v8；强停后再次发出 `DURABLE_ROOM_COLD_BACKGROUND_OK`，服务端先保持未 ACK，冷启动立即退后台后系统通知出现，cursor 最终收束为 `next=10463/sent=10462/ack=10462`。证据为 `/tmp/pixel7-durable-room-cold-background-final.png` 与通知管理器读回。
8. 独立 Review 首轮发现 state 更新可能取消并重订阅同一通知 Flow；补齐 `serverId.distinctUntilChanged()` 与 `onlyAlertOnce` 后复核无 Blocker / High / Medium。

## 边界

- 这组只补齐移动投递和通知，不改变 proactive 的触发、判断、频率、去重或插件数据。
- 通知仍遵守现有策略：应用前台且正在看同一会话时不重复提醒；后台或其他会话才提醒。
- 通知内容使用 Android private public version；锁屏只显示“解锁后查看新消息”。

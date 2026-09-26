# Android 极薄壳通知实验

2026-09-26：维护者确认先实验远程 Web 页面、免登录连接与消息通知；不实现账号密码。
这是独立 `com.akashic.shell` 客户端，不替换 `com.akashic.mobile` 的配对、本地历史和 OTA。

## 结构与 owner

```text
┌───────────────────────────┐
│ Session：已提交的 Message   │
└─────────────┬─────────────┘
              ▼ 只读目录和消息页
┌───────────────────────────┐
│ akashic_clients：通知 SSE  │
└─────────────┬─────────────┘
              ▼ session_id + message_id + seq
┌───────────────────────────┐
│ Android：系统通知 → 保存进度 │
└───────────────────────────┘
```

- Web 页面拥有聊天交互。Android 只拥有服务器地址、后台连接、系统权限和通知进度。
- 通知筛选由客户端插件负责：listed Session 中完成且含文字的 Output。其他渠道和 quiet 输出不提醒。
- 通知读取每页独立取得和释放 catalog scope；网络等待不持有插件租约。
- 本轮不增加 Core API、通知数据库、消息队列、FCM 或 UnifiedPush。

## 协议与恢复

`POST /api/chat/notifications/stream` 接收 `{"cursor": null}` 或
`{"cursor": {"akashic:session": 12}}`。首次连接先记录各会话现有 head，从此后的消息开始提醒。
已有进度按会话 seq 向前分页读取；目录页和消息页的大小不构成保留窗口。新会话从头读取。
使用 POST 是为了不把随会话数增长的进度放进 URL；请求只读，不创建服务端消费状态。

SSE 事件：

- `ready {cursor}`：确认起始进度。
- `message {session_id, message_id, seq, recorded_at, title, preview}`：完整消息的提醒。
- `cursor {session_id, seq}`：该消息页已检查到的位置，包括不需要提醒的消息。
- `heartbeat {}`：保活，不推进任何消息进度。

Android 先提交系统通知，再同步保存 seq。进度保存前退出允许重放；同会话通知使用稳定 tag，
当前通知含相同 message_id 时不重复提交。每个会话保留最新提醒，正文仍从服务端查看。
进入 App 不等于读过所有会话，不自动清除或压掉其他提醒。

系统通知和本地进度没有跨服务事务；这套协议提供可重放消费，不宣称严格 exactly-once，
也不把系统接纳通知当作用户已读。权限关闭、协议异常、进度写入失败均不越过未显示消息；
状态栏显示失败并保留进度。重新打开 App 或手动重连可重试。

## 持久化与回滚

- `sessions.db/messages`：通知路径只读，正常增加仍归原消息 owner；不更新或删除正文。
- 手机通知进度：消费成功后原位推进；首次连接保存基线，用户切换服务器时清除旧服务器进度。
  它是派生消费位置，不是聊天正文、消息投递回执或已读状态。
- 卸载实验壳会删除该 App 的设置和通知进度；不影响正式 Mobile 或服务端消息。
- 服务端无 schema 迁移。回退通知接口只使实验壳显示“服务器暂不支持通知”，聊天仍可进入。

## Android 后台边界

个人自托管实验采用原生前台服务和网络恢复重连，与
[ntfy instant delivery](https://docs.ntfy.sh/subscribe/phone/#instant-delivery) 的连接方式相同。
需允许通知和后台运行；[Android Doze 文档](https://developer.android.com/training/monitoring-device-state/doze-standby)
说明电池优化豁免允许网络访问，但其他限制仍存在。强行停止、关机、无网络或无法路由到服务端
都不能保证即时提醒；恢复连接后按已保存进度补发。局域网地址不会因开启通知就获得外网可达性。
FCM 和 UnifiedPush 是不同的传输部署选择，未来引入时不能改变消息 owner 和消费语义。

## 验收

- `python docker/debug/notification_feed.py` 使用一次性真实 SQLite 库，检查基线、跨页、跨会话、
  旧时间戳、断线重连、quiet/渠道过滤和读取前后完整性。
- `--serve PORT` 只监听回环地址，提供真实 Chat API/通知读取及临时消息写入入口，用于隔离 APK。
  夹具页面明确标记为验收页，不冒充正式聊天 UI。
- 真机分开记录：真实 LAN 页面连接、通知夹具的后台/锁屏/断线恢复/点击跳转。
  APK 构建、CI、正式部署和主动来源的真实送达各自取证。

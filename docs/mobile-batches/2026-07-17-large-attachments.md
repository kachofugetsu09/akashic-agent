# 大附件按需下载

## 问题证据

隔离 Mobile Lab 中存在一个 47,381,751-byte 的历史文档。服务端数据库确认它只有一条 ready 附件记录；日志中的连续 `attachment.download` 来自 Android 按固定分片推进同一下载，并非服务端重复创建附件或 WebSocket 重连。

旧客户端把历史投影中新发现的每个附件都写成 `pending`。因此只要恢复这段历史，手机就会立刻下载完整文档，抢占移动网络和同连接发送窗口。

## 最小状态链

```text
服务端附件描述
      │
      ├─ < 10 MiB ─────────────→ pending → downloading → cached
      │
      └─ ≥ 10 MiB ─→ remote ──点击下载──→ pending → downloading → cached
```

- `remote` 只表示服务端可取、本机尚未请求；不是失败、驱逐或离线状态。
- 显式下载复用已有 `retryDownloadedAttachment` 原生边界和 `AttachmentDownloadCoordinator`，没有新增协议命令或第二套下载器。
- 已缓存、下载中或失败的既有记录保持原状态，历史幂等刷新不会破坏断点或删除本地文件。
- 阈值与计费网络大上传现有 10 MiB 语义一致；恰好 10 MiB 视为大附件。

## Material 3 交互

附件继续属于消息内容，不升格成独立卡片或 sheet。文件名是主信息；`大小 · 尚未下载` 使用既有 `on-surface-variant`；右侧 44dp 文字按钮使用既有 primary，点击后原位变为等待/进度状态。没有新增色值、阴影或圆角层级。

## 网络抖动约束

插件目录更新使用 connection-scoped 控制帧。控制帧先给合法在途帧最多 30 秒释放同连接写锁，取得锁后才启动 3 秒 WebSocket 写超时：正常附件二进制或 durable event 不会再被 3 秒阈值误删；长期无法释放写锁或真正卡住的控制写入仍会被 4408 明确关闭并由客户端恢复。

## 验证记录

- `pytest -q tests/mobile_realtime/test_gateway.py`：23 passed。
- `pyright infra/mobile_realtime/gateway.py`：0 errors。
- `npm run typecheck && npm run lint`：通过。
- Android 定向 debug unit 通过；release unit、Lint、R8、assemble 和 APK v2 签名通过。
- Pixel 7 真机执行完整 `LocalDeliveryStoreTest`：25 passed。边界用例断言 `10 MiB - 1 byte` 自动排队、`10 MiB` 保持 remote；`MediaCacheStore.reconcile()` 后仍保持 remote；快速连续请求两次只发送一条 `attachment.download`。
- 隔离 Mobile Lab 注入真实 11.0 MiB 历史附件后，Pixel 7 先保持“尚未下载”；连续点按后只运行一条 offset 分片链，界面原位显示 32%→69%，最终变为“已下载”并出现分享操作。证据为 `/tmp/pixel7-release-reconnected.png`、`/tmp/pixel7-large-download-progress2.png`、`/tmp/pixel7-large-download-finished2.png`、`/tmp/pixel7-large-download-complete.png`；应用日志无 FATAL、AndroidRuntime 或 RenderProcessGone。

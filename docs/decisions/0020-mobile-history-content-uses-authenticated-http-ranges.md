# 0020 · Mobile 历史长正文使用认证 HTTP Range 恢复

- 状态：accepted
- 日期：2026-08-02
- 关联条款：MOB-001、MOB-003、MOB-006、MOB-007、SES-001、TST-005、TST-008

## 背景

第一阶段用 `answer.delta` 和紧凑 `message.final` 解除了实时长回复的单帧限制，但远端恢复仍由 `history.page` 把一页完整正文编码成一个 JSON。Android 在 WebSocket message 重组后执行 256 KiB 应用帧检查，因此传输层 fragmentation、把 page size 降为 1 或提高底层 frame 上限都不能恢复一条本身超限的消息。

历史恢复还必须保持 thinking、tool call、tool result、消息身份和顺序。把正文切成大量 durable WebSocket 事件会放大 inbox、ACK 和重放，并让批量历史阻塞实时聊天。

## 决定

1. WebSocket 继续拥有认证、历史游标、manifest 和短期票据控制面；正文 byte range 由同一已验证 endpoint 派生的 HTTPS 数据面提供。
2. 支持方在 `history.get` 显式声明 `content_ref_version=1`。Core 在固定 `snapshot_max_seq` 内按 `after_seq` 读取，页面超出安全预算时用 `content_ref` 替换最大正文；旧客户端仍走 page 协议，无法安全内联时返回 `upgrade_required`。
3. `content_ref` 使用 UTF-8 总字节数、SHA-256、编码和预览标识不可变正文。Core 不创建第二份持久正文，Range 请求每次从 SessionDB 重新读取并核对同一消息身份、session、长度和摘要。
4. 已认证 WebSocket 为 `message.content.prepare` 签发独立 audience 的短期票据。票据绑定 server、device、connection epoch、session、message、byte length 和 SHA-256；HTTP 执行前重新检查设备撤销和当前连接代际。
5. HTTP 只接受单个有界 `Range`，返回 `206`、`Content-Range`、强 ETag、`Content-Digest` 与 `Repr-Digest`。Android 先把原始字节 fsync 到临时文件，再推进 Room 中已确认的连续 offset。
6. Android 只有在总长度、整篇 SHA-256 和严格 UTF-8 解码全部通过后，才在一个 Room 事务中替换消息正文并消费恢复任务。临时文件与恢复行是可重建投影，不反向修改 SessionDB。

## 理由

控制面与大数据面分离沿用 MOB-006 已建立的设备授权和 TLS origin 边界。HTTP Range 直接提供标准的偏移、部分响应、验证器和断点续传，不需要把历史正文复制成 durable inbox 事件，也不会让单条 WebSocket JSON 承担逻辑消息总长。

请求级版本协商不依赖重新配对，已经配对的设备升级 APK 后可以立即启用；旧 APK 不会收到无法解析的 `content_ref`。

## 影响

- SessionDB 消息只读且仍是唯一权威正文，不新增迁移或服务端 blob。
- Core Mobile schema 增加 `message.content.prepare`；Android Room 增加可级联删除的正文恢复任务表和临时文件目录。
- 历史列表同步完成与长正文全部恢复完成是两个时点。客户端先落消息身份、预览和 thinking/tool 投影，连接可用时继续后台恢复正文。
- ticket 过期或连接变化时保留已确认 offset 并重新 prepare；摘要、Range 或 UTF-8 不一致时拒绝提交并显式失败。
- 回滚 Core commit 不改变 SessionDB；回滚 Android 需要按 Room 迁移合同保留 v12 数据库，不能降级打开。

## 验收

- 至少一条超过 256 KiB、包含跨 Range 边界 Unicode 的正文能在进程中断后续传，并逐字节等于 SessionDB。
- 首次投影与清除本地投影后的恢复结果在 message ID、seq、role、正文摘要、thinking/tool block 类型、顺序和关联上相同。
- 旧客户端请求不会收到 `content_ref`；无法安全内联时服务端明确拒绝，不发送超限帧。
- 正式 workspace、SessionDB 消息、线上 Gateway 和设备正式包在隔离验证前后保持不变。

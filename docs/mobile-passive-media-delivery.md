# Android 被动媒体链路交付记录

本文记录 Android 与服务端被动消息链路的能力边界、分组提交和真实验证证据。主动推送不在本轮范围内。

## 目标链路

```text
Android 文件选择器
    └── 附件草稿与本地预览
          └── WebSocket 二进制分片上传
                └── 服务端校验、落盘与 attachment_id
                      └── message.send(media_refs)
                            └── Agent 图片/文件输入

Agent / meme 插件输出媒体
    └── 服务端创建可下载 attachment descriptor
          └── message.final / history.page
                └── Android WebSocket 分片下载与缓存
                      └── 图片预览 / 文件打开 / meme 展示
```

## 安全与隔离约束

- 自动化测试使用临时 workspace、临时 SQLite 和临时附件目录，不连接线上 workspace。
- 模拟器只连接隔离 Gateway；只有最终验收才重启真实 runtime，并使用专用测试会话。
- 服务端不向客户端暴露绝对文件路径，只传递不透明 `attachment_id` 和展示元数据。
- 二进制帧为 `4-byte big-endian header length + UTF-8 JSON header + payload`；header 包含 `attachment_id` 与绝对 byte `offset`，payload 最大 128KiB。
- 上传 offset 在文件 `fsync` 后才推进 SQLite；断线后 `attachment.begin.ok.next_offset` 是唯一续传依据。
- 上传在协议边界校验文件数量、声明大小、实际大小、MIME、SHA-256 和会话归属。
- 普通 turn、短路命令和传输失败保持不同终态，不用空值或假成功掩盖错误。

## 能力矩阵

| 能力 | 当前基线 | 目标 | 状态 |
| --- | --- | --- | --- |
| 文字、思考与工具流 | Android 已支持 | 保持现有流畅度 | 已有 |
| 停止生成 | UI 按钮为空回调 | 接通 `turn.stop` | 待实现 |
| 文件与图片选择 | 附件按钮为空回调 | 系统多选、预览、移除 | 待实现 |
| 上传传输 | 协议只有 begin/finish 名称 | 全 WebSocket 二进制分片 | 服务端完成 |
| 上传进度 | Room 有 transfer 表但未使用 | 乐观进度加服务端确认 | 服务端完成 |
| 弱网续传 | 无 | 按确认 offset 恢复 | 服务端完成 |
| 图文和纯附件消息 | 服务端拒绝 media_refs | 两者均支持 | 服务端完成 |
| Agent 图片理解 | Web 可进入 VL 链路 | Android 上传图片等价接入 | 服务端完成 |
| 助手图片、文件、meme | 服务端发送路径，Android 丢弃 | descriptor、下载、缓存、展示 | 待实现 |
| 历史媒体同步 | 历史 DTO 裁掉 media | 全 mobile 会话恢复媒体 | 待实现 |
| 图片预览 | 无 | Material 3 全屏预览、缩放、GIF | 待实现 |
| 普通文件操作 | 无 | 类型、大小、进度、系统打开与分享 | 待实现 |
| Telegram 被动媒体等价 | Telegram 支持图片入站和图片出站 | Android 覆盖同类能力 | 待实现 |

## 设计约束

### Material Design 3

- 保留当前浅蓝背景、深蓝主操作和亮紫过程状态的语义映射。
- 卡片只用于附件草稿组、传输状态组和独立预览 sheet，不把每个文件机械包成卡片。
- 图片使用 1dp 内描边；浅色纯黑 10%，深色纯白 10%。
- 所有触控目标至少 44dp，按压缩放为 0.96。
- 上传、失败、重试和完成状态使用可中断动画，不使用不可逆的长关键帧。

### 色彩与排版

- 以 OKLCH 的感知亮度和色相稳定性审阅现有 Compose 色板，最终落为 Android `Color` token。
- 正文、辅助文字和状态文字分别校验实际背景上的对比度；修正时优先调整亮度。
- 正文保持舒适行高；动态进度、大小和耗时使用等宽数字，避免状态更新导致跳动。
- 文件名允许安全断行，截断时必须可在预览或系统详情中看到完整值。

## 分组与提交记录

### 0. Control 短路兼容

- Commit：`d66825dc fix(control): accept plugin-short-circuited turns`
- 功能：插件命令短路不再错误要求 `TurnCommitted`；普通 turn 的完整提交检查保留。
- 单测：control、lifecycle、AgentCore、Akasha 共 82 项通过；三个外部命令插件分别通过。
- E2E：重启真实 runtime 后，对 `telegram:7674283004` 执行 `/memorystatus`，返回完整状态且无 `turn 缺少 TurnCommitted`。

### 1. 服务端附件协议与存储

- Commit：`feat(mobile): add resumable websocket uploads`（本组提交）。
- 协议：JSON `attachment.begin/finish` 加 WebSocket binary chunk；单片 128KiB，按绝对 byte offset 串行续传。
- 一致性：同附件条带锁串行化 begin/chunk/finish；文件 fsync 后提交 SQLite offset；文件短于 offset 或摘要失败时进入 failed，并在相同声明再次 begin 时从 0 恢复。
- 边界：纯文件名、MIME、大小、SHA-256、设备/会话归属、重复 media ref、单消息最多 10 个附件及总字节上限。
- 验证：Pyright 0 error / 0 warning；目标 mobile 测试 51 项通过；完整 `pytest tests/` 2184 项通过；schema 生成一致性与 `git diff --check` 通过。
- E2E：隔离临时 workspace 中完成认证 WSS、上传至服务端明确确认 1MiB、断开、新连接 resume、从 1MiB offset 续传、finish、纯附件 message.send，并验证 `InboundMessage.media` 文件内容一致。

### 2. Android 上传与发送

- Commit：待完成。
- E2E：待完成。

### 3. 出栈媒体、meme 与历史

- Commit：待完成。
- E2E：待完成。

### 4. Material 3 预览与文件交互

- Commit：待完成。
- E2E：待完成。

### 5. 弱网、模拟器与 APK 发布

- Commit：待完成。
- E2E：待完成。

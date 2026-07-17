# 核心实时附件上传能力

本文只描述核心仓库拥有的附件上传语义。客户端的文件选择、草稿、本地持久化、缓存、重试界面、通知和预览由各客户端仓库负责，不属于本能力。

## 能力边界

核心拥有以下权威判断：

- 已认证连接的 device identity，以及 attachment 与 session 的绑定。
- `attachment_id` 绑定的设备、session、文件名、MIME、大小和 SHA-256 是否与首次声明一致。
- 服务端已持久化到哪个 offset，以及上传当前处于 `transferring`、`ready` 或 `failed`。
- `message.send.media_refs` 引用的附件是否完整、属于当前设备和 session，并且可以作为 Agent 输入。

核心不定义客户端怎样保存待发送草稿、怎样展示进度、怎样安排后台任务，也不要求任何具体平台或 UI framework。当前模块和表名中的 `mobile` 是历史实现名称，不扩大协议语义。

```text
┌──────────────────────────────┐
│ Client-owned preparation     │  file selection, draft, retry policy
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│ Core attachment upload       │  identity, ownership, offset, digest
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│ Agent media input            │  ready attachments only
└──────────────────────────────┘
```

## 协议顺序

1. 客户端发送 `attachment.begin`，声明 `attachment_id`、session、文件名、MIME、大小和 SHA-256。
2. 核心创建传输记录，或在全部声明一致时返回已有记录的 `next_offset`。
3. 客户端从 `next_offset` 开始发送 WebSocket 二进制分片。每帧由 4 字节大端 header 长度、UTF-8 JSON header 和 payload 组成；单个 payload 最大 128 KiB。
4. 核心按严格连续 offset 落盘。文件 `fsync` 成功后，才以 compare-and-set 推进数据库 offset。
5. 客户端发送 `attachment.finish`。核心核对已传字节、实际文件大小和完整 SHA-256，再把记录推进为 `ready`。
6. `message.send` 只接受属于当前设备和 session 的 ready 附件；核心把内部路径交给 Agent，不通过协议暴露服务端绝对路径。

`attachment.progress` 是服务端已提交 offset 的观察结果，不是客户端本地读取进度。进度在跨过 1 MiB 边界或传输达到声明大小时发布。

## 持久化与恢复

| 对象 | 正常增加 | 允许原位更新 | 逻辑失效 | 物理减少 |
|---|---|---|---|---|
| 附件记录 | 首次 `attachment.begin` 创建一行 | 已确认 offset、状态和更新时间按传输协议推进 | 摘要失败或文件短于已提交 offset 时进入 `failed` | 本能力未定义自动删除协议 |
| 附件文件 | 首次 begin 创建空文件，分片按 offset 追加 | 未提交尾部在恢复时截断到数据库 offset；failed 重试先截断到 0 | `failed` 记录不能作为 Agent 输入 | 仅在“新文件已创建但记录创建失败”时清理孤儿文件；正式保留策略不由本 PR 定义 |

恢复以数据库提交的 offset 为准：

- 文件长于数据库 offset，说明文件写入完成但 offset 尚未提交；核心截断未提交尾部后继续。
- 文件短于数据库 offset，说明持久状态损坏；核心 fail-loud，将记录标记为 `failed`，同一声明再次 begin 时从 0 重传。
- 相同 `attachment_id` 的设备、session 或元数据发生变化时拒绝复用，不能把新内容写进旧身份。
- 已 ready 的相同声明可以幂等返回；摘要失败的记录只有再次 begin 才会显式重置。

## 信任边界

核心在协议入口集中校验：

- 文件大小在配置上限内，文件名是 1～255 字符的纯文件名。
- MIME 形状和 64 位十六进制 SHA-256 合法。
- 分片非空、不超过 128 KiB、offset 连续且不超过声明大小。
- `media_refs` 不重复、单条消息最多 10 个，并且总大小不超过配置上限。
- attachment、设备和 session 的所有权一致。

违反声明、所有权、offset 或终态的不变量时返回明确协议错误。缺失文件、数据库与文件矛盾等内部损坏保持 fail-fast、fail-loud，不用空结果或假成功继续。

## 本层验收

- 协议 schema 生成结果与 `schema/mobile-realtime-v1.json` 一致。
- 单元测试覆盖声明校验、身份冲突、严格 offset、fsync 后提交、断点恢复、摘要失败和 ready-only 引用。
- Gateway 场景穿过认证 WebSocket、断线重连、二进制分片、finish 和 `message.send(media_refs)`，证明 ready 附件能进入 Agent 输入。
- 最终 diff 不包含客户端 UI、本地数据库、通知、缓存或平台构建工程。

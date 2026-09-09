# 插件 v3 Channel 附件持久化任务合同

> 2026-09-09：本文历史执行状态 unknown/UNKNOWN/uncertain 已被 [0063](../decisions/0063-execution-failures-have-terminal-results.md) 的明确失败终态和收尾规则取代；其他合同不变。


本文定义 C23：为 v3 Channel、Core built-in channel 与 Session message 提供唯一附件事实。
它是 [Channel capability 合同](plugin-v3-channel-capability-task-contract.md) 从 text-only 进入能力等价的前置，
也是 [持久化状态地图](persistence-state-map.md) 中 `uploads/` 禁止自动清理规则的可执行补充。

## 1. 目标与边界

本批次只建立以下事实：

1. Channel wire 只传 opaque `AttachmentRef`，不传任意绝对路径；
2. Core 原子导入 bytes/stream/local file，发布不可变 artifact；
3. Session message 与 artifact binding 在同一个 SessionDB transaction 提交；
4. provider 发送期间由 exact read lease 保持 artifact 可读；
5. 已发布 artifact 与已提交 message binding 不自动删除；
6. Telegram、QQ、Web、Mobile 现有附件协议经 adapter 接入，不以重写旧数据开始迁移。

本批次不实现按年龄、容量或“当前 prompt 是否使用”做 GC；不改写既有 `messages.extra.media`；
不把 Mobile 的 chunk、device inbox、cursor 和 resumable upload 状态搬进 SessionDB；不允许 candidate
读取正式附件或 provider credential。

## 2. 当前代码事实

- `AttachmentStore.write_bytes()` 已提供 `.part → fsync → os.replace`，但只有路径，没有 artifact id、hash、
  Session binding、read lease 或恢复记录。
- Telegram、QQ、Web 的入站媒体最终保存到 `<workspace>/uploads/`；Session message 的 `extra.media`
  保存字符串路径。已有消息仍引用文件时必须保持可读。
- Mobile 的上传、outbound snapshot、device inbox 与 message binding 由 `mobile.db` 自己持有；它是受保护的
  协议，不是可以被通用 Channel refactor 替换的缓存。
- 当前 v3 `ChannelInboundMessage`、`OutboundEnvelope` 与 `ProviderDeliveryRequest` 是 text-only；
  `MessagePush` 对 v3 附件在读取源路径前返回 `REJECTED`。C23 完成前保持该行为。

## 3. 目标所有权

```text
provider / Web / Mobile finalized upload
                  │ bytes / verified local file
                  ▼
┌──────────────────────────────────────────┐
│ Core AttachmentStore                    │
│ stage → hash/size/MIME → fsync → publish │
│ sessions.db attachments = ready          │
└──────────────────┬───────────────────────┘
                   │ opaque AttachmentRef
                   ▼
┌──────────────────────────────────────────┐
│ exact Channel envelope / Session append │
│ messages + message_attachments 同事务    │
└──────────────────┬───────────────────────┘
                   │ AttachmentReadLease
                   ▼
             provider / model read

Mobile 原协议 ── finalized file ── copy/adopt ──► Core AttachmentStore
             └─ mobile.db/inbox/cursor 保持原 owner，不被本批重写
```

唯一事实 owner 如下：

| 事实 | owner |
|---|---|
| artifact identity、hash、size、storage key、ready state | SessionStore `attachments` |
| message 的附件顺序与方向 | SessionStore `message_attachments` |
| artifact bytes | `<workspace>/uploads/artifacts/` |
| staging write | Core `AttachmentWriter`，未 publish 前唯一 cleanup owner |
| import crash boundary | SessionStore `attachment_imports` |
| provider 发送中的可读性 | `AttachmentReadLease` |
| Mobile chunk/inbox/device delivery | 现有 `mobile.db` owner |

## 4. 类型与窄接口

```python
@dataclass(frozen=True, slots=True)
class AttachmentRef:
    artifact_id: str
    kind: Literal["image", "file"]
    filename: str | None
    media_type: str | None
    size_bytes: int
    sha256: str

class ChannelAttachmentImportPort(Protocol):
    async def import_bytes(
        self,
        data: bytes,
        *,
        kind: Literal["image", "file"],
        filename: str | None,
        media_type: str | None,
    ) -> AttachmentRef: ...

class AttachmentReadLease(Protocol):
    @property
    def ref(self) -> AttachmentRef: ...
    async def read_bytes(self, *, max_bytes: int) -> bytes: ...
    async def aclose(self) -> None: ...

class ChannelAttachmentReadPort(Protocol):
    async def acquire(self, ref: AttachmentRef) -> AttachmentReadLease: ...
```

`ChannelFactoryContext` 只在 formal runtime 获得 per-binding import/read ports；candidate 不构造 adapter，也不得取得 ports。
插件不能拿 Session repository、任意 workspace path、删除方法或 SQL。Core internal adapter 另可调用 streaming writer 与
`adopt_file()`；公开插件首批只用 bounded `import_bytes()` 和 read lease。
read port 由 Host 包装并冻结当前 `binding_token`；每次 acquire 先登记到该 binding 的 in-flight owner，lease terminal
`aclose()` 后才释放，`close_admission/await_quiescence` 因而能等待附件读取。Store 的裸 read lease 不直接发给插件，插件
不能传入任意 token 或按 channel name/current snapshot 重查。

`AttachmentRef` 加入 `ChannelInboundMessage.attachments`、`OutboundEnvelope.attachments` 与
`ProviderDeliveryRequest.attachments`。PassiveWorker 在 exact envelope 内通过 Core resolver 构造模型所需的只读路径投影；
插件和 Channel wire 不接触该路径。历史 `extra.media` 仍由 Session read adapter 投影，不能把旧绝对路径冒充 artifact id。

## 5. SessionDB schema 与提交协议

```sql
CREATE TABLE attachments (
    artifact_id TEXT PRIMARY KEY,
    storage_key TEXT NOT NULL UNIQUE,
    kind TEXT NOT NULL,
    filename TEXT,
    media_type TEXT,
    size_bytes INTEGER NOT NULL,
    sha256 TEXT NOT NULL,
    state TEXT NOT NULL CHECK (state = 'ready'),
    created_at TEXT NOT NULL
);

CREATE TABLE attachment_imports (
    artifact_id TEXT PRIMARY KEY,
    storage_key TEXT NOT NULL UNIQUE,
    expected_size_bytes INTEGER NOT NULL,
    expected_sha256 TEXT NOT NULL,
    phase TEXT NOT NULL CHECK (
        phase IN ('prepared', 'file_published', 'artifact_committed')
    ),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    error TEXT
);

CREATE TABLE message_attachments (
    message_id TEXT NOT NULL,
    ordinal INTEGER NOT NULL,
    artifact_id TEXT NOT NULL,
    direction TEXT NOT NULL CHECK (direction IN ('inbound', 'outbound')),
    PRIMARY KEY (message_id, ordinal),
    FOREIGN KEY (message_id) REFERENCES messages(id) ON DELETE CASCADE,
    FOREIGN KEY (artifact_id) REFERENCES attachments(artifact_id)
);
```

提交顺序固定为：

1. 在写文件前先插入 `attachment_imports(prepared)`，identity 与最终 storage key 已固定；staging 文件仅由 writer 持有；
   校验 regular file、nofollow、root containment、size、MIME 与 SHA-256；
2. fsync file，以 no-replace hard-link publication 原子建立 `<artifact_id>.bin` 后移除 staging、fsync artifact directory，
   禁止 `os.replace` 覆盖任何既有 artifact，再把 intent 推进 `file_published`；
3. 在同一 SessionDB transaction 插入 `ready` artifact 并把 intent 推进 `artifact_committed`。任一步失败把错误写在原
   非终态 intent；rename 后 DB 失败时 bytes 保留为“无 ready row、不能被 message/provider 引用”的物理 orphan，
   不是 visible artifact。恢复审计按 intent + 固定文件名报告，普通启动不得猜测删除、自动重绑或伪造 ready；
4. message append 与 `message_attachments` 在同一 SQLite transaction；引用非 ready/不存在 artifact 时整批回滚；
5. transaction 成功后才更新 Session cache；取消发生在 commit 后时先完成 cache adoption，再恢复 `CancelledError`。

`message_attachments` 是唯一权威 binding；`extra.attachment_ids` 只是同一事务写入并由 Store 校验一致的兼容投影，
`message_edit` 不得改写附件集合。物理删除 message 时只级联 binding，不删除 artifact。artifact 物理删除仍没有普通运行协议；未来若增加，必须是用户明确
发起的命名操作，先完整引用扫描、dry-run、不可覆盖备份、hash/integrity 检查，再单独批准。

## 6. 导入、读取与失败语义

- `import_bytes/import_stream` 有按 channel/provider 配置的单项与批次上限；超限、MIME 不支持、hash 不符在 publish 前
  fail-loud，staging cleanup 必须 critical completion。
- `adopt_file()` 只接受 regular file，拒绝 symlink、root escape、变化中的 source；复制后重新 hash，不直接把来源路径
  当稳定 artifact。
- 同 bytes 不要求内容去重；artifact id 每次导入唯一，避免错误共享生命周期。SHA-256 是完整性证据，不是身份。
- artifact 是 workspace 级不可变事实，不按 Session 独占；同一 artifact 可被多个 message 显式绑定。opaque id 不是访问
  secret，授权来自 exact envelope/Host per-binding port，插件不得枚举 Store 或任意 artifact id。
- read lease 只打开 `ready` artifact，核对 storage key containment 与 size/hash。provider callback terminal 后 finally close；
  lease close 不删除 bytes。
- 首个公开 read API 是有明确 `max_bytes` 的 bounded `read_bytes()`；进入 Feishu/Telegram 大文件迁移前必须增加
  Core-owned chunk/stream reader，并以常量内存 oracle 验证，不能把 50 MiB 文件整体读入 plugin heap 后仍称能力等价。
- provider 已可能产生外部效果后失败仍返回 `UNKNOWN` 且不重试；附件 read/import 错误发生在 provider 调用前可返回
  `REJECTED`。
- Session append、Turn fail/cancel、candidate discard、插件卸载与 generation drain 均无权删除已发布 artifact。
- `adopt_file()` 的允许根必须由调用方类型固定：legacy Channel 只可从 `<workspace>/uploads/`，Mobile 只可从其 finalized
  attachment owner，MessagePush 本地文件仍按现有用户授权 path 单次打开；全部使用 nofollow fd，copy 前后核对 inode/
  size/mtime/hash，变化时 fail-loud。插件公开 port 不接受 path。

## 7. Channel 迁移规则

1. Feishu/QQ v3 factory 用 formal import port 保存入站 bytes；QQ outbound 不支持附件时在 provider 调用前返回
   `REJECTED`，Feishu 通过 read lease 上传。
2. Telegram/QQ/Web Core adapter 保留现有 provider 限制、文件名、诊断和回复媒体行为，但先导入 artifact，再产生
   exact inbound envelope。
3. Mobile `attachment.finish` 与原 `mobile.db` transaction 保持不变；新增 durable
   `mobile_attachment_imports(device_id, session_id, client_message_id, ordinal, mobile_attachment_id, artifact_id, phase)`
   作为跨库恢复/idempotency owner。同一个 finalized Mobile attachment + message ordinal 重试只解析到同一 Core
   artifact；`media_refs` 在进入 Bus 前完成 copy/adopt 与 mapping commit。Mobile 原文件、row、inbox 不因 Core artifact
   成功或失败而删除。
4. Web 任意客户端路径不直接成为 ref；已有 upload id 经 Core store resolve。远程 URL 只有先由 Core bounded fetch/import
   成功后才能成为 artifact。
5. `MessagePush` 本地 file/image 在排队前 adopt；URL 由 Core bounded fetch/import。snapshot 失败时不读源；
   import 成功后即使 provider 失败也保留 artifact。

## 8. 验证与集中 E2E

每个实现 PR 只跑 domain/Store/Manager tests，不启动完整服务。必须能杀死：

- staging write/DB insert failure：无 ready row/无 message binding；rename 后 orphan 允许保留，但必须有非终态 durable
  import intent 且可审计，`.part` 必须清零；caller cancel 在 import admission 后必须先完成同一 critical attempt，再恢复取消，
  结果只能是 committed artifact 或带 intent 的非终态，不能丢 owner；
- symlink/root escape/source mid-copy mutation：fail-loud，无 ready row；
- message + attachment binding 任一点失败：两张表整批回滚、Session cache 不 adoption；
- post-commit cancel：DB/cache/ref 完全一致后才恢复取消；
- provider read 中 hot reload/Bus close：exact read lease 收束、receipt `UNKNOWN`、artifact 仍可读；
- source 文件删除后，adopted artifact 仍可读且 hash 不变；
- Mobile 多设备 transaction 失败仍按旧协议整体回滚，Core 不删 Mobile owner；
- legacy message 只有 `extra.media` 时仍可读，不创建伪 artifact、不改写 DB；
- message 删除只删 binding，artifact 保留；启动、plugin unload、candidate discard 均零 artifact delete；
- `PRAGMA foreign_key_check`、attachment projection-vs-binding integrity、orphan intent audit 全部通过；
- 复制 workspace 恢复后，artifact 文件、metadata、message binding 的数量/hash/可读性一致；legacy `extra.media` 的绝对路径
  通过显式 old-workspace→restored-workspace relocation view 解析，不 UPDATE 原消息，也不要求旧绝对根仍存在。

最终只在 E3 做一次 recording Feishu/QQ 文本+附件组合，在 E4 的 workspace 副本做一次 restore/integrity/readback；
不为每个 channel 单独启动 Docker E2E，不读取 hua-home 正式 credential，不向真实 provider 发送。

E4 前必须先扩展正式 backup owner，而不是只复制一份代码 worktree：source manifest 至少同时声明 `sessions.db`、
`mobile.db`、`uploads/artifacts/`、legacy `uploads/` 与 Mobile finalized files；目录 snapshot 先生成 immutable file manifest
（relative path、mode、size、SHA-256），SQLite 使用 online backup。由于多 DB + files 没有全局事务，备份记录每个 source
的开始/结束时间和应用 commit，并在隔离恢复后以 durable binding 为起点逐项 readback。代码分支
`backup/plugin-v3-pre-attachment-contract-20260817` 只负责源码回滚，不是运行数据备份证据。

## 9. 删除条件与回滚

C23、Feishu/QQ v3 与 Core transition adapters 全部通过前：

- v3 attachment 保持确定性 `REJECTED`；
- 旧 Core channel attachment path 继续服务生产兼容；
- 不删除 `AttachmentStore`、`messages.extra.media`、Mobile attachment tables 或旧文件。

zero-consumer Gate 必须覆盖 Telegram photo/document/reply media、QQ image、Web upload/media route、Mobile media_refs 与
MessagePush file/image。删除兼容投影前还要证明所有新 message 已有 binding、所有历史 message 仍可读。

Core 源码回滚点为 `backup/plugin-v3-pre-attachment-contract-20260817`（`fc1a2a76`）；当前开发机没有向正式
workspace 写入任何 C23 数据，因此此刻不存在运行数据 rollback artifact。未来首次 workspace 副本/试运行前必须按上节
先产生独立不可覆盖 backup manifest。回滚代码不得删除已经由 C23
写入的 artifact、metadata 或 binding；旧 reader 必须继续通过 `extra.media` 兼容投影读取。

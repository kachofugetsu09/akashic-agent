# Android Mobile Realtime Client 设计规格

- 日期：2026-07-14
- 状态：设计已确认，待实施
- 范围：Android 客户端、Mobile Realtime Gateway、配对认证、WebSocket 协议、弱网恢复、后台主动推送与 Material 3 UI

## 1. 背景

Akasic 已有本机 WebChat：`bootstrap/chat_api.py` 提供 `127.0.0.1:6322` 的页面、WebSocket、会话、上传和媒体入口，`infra/channels/web_chat_channel.py` 把 `InboundMessage`、`OutboundMessage`、`TurnStarted`、`StreamDeltaReady`、`ToolCallStarted`、`ToolCallCompleted` 映射成实时聊天事件。当前 WebChat 能展示 thinking、tool 和 answer 的交替过程，也能接收 `message_push` 产生的主动消息。

现有入口只面向本机，不具备设备身份、一次性配对、撤销、断线续传、离线队列或公网暴露所需的安全边界。直接把 `6322` 暴露到 Cloudflare Tunnel 会把无认证的本地 WebChat 变成公网入口，不可接受。

本设计新增一个独立的 Mobile Realtime Gateway，并开发仅支持 Android 的原生客户端。客户端优先通过局域网连接电脑，也可以经 Cloudflare Tunnel 使用移动网络。移动端所有业务通信均走一条 WebSocket；二维码只用于首次配对，配对成功后通过 Android Keystore 中的设备密钥自动认证，不要求反复扫码。

## 2. 产品目标

### 2.1 用户目标

- Android 手机能像现有 WebChat 一样与本机 Akasic 对话。
- thinking、tool、answer 按真实发生顺序交替流式显示。
- 首次扫码并在电脑端确认后，设备长期记住连接，不重复扫码。
- 在局域网和 Cloudflare Tunnel 之间自动选择可用路径。
- 网络抖动、切网、进程重启和 Gateway 重启后能够续传，不重复消息。
- 支持 agent 通过现有主动消息链路向手机实时推送文本、文件和图片。
- UI 使用原生 Jetpack Compose 和明确的 Material 3 设计系统。

### 2.2 工程目标

- 保留本机 WebChat `6322` 的现有行为和无公网暴露边界。
- Mobile Gateway 与现有 Agent、EventBus、MessageBus、SessionManager、InterruptController、AttachmentStore 和 MessagePushTool 共享运行时。
- 协议具有显式版本、类型、顺序、ACK、恢复、幂等和背压语义。
- 外部输入只在 WebSocket、二维码、配置、数据库和二进制帧边界校验；通过边界后信任类型与不变量。
- 协议错误、安全错误和内部契约错误保持 fail-fast、fail-loud，不用静默 fallback 掩盖。
- 首版可在本机 Android Emulator 完成无真机开发、截图和弱网验证。

## 3. 非目标

- 首版不支持 iOS、桌面原生客户端或 Kotlin Multiplatform。
- 不建设公共多用户 IM、群聊、好友关系、已读回执或组织权限系统。
- 不引入 OpenIM、Matrix、Socket.IO 或 Centrifugo 作为核心消息层。
- 不让手机直接连接模型厂商，也不绕过 Akasic Agent loop、工具、记忆和主动推送。
- 不把现有 WebChat `6322` 改成公网认证入口。
- 不要求 WebChat 会话与 Mobile 会话跨渠道合并；首版 Mobile 会话使用独立的 `mobile:<uuid>` session key。
- 不依赖 FCM。关闭 Android 后台实时服务时，应用退到后台不承诺即时主动推送，只在下次打开时同步。
- 不在首版实现开机后静默拉起长连接。

## 4. 仓库与交付边界

### 4.1 决策

Android 工程和 Mobile Gateway 都放在当前 `akasic-agent` 仓库，不创建独立仓库，不使用 Git submodule。

```text
akasic-agent/
├── bootstrap/
│   └── mobile_api.py
├── infra/
│   └── mobile_realtime/
│       ├── gateway.py
│       ├── protocol.py
│       ├── auth.py
│       ├── pairing.py
│       ├── key_protection.py
│       ├── inbox.py
│       └── storage.py
├── clients/
│   └── android/
│       ├── app/
│       ├── gradle/
│       ├── build.gradle.kts
│       ├── settings.gradle.kts
│       └── gradle.properties
├── schema/
│   └── mobile-realtime-v1.json
├── scripts/
│   └── generate_mobile_realtime_schema.py
├── tests/
│   └── mobile_realtime/
└── tests_scenarios/
    └── mobile_realtime_live.py
```

这样安排的原因：

- Gateway 与 Agent 事件模型、session key、附件存储和主动推送会同步变化。
- 一次协议修改应在同一个提交中同时更新 Python、schema、Kotlin 和契约测试。
- 单仓库 CI 可以原子地阻止“服务端已升级、客户端未升级”的协议漂移。
- 当前只有一个服务端和一个 Android 客户端，没有独立版本治理带来的收益。

### 4.2 为什么不用 submodule

submodule 只固定另一个仓库的提交指针，不解决协议兼容、跨仓发布或 CI 原子性，反而增加 clone、分支、PR、版本升级和本地开发摩擦。即使未来拆分 Android，也应让独立仓库消费版本化 schema 或发布物，而不是把应用源码作为 submodule 挂回服务端仓库。

### 4.3 未来拆仓门槛

只有满足至少一项时才重新评估拆仓：

- Android 拥有独立维护团队、访问权限或发布节奏。
- 一个 Android 版本需要同时兼容多个服务端版本。
- Mobile Realtime 协议成为对外发布的公共 SDK 或产品边界。
- Android CI、签名和商店发布需要与服务端仓库进行权限隔离。

## 5. 总体架构

```text
Android App
├── Compose UI
├── Room / DataStore / Keystore
├── RealtimeConnectionService
└── OkHttp WebSocket
        │
        ├── LAN WSS :6323
        └── Cloudflare Tunnel WSS
                 │
                 ▼
Mobile Realtime Gateway
├── device auth / pairing / revocation
├── versioned protocol / ACK / resume
├── durable device inbox
├── transient active-turn ring
└── Mobile channel adapter
        │
        ├── MessageBus / EventBus
        ├── SessionManager
        ├── InterruptController
        ├── AttachmentStore
        └── MessagePushTool
                 │
                 ▼
            existing Agent loop
```

进程监听边界：

```text
127.0.0.1:2236  Dashboard
127.0.0.1:6322  existing WebChat, local only
0.0.0.0:6323    authenticated Mobile Gateway, TLS only
                         │
                         └── Cloudflare Tunnel 只映射 6323
```

`6323` 对局域网开放，但任何业务帧之前都必须完成服务器身份验证和设备签名认证。Cloudflare Tunnel 只改变传输路径，不成为应用身份系统，也不能绕过设备撤销。

## 6. 组件所有权

### 6.1 `bootstrap/mobile_api.py`

只拥有：

- TLS WebSocket server 的创建、监听与关闭。
- 配置加载后的组件装配。
- 把已建立的 WebSocket 交给 `MobileRealtimeGateway`。
- 健康检查和不包含敏感信息的本机诊断。

它不直接解析 agent 事件，不直接操作 session，不拥有配对规则。

### 6.2 `infra/mobile_realtime/gateway.py`

拥有：

- WebSocket 生命周期和连接状态机。
- 握手、能力协商、认证、恢复和关闭码。
- command/reply/event/ack 路由。
- P0/P1/P2 调度、背压和附件二进制帧。
- 把移动端命令投影到现有 channel runtime。

### 6.3 `auth.py` 与 `pairing.py`

`auth.py` 拥有已配对设备的 challenge/response、服务器应用身份、设备撤销和密钥指纹验证。`pairing.py` 拥有短期配对 session、一次性 secret、电脑端确认和设备注册。

原始二维码 secret 不进入普通设备表，不写 Android Room/DataStore，也不出现在日志。

### 6.4 `key_protection.py`

拥有服务器应用身份私钥和 LAN TLS 私钥的静态加密、解锁、轮换与内存生命周期。它通过 Linux Secret Service 取得 master key，使用 AES-256-GCM 读写密钥密文，并向 `auth.py` 和 TLS bootstrap 返回已解析的运行时密钥对象。

调用方不能读取 master key、nonce 或明文私钥文件，也不能自行实现第二套 fallback。Secret Service 不可用、锁定或密文认证失败时，Mobile Gateway 必须启动失败。

### 6.5 `inbox.py` 与 `storage.py`

`storage.py` 是 Mobile Gateway SQLite schema 和事务边界的 owner。`inbox.py` 拥有每设备 `event_seq`、未 ACK 的 P0 事件、恢复窗口和清理策略。

### 6.6 Mobile channel adapter

Gateway 以 channel 名称 `mobile` 接入现有 `ChannelContext`，负责：

- 发布 `InboundMessage(channel="mobile")`。
- 订阅 `OutboundMessage`。
- 监听 `TurnStarted`、`StreamDeltaReady`、`ToolCallStarted`、`ToolCallCompleted`。
- 调用 `InterruptController` 中断 turn。
- 向 `MessagePushTool` 注册 text、stream_text、file、image sender。

Agent core、工具循环和记忆链路不感知 WebSocket、二维码或 Android。

## 7. 协议总则

### 7.1 单连接与版本

移动端所有业务操作都使用同一条 WebSocket。除 TLS 握手外，手机不调用 REST。

JSON envelope：

```json
{
  "v": 1,
  "kind": "command",
  "type": "message.send",
  "id": "01J...",
  "connection_epoch": 7,
  "session_id": "mobile:...",
  "turn_id": "...",
  "payload": {}
}
```

服务端事件额外包含：

```json
{
  "v": 1,
  "kind": "event",
  "type": "answer.delta",
  "id": "01J...",
  "connection_epoch": 7,
  "event_seq": 1842,
  "session_id": "mobile:...",
  "turn_id": "...",
  "payload": {"delta": "..."}
}
```

约束：

- `v` 是 wire protocol major version。
- `id` 是 command/event 唯一 ID，使用单调可排序 ID 或 UUIDv7。
- `event_seq` 在单个 `device_id` 范围内严格递增。
- `connection_epoch` 每次认证成功后递增；客户端丢弃旧 epoch 的回调。
- command 通过 `id` 幂等，重复发送只返回原 reply，不重复产生消息。
- 未协商的必需事件类型或不支持的 major version 直接返回协议错误并关闭，不静默忽略。

### 7.2 帧类型

```text
kind
├── command  client -> server operation
├── reply    server -> client command result
├── event    server -> client ordered state change
├── ack      client -> server cumulative acknowledgement
└── control  unauthenticated/authenticated connection control
```

### 7.3 连接状态机

```text
CONNECTING
   │ TLS established
   ▼
SERVER_CHALLENGE
   │ verify server signature + fingerprint
   ▼
DEVICE_PROOF
   │ verify device signature + revocation state
   ▼
AUTHENTICATED(epoch=N)
   │ send resume(last_ack, active_turns)
   ▼
SYNCING
   │ replay P0 + P1 ring or turn.snapshot
   ▼
READY
```

任何认证前业务 command 都以 `4401` 关闭。设备被撤销使用 `4403`，协议版本不兼容使用 `4406`，服务端过载使用 `4413` 并允许客户端退避重连。

## 8. 命令与事件

### 8.1 客户端 command

```text
session.list
session.create
session.open
history.get
message.send
turn.stop
attachment.begin
attachment.finish
device.update
ping
```

`message.send` 至少包含：

```text
client_message_id
session_id
text
media_refs[]
client_created_at
```

服务端以 `client_message_id` 保证幂等。客户端在收到正式 reply 前可以安全重发。

### 8.2 服务端 event

```text
session.created
session.updated
history.page
turn.started
react.thinking.delta
react.tool.started
react.tool.completed
answer.delta
turn.snapshot
message.final
turn.interrupted
message.proactive
attachment.progress
attachment.ready
connection.degraded
sync.completed
sync.reset_required
device.revoked
```

`turn.snapshot` 是恢复边界，包含当前 turn 已知的 ordered blocks：

```text
turn.snapshot
├── turn_id
├── status
├── blocks[]
│   ├── thinking
│   ├── tool
│   └── answer
├── content_so_far
└── last_source_event_id
```

客户端 reducer 以 `turn_id + block ordinal` 合并快照与后续 delta，不把快照当成第二条消息。

## 9. ACK、恢复与优先级

### 9.1 累计 ACK

客户端每 `100ms` 或累计收到 `32` 个事件时发送一次 ACK：

```json
{
  "v": 1,
  "kind": "ack",
  "type": "event.ack",
  "connection_epoch": 7,
  "payload": {"through_event_seq": 1842}
}
```

ACK 只允许前进。倒退 ACK 或超过已发送上限的 ACK 属于协议错误。

### 9.2 三档优先级

```text
P0 durable
├── command replies
├── session/message final state
├── turn start/final/interrupted/snapshot
├── proactive messages
├── attachment ready/error
└── security/device events

P1 resumable transient
├── thinking/answer delta
├── tool started/completed live projection
└── attachment progress

P2 latest-wins
├── presence
├── typing/activity hints
└── diagnostics not required by UI state
```

连接调度按 `P0:P1:P2 = 8:4:1` 加权轮转。P0 不允许静默丢弃；发送队列超过上限时关闭当前连接，让客户端从 durable inbox 恢复。P2 在压力下只保留最新状态。

### 9.3 Durable inbox 与 transient ring

- P0 写入每设备 durable inbox，收到累计 ACK 后批量删除。
- 活跃 turn 的 P1 保存在内存 ring，默认每 turn 上限 `8MiB`。
- P1 ring 不足以覆盖 `last_ack` 时，服务端发送一个 P0 `turn.snapshot`，随后继续实时 delta。
- Gateway 重启后不尝试伪造丢失的 token delta；从 session store 和 tool chain 重建 snapshot 或 final message。
- unacked P0 默认保留 7 天；超过保留期时发送 `sync.reset_required`，客户端用 session/history command 重建本地投影。

## 10. 流式性能与背压

### 10.1 Delta 合并

Gateway 对同一 `turn_id + block` 的连续文本增量按以下任一条件 flush：

- 到达 `50ms`；
- 累计 UTF-8 内容到达 `4KiB`；
- block 类型变化；
- tool 事件、turn 完成、中断或连接关闭。

这减少系统调用、JSON 编码、Compose recomposition 和 SQLite 写放大，同时保持肉眼连续的流式体验。

### 10.2 心跳与重连

- OkHttp WebSocket ping interval：`25s`。
- Android `NetworkCallback` 检测到新网络后立即尝试连接，不等待旧 backoff。
- 失败重连采用 full jitter：基础 `0.5s`，指数增长，上限 `30s`。
- LAN endpoint 先启动；`750ms` 内未完成认证则并行尝试 Tunnel endpoint。
- 第一个完成认证的连接胜出，其余连接立即关闭。
- 每次新连接生成新的 `connection_epoch`，旧 socket 的异步回调不能更新 UI 或 ACK cursor。

### 10.3 连接内存上限

- 单条 JSON frame 默认上限 `256KiB`。
- 单条文本消息默认上限 `64KiB`。
- P0 待发送内存窗口默认上限 `8MiB`；超出后关闭并恢复，不丢 durable 数据。
- P1 ring 默认每活跃 turn `8MiB`。
- P2 使用 latest-wins slot，不形成无界队列。

## 11. WebSocket 二进制附件

附件不使用 HTTP 上传。流程：

```text
attachment.begin(JSON)
├── attachment_id
├── filename
├── content_type
├── size
└── sha256
       │
       ▼
binary chunk frames
       │
       ▼
attachment.finish(JSON)
       │
       ▼
attachment.ready(media_ref)
```

实施阶段将固定 chunk index 头调整为绝对 byte offset 头，以便服务端在断线重连后直接返回持久化 offset，客户端无需推断最后完整 chunk。二进制 frame 布局：

```text
4 bytes   JSON header 长度，unsigned big-endian
N bytes   UTF-8 JSON header，最大 1024 bytes
          {"attachment_id":"<ULID/UUIDv7>","offset":<absolute byte offset>}
M bytes   payload，最大 128KiB
```

客户端按服务端 `attachment.begin.ok.next_offset` 串行续传；服务端按 1MiB 边界和完成点确认进度。服务端在 `attachment.finish` 时校验总大小和 SHA-256，再交给现有 `AttachmentStore`；摘要失败标记为 failed，同一声明再次 begin 时从 offset 0 重传。默认单附件上限 `50MiB`，由服务端配置收紧或放宽。

## 12. 首次配对与缓存认证

### 12.1 电脑端入口

现有本机 WebChat 增加“连接手机”入口。该入口只在 `127.0.0.1:6322` 上生成二维码和执行确认，不随 Mobile Gateway 暴露到 Cloudflare Tunnel。

```text
WebChat 本机 UI
├── 创建一次性 pairing session
├── 显示 QR
├── 显示手机设备名与 6 位确认码
├── 用户确认 / 拒绝
└── 已配对设备列表与撤销操作
```

### 12.2 QR 内容

QR 包含：

```text
protocol_version
server_id
server_application_key_fingerprint
lan_endpoints[]
tunnel_endpoints[]
tls_spki_pins[]
pairing_id
one_time_secret
expires_at
```

一次性 secret 使用至少 256 bit 随机数，默认 120 秒过期，只允许一次成功 claim。服务端数据库只保存 secret 的不可逆校验值。

### 12.3 Android 密钥

扫码后 Android 在 Keystore 中生成不可导出的 ECDSA P-256 设备密钥，限制用途为 ECDSA/SHA-256 签名。设备私钥不导出到 Room、DataStore、文件或应用备份。后台自动重连需要无交互签名，因此该 key 不要求每次使用时进行生物识别；设备锁和 Keystore 安全级别仍由系统执行。客户端通过 WebSocket 发送 `pair.claim`：

Android 官方说明 Keystore key material 保持不可导出，并可绑定到 TEE 或 Secure Element：[Android Keystore system](https://developer.android.com/privacy-and-security/keystore)。

- pairing ID；
- one-time secret proof；
- 设备公钥；
- 设备名称和客户端能力；
- 配对 transcript 签名。

电脑与手机显示由 transcript 派生的同一个 6 位确认码。电脑端明确确认后，服务端创建 `device_id` 并登记公钥。

### 12.4 配对完成后缓存

Android 持久化：

- `server_id` 和展示名称；
- LAN/Tunnel endpoints；
- 服务器应用密钥指纹与 TLS pin；
- `device_id`；
- Keystore alias；
- `last_ack`；
- 当前会话和后台实时设置。

原始二维码 secret 立即从内存释放，不写入 Room、DataStore、日志或备份。

### 12.5 后续自动认证

每次连接：

1. 服务端发送带签名的随机 challenge。
2. Android 先验证服务器应用签名和已缓存 fingerprint。
3. Android 用 Keystore 私钥签名 challenge、device ID 和连接 transcript。
4. 服务端校验设备公钥和撤销状态。
5. 认证成功后返回新的 `connection_epoch`，随后执行 resume。

普通凭据轮换、断线、切网和应用重启不要求扫码。只有设备被撤销、服务器身份重置或用户主动删除服务器配置时才重新配对。

## 13. TLS、局域网与 Cloudflare Tunnel

正式构建只允许 `wss://`：

- Tunnel endpoint 使用公共 CA 和正常 hostname 校验。
- LAN endpoint 使用稳定本地域名和 QR 固定的 TLS SPKI pin。
- 客户端还会验证应用层 server identity；TLS pin 与应用身份分别拥有传输和产品身份边界。
- `ws://` 仅允许 debuggable build 显式开启，不能被 release 配置或运行时 fallback 启用。

局域网发现可使用稳定的 `.local` 名称加 Android NSD。二维码同时携带当时可用的 LAN 地址；mDNS/NSD 失败时尝试缓存地址，再退到 Tunnel，不要求用户重新扫码。

Cloudflare Tunnel 只映射 Mobile Gateway，不映射 Dashboard 或现有 WebChat。Cloudflare Access 可以作为额外防护，但不能替代设备签名认证。

## 14. 服务端存储

Mobile Gateway 使用独立 SQLite 数据库，开启 WAL。建议表：

```text
mobile_server_identity
├── server_id
├── keyset_manifest_path
└── public_key_fingerprint

mobile_pairing_sessions
├── pairing_id
├── secret_hash
├── expires_at
└── status

mobile_devices
├── device_id
├── public_key
├── display_name
├── created_at
├── revoked_at
└── capabilities

mobile_device_cursors
├── device_id
├── next_event_seq
└── acknowledged_event_seq

mobile_device_inbox
├── device_id
├── event_seq
├── event_id
├── priority
├── envelope_json
└── created_at
```

数据库不保存 master key、明文私钥、nonce 或 Secret Service 返回值，只保存当前 keyset manifest 引用和公钥指纹。

事务不变量：

- 分配 `event_seq` 与插入 durable event 在同一事务完成。
- ACK cursor 前进与已确认 inbox 删除在同一事务完成。
- 设备撤销与拒绝后续认证在同一事务可见。
- pairing success 与 one-time secret 作废在同一事务完成。

这些不变量由 storage 层拥有；Gateway 和 UI 不重复添加无法恢复的防御性检查。

### 14.1 密钥静态加密

需要静态加密的长期私钥：

```text
Linux Secret Service
└── mobile-realtime-master-key-v1, random 32 bytes
          │
          ▼
AES-256-GCM
└── data/mobile/keys/keyset-v1/
    ├── server-identity.key.enc
    ├── lan-tls.key.enc
    └── manifest.json

data/mobile/keys/current.json
└── atomic pointer to keyset-v1
```

- master key 首次启用时使用 CSPRNG 生成，只保存在 Linux Secret Service。
- 不把 master key 写入 TOML、SQLite、环境变量、命令行、日志或普通备份。
- 每个密钥 blob 使用独立随机 96-bit nonce，禁止在同一 master key 下复用 nonce。
- AAD 使用规范编码绑定 `server_id`、`key_purpose`、`keyset_version` 和 public fingerprint。
- AES-GCM 认证标签校验失败时直接拒绝解密，不能尝试忽略完整性错误。
- 密文文件仍使用 `0600`，目录使用 `0700`，作为加密之外的第二层访问控制。

每个 keyset manifest 记录 `keyset_version`、`master_key_id`、两个 blob 的相对路径、purpose、公钥 fingerprint 和内容 hash。`current.json` 是启动时唯一的版本选择入口，但它和 manifest 都属于不可信文件输入，必须完成 schema、路径约束、blob tag 和 fingerprint 校验；数据库中的 server fingerprint 用于交叉验证身份没有被替换。

建议的密文格式：

```text
magic              "AKKEY"
format_version     uint8 = 1
key_purpose        identity | lan_tls
keyset_version     uint32
nonce              12 bytes
ciphertext         variable
auth_tag           16 bytes
```

`cryptography` 和 `secretstorage` 是必需依赖，缺失时正常 import/startup 失败；不得通过动态导入或明文模式让 Mobile Gateway 看起来可用。

参考：

- [Cryptography authenticated encryption and AES-GCM](https://cryptography.io/en/latest/hazmat/primitives/aead/)
- [Freedesktop Secret Service API](https://specifications.freedesktop.org/secret-service/latest/)

### 14.2 启动与内存生命周期

启动顺序：

1. 读取并校验 `current.json` 和目标 keyset manifest。
2. 从 Secret Service 获取 manifest 指定的 master key item。
3. 读取并完整解析密钥 blob header。
4. 用 AAD 执行 AES-256-GCM authenticated decryption。
5. 验证解出的私钥、manifest 和数据库 public fingerprint 一致。
6. 构造应用身份签名对象和 TLS `SSLContext`。
7. 清理临时明文 buffer，Gateway 才进入监听状态。

应用身份私钥直接从内存解析。LAN TLS 私钥通过 Linux anonymous `memfd` 交给 `SSLContext.load_cert_chain()`，不得在 workspace、`/tmp` 或状态目录生成明文 PEM。`SSLContext` 完成加载后立即关闭 memfd，并对调用方持有的可变明文 buffer 做 best-effort zeroization。

以下任一情况都阻止 Mobile Gateway 启动并输出稳定错误码：

- Secret Service 不可用、锁定或 item 缺失；
- master key 长度或类型错误；
- blob magic、版本、purpose 或长度错误；
- AES-GCM tag 无效；
- 解密出的公钥 fingerprint 不匹配；
- 无法建立不落盘明文的 TLS context。

不得回退到明文私钥、临时 PEM、默认密钥或重新生成身份。核心 Akasic 是否继续运行由启动编排统一决定，但 Mobile Gateway 不能报告 ready。

### 14.3 轮换与恢复

master key 轮换使用两阶段写入：

1. 用旧 master key 解密现有 private key blobs。
2. 在 Secret Service 创建新版本 master key item。
3. 在新的 `keyset-vN/` 目录中，用新 master key、全新 nonce 和递增 `keyset_version` 生成两个密文及 manifest。
4. 重新解密新 keyset，并验证两个公钥 fingerprint 和 manifest hash。
5. `fsync` 新 keyset 文件与目录。
6. 原子替换 `current.json`，一次性切换整个 keyset。
7. 新 keyset 完成一次真实启动后才删除旧 keyset 和旧 Secret Service item。

轮换 master key 只改变静态加密层，不改变 server identity 或 TLS public key，因此手机不重新配对。真正轮换 server identity key 会改变 fingerprint，必须走显式设备迁移或重新配对，不能伪装成 master key 轮换。

普通备份只包含 keyset 密文和 manifest，不包含 Secret Service master key。v1 不提供 master key 自动导出；把状态目录恢复到另一台没有对应 Secret Service item 的机器时，Gateway 必须拒绝启动并要求重置 server identity、重新配对。凭据灾难恢复导出属于后续单独设计，不在本规格中暗示可用。

此设计保护磁盘、状态目录或普通备份被复制后的私钥机密性；它不声称能够抵御已经控制当前用户会话、可解锁 Secret Service 或可读取 Gateway 进程内存的攻击者。

## 15. Android 技术栈与工程结构

### 15.1 技术栈

- Kotlin。
- Jetpack Compose + Material 3。
- AndroidX Navigation、Lifecycle、ViewModel、Room、DataStore，以及 Android 平台 Keystore API。
- OkHttp WebSocket。
- kotlinx.serialization JSON。
- Coil 3 图片加载与缓存。
- `multiplatform-markdown-renderer` 的 Material 3 renderer 展示 Markdown。
- CameraX + ZXing core 扫描 QR，不依赖 Google Play Services。

首版不引入第二套网络栈、Socket.IO、RxJava 或 DI 框架。依赖在 Gradle Version Catalog 中统一锁定；升级必须同时运行协议和 UI 验证。

### 15.2 Android 工程边界

```text
clients/android/app/src/main/java/.../
├── App.kt
├── MainActivity.kt
├── data/
│   ├── local/
│   ├── realtime/
│   └── repository/
├── domain/
│   └── model/
├── service/
│   └── RealtimeConnectionService.kt
└── ui/
    ├── pairing/
    ├── conversation/
    ├── conversations/
    ├── settings/
    └── design/
```

单 Activity + Compose。UI 使用单向数据流：

```text
Composable
   │ intent
   ▼
ViewModel
   │ command
   ▼
Repository ──► Realtime connection / Room
   │ state
   ▼
StateFlow ──► Composable
```

Room 是会话、消息、block、outbox 和附件状态的持久化真相源；当前活跃 turn 的高频 delta 先进入内存 StateFlow，再批量落盘。

### 15.3 本地数据

```text
ServerProfile
Conversation
Message
TurnBlock
OutboxCommand
AttachmentTransfer
RealtimeCursor
```

DataStore 只保存当前 server、当前 session、主题和后台实时开关等轻量设置。设备私钥只存在 Keystore。

## 16. Android 后台实时连接

普通后台进程无法保证 WebSocket 长驻。Android 14+ 要求前台服务声明类型和权限；该场景使用 `remoteMessaging`，并由用户在应用可见时开启“保持实时连接”。

```text
保持实时连接 = ON
├── start remoteMessaging foreground service
├── 持续显示“Akashic 已连接”通知
├── 维护 WSS、ACK、resume 和主动推送
└── 通知操作：打开对话 / 暂停连接

保持实时连接 = OFF
├── 前台期间保持实时
├── 退到后台允许系统回收连接
└── 下次打开通过 last_ack 同步
```

Android 13+ 在启用时请求通知权限。拒绝后不伪装为后台实时成功；设置页明确显示“仅前台实时”。首版不从 `BOOT_COMPLETED` 静默启动，设备重启后需打开一次应用。

官方约束参考：

- [Android foreground service types](https://developer.android.com/develop/background-work/services/fgs/service-types)
- [Restrictions on starting a foreground service from the background](https://developer.android.com/develop/background-work/services/fgs/restrictions-bg-start)

## 17. Material 3 UI 规格

### 17.1 信息结构

```text
Conversation Screen
├── Top App Bar
│   ├── conversation drawer
│   ├── title
│   ├── encrypted connection label
│   └── overflow
├── Message List
│   ├── user message bubble
│   └── assistant turn
│       ├── assistant text
│       ├── shared process rail
│       │   ├── thinking
│       │   ├── tool
│       │   ├── thinking
│       │   └── tool
│       └── streamed answer
└── Composer
    ├── attachment
    ├── text input
    └── send / stop
```

thinking、tool 和 answer 是一个连续 turn，不拆成独立卡片墙。用户消息使用 primary container；assistant 正文直接处于页面 surface；活动轨道共享一条时间线，只有当前工具步骤获得 tertiary state layer。

### 17.2 色彩语义

规范色以 OKLCH 表达，Compose 中使用预计算、经 sRGB gamut 校验的 ARGB 值：

```text
primary H=255      user / main action / send
tertiary H=315     active tool execution only
neutral H=255      surfaces / text / outlines with low chroma
error H=25         actual failure only
```

亮色主要 token：

```css
--primary: oklch(0.500 0.135 255);
--primary-container: oklch(0.910 0.042 255);
--on-primary-container: oklch(0.285 0.075 255);
--tertiary: oklch(0.480 0.105 315);
--tertiary-container: oklch(0.900 0.042 315);
--on-tertiary-container: oklch(0.285 0.060 315);
--surface: oklch(0.985 0.006 255);
--on-surface: oklch(0.215 0.012 255);
--on-surface-variant: oklch(0.345 0.025 255);
```

关键正文组合 APCA 不低于 75，主要正文目标不低于 90。暗色主题从同一 hue/chroma 体系反向映射 L，不手工选择另一套颜色。默认跟随系统主题，禁用 Material You 动态取色，保证动作蓝和工具紫的产品语义稳定。

### 17.3 形状、层级和 motion

- 形状使用明确的 4/8/12/16/20/28dp 层级，不把所有组件做成相同大圆角。
- 嵌套圆角满足 `inner radius = outer radius - padding`。
- 触控区域不小于 44dp，Android 实现默认使用 48dp。
- 按下反馈缩放到 `0.96`，只用于按钮等直接操纵控件。
- hover 不适用于手机；pressed/selected/running 使用 Material state layer。
- 展开、收起和状态替换必须可中断，不使用无关的弹跳和整卡放大。
- 除浮动层和系统 sheet 外，主要层级依靠 tonal surface、位置和留白，不依赖大阴影。
- 系统开启“减少动画”时取消 stagger、位移和循环进度动画。

### 17.4 弱网与错误呈现

- 连接退化在 composer 上方显示一行状态：“网络不稳 · 消息已缓存，正在续传”。
- 普通重连不弹重复 Toast。
- 发送中、已发送、待重试属于消息状态，不产生独立彩色卡片。
- tool error 在活动轨道中使用 error icon、明确文字和中性 surface。
- 设备撤销或 server identity 变化使用阻断页面，因为用户必须采取安全动作。

## 18. Android 渲染与存储性能

- `LazyColumn` 使用稳定 message ID 和 block ID。
- 活跃 turn 在内存中每 `50ms/4KiB` 合并一次 UI state。
- Room 每 `250ms`、累计 `8KiB` 或 turn 结束时批量写入，以最先满足者为准。
- 最终 `message.final` 必须以事务替换流式临时内容并标记 turn 完成。
- Markdown parse 结果按 message content hash 缓存；只重新解析发生变化的活跃 answer block。
- 图片使用 Coil memory/disk cache；原图解码服从界面尺寸，不在主线程读取大文件。
- 附件传输和 SHA-256 在 `Dispatchers.IO`，Compose state 只接收小型进度模型。

## 19. 安全边界

集中校验的位置：

```text
QR decoder
TLS / server identity handshake
encrypted key blob + Secret Service unlock
WebSocket JSON envelope decoder
binary attachment header decoder
command payload schema
SQLite row deserialization
Android Room migration
```

边界之后使用 typed model，不在每个下游函数重复检查同一个字段非空或类型正确。

必须拒绝：

- 过期、重复或未知 pairing ID。
- server fingerprint 与已缓存值不一致。
- 被撤销设备的签名。
- event ACK 倒退或越界。
- command ID 与历史 payload 冲突。
- attachment UUID、index、size 或 hash 冲突。
- 未协商的 major protocol version。
- release build 中的 `ws://` endpoint。
- Secret Service master key 缺失、锁定或长度错误。
- encrypted key blob 的 header、AAD、tag 或 public fingerprint 不一致。

日志默认不记录消息正文、thinking、tool 参数、二维码 secret、master key、nonce、明文私钥、签名、完整公钥或附件内容。诊断日志记录 event ID、seq、类型、字节数、耗时、keyset version 和错误码。

## 20. 可观测性

服务端至少暴露：

```text
mobile_connections_active
mobile_auth_success_total
mobile_auth_failure_total{reason}
mobile_resume_total{mode=replay|snapshot|reset}
mobile_ack_lag_events
mobile_ack_lag_seconds
mobile_outbox_bytes{priority}
mobile_delta_batch_bytes
mobile_delta_batch_delay_ms
mobile_p2_replaced_total
mobile_attachment_inflight_bytes
mobile_key_unlock_failure_total{reason}
mobile_keyset_version
```

Android debug diagnostics 页面显示当前 endpoint、connection epoch、last ACK、重连次数、pending outbox 和最近错误码，不显示 secret。

## 21. 配置

建议的 typed config：

```toml
[mobile_realtime]
enabled = true
host = "0.0.0.0"
port = 6323
database = "data/mobile_realtime.db"
lan_hostname = "akashic.local"
public_url = "wss://agent.example.com/ws"
max_attachment_mb = 50
inbox_retention_days = 7

[mobile_realtime.key_encryption]
provider = "secret_service"
master_key_namespace = "akasic/mobile-realtime"
keyset_manifest = "data/mobile/keys/current.json"
```

配置中只有 Secret Service item namespace 和 keyset manifest 路径，不包含 master key。具体的版本化 item ID 由 manifest 指定。TLS 证书、密钥密文、server identity key 和 Cloudflare Tunnel 凭据不写入仓库。配置加载边界验证 provider、namespace、manifest、endpoint scheme、端口、路径和上限；错误配置阻止 Mobile Gateway 启动并输出明确错误，不静默退化到明文、环境变量密钥或无认证模式。

## 22. Schema 与兼容策略

Python typed protocol model 是 schema 的 source of truth，`scripts/generate_mobile_realtime_schema.py` 生成 `schema/mobile-realtime-v1.json`。生成结果进入仓库。

兼容规则：

- major version 不同：拒绝连接。
- 同 major 的新增 optional 字段：向后兼容。
- 字段语义变化、必填字段变化或事件顺序变化：提升 major。
- 客户端在 auth hello 中声明 event/feature capabilities，服务端只发送已协商能力。
- Python 和 Kotlin 都使用同一组 golden frames 做 encode/decode 契约测试。
- CI 重新生成 schema 后必须保持 git clean，防止忘记提交 schema 变更。

## 23. 测试策略

### 23.1 Python 单元与集成测试

```text
pairing
├── one-time secret expires and cannot be reused
├── desktop confirmation is required
├── device signature succeeds
└── revoked device fails loudly

key protection
├── generated private keys never appear as plaintext files
├── Secret Service item missing/locked fails startup
├── wrong master key, nonce, AAD, tag, or purpose fails startup
├── decrypted public fingerprint must match database
├── TLS key reaches SSLContext through memfd only
├── master-key rotation keeps server fingerprint stable
├── interrupted rotation leaves the old key usable
└── backups contain ciphertext but no master key

protocol
├── invalid envelope rejected at boundary
├── unsupported version closes 4406
├── duplicate command returns same reply
├── ACK advances cumulatively
├── ACK rollback/overflow rejected
└── stale connection epoch cannot publish

recovery
├── durable P0 replay
├── P1 ring replay
├── missing P1 produces turn.snapshot
├── gateway restart reconstructs final state
└── queue overflow disconnects without P0 loss

attachments
├── persisted byte-offset resume
├── offset conflict rejected
├── size/hash mismatch rejected
└── completed file enters AttachmentStore
```

集成测试使用真实 FastAPI WebSocket、真实 SQLite 临时数据库和真实 event bus，不通过“假成功” transport 绕过协议。

### 23.2 Android 测试

- protocol encode/decode golden tests。
- Room migration、outbox、cursor 和 reducer 单元测试。
- Keystore challenge 签名 instrumented test，并断言私钥不可导出且不进入 Room/DataStore。
- Compose screenshot：扫码、空会话、流式 thinking、工具运行、工具完成、工具失败、弱网、长正文、字体放大、亮色和暗色。
- reducer 验证 `thinking -> tool -> thinking -> answer` 的严格顺序。
- process death 后从 Room 恢复，不产生第二条 assistant message。

### 23.3 Emulator 集成测试

本机已有 Android SDK platform 36 和 `Medium_Phone_API_36.1` AVD。验证链路：

```text
start real Akasic + Mobile Gateway
start emulator
pair once
send message
observe thinking/tool/answer
toggle airplane mode for 30s
restore network
verify resume and no duplicate
switch LAN/Tunnel endpoint
restart Gateway
kill/relaunch Android process
verify proactive foreground-service notification
capture screenshot + logcat + gateway metrics
```

Cloudflare Tunnel 属于 `tests_scenarios` live smoke，不放入默认 pytest，避免外部网络依赖污染本地单测。

## 24. 实施阶段

```text
1. protocol models + generated schema + golden frames
2. Secret Service + AES-256-GCM key protection + rotation tests
3. server identity + pairing storage + local WebChat pairing UI
4. authenticated Gateway handshake + device revocation
5. Mobile channel adapter + session/message/turn commands
6. event_seq + durable inbox + ACK/resume/snapshot
7. Android project skeleton + Room/DataStore/Keystore
8. QR pairing + endpoint race + reconnect state machine
9. conversation UI + shared process rail + Markdown
10. binary attachments + Coil rendering
11. remoteMessaging foreground service + proactive notification
12. weak-network, process-death and Gateway-restart verification
13. Cloudflare Tunnel live smoke and release hardening
```

每个阶段先完成对应契约测试再进入下一阶段。不得用未认证临时入口、假消息或静默 fallback 提前宣称链路完成。

## 25. 验收标准

- 现有 WebChat 继续只监听 `127.0.0.1:6322`，行为不变。
- Mobile Gateway 使用独立 `6323`，认证前不能发送或订阅业务消息。
- server identity 与 LAN TLS 私钥在磁盘和普通备份中只以 AES-256-GCM 密文存在，master key 只在 Secret Service。
- Secret Service 锁定、master key 缺失或密文被篡改时 Mobile Gateway fail-loud，且不会生成明文 fallback。
- master key 轮换后 server fingerprint 不变，已配对手机不重新扫码；轮换中断时旧密钥仍可恢复。
- 手机扫码一次并在电脑确认后，冷启动、切网和进程重建均不再扫码。
- 删除 QR secret 后，设备仍能使用 Keystore challenge 自动认证。
- 局域网、移动网络和 Cloudflare Tunnel 之间切换不丢消息、不重复消息。
- 飞行模式保持 30 秒后恢复，继续同一条流式回答。
- Gateway 重启后通过 durable inbox、snapshot 或 final history 恢复一致状态。
- thinking、tool、thinking、answer 按真实事件顺序显示在共享活动轨道中。
- 主动消息在应用前台实时到达；启用 remoteMessaging 服务后在后台实时到达并产生系统通知。
- 关闭后台实时后，UI 明确显示“仅前台实时”，下次打开能补齐离线 P0。
- 设备撤销、server identity 变化、协议版本不兼容和附件损坏均 fail-loud。
- 单次配对、弱网续传和主动推送均可在 `Medium_Phone_API_36.1` 模拟器复现并留存截图、日志和指标。
- `pytest` 目标测试、Android unit/instrumented tests、Compose screenshots、Gradle lint 和 assemble 全部通过。

## 26. 已确认决策摘要

```text
platform              Android only
repository            current akasic-agent monorepo
android path           clients/android/
transport              one versioned WebSocket
local path             LAN WSS first
remote path            Cloudflare Tunnel WSS fallback
existing WebChat       unchanged, local-only :6322
mobile gateway         authenticated :6323
pairing                one-time QR + desktop confirmation
cached auth            Keystore ECDSA P-256 device key
server key at rest     Secret Service + AES-256-GCM encrypted blobs
recovery               event_seq + cumulative ACK + resume/snapshot
stream batching        50ms or 4KiB
heartbeat              25s
reconnect              full jitter 0.5s -> 30s
endpoint stagger       750ms
priority scheduler     P0:P1:P2 = 8:4:1
attachment chunk       128KiB, sequential offset
background realtime    remoteMessaging foreground service
UI                     native Compose, restrained Material 3
color                  stable OKLCH semantic palette
```

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
| 停止生成 | UI 按钮为空回调 | 接通 `turn.stop` | 完成 |
| 文件与图片选择 | 附件按钮为空回调 | 系统多选、预览、移除 | Android 完成 |
| 上传传输 | 协议只有 begin/finish 名称 | 全 WebSocket 二进制分片 | 服务端完成 |
| 上传进度 | Room 有 transfer 表但未使用 | 乐观进度加服务端确认 | 服务端完成 |
| 弱网续传 | 无 | 按确认 offset 恢复 | 服务端完成 |
| 图文和纯附件消息 | 服务端拒绝 media_refs | 两者均支持 | 服务端完成 |
| Agent 图片理解 | Web 可进入 VL 链路 | Android 上传图片等价接入 | 服务端完成 |
| 助手图片、文件、meme | 服务端发送路径，Android 丢弃 | descriptor、下载、缓存、展示 | 完成 |
| 历史媒体同步 | 历史 DTO 裁掉 media | 全 mobile 会话恢复媒体 | 完成 |
| 图片预览 | 无 | Material 3 全屏预览、缩放、GIF | 完成 |
| 普通文件操作 | 无 | 类型、大小、进度、系统打开与分享 | 完成 |
| Telegram 被动媒体等价 | Telegram 支持本地路径与 HTTP(S) 图片出站 | Android 覆盖同类能力 | 服务端完成 |
| 长离线全量重建 | `sync.reset_required` 会卡在 event gap | 仅重建当前 server 投影并保留 outbox/draft | 完成 |
| 后台被动送达 | 仅 Activity 可见时保持连接 | 单例 foreground service 与隐私通知 | 完成 |

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

- Commit：`028ed711 feat(mobile): add resumable websocket uploads`。
- 协议：JSON `attachment.begin/finish` 加 WebSocket binary chunk；单片 128KiB，按绝对 byte offset 串行续传。
- 一致性：同附件条带锁串行化 begin/chunk/finish；文件 fsync 后提交 SQLite offset；文件短于 offset 或摘要失败时进入 failed，并在相同声明再次 begin 时从 0 恢复。
- 边界：纯文件名、MIME、大小、SHA-256、设备/会话归属、重复 media ref、单消息最多 10 个附件及总字节上限。
- 验证：Pyright 0 error / 0 warning；目标 mobile 测试 51 项通过；完整 `pytest tests/` 2184 项通过；schema 生成一致性与 `git diff --check` 通过。
- E2E：隔离临时 workspace 中完成认证 WSS、上传至服务端明确确认 1MiB、断开、新连接 resume、从 1MiB offset 续传、finish、纯附件 message.send，并验证 `InboundMessage.media` 文件内容一致。

### 2. Android 上传与发送

- Commit：`bb8e9609 feat(android): add resumable attachment sending`。
- 文件入口：`OpenMultipleDocuments` 选择后立即复制进 app 私有目录，复制时流式计算大小和 SHA-256；不依赖重启后可能失效的 Content URI。
- 续传：每次连接生成新的 begin command ID，以服务端 `next_offset` 覆盖本地确认值；每个 1MiB 确认窗口由 8 个 128KiB binary frame 组成，未确认字节不写入 Room offset。
- 一致性：`attachment.progress/ready` 与 durable cursor 在同一 Room 事务提交；完整 offset 重连直接恢复 finish；消息入 outbox 时附件从 ready 原子推进 sending，reply 后转 sent 或恢复 ready。
- 消息：支持图文和纯附件；发送确认后清理 app 私有源文件。
- 版本：Android `0.3.0`（versionCode 3），对应本组可安装 APK。
- APK：私有 Release `kachofugetsu09/akashic-mobile-releases` 的 `v0.3.0`，本机 release 测试、Lint、R8 构建和 v2 签名验证通过。
- 验证：Android JVM 21 项通过；`compileDebugAndroidTestKotlin`、`lintDebug`、`assembleDebug` 通过；无窗口 API 36.1 模拟器 11 项 Room、Compose、Keystore instrumentation 全部通过，随后已关闭模拟器。
- 跨端 E2E：服务端组已独立证明认证 WSS 到 `InboundMessage.media`；Android 组已证明 1MiB 窗口、断线新 begin ID、完整 offset 恢复 finish、纯附件发送门控。真实 Android→隔离 Gateway 联合链路留在第 5 组执行。

#### UI skill 约束落实

better-colors：

| Before | After |
| --- | --- |
| 附件没有颜色语义 | 上传沿用主蓝色，失败只在图标与状态文字使用 error；亮紫仍专属 agent 思考/工具运行 |
| 容易新增一套 pastel 附件色 | 全部复用 Material `surfaceContainerLow/primary/error`，进度轨道为同色 16% state layer |

better-typography：

| Before | After |
| --- | --- |
| 没有文件名和状态层级 | 文件名用 `labelLarge` 单行省略，大小与状态用 `labelMedium` |
| 百分比变化可能跳动 | 大小和进度使用 `tnum`，完整文件名保留在无障碍语义中 |

better-ui：

| Before | After |
| --- | --- |
| 附件按钮为空回调 | 接通系统多选、私有草稿、自动上传、失败重试；待上传、失败和已就绪草稿可移除，活动上传不伪装成可取消 |
| 容易形成卡片墙 | 仅每个独立附件使用无 elevation 的低层 Surface，附件区不再套外层卡片 |
| 状态突变、触控目标不明确 | 状态图标可中断交叉淡入；草稿区展开/收起；可用的重试和移除保持 48dp 与 0.96 按压反馈 |
| 纯附件无法发送 | 仅当所有草稿 ready 时允许发送；上传中和等待网络保持禁用 |

### 3. 出栈媒体、meme 与历史

- Commit：`71beaab0 feat(mobile): add websocket media downloads`。
- 身份修复：`1ab49f0a fix(mobile): unify live and history message ids`。
- 消息身份：用户消息以同一个 ULID 同时作为 command ID 与 `client_message_id`；assistant 的实时 final 直接携带 Session SQLite 回填的消息 ID，历史同步复用同一 ID，不再按文本、附件或“最后一条消息”反查。
- 所有权：after-reasoning phase 持有本轮 user/assistant 的精确消息对象，批量持久化在原对象上回填 ID；插件追加媒体同时进入 assistant 历史与实时 descriptor。
- 协议：`message.final/history.page` 使用统一 descriptor；客户端以 `attachment.download(offset)` 请求 128KiB binary chunk，服务端先发二进制再发 reply，同 command ID 可重放。
- 持久化：本地媒体先复制为只读 canonical 文件，再以随机 opaque ID 注册；同会话内容身份在 SQLite 写事务内稳定复用，批次失败不留记录或孤儿文件。
- 元数据：手机上传后再由 Agent 返回的文件，按 `session_id + local_path` 恢复原文件名和 MIME；descriptor 不含服务端路径。
- 网络媒体：被动回复中的 HTTP(S) 资源逐跳校验 DNS 全部地址、固定公网 IP 连接并保留 TLS hostname，限制重定向和流式总字节，原子快照后再进入统一附件存储。
- 性能：复制与 SHA-256 移到工作线程；每连接独立串行发送，弱网设备不应占用全局投递锁。
- 保留：已进入 Session 历史、durable inbox 或完成下载 receipt 的出站附件不会按简单 TTL 删除。后续清理必须先引入显式引用表和 session/receipt 删除钩子，不能制造永久 descriptor 指向缺失文件。
- 验证：媒体协议提交时 `tests/mobile_realtime` 105 项通过；身份修复扩大覆盖 mobile、lifecycle、turn、control 和 AgentCore 后 189 项通过；Pyright 0 error，只有存量类型告警；schema 与 `git diff --check` 通过。真实认证 WSS 已验证 binary-before-reply、重复 command 重放、分页 resume、旧 epoch ACK 和慢连接隔离。

### 4. Material 3 预览与文件交互

- Commit：`d04ce0f9 feat(android): receive realtime attachments`。
- 身份合并：发送 command 与用户 `client_message_id` 复用一个 ULID；assistant live final 迁移到服务端 canonical ID，history 同 ID 原子替换消息、附件链接和 thinking/tool blocks，不重复显示。
- 即时回复：无持久化 ID 的 control/plugin final 使用 `ephemeral:<event-id>` 独立命名空间，只负责本机展示，不伪装成可同步历史。
- 下载：单 active coordinator 按持久 offset 串行续传；协议、文件、Room 与溢出错误关闭当前连接并重新调度，失败项可手动重下。
- 缓存：受限私有目录、SHA-256 文件名、启动 reconcile、512MiB 配额与真实 LRU；重连只恢复待下载队列，不重复扫描全缓存。
- 迁移：Room v1→v2 显式 migration，并以 `MigrationTestHelper` 创建真实 v1 数据后升级验证。
- 展示：图片和 GIF 使用 Coil inline 预览及全屏缩放；解码失败回退到文件行。普通文件显示类型、大小和进度，通过非导出 FileProvider 交给系统打开。
- 验证：独立 reviewer 两轮检查后批准；JVM 34 项通过，`compileDebugAndroidTestKotlin`、Lint 与 debug APK 构建通过；被根 `data/` 规则忽略的核心源码、测试与 Room schema 已 force-add，并用 `git ls-tree` 确认进入提交。
- E2E：数据层、Compose 与 migration instrumentation 已编译；真实模拟器联合链路留在第 5 组执行。

#### UI skill 约束落实

better-colors：

| Before | After |
| --- | --- |
| 已缓存和下载中都使用 `primary` | 已缓存使用 `onSurfaceVariant`；下载、重试等主动状态保留 `primary` |
| 图片边缘在浅色背景上不稳定 | 浅色使用纯黑 10% outline，深色使用纯白 10% outline |
| 容易为附件再造 pastel 色系 | 复用现有 Material 3 token；亮紫仍只表达思考和工具活动 |

better-typography：

| Before | After |
| --- | --- |
| 附件没有信息层级 | 文件名使用 `labelLarge`，类型、大小与状态使用 `labelMedium` |
| 百分比更新会造成字宽跳动 | 大小与进度启用 `tnum` |
| 长文件名挤压操作 | 单行省略，完整名称保留在无障碍语义中 |

better-ui：

| Before | After |
| --- | --- |
| 被动附件不可见 | 附件进入 user/assistant 消息内容层 |
| 每个文件容易形成独立卡片墙 | 普通文件共享同一列表平面，以浅分隔线表达关系 |
| 图片只能当普通文件 | inline 图片/GIF，点击进入全屏 Material 3 预览 |
| 缓存淘汰或图片损坏没有恢复路径 | `EVICTED` 提供真实重下；Coil 失败回退为可打开文件行 |
| 操作反馈与系统边界不明确 | 48dp 触控区、0.96 按压、state layer、`safeDrawing` 与缩放平移限界 |

### 5. 弱网、模拟器与 APK 发布

- 生命周期 Commit：`c2a9f88d feat(mobile): complete passive delivery lifecycle`。
- 隔离 harness Commit：`ed7549e2 test(mobile): add isolated gateway harness`。
- 停止：`turn.stop` 绑定当前 `session_id + turn_id`；客户端在重连时复用同 command ID，重复点击不产生第二次中断。
- reset：`sync.reset_required` 在普通连续序号校验前接管，只删除当前 server 的远端投影；保留 outbox、草稿、本地发送状态和其他 server，再以新 generation 全量拉取 session/history。
- 已发送附件：发送前原子导入 received cache，消息、附件、link、outbox 与 transfer 状态在同一 Room 事务提交；发送确认后删除草稿源但保留气泡附件。
- 后台：`START_STICKY` foreground service 复用 `AppContainer` 唯一的 Room、RealtimeSession 和 WebSocket；Android 13+ 请求通知权限，前台当前会话不重复通知，锁屏 public version 不暴露消息、thinking、tool 或密钥。
- 交互：已缓存文件可经受限 FileProvider 分享；READY 状态错误可关闭；流式文字和展开动画使用 stick-to-bottom，用户主动上滑后不抢位置。
- 自动验证：`tests/mobile_realtime` 110 项通过；Android JVM、Kotlin、AndroidTest 编译、Lint 与 debug APK 构建通过；主仓库完整 `pytest tests/` 在前一提交点 2221 项通过。
- 隔离 E2E：临时 TLS Gateway 的 17 项 gateway/isolated 测试通过，覆盖两次 history 去重、换 epoch 补发、固定 GIF 二进制/SHA 和单次 Agent 入站；真实启动可生成单次 pairing JSON/PNG 并干净退出，所有数据库与附件均位于临时根。
- 模拟器 E2E：API 36.1 无窗口模拟器运行 29 项 Room migration、Compose、Keystore instrumentation 全部通过；真实隔离 Gateway 完成配对、历史、发送、固定 GIF 下载与缓存，强停进程后恢复不重复。移除 `adb reverse` 后恢复，客户端 13.2 秒内经 full-jitter 自动回到 READY。
- 版本：Android `0.4.0`（versionCode 4）；私有 Release [`v0.4.0`](https://github.com/kachofugetsu09/akashic-mobile-releases/releases/tag/v0.4.0)，资产 `Akashic-Mobile-v0.4.0.apk`（5,007,406 bytes）。
- 发布验证：release JVM、Lint、R8 与 assemble 在 1m54s 内完成；APK v2 签名有效，版本清单为 `0.4.0/4`；本地与 GitHub 资产 SHA-256 均为 `f9a43a3981bae767e952c00271549e0655f7f4f4ed6d349ab5abb3993b1dc853`。
- 资源清理：隔离 Gateway 根目录确认无符号链接后删除；无窗口模拟器已关闭，`adb devices` 无残留设备。

#### 本组 UI skill 约束落实

| Skill | Before | After |
| --- | --- | --- |
| better-colors | 新状态容易继续增加彩色容器 | 停止沿用主操作语义，错误只用 `errorContainer`；通知权限提示复用 M3 inverse surface，不新增色系 |
| better-typography | 连接、错误与文件操作层级混杂 | 状态沿用现有 body/label 层级，文件元信息继续使用 `labelMedium + tnum` |
| better-ui | 停止为空回调、错误不可见、滚动失跟 | 真实停止与分享、单一可关闭 Snackbar、尊重用户上滑的 stick-to-bottom；未增加卡片或阴影 |

### 6. 中止反馈与快捷命令

```text
┌──────────────────────────────────────────────┐
│ 对话与思考 / 工具时间线                      │
│                                              │
│  生成中：正在中止本轮…                       │
│  终态：  已中止 · 7s                         │
│          ■ 生成已中止，可继续补充             │
├──────────────────────────────────────────────┤
│ [命令] [输入消息………………] [附件] [发送 / 中止] │
└──────────────────────────────────────────────┘
                 │
                 └── 快捷命令 ModalBottomSheet
                     /undo          撤销上一轮对话
                     /memorystatus  查看记忆整理状态
                     /kvcache       查看 KVCache 状态
```

- 服务端：新增 `command.list`，命令目录直接读取 `ChannelContext.bot_commands`，因此插件增删命令后无需重新发布 APK；`/stop` 仍由生成控制按钮拥有，不混入文本命令。
- 中止一致性：真实 interrupt 与“请求抵达时 turn 已结束”的竞态都发布 `turn.interrupted`，清除 active/process 映射；未知 controller 状态直接失败，不伪装成功。
- 即时反馈：点击后按钮进入不可重复点击的进度态，并在输入区上方显示“正在中止本轮…”；终态保留已有的部分思考、工具与回答，显示“已中止 · Ns”和可继续补充提示。
- 命令交互：左侧菜单打开原生 Material 3 bottom sheet；附件移到输入框右侧。点击命令只填入输入框并唤起键盘，用户确认后发送，避免 `/undo` 等破坏性命令误触执行。
- 自动验证：服务端 mobile realtime 33 项通过；Android JVM、AndroidTest 编译通过；无窗口 API 36.1 模拟器中 31 项非外部 Gateway instrumentation 与 3 项 UI 证据测试通过。全量 instrumentation 的 2 项隔离 Gateway 测试因本轮未启动外部 harness 而按预期缺少连接参数，其余 31 项通过。
- UI 证据：[快捷命令面板](assets/mobile-v0.5.0-command-sheet.png)、[中止请求中](assets/mobile-v0.5.0-stop-pending.png)、[中止终态](assets/mobile-v0.5.0-stop-terminal.png)。取图完成后无窗口模拟器已关闭。
- 版本：Android `0.5.0`（versionCode 5）；私有 Release [`v0.5.0`](https://github.com/kachofugetsu09/akashic-mobile-releases/releases/tag/v0.5.0)，资产 `Akashic-Mobile-v0.5.0.apk`（5,040,174 bytes）。
- 发布验证：release JVM、Lint、R8 与 assemble 通过；APK v2 签名有效，版本清单为 `0.5.0/5`；本地与 GitHub 资产 SHA-256 均为 `ff56ee255798e09dac5aba2784a73be4e4c22df00715a4f738aa2191784b50d1`。

#### 本组 UI skill 约束落实

better-colors：

| Before | After |
| --- | --- |
| 中止点击后没有持续色彩语义 | 正在中止和中止终态统一使用现有亮紫 `tertiary`，不占用错误红色 |
| 命令容易被做成多色功能卡 | 命令名只使用 `primary`，说明使用 `onSurfaceVariant`，sheet 复用 `surfaceContainerLow` |

better-typography：

| Before | After |
| --- | --- |
| 中止结果与普通正文无法区分 | 状态标题使用 `labelLarge`，持续时间以紧凑 `· Ns` 进入同一阅读句 |
| 命令与说明缺少扫描锚点 | `/command` 使用 monospace 与固定列宽，说明使用 `bodyMedium`，两行内安全省略 |

better-ui：

| Before | After |
| --- | --- |
| 中止按钮只有瞬时 spinner，终态无痕 | 即时、处理中、终态三阶段反馈；保留部分结果，不制造错误卡片或假助手消息 |
| 附件占据输入框左侧，命令无入口 | 固定为 `[命令][输入][附件][主操作]`，四个角色位置稳定 |
| 命令可能演变成 Telegram 式浮层菜单或卡片墙 | 使用原生 ModalBottomSheet 与共享列表平面，每行 64dp、触控目标至少 48dp |

### 7. 历史身份、原生选词与富文本阅读

```text
服务端 canonical 历史消息
          │
          ├── 有 client_message_id ── 精确合并 optimistic identity
          │
          └── 旧消息缺 client_message_id
                 │
                 └── 同 session + 同正文 + 已发送 + 仅一个候选
                     + 本地时间不晚于 canonical 完成时间
                          │
                          ├── 成立：迁移附件/块并删除旧 identity
                          └── 不成立：保留两条，不猜测、不丢消息

消息正文
  ├── Markdown：原生 SelectionContainer，可拖动选区并复制
  └── display math：识别 \[ ... \] / $$ ... $$，交给 Compose LaTeX
```

- 历史重复根因：真实 `sessions.db` 中截图对应的用户消息只有一条；旧 APK 的本地 optimistic message 没有服务端可回传的 `client_message_id`，更新后全量历史又插入 canonical identity，造成手机 Room 中双份投影。
- 兼容修复：只在唯一、同文、同 session、已发送且发生在 canonical 完成时间之前的一小时候选上迁移身份。相同问题稍后再次发送、同时存在多个候选或时间不成立时均不合并。
- 文本选择：用户与助手正文进入 Compose `SelectionContainer`；长按显示系统浮动选择栏，可拖动原生光标做局部复制，不给每条消息增加复制卡片或操作按钮。
- Markdown：正文、列表、表格继续使用现有 renderer；标题改为 22/20/18/16sp 的递减层级，避免 `第一步`、`第二步` 抢占整屏。块级公式使用独立的原生 LaTeX renderer，覆盖真实历史里的单行 `\[ formula \]`、多行公式、代码围栏和流式未闭合状态。
- 图标：用户提供的 Akashic 插画裁成 adaptive launcher icon，并提供各 density fallback；通知栏继续使用职责独立的单色 small icon。
- 提交：`9ada3cec` 历史身份修复；`e12270c9` 选择、Markdown 与 LaTeX；`10762cd4` adaptive icon；`eff1c032` 选择交互测试；`3ca4a7c3` 通知图标分离；`dba5b70c` Android 0.6.0。
- 自动验证：Android JVM 51 项通过；API 36.1 无窗口模拟器的 16 项 `LocalDeliveryStoreTest`、真实单行公式/标题和长按选择路径通过。取图与验证完成后模拟器已关闭。
- UI 证据：[长按原生选词](assets/mobile-v0.6.0-text-selection.png)、[紧凑标题与 LaTeX](assets/mobile-v0.6.0-markdown-latex.png)、[真实 Launcher 图标](assets/mobile-v0.6.0-launcher-icon.png)。
- 版本：Android `0.6.0`（versionCode 6）；私有 Release [`v0.6.0`](https://github.com/kachofugetsu09/akashic-mobile-releases/releases/tag/v0.6.0)，资产 `Akashic-Mobile-v0.6.0.apk`（6,373,822 bytes）。
- 发布验证：release JVM、Lint、R8 与 assemble 在 1m58s 内通过；APK v2 签名有效，版本清单为 `0.6.0/6`；本地与 GitHub 资产 SHA-256 均为 `72311e89acbb38757da02a6c1801c4b6054f181bd929c10c8d29e2f7365ee2ab`。

#### 本组 UI skill 约束落实

better-colors：

| Before | After |
| --- | --- |
| Launcher 使用通用蓝 `#2463AE` 勾选 vector | 使用插画本身的深紫、薄荷绿、叶绿与玫红语义；adaptive 底色为 `#251334` |
| 阅读页容易因公式再增加彩色容器 | 正文、公式和选区沿用现有 Material 3 scheme，不新增状态色或卡片色 |

better-typography：

| Before | After |
| --- | --- |
| renderer 默认标题使 `### 第一步` 接近 display 尺寸 | H1/H2/H3/H4 收敛为 22/20/18/16sp，正文保持 16sp |
| 公式以反斜杠原文混入段落 | 数学内容使用原生数学字形、自动换行与 TalkBack 描述 |
| 正文只能整体观看 | 长按后可拖动系统选区，复制需要的局部文字 |

better-ui：

| Before | After |
| --- | --- |
| 更新 APK 后旧 optimistic 与 canonical 同时出现 | 在 Room 事务内迁移唯一旧身份；歧义时保留数据而不是猜测 |
| 复制功能容易演变成每条消息一个按钮 | 复用 Android 原生选择工具栏与光标，不增加卡片或常驻操作 |
| Launcher 与通知共用一个 vector | Launcher 使用 adaptive illustration，通知使用独立单色 small icon |

### 8. 历史时序、保留配对的重同步与固定命令抽屉

```text
服务端 history.page
      │
      ├── seq 0  user       ┐
      ├── seq 1  assistant  ├── Room 按 serverSeq 排序
      └── seq N  ...         ┘       │
                                     └── createdAt 只负责计算思考耗时

会话抽屉
┌────────────────────────────────────┐
│ 最近                               │
│   会话 A                           │
│   会话 B                           │
│                                    │
│ 连接与数据                         │
│   ↻ 重新同步消息                   │
│     保留连接并重新拉取历史         │
│   ▣ 重新扫码连接                   │
│                                    │
│ [✎ 聊天]                           │
└────────────────────────────────────┘
```

- 时序根因：真实 `sessions.db` 中同一轮 user/assistant 的持久化时间只差几十微秒；旧客户端为了显示 `turn_duration_ms` 把 assistant `createdAt` 回拨几十秒，又使用同一字段排序，导致回答跑到问题前面。
- 时序修复：Room v2→v3 增加 nullable `serverSeq`；完整历史严格按服务端 `seq`，实时、pending 和 failed 消息在历史之后按本地创建时间稳定排序。`createdAt/updatedAt` 继续只承担思考耗时，两个不变量不再互相污染。
- 手动重同步：会话抽屉的“连接与数据”区增加低频维护入口。确认后删除可从电脑恢复的 committed projection 和失去引用的附件缓存，保留 ServerProfile、cursor、Android Keystore 设备密钥、pending/failed、outbox 和附件草稿，再沿当前已认证 WebSocket 全量拉取所有 mobile 会话。
- 终态：同步期间顶部和抽屉都显示“正在同步消息”；history error 会结束本轮重建并保留错误提示。缓存契约损坏等未捕获异常会先恢复 READY 再 fail-loud，不会永久卡在 SYNCING。
- 命令抽屉：改为固定 `440dp` 的 Material 3 sheet；标题留在固定区域，命令使用内部 `LazyColumn` 滚动。顶层使用不透明 `surfaceContainerHighest`，不再透出背后的对话正文；点击仍只回填输入框，不直接执行命令。
- 提交：`23adcb27` 修复时序、重同步和命令抽屉；`69da9950` 准备 Android 0.6.1。
- 自动验证：Android JVM、AndroidTest 编译、debug Lint/assemble 通过；API 36.1 无窗口模拟器中 19 项 Room migration/history/cache 测试和 2 项命令交互测试通过。独立 reviewer 复核后批准，无阻塞项；模拟器取图后已关闭。
- UI 证据：[固定且不透明的命令抽屉](assets/mobile-v0.6.1-command-sheet.png)。
- 版本：Android `0.6.1`（versionCode 7）；私有 Release [`v0.6.1`](https://github.com/kachofugetsu09/akashic-mobile-releases/releases/tag/v0.6.1)，资产 `Akashic-Mobile-v0.6.1.apk`（6,373,822 bytes）。
- 发布验证：release JVM、Lint、R8 与 assemble 在 1m54s 内通过；APK v2 签名有效；本地与 GitHub 资产 SHA-256 均为 `fc0923018573b327d93c8ef6671e3f3b93723b8e35430595297a37c4d0921f52`。

#### 本组 UI skill 约束落实

better-colors：

| Before | After |
| --- | --- |
| 命令 sheet 使用带 alpha 的 `surfaceContainerLow`，背后正文穿透 | 顶层 sheet 使用完全不透明的 `surfaceContainerHighest`；命令行仍共享同一平面 |
| 同步入口容易再造新的状态色 | 只复用现有 `primary/onSurfaceVariant` 和系统进度语义，不增加色系 |

better-typography：

| Before | After |
| --- | --- |
| 标题随命令列表一起滚走，长目录缺少固定阅读锚点 | `titleLarge + bodyMedium` 标题区固定，只有命令目录滚动 |
| 命令数量变化会改变整个 sheet 的阅读位置 | 固定 440dp 视窗；monospace 命令列和说明列位置保持稳定 |

better-ui：

| Before | After |
| --- | --- |
| 命令数量直接撑高 sheet，内容与对话争夺层级 | 固定半高 sheet、实体背景、内部独立滚动；向下拖动仍由原生 ModalBottomSheet 关闭 |
| 清缓存只能靠卸载或重新扫码，破坏连接状态 | 抽屉低频维护区提供带确认的“重新同步消息”，保留配对和本地未发送工作 |
| 时序问题容易被列表 `reverse()` 暂时遮掩 | 持久化服务端序号并让 DAO 拥有排序不变量，实时与历史统一稳定 |

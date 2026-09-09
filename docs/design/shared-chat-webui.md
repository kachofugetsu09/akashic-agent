# 共享对话 WebUI 试点设计

- 状态：implemented pilot
- 日期：2026-08-01
- 决策：[0018](../decisions/0018-chat-webui-has-one-source-and-two-adapters.md)
- 关联条款：WEBUI-001～WEBUI-007、MOB-001、TST-007～TST-008
- 视觉系统：[0043](../decisions/0043-paper-brand-tokens-replace-material-visual-semantics.md)；[纸张品牌系统](akashic-paper-brand-system.md)

## 1. 用户意图

两端采用移动端现有的浅蓝主题与界面质感，同时保留桌面 Web Chat 更自然的流式正文生长。以后对话前端只在 `akasic-agent` 修改；Android 仍保留比浏览器更丰富的原生能力，桌面仍可提供扫码配对等仅 Web 能力。

## 2. 当前事实与边界

- **F：** 两端都使用 React、Vite、`ChatMessageView` 和 `MessageResponse`。
- **F：** Android 通过 `WebViewAssetLoader` 加载 APK 内静态资产，以 Native bridge 发送完整 snapshot 和 streaming patch。
- **F：** Android 原生层拥有 Room、outbox、附件传输、通知、Keystore、相机扫码和生命周期。
- **C：** 共享 WebUI 源码迁入本仓库；视觉采用移动端色调，流式正文使用 Web 端呈现路径。
- **C：** 桌面扫码配对继续存在，移动端不需要伪装支持它。
- **U：** 本试点不定义 iOS 容器、远程动态下发 WebUI 或线上灰度更新协议。

## 3. 目标结构

```text
┌──────────────────────── akasic-agent ─────────────────────────┐
│ frontend/theme                                               │
│ ├─ theme-catalog.json       主题色值与领域状态目录             │
│ ├─ brand-tokens.css         paper / ink / rule / type          │
│ └─ material-tokens.css      迁移期兼容与既有适配器             │
│ frontend/chat                                                │
│ ├─ theme.css                共享 WebUI token 入口             │
│ ├─ message-view.tsx          共享消息、工具、流式正文          │
│ ├─ message-view.css          共享消息、工具与引用视觉          │
│ ├─ message-actions.tsx       共享引用、复制与引用预览          │
│ ├─ conversation-navigation.* 共享功能入口、会话与底部操作      │
│ ├─ main.tsx                  桌面适配器 + QR 配对能力          │
│ ├─ mobile-native.tsx         Mobile React 应用                │
│ └─ mobile-entry.tsx          Android transport + 挂载入口     │
└───────────────┬──────────────────────────────┬─────────────────┘
                │ desktop Vite build           │ clean commit build
                ▼                              ▼
       ┌─────────────────┐          ┌──────────────────────────┐
       │ static/chat     │          │ akashic-mobile-web.zip   │
       │ HTTP + WebSocket│          │ manifest + SHA-256       │
       └─────────────────┘          └────────────┬─────────────┘
                                                │ pinned consumer
                                                ▼
                                   ┌──────────────────────────┐
                                   │ akashic-mobile           │
                                   │ Gradle verify + unzip    │
                                   │ WebViewAssetLoader       │
                                   └──────────────────────────┘
```

## 4. 能力矩阵

| 能力 | 共享 WebUI | 桌面适配器 | Android 适配器 / 原生层 |
|---|---|---|---|
| 主题、消息、Markdown、工具轨迹 | 拥有 | 使用 | 使用 |
| 流式正文生长 | 单消息 rAF 发布器 | WebSocket delta 提交权威目标 | native patch 提交权威目标 |
| 会话侧栏、引用、复制 | 拥有 | 使用 | 使用 |
| 知识与插件入口 | 共享导航结构 | 跳转 Dashboard 公网端口 | 打开 Native bridge 页面 |
| 新聊天 | 共享导航结构 | Web session | Native bridge session |
| 扫码配对展示 | 复用视觉组件 | 生成 QR、确认设备 | 不挂载 |
| 相机扫码 | 无 | 无 | 原生 CameraX / ZXing |
| 设置、诊断、清理同步、重新扫码 | 无 | 不挂载 | Native bridge 拥有 |
| 离线队列、重试、阅读位置 | 只展示已验证状态 | 无 | Room 与 Native bridge 拥有 |
| 通知、分享、Keystore | 无 | 无 | Android 原生拥有 |

## 5. 性能合同

1. 历史消息保持 `content-visibility: auto`，streaming 行不启用该隔离，避免正在增长的消息高度估算错误。
2. Native patch 继续按 `requestAnimationFrame` 合并；React 消息行继续 memo，未变化历史行不重渲染。
3. 桌面与 Android 的权威增量都先进入单消息展示投影。同一帧内的多次更新合并为最新 target，每帧只通知一次对应消息行；不创建逐字队列，也不扫描或重建稳定历史行。
4. `message.final` 和 Android `streaming=false` 立即显示权威终稿并取消剩余展示帧；已经调度的旧帧不得在 terminal 后再次通知或覆盖终稿。
5. `MessageResponse` 只在正文或 `isAnimating` 改变时更新。流式 Markdown 由 Markstream 的 append-tail parser 接管并复用稳定顶层节点；terminal 使用同一组件完成最终解析，不再维护第二套 block 冻结和未闭合修复补丁。
6. 代码块、数学公式与 Mermaid 在 streaming 阶段保持轻量源码节点，terminal 后交回 Markstream 内建 renderer 完成富化；历史行继续按 viewport 延后富化。
7. 桌面打开会话先按 `seq` 游标读取最新尾页；读取更早页时按稳定消息 identity 恢复阅读锚点。分页只读 SessionDB，不修改、压缩或删除权威消息。
8. 不为动效新增依赖；交互状态使用可中断 transition。产物按构建入口分离，桌面不会加载 Android bridge、Room 投影或移动插件目录代码。

### 5.1 已验证的更新放大故障

两端共享 `Message`、React 组件和最终视觉，只保证展示语义同构，不保证执行成本同构。旧 Android 流式路径在每个 delta 写入 Room 后，使活动会话的完整 `MessageWithBlocks` 查询失效；客户端随后重新物化稳定历史并跨 Native bridge 提交 WebView。桌面 adapter 直接把 WebSocket delta 写入单消息展示投影，不经过 Room 与 bridge，因此相同 TPS 隐藏了完全不同的单次更新成本。旧 Markdown 渲染增加了活动消息的主线程工作，但不是这次全量更新放大的 owner。

高频局部变化经过每个 adapter 后都必须保持局部：正文 delta 不得重新查询、物化、序列化或提交未变化历史；稳定消息保持对象身份；只有 terminal、history heal 或明确的会话切换可以用权威 snapshot 校准展示。性能验收除 TPS 与总耗时外，还要记录每个 delta 触发的查询行数、bridge 字节数、React 通知次数和长任务，避免把“结果一样”误判成“成本一样”。

### 5.2 观测与归因

观测以 `session_id + turn_id + client_message_id` 为主身份，日志只记录阶段、耗时、计数与 outcome，不记录 prompt、正文或工具参数。Provider 原始首块、Core 首增量、Mobile durable inbox、真实 socket、Room、React commit、下一帧与 composer-ready 分层记录，不能用下游首字倒推 Provider TTFT。

```text
用户发送
  │
  ├─ send.received → send.ack → reply_sent
  │
  ├─ Akasha query → Provider raw first → Core first delta
  │                                      │
  │                                      ▼
  │                         durable queued → socket sent
  │                                      │
  │                                      ▼
  │                         Room → React commit → next frame
  │
  └─ Provider done → Akasha turn commit → runtime terminal
                                           │
                                           ▼
                              durable final → composer ready
```

定位规则：`send → provider.call.start` 属于准入、上下文与 Akasha 前置段；`provider.call.start → raw.first` 才是供应商首块段；`raw.first → next frame` 属于 Core、网络、Room 与 WebView 消费段。尾部同理拆成 Provider 完成、Akasha/AfterTurn、worker durable terminal 和客户端 composer 四段。

## 6. 产物、失败和回滚

### 桌面连接与异步任务

桌面 controller 使用 Effect v3 管理 WebSocket 重连、发送等待和状态轮询。选用稳定版，
先固定资源泄漏与轮询重叠的回归，再替换手写 timer 和取消记账；不增加独立连接管理类。
TypeScript 固定为 5.9.3，沿用现有 strict、ES2022 和 bundler 配置。

```text
┌─ React controller 生命周期 ─────────────────┐
│ 连接任务 → socket / 监听器 → 释放 → 重试等待 │
│ 轮询任务 → fetch 完成 → 等待 → 下一次 fetch │
└───────────────────┬────────────────────────┘
                    └─ 卸载：中断任务并释放资源
```

自动重试保留十二次上限、指数退避、三十秒上限和随机延迟；成功连接重置计数。
主动发送可以提前结束重试等待，但不重放已经发送的消息。发送等待的成功只表示
`WebSocket.send` 完成，不代表服务端确认；超时、关闭和取消仍以错误返回。
状态轮询在上次请求结束后等待 1.2 秒，卸载时中断 fetch，旧请求不能在卸载后发布状态。

消息解析、Session/Turn、流式投影、插件 ABI 和 Android 原生连接 owner 保持不变。
`desktop-chat-lifecycle.test.mjs` 挂载真实 controller，使用受控 socket、HTTP 完成顺序和
时钟验证重连后卸载、串行轮询、重试上限及 React StrictMode。

### 构建与恢复

- `npm run package:mobile-web` 只接受干净 Git tree，ZIP 内写入 source repository、commit、tree 和资产摘要。
- Android 在解包前核对外部 SHA-256；不匹配时 Gradle 失败，不使用旧缓存或网络 fallback。
- WebUI 构建失败不会改变移动端原生状态；产物升级只替换 APK 构建输入。
- 回滚主仓库到上一个 WebUI commit并重新打包；移动仓库恢复上一个 ZIP、摘要和 source lock。两边都不需要迁移数据库或 workspace。

## 7. 视觉语法

桌面与 Mobile 使用同一纸张品牌 token、`ChatMessageView` 和 `message-view.css`。用户气泡、Akashic 正文、Markdown 与工具过程由共享 WebUI 拥有；Mobile 只补 viewport、触摸、抽屉、Bridge、草稿、outbox 与离线状态，不增加装饰性角色标题或平行消息组件。Android 原生能力和 Bridge owner 不随视觉变化。

## 8. 试点验收

- 主仓库：typecheck、chat build、mobile web build、mobile state tests、lint。
- 视觉：只在生产桌面 Chat 与移动 Web 构建中核对主题 token、布局、消息和工具轨迹；不得为验收复制第二套消息 DOM、静态数据或交互状态。
- 离线与降级场景：通过生产 Chat 的静态响应、fixture transport 或现有状态测试注入输入，继续使用正式 `ChatMessageView`；不维护平行方案页或独立消息实现。
- 移动仓库：ZIP 正向校验、篡改失败测试、Gradle debug build。
- 报告两仓库 commit/tree、ZIP digest；真机 WebView、内存、掉帧和冷启动单独列为未验证或设备证据。


## 历史附件、聊天投影与发送待办（2026-09-08，已确认）

```text
┌─ Core Message / Artifact ──────────────────────────────┐
│ 原始 Message + seq；不可变 artifact_id / metadata / bytes │
└────────────┬──────────────────────────────┬─────────────┘
             │ 完整同步                      │ Message 引用授权下载
             ▼                              ▼
┌─ 共享 WebUI ─────────────────┐  ┌─ Android ──────────────────┐
│ 只读聊天投影；隐藏迁移诊断     │  │ Room / outbox / 文件缓存    │
│ 思考与工具记录留在原 Message  │  │ cacheId = H(server, artifact)│
└──────────────────────────────┘  └────────────────────────────┘
```

- `attachment.download` 请求携带 `message_id`、`artifact_id` 和 `offset`。Session reader 先确认该 Message 引用了文件，再由 Core ArtifactStore 核验和读取。回复保留完整附件 metadata；下载二进制头使用 `artifact_id`，上传头继续使用 Frame ID `attachment_id`。空文件允许零字节分片并按 SHA-256 验证。
- Android 缓存键与远端 Artifact ID 分开。Native→Web snapshot v11 的下载状态同时提供 `artifactId`（匹配 Message 引用）和 `cacheId`（调用本地重试、打开和分享）。会话仅从 Message link 取得授权，不拥有共享文件缓存。
- `history.provenance`、`history.record`、`history.turn_input` 不进入普通聊天；纯归档 Message 不占布局和可见未读数量。`history.transcript` 的已知旧格式按原组顺序展示思考、说明和工具记录，不生成新消息或执行状态。原始数据和同步进度不减少。旧阅读或导航锚若指向隐藏行，定位到后续首个可见行；末尾则定位前一可见行，不能直接跳到最新消息。
- 明确拒绝删除本地 outbox、保留失败正文并释放本地上传占用。结果未知保留原命令及其附件占用；核对复用原 ID。新一次发送创建新 Message，不迁移旧视觉身份。已落地 Input 或 ACK 都是接受证据，迟到错误不得将其降级。
- 文件缓存写入失败只结束该下载并消费对应回复。Room 持久化失败停止消费和 ACK，等待用户处理存储后重连；自动重连不作为本地数据修复。

验证入口：`tests/test_mobile_message_log.py`、共享聊天投影测试、Android Room 18→19 迁移与下载测试；`tests_scenarios/mobile_artifact_history.py` 提供全新目录中的真实 TLS Gateway，用于 Android Room→文件→共享 WebView 的完整验证。测试不读取正式 workspace 或正式手机应用。

## 9. 正文接替与历史恢复（2026-09-09）

### 9.1 已确认的正文展示

同一 source 的 `continue` 正文只是执行中的当前正文；新正文接替旧正文，`complete` 结束后只展示最终正文。思考与工具合为一条过程轨迹，仍沿原 `message_id + part_index` 查看，最后正文使用最终 Message 的复制、引用和时间。`pause`、`failure`、`abandon` 按 `through_seq` 隔开前后展示过程，停止前的过程保留在末条输出上；这不改变持久 Turn 的划分。其他 source 的输出不替换本来源正文。分页和实时追加使用同一展示规则，不改写任何 Message。

```text
Input → 等待 → 思考 / 工具 + 当前正文 → 最终正文
                  └─ 原过程引用仍保留 ──────┘
```

本次修复由共享 WebUI 的 `message-timeline.ts` 和 `TimelineMessageView` 拥有，桌面与 Android WebUI 使用同一规则。`runtime_patch=false`；SessionDB 和 Akasha 不变。只修改 Git worktree 的源码与本说明，验证使用一次性 fixture；没有正式 workspace、外部消息或部署副作用。恢复点为任务开始前的源码归档与基线 `b5967641`。

### 9.2 当前窗口与按需历史

维护者于 2026-09-09 选择按需方案并授权实现、合并、部署与必要 APK 发布。原 Android `28bdd84` 的全会话前向补齐已被替换；桌面本来就使用最近 50 条和 `before_seq`，本次同时收窄它的展示响应。

只读基线：正式会话末尾 50 条 HTTP 响应为 6,718,701 bytes，其中不可见 `history.record` 占 6,655,426 bytes；loopback 三次完整读取为 854 / 242 / 159 ms，不含公网与手机渲染。4,204 条持久记录的正文与 metadata 共 160,616,603 bytes。行数上限不能代替字节边界。

```text
┌───────────────┐    ┌─────────────────────┐
│ 认证与会话目录 │───▶│ 当前会话尾页，head=H │
└───────────────┘    └──────────┬──────────┘
                              ▼
                   ┌──────────────────────┐
                   │ 从 H 订阅 + 回复状态 │──▶ 可以发送
                   └──────────────────────┘
┌───────────────┐    ┌─────────────────────┐
│ 上翻 / 引用跳转│───▶│ 旧页 / 目标所在窗口 │
└───────────────┘    └─────────────────────┘
```

#### 分页与展示合同

- `history.get(direction=backward)` 无 cursor 时读尾页；`before_seq` 排他地读更早消息；`around_id` 由服务端定位并返回以目标结尾的窗口，目标不存在返回 `message_not_found`。三者都复用 `MessageReader.read_tail`，不创建消息身份或改写日志。
- 页内 `(after_seq,next_after_seq]` 是完整收到的记录/下载清单范围；`through_seq` 是快照 head。向后页的 `next_after_seq=before_seq-1`；`next_before_seq` 是首条 seq，`has_more` 指更早记录。没有更早记录时下界为 -1。`request_id` 只关联当前请求，不成为历史进度。
- 初始页数 50 是调节值，不是聊天准入条件。单帧仍有 240 KiB 预算，尾页缩小时只舍去左侧整条记录，不能丢掉最新消息。超大可见正文沿既有整条 JSON 清单和 Range 传输，不截断内容。
- 新 Mobile 显式请求 `display_only=true`，桌面默认用展示表示：不可见的 `history.provenance / history.record / history.turn_input` 只留 kind 和 unavailable 标记，原 part 下标与可见 `history.transcript` 不变。旧客户端保持旧表示。整条下载仍由原 byte_length 与 SHA-256 精确选择两种已知表示，升级前的未完成下载可以继续。
- Room 19→20 只新增 `message_ranges(sessionId,afterSeq,throughSeq)` 与下载记录的表示标记；不删除旧消息、附件、草稿、outbox 或配对。范围和完整记录/清单在同一事务提交，之后才 ACK。重叠范围合并不丢覆盖；没有范围证据的旧缓存不推断为完整前缀。
- Native 拥有订阅、Room、下载与唯一连续显示窗口。上翻扩展左边界；引用跳转换成目标窗口；旧窗口中的新实时消息只进入缓存，回最新时重新取尾页。窗口替换推进 WebUI projection generation，旧代际事件不能混入新窗口。
- 初次 READY 等待当前尾页清单、当前订阅确认和已知回复状态，不等待全部旧正文、附件、其他会话或通知定位。outbox 仍持久化用户输入，在 READY 后由原 owner 发送。保存的阅读锚在近期窗口外时，单独定位该锚，不枚举它之前的全部记录。
- 缓存未命中不能判定通知过期；必须取得服务端明确不存在证据。通知仍在精确 `Output(finish=complete)` 的完整正文落地后发布。分页错误不把已就绪连接降为不可聊天。

#### 所有权与持久化

Core Message 日志和附件保持 append-only，只有既有 adapter 的读协议变化；不新增 SessionDB 状态或删除权限。Native 本地范围只增加或合并，明确清理投影时与对应缓存一起减少；事件 `reset_required` 只更新事件 cursor 并重读目录/当前尾页，不清空 Message。旧缓存重放时只允许去掉上述不可见归档展示值，并逐字段核对其余消息事实；权威归档仍在服务端。旧、新清单交错时保留当前下载 owner，不改写其已确认片段。

恢复点：Core 基线 `b5967641`、Mobile 基线 `28bdd84` 及任务目录外源码归档；正式部署前另备份整个运行 workspace。Room schema 升级后不支持直接降级 APK；回退需提供兼容 schema 的修复版。所有设备验证使用隔离 application ID。

验收覆盖尾页帧预算、旧/新表示摘要、完整数据库未变、部分窗口与实时追加、Room 接收范围/ACK 原子性、旧引用定位、阅读锚、断线和未下载正文。性能分别记录首屏 bytes、接收行数和发送时刻，不能把本地输入接受当成服务器已发送。

参考：[Matrix limited timeline 与向前补页](https://spec.matrix.org/latest/client-server-api/#syncing)、[Stream 消息 ID 分页](https://getstream.io/chat/docs/javascript/channel-pagination/)；使用本项目已有 `message_id + seq`，不引入第三方 token 模型。

### 9.3 回复过程与调用统计

每条过程轨迹只在开头挂载一次 `turn.before_reasoning`，工具调用后的消息和后续草稿不重复挂载。实时回复从等待首段起就把插槽放在同一个思考面板内；思考到达后不移动插槽，避免 Akasha 卡片卸载重查。`model.selection` 是内部选择记录，不占正文布局，未知插件内容仍明确显示不可展示。

统计由 models 的调用记录拥有。Web 通过公共 `/api/settings/model/calls/{call_id}` 读取；Android build 79 起用 `readModelCallStats` 转发已有 `model.call.get`，共享页面校验与计算数值；没有数据或查询失败均不估算。

Android build 80 起，断线、等待订阅或查看历史造成的观察缺失通过 Native→WebUI 的 `reply.clear` 事件清除临时回复状态；事件绑定当前 session 与 projection generation。服务端 `reply.status.available=false` 只表示真实能力不可用，两者不能互相代替。该事件不进入服务端协议或 Message 日志。最低原生 build 为 80。

“加载更早的消息”位于已加载历史的最上端，占有独立行并随消息滚动，不悬浮遮挡正文。插件卡片查询从实际发出时开始计算 30 秒期限；本地排队不消耗传输期限，超时仍取消所属 UI owner 并释放容量。

Native→Web snapshot v11 包含本地 `reply.clear` 事件。APK 通过已有 manifest 兼容检查拒绝 snapshot v10 的 OTA 界面，改用内置界面，防止升级后向旧界面发送未知事件。


### 9.4 召回卡片的页面缓存

`cache: "memory"` 由共享 Web Host 拥有：复用现有最多 128 项、8 MiB 的结果缓存，只保存 `pending !== true` 的成功结果。相同插件 revision、方法、参数、session/turn 身份的在途读取共享一个请求。该模式向 Native 发送既有 `cache: "none"`，不改变原生磁盘缓存合同，也不把进行中的结果持久保存。插件通过 `capabilities.queryCacheModes` 判断 Host 是否支持该优化；旧 OTA Host 继续使用原来的无缓存读取。

UI owner 只拥有订阅。最后一个订阅卸载时，尚未发送的读取取消；已经发送的页面缓存读取由独立 wire owner 完成，最多等待现有 30 秒期限。catalog 切换取消所有在途读取，旧结果不能污染新版本。普通非缓存查询仍随原 UI owner 撤销。

renderer 可提供 `prefetch(context)`，只执行一次轻量读取，不挂载卡片内容；Host 仅保留近视口观察节点。已提交过程在邻近可视区域预取首个思考卡片；未展开的预取不轮询，主动展开可以提升尚在网页队列中的同一读取优先级。进行中的可见卡片仍按已有间隔刷新，失败保留已显示内容并提供局部重试。页面重载会丢弃内存结果，缓存不是新的召回权威状态。

网页 `[akashic-trace] webui.plugin_query.*` 与服务端 `mobile.plugin_query.*` 以 `owner_id` 对齐，分别记录排队、实际发出/执行和回执阶段，不记录正文、参数或授权材料。服务端执行成功不等于网页收到；缺失阶段仍需结合手机日志定位。

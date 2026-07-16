# Akashic Mobile Agent IM 实施日志

本日志按语义完整的功能组记录设计、验收、实现、真实证据和提交。长期约束见 [mobile-agent-im-original-request.md](mobile-agent-im-original-request.md)。

## 总体状态

| 领域 | 当前基线 | 下一闭环 |
|---|---|---|
| 消息基本功 | 已有真实时间、日期分组、双向引用、单条/批量复制、消息选择、搜索、跳转、未读、失败重试和乱序归位 | 删除/转发的服务端所有权与协议语义 |
| 实时 Agent | 有流式回答、思考/工具时间线、停止、可展开的安全工具详情和结构化复制 | 失败态解释与长结果二次操作 |
| 媒体 | 有上传、进度、缓存、预览、GIF/meme、重试、按需大文件下载和分享 | 相机/系统分享入口与媒体发送一致性 |
| 网络 | 有认证、resume、durable inbox、连接状态 | 抖动场景矩阵和用户可恢复动作 |
| 会话 | 有 mobile 全量同步、抽屉、切换、新建、当前会话搜索和阅读位置恢复 | 失效会话解释与本地清理 |
| 扩展 | 已有受控 `plugin.ui.*`、热更新和 Observe/KV Cache 移动 Dashboard 试点 | 第二个任务型插件试点与跨插件导航 |
| 质量 | 有 Android 测试、隔离 Gateway、签名 release 与 Pixel 7 闭环 | 可复用的 release WebView 交互驱动 |

## Cycle 1：时间、滑动引用与复制

状态：已实施，通过 Pixel 7 隔离环境验收与三轮独立 Review。

### 缺口与任务

- 消息实体已有 `createdAt`，但 WebView 快照丢弃，用户无法判断发送和回答时间。
- 没有引用协议、持久化、Agent 上下文或历史恢复；不能精确回应旧主动推送和旧回答。
- 已有系统文字选择，但最终回答没有稳定的独立复制动作。
- 当前 UI 需要在不增加“消息卡片墙”的前提下承载这些动作。

### Better UI

| Before | After |
|---|---|
| 消息只静态显示，引用依赖未来菜单 | 复用 `motion` 实现仅向左的跟手拖动，`50dp` 阈值点亮引用 state layer 并触发原生轻触反馈，释放后归位 |
| 输入区只有附件和发送状态 | 引用时在输入区内部形成附着的引用条，可关闭，发送后原子清空 |
| 最终回答没有显式动作 | 回答后增加低强调度圆形复制动作，复制成功给短暂状态反馈，不新增动作卡片 |
| 消息之间只靠大间距 | 时间直接落在消息平面；跨日才增加文字与细线组成的日期分隔 |

### Better Colors

| Before | After |
|---|---|
| 蓝色承担普通主操作，紫色承担 Agent 过程 | 保持映射；引用预览使用 primary container，滑动越阈值使用更明确的 primary state layer |
| 次要元信息只有通用灰色 | 时间继续使用 on-surface-variant，不引入新的装饰色 |
| 复制没有状态语义 | 成功只在原动作位置切换为 primary/完成图标，不创建绿色成功徽章 |

### Better Typography

| Before | After |
|---|---|
| `createdAt` 未展示 | 时间使用 12px、tabular figures、短本地格式；日期使用 12px 中等字重 |
| 引用来源无排版层级 | 来源用 12px 中等字重，预览用 13px 单行省略，正文保持原有阅读尺度 |
| 动作依赖图标猜测 | 图标保留可访问名称，短暂反馈使用简短“已复制”状态文本或 aria-live |

### Pixel 7 可复用验收标准

1. 在一条已完成消息上左滑：纵向轻微移动不触发；超过约 `50dp` 只振动一次，退回后再次越界可再振动。
2. 越界松手后输入区出现正确角色和单行预览；关闭可撤销；发送后键盘收起、引用条消失。
3. 服务端收到引用目标，Agent 能根据被引用内容回答；断开重连和全量历史同步后引用关系仍显示。
4. 用户消息和助手回答显示本地时间；跨日样本只出现一个日期分隔，不为每条消息加卡片。
5. 点击最终回答复制动作后，Android 剪贴板内容与 Markdown 源文一致，并出现短暂成功反馈。
6. 文字长按选择、纵向滚动、时间线自动跟随和附件点击不被滑动手势破坏。
7. TalkBack 能读出“引用此消息”“取消引用”“复制回答”；触控目标不小于 `44dp`。

### 验证记录

#### 实现范围

- WebView 快照升级为 v2，消息携带 `createdAt` 与引用关系；同日消息显示本地时间，跨日使用一条平面日期分隔。
- 用户消息与已完成的 Agent 回答都支持仅向左滑动引用；复用 `motion/react` 的约束拖动与 spring 归位，跨过阈值调用原生 haptic。
- 引用预览附着在 composer 内部上方，发送消息中内嵌同一关系；消息动作行直接贴近所属消息，用户侧右对齐、Agent 侧左对齐。
- 原生桥接系统剪贴板和触觉设置；Room 升级到 v4，保存创建时间和引用字段，并提供 v3 → v4 迁移。
- 移动协议只允许客户端提交 canonical ID 或 client message ID；角色和预览由服务端同会话消息重新解析，不能由客户端伪造。
- 复用 Telegram 的历史消息引用文本格式，把引用同时写入显示历史与 `llm_user_content`。
- 修复统一 `PassiveMessageWorker → ConversationRuntime` 桥丢弃渠道元数据的问题；桥接层现在整体转发受控 inbound metadata，因此 `client_message_id` 和引用关系能由既有 AfterReasoning 持久化 owner 落库。
- 用户消息只有在 `message.final` 返回服务端规范消息 ID 后才变为可引用；仅收到传输 ACK、发送失败或仍在等待提交时都不会暴露引用动作，避免引用尚未落库的乐观消息。
- `message.final` 中的 canonical ID 与 client ID 都从同一条已持久化用户记录投影，插件不能把两者改写成错误配对。
- 客户端创建时间改为带时区的 RFC 3339 instant，并由 AfterReasoning 原样写入用户消息时间；历史显示不再被服务端处理耗时改写。
- 引用预览限制为 512 字符，但模型上下文保留目标消息全文；UI 截断不再损失 Agent 实际理解所需的内容。
- 仅附件消息使用 `[附件]` 作为引用语义文本，避免 UI 显示可引用但服务端拒绝空正文。
- 过程紫色继续用于活动节点；另设同色相、较低明度的 `trace-text`，保证小字号状态文本在 surface 上达到 4.70:1，不牺牲点亮节点的明亮感。
- 协议拒绝按可恢复性分流：outbox 改为逐条单飞，Gateway 用 `4410` 明确标记坏 `message.send`，客户端只隔离当前命令并自动恢复后续队列；无法归因的 `4400` 和版本错误 `4406` 保留待发内容，不再伪装成网络抖动循环重连。
- Gateway 对手机和 WebSocket close reason 只返回静态协议原因；Pydantic 字段位置、判别值和用户正文仅留在受控服务端诊断边界，不回显到客户端。
- resume 会核对冻结的 durable inbox 是否逐号连续；缺号时只发送 `sync.reset_required`，由既有 Android reset 协议跳过损坏区间、清理可重建投影并全量同步，不放宽客户端严格顺序校验。

#### 自动化验证

- `.venv/bin/pytest -q tests/`：`2269 passed`。
- `.venv/bin/pytest -q tests/control/test_channel_adapter.py tests/control/test_control_execution.py tests/mobile_realtime/test_channel.py tests/mobile_realtime/test_mobile_realtime_protocol.py tests/test_lifecycle_phases.py`：`89 passed`。
- `npm run typecheck && npm run lint`：通过。
- `pyright`（本组改动 Python 文件）：`0 errors, 136 warnings`；warning 为仓库既有类型债务。
- `clients/android/scripts/build-release.sh`：release unit test、lint、R8、签名 APK 构建和 `apksigner` 校验通过；签名证书 SHA-256 为 `49bf31ed…40bc`。
- `ANDROID_HOME="$HOME/Android/Sdk" ./gradlew :app:assembleDebugAndroidTest`：通过，Room 迁移与 LocalDeliveryStore instrumentation 源码完成真实编译。
- `ANDROID_HOME="$HOME/Android/Sdk" ./gradlew testDebugUnitTest`：通过；协议关闭策略区分坏命令与版本不兼容。
- Gateway 回归用例真实完成认证、resume、坏 `reply_to` 投递、`protocol.error` 和 WebSocket close，确认控制帧与 close reason 均不包含用户载荷。
- durable inbox 回归覆盖首条缺失、中间缺失、reset ACK 后再次正常 resume；既有用例继续覆盖 600 条分页和 resume 期间并发事件位于 terminal 之后。
- 用 v3 Room schema 创建临时 SQLite，执行迁移 fixture 和 `MIGRATION_3_4` 三条 SQL：原消息结果为 `保留我|||`，证明正文保留且新增引用列为空。
- WCAG fallback 色值检查：正文/surface `14.72:1`，次要文字/surface `6.66:1`，引用文字/container `8.82:1`，白色/primary `4.87:1`，error/container `4.79:1`，trace-text/surface `4.70:1`。

#### Pixel 7 隔离网关证据

- 使用 `docker/mobile-lab` 独立 workspace、独立插件目录和 `wss://mobile-lab.wangyuanzhe28.site/ws`；正式 workspace 未写入测试消息。
- 真机建立 `mobile:c77b8b2f-7de6-40d8-9918-6323535bcb2c` 测试会话；输入法弹出时 composer 紧贴 IME，引用预览继续附着在 composer 上方。
- 左滑用户消息 `quote_user_source_v2` 后发送 `quote_user_reply_v2`：服务端落库 `reply_role=user`，目标为 canonical `:4`，状态为“已发送”。
- 左滑 Agent 回答“嗯。”后发送 `quote_agent_reply_v2`：服务端落库 `reply_role=assistant`，目标为 canonical `:7`，状态为“已发送”。
- 两条消息的 `llm_user_content` 分别包含“被回复消息（来自 你）”与“被回复消息（来自 Akashic）”，证明引用进入真实模型上下文，不是仅有视觉预览。
- 真机截图：`/tmp/akashic-pixel7-quote-user-v2-sent.png`、`/tmp/akashic-pixel7-quote-agent-v2-sent.png`；logcat 中对应 `message.send` 均携带 `reply_to`，无协议校验或断线错误。
- 重载隔离 Agent 进程后发送 `canonical_reply_probe_0501`：传输 ACK 阶段不显示引用动作，收到 `message.final` 后用户消息升级到 canonical `:12` 并出现引用动作。
- 引用该用户消息发送 `quote_user_canonical_reply_0503`：落库目标为 `:12`，`reply_role=user`；随后引用 Agent 回答“嗯。”发送 `quote_agent_canonical_reply_0506`：落库目标为 `:15`，`reply_role=assistant`。
- 两次新探针均在真实模型请求中分别呈现“来自 你”和“来自 Akashic”；截图为 `/tmp/akashic-pixel7-probe-0501-final2.png`、`/tmp/akashic-pixel7-quote-user-sent-0503b.png`、`/tmp/akashic-pixel7-quote-agent-sent-0506.png`。
- Pixel 7 使用签名 release APK；重启应用、断开并恢复 WebSocket 后引用关系仍由服务端历史恢复，logcat 无 FATAL、Room、协议反序列化或证书链错误。

#### 正式网关序列缺口恢复

- Pixel 9 的真实 cursor 为 `ack=1994`，durable inbox 首条却是 `1996`；客户端按协议等待 `1995`，旧服务端每次重连又追加 `sync.completed`，最终把序号推到 2.7 万并形成永久重连。
- 重启前使用 SQLite online backup 保存 `mobile_realtime.db` 和 runtime 日志，备份位于 `workspace/backups/mobile-sequence-gap-before-fix-20260716-0855/`，`integrity_check=ok`。
- 新 Gateway 首次 resume 发出 `sync.reset_required` 后，Pixel 9 cursor 推进到 `ack=27789`，durable inbox 清为 `0`；3 秒复查 cursor 和 inbox 均保持不变，重连风暴停止。
- 重启同时发现 2 条已无消息本体的 embedding 缓存阻塞 Akasha fail-fast 启动；先备份完整 `sessions.db` 到 `workspace/backups/sessions-orphan-embeddings-before-repair-20260716-0859/`，再只删除 2 条孤儿缓存。修复后 `integrity_check=ok`、孤儿数为 `0`，未删除消息或会话。

#### 独立 Review

- 三轮 Review 均按可达失败路径检查协议、持久化、Room 迁移、WebView 投影和真机状态语义；最终无 blocker、高风险或中风险问题。
- Review 发现并修复：插件改写 client ID 造成 canonical 配对错误、v3 fixture 使用不存在的 `turnId`、仅附件引用抛内部异常。
- 新增 Room instrumentation 用例覆盖 `message.final` 将用户 optimistic 行迁移为 canonical，并保留附件关联与引用投影。用户消息不会产生 Agent `turn_blocks`，因此没有为不可达状态制造测试 fixture。

#### Kill AI Slop 与位置复核

- 没有引入渐变标题、玻璃拟态、发光状态点、彩色图标方块或独立消息卡片墙。
- 引用使用一个有状态意义的同色区域，不在引用条外再套卡片；复制和回复保持 44dp 圆形 state layer。
- 日期分隔、时间、动作和状态直接落在会话平面；只有用户消息本身使用方向性气泡。
- 真机逐项确认动作所属关系：用户动作跟随右侧气泡，Agent 动作跟随左侧回答，引用预览紧贴输入区，没有漂到顶部、抽屉或独立浮层。

#### 已知问题与下一闭环

- Pixel 7 上一条修复前产生的本地失败消息会附加在较新的服务端历史之后，没有按创建时间归位；这不影响本轮成功引用的持久化与恢复，但应在“失败重试与乱序合并”闭环统一修复，避免在引用功能里顺手引入第二套消息排序规则。

## Cycle 2：会话搜索、消息跳转与未读锚点

状态：已实施，通过 Pixel 7 隔离环境搜索、弱网恢复、未读、会话切换、附件和破坏性重建闭环。

### 缺口与任务

- 长会话只能手动滚动，不能从用户提问、Agent 最终回答或附件名定位目标。
- 用户离开底部后，新的 Agent turn 没有稳定未读边界；token、thinking 和 tool 增量又不能被误算成多条消息。
- 既有回到底部按钮位于中间，没有成熟 IM 的阅读锚点和计数语义。
- 搜索期间流式输出仍可能抢回底部，导致正在阅读的旧消息失去位置。

### 交互结构

```text
┌─ 顶栏 ──────────────────────────────────┐
│ 常态：会话抽屉  连接状态        搜索    │
│ 搜索：返回  [ 搜索这段对话 · 清除 ]     │
├─────────────────────────────────────────┤
│ 日期分隔                                │
│ ───────────── 3 条新消息 ─────────────  │
│ 消息语义区 ← 跳转后短暂 state layer     │
│                                  ↓  3   │
├─────────────────────────────────────────┤
│ 2 / 7                         ↑   ↓     │
└─────────────────────────────────────────┘
```

搜索模式用顶栏输入和底部导航平面替换输入区，不新增搜索卡片或结果卡片。首版复用 Android 已同步到 Room、再完整投影给 WebView 的当前会话历史，不提前增加服务端搜索协议；开始虚拟化历史后再下沉到 Room FTS。

### Better UI

| Before | After |
|---|---|
| 只能连续手动滚动 | 顶栏 44dp 搜索入口；按稳定消息 ID 导航结果，近距离平滑居中，超过两个 viewport 直接定位，避免长时间飞屏 |
| 流式更新一直拥有滚动锁 | 进入搜索的 layout 阶段调用现有 `use-stick-to-bottom.stopScroll()`；退出时只有“原本在底部、没有跳转、没有手动滚动”才恢复底部 |
| 居中通用回底按钮 | 右下 48dp 圆形动作紧邻输入区；未读计数附着在按钮上，第一次点击到首条未读，之后才到底部 |
| 新内容没有阅读边界 | 日期、未读、角色分隔和正文按语义顺序直接落在消息平面，不把未读塞进消息或时间线 |

### Better Colors

| Before | After |
|---|---|
| 紫色同时面临过程与选中语义竞争 | 紫色继续只表达 Agent thinking/tool 过程；搜索目标、结果命中和未读统一使用主蓝色 |
| 跳转位置只能靠滚动猜测 | 目标消息使用 1 秒主蓝 state layer，命中文字使用 primary container；不增加边框或永久色块 |
| 未读可能被做成状态卡 | 一条同色浅线与主色文字表达未读边界，计数只附着在回底动作上 |

fallback 主色文字/页面对比为 `4.52:1`，on-primary-container/primary-container 为 `8.82:1`；12px 未读标签与 11px 按钮计数均使用中粗或粗字重。

### Better Typography

| Before | After |
|---|---|
| 无搜索输入层级 | 搜索输入保持 16px，避免 Android WebView 输入缩放；placeholder 使用 on-surface-variant |
| 结果位置和未读数无稳定数字排版 | `2 / 7`、`99+` 与未读标签使用 tabular figures，分别为 13px 与 11–12px |
| 搜索命中依赖整块染色 | 正文尺度和行高不变，只用浏览器原生 CSS Highlight 标记当前目标内的命中词 |

### Pixel 7 可复用验收标准

1. 搜索用户提问、Agent 最终回答和附件名，结果总数、当前位置和上下边界正确；连续改查询不会残留旧结果。
2. 输入第一个命中字符后搜索框不失焦、键盘不收起；只有键盘明确提交搜索时才把无障碍焦点交给目标消息。
3. 近距离结果平滑居中，远距离结果直接居中；目标 state layer 保持约 1 秒并平滑退场。
4. 搜索期间让 Agent 持续流式输出，屏幕不被 ResizeObserver 或自定义 auto-scroll 拉回底部；关闭后保留已跳转位置。
5. 从会话 A 的旧位置切到长会话 B，B 重新建立独立滚动上下文并落在最新消息，不继承 A 的 `scrollTop`。
6. 离开底部后产生一个新 Agent turn，未读只增加 `1`；后续 thinking/tool/token 更新不重复增加。
7. 首条未读跨日期时顺序固定为“日期 → 未读 → 角色 → 正文”；到达真正底部后才清空未读。
8. 搜索、清除、上下结果和回底动作的触控区均不小于 44dp；TalkBack 能读出搜索、结果位置、未读数量和回底动作。
9. 长会话滚动与流式回答无明显掉帧；截图确认回底按钮不遮时间、工具时间线或输入区，logcat 无 WebView console error。

### 当前验证记录

- `npm run typecheck`、`npm run lint -- --max-warnings=0`、`git diff --check`：通过。
- `npm run test:mobile-web-state`：`11 passed`；覆盖 streaming → canonical ID 迁移、thinking/tool 增量不重复计数、下一 turn 只新增一次、脱离底部后的滚动锁、附件名搜索、搜索增量索引、普通 reconnect 保留未读，以及破坏性投影重建建立已读基线。
- `npm run build:mobile-web`：5267 个模块完成生产构建；只有仓库既有 bundle size warning。
- `ANDROID_HOME="$HOME/Android/Sdk" ./gradlew :app:testDebugUnitTest --tests com.akashic.mobile.data.realtime.AttachmentDownloadCoordinatorTest --tests com.akashic.mobile.ui.web.MobileWebSnapshotTest`：通过；覆盖投影重建代际序列化、同连接重复 ready 不重复下载、断线从 fsync offset 续传、末分片 reply 丢失后的本地校验发布，以及下载占用状态只发生一次 `true → false` 迁移。
- `clients/android/scripts/build-release.sh`：release unit、lint、R8、签名 APK 构建和证书校验通过。
- 最终修复后重新执行 `clients/android/scripts/build-release.sh`：`66 actionable tasks`，构建成功；`apksigner` 确认 v2 签名和证书 SHA-256 `49bf31ed…40bc`。
- 最新签名 release APK 已用 `adb install -r` 无损覆盖 Pixel 7；未卸载应用、未清除数据。版本仍为 `0.7.7 (16)`，手机内 `base.apk` 与构建产物 SHA-256 均为 `afb2a436…bc0e7`。
- 隔离 `docker/mobile-lab` 的 Agent、chat proxy 和 Cloudflare tunnel 已连续运行约 4 小时；Agent health 为 healthy，独立 workspace 加载 `akasha` 与 `default_memory`，未启用主动推送。
- kill-ai-slop 最终扫描 `frontend/chat/src` 为 36 个文件、9 组、50 个命中；命中的是既有 shimmer、工具错误/脉冲、showcase、语义圆形图标动作、代码等宽字体和 24px 半径。本轮没有新增渐变、玻璃拟态、发光状态点、卡片墙或胶囊堆叠。搜索输入的胶囊形状属于单一任务容器，回底和导航按钮属于圆形动作类别。
- 独立 Review 首轮发现并修复：自动搜索跳转抢走输入焦点、切会话继承旧滚动上下文、搜索期间未解除 `ResizeObserver` 底部锁、NFKC 搜索与 DOM Range 无法共享位置、清除动作只有 36dp。
- 独立 Review 复核继续发现并修复：公开 `isAtBottom` 合并了 near-bottom 状态，不能表示库内部是否重新锁定；现改用 `escapedFromLock` 监听真实锁状态，搜索期间每次重新锁定只纠正一次，不形成 effect 循环。
- 后续独立 Review 发现并修复：streaming 助手 ID 在 `message.final` 后迁移为 canonical ID 导致同一 turn 重复未读；16ms 快照合并导致发送后回底漏判；搜索期间每个 token 全量扫描长历史；目标 state layer 被不透明消息平面遮挡。当前 canonical 对齐已有独立回归测试，发送动作显式产生一次性回底 token；搜索索引只在搜索模式存在，查询变化才全量匹配，后续快照只重算 revision 变化的消息；state layer 位于消息内容上层。
- 最终边界 Review 又发现普通 reconnect 和破坏性重建共用 `SYNCING`、下载中允许清缓存、`sessionId + createdAt` 锚点可能同毫秒碰撞。现由原生 `projectionGeneration` 明确标记破坏性重建；下载协调器上报真实 active 状态并同时在 UI 与 `RealtimeSession` 边界禁止重载；未读锚点改为每个逻辑消息生成身份，只在明确的 old ID → canonical ID 迁移时转移已访问状态。
- 上述三项修复经同一 Reviewer 强制重跑 Web 11 条状态测试、typecheck、lint、diff-check 和 Android 两组测试后复核通过，最终无 blocker、high、medium 或 low finding。

### 真机证据

- 流式对话 `cycle2alpha`：发送后输入法自动收起，thinking 逐 token 生长、时间线跟随，最终折叠为“已思考 8s”；截图为 `/tmp/pixel7-cycle2-stream-early.png`、`/tmp/pixel7-cycle2-stream-mid.png`、`/tmp/pixel7-cycle2-stream-final.png`。
- 搜索用户正文、Agent 回答和附件名均得到正确结果数和跳转位置；键盘与输入焦点保持，前后结果可从 `1/2` 切到 `2/2`。截图为 `/tmp/pixel7-cycle2-search-user.png`、`/tmp/pixel7-cycle2-search-assistant.png`、`/tmp/pixel7-cycle2-search-prev.png`、`/tmp/pixel7-cycle2-search-attachment-keyboard.png`。
- Wi-Fi 关闭后消息进入缓存队列，离开底部再恢复 Wi-Fi：认证、`resume`、queued send、history 和 ACK 完成，助手只产生 1 条未读；首次点击定位未读锚点，回到真正底部后计数清零。截图为 `/tmp/pixel7-cycle2-offline-queued-away.png`、`/tmp/pixel7-cycle2-offline-unread-arrived.png`、`/tmp/pixel7-cycle2-unread-anchor.png`、`/tmp/pixel7-cycle2-unread-cleared.png`。
- 最新 APK 再次关闭/恢复 Wi-Fi后，logcat 显示 `resume epoch=26`，随后 `message.send`、流式事件和 ACK 正常完成；历史没有被普通 reconnect 当成破坏性重建清空。截图为 `/tmp/pixel7-cycle2-reconnect-offline.png`、`/tmp/pixel7-cycle2-ordinary-reconnect-response.png`。
- 会话抽屉在 `cycle2alpha` 与 `quote_user_source` 间切换并返回；搜索、未读和滚动上下文没有跨会话泄漏。截图为 `/tmp/pixel7-cycle2-drawer.png`、`/tmp/pixel7-cycle2-session-switch.png`、`/tmp/pixel7-cycle2-session-return.png`。
- 真实选择 `/sdcard/Download/cycle2-file-probe.md`，先显示上传进度再完成 attachment-only send；Agent 调用 `read_file` 并返回结果，附件名可被搜索。截图为 `/tmp/pixel7-cycle2-attachment-staged.png`、`/tmp/pixel7-cycle2-attachment-sent.png`、`/tmp/pixel7-cycle2-attachment-final.png`。
- 真机首次破坏性重建暴露同一连接重复 `attachment.download`，服务端以 4406 关闭；修复后重建只发送一次下载命令，不再出现 close/error/fatal，完整 fsync 但缺 reply 的末分片会先做 SHA-256 再原子发布。
- 最终 APK 再次执行“清理缓存并同步”：历史恢复时没有巨量未读，附件最终显示“已下载”，logcat 只有一个 `attachment.download`，没有 4406、`invalid attachment download state` 或 FATAL。截图为 `/tmp/pixel7-cycle2-final-resync-confirmed-early.png`、`/tmp/pixel7-cycle2-final-resync-done.png`、`/tmp/pixel7-cycle2-final-resync-attachment.png`。

## Cycle 3：工具调用详情与安全历史投影

状态：已实施，通过 Pixel 7 隔离环境实时调用、展开/收起和破坏性历史重建闭环。

### 缺口与任务

- 时间线只显示工具名称和一句描述，不能核对真实参数、结果摘要、失败内容或单次耗时。
- 实时 `react.tool.started` 原本携带完整参数，历史同步却只保留描述；同一工具在重连前后语义不一致。
- 直接把工具参数持久化到手机会把 token、cookie、authorization 等凭据留在 Room，且无界参数会放大 WebSocket、数据库和 WebView 压力。
- 长工具结果如果跟正文一样完整铺开，会把当前任务和最终回答推离视口。

### 交互结构

```text
○  思考内容
│
◇  shell                   完成 · 5.1s  ⌄
│  sleep 5秒然后echo
│  ├─ 参数
│  │  command    sleep 5 && echo LIVEDETAILOK
│  └─ 结果             ← 局部结果区，长内容内部滚动
│     { "exit_code": 0, "output": "LIVEDETAILOK" }
│
○  后续思考
```

工具详情仍属于 Agent 生长时间线，不另起详情页，也不把每个调用改成独立卡片。菱形继续表达工具节点，圆形继续表达思考节点；展开层只是附着在当前节点上的 state layer。

### Better UI

| Before | After |
|---|---|
| 工具行只能阅读名称和描述 | 整行形成至少 44dp 的 disclosure 触控目标；状态、耗时和 chevron 位于同一视觉行 |
| 参数和结果不可见 | 点击同一行在时间线内展开参数与结果；不跳页、不弹 modal、不打断上下文 |
| 长结果可能形成整屏文本 | 结果区最多约 11rem，高度之外内部滚动；工具标题、前后思考和最终回答仍可建立相对位置 |
| 运行与完成只靠节点猜测 | `运行中`、`完成 · 5.1s`、`失败`直接写出；展开/收起使用 180–200ms Material easing |

### Better Colors

| Before | After |
|---|---|
| 紫色只点亮节点，工具行缺少状态映射 | 活动菱形、扳手、`运行中`和 chevron 共用高亮紫；完成态回到中性 on-surface-variant |
| 详情容易被做成有边框的第二张卡 | 使用 surface-container-low 与一条低色度紫色附着线形成 state layer，不增加阴影和外围边框 |
| 错误可能继续沿用活动紫 | 失败节点、状态和错误文本只使用 Material error；蓝色仍只属于连接、搜索和普通主操作 |

### Better Typography

| Before | After |
|---|---|
| 名称、描述和状态层级不完整 | 工具名 14px 等宽、描述 13px 正文字体、状态 12px tabular figures；一眼先读任务再读元信息 |
| 复杂参数可能被压成一段 JSON | 顶层键使用 12px 正文字体，值使用 12px 等宽并保留换行；`description` 不在参数区重复显示 |
| 耗时只能从整轮“已思考 Ns”推断 | 实时工具完成后按 `ms / 0.1s` 精度显示单次耗时；历史没有可靠起止时间时明确省略，不伪造 |

### 数据与安全边界

- 服务端在 mobile 协议投影 owner 统一处理实时事件和历史 `tool_chain`，两条路径都发送相同的 `arguments`。
- 参数按键名递归隐藏 `secret`、`token`、`authorization`、`cookie`、`apiKey`、`privateKey`、凭据等字段；字符串值和 argv 列表中的常见 Authorization、Bearer、API key、password 等形式也在统一边界隐藏，显示值固定为 `[已隐藏]`。
- 投影限制为最多 5 层、256 个节点、每容器 64 项、单字符串 2000 字符，并按真实 UTF-8 JSON 字节把实时与历史的单次调用参数统一限制为 8 KiB；越界位置显式写入 `[已截断]`，不静默伪装完整数据。当前 Android 热快照仍按当前会话整体投影，8 KiB 上限避免已完成参数在后续流式帧中反复放大；更大详情应通过未来的按需协议获取，不进入热快照。
- 历史页以 240 KiB 为安全目标；只有页面逼近帧上限时，才从末尾依次移除完整参数和参数派生描述，保留消息正文、工具身份和既有结果摘要。该策略同时覆盖多字节 emoji，不用字符数冒充 WebSocket 字节数。
- Android 继续复用既有 `tool.v1:` 内容编码，不增加 Room 表或迁移；新增字段带默认值，旧记录仍可解码。
- 单工具耗时由服务端在 `react.tool.started` 使用 monotonic clock 记录，并随 `react.tool.completed` 发送；Android 只持久化服务端给出的 `duration_ms`，不把手机收帧间隔伪装成执行时间。服务端历史没有单工具时间戳，因此重建后只恢复参数和结果。

### Pixel 7 可复用验收标准

1. 真实触发 `list_dir`、`shell` 等工具：运行中节点使用亮紫和文字状态，完成后回到中性色并显示真实耗时。
2. 点击工具整行可展开/收起；参数名称、值、结果和错误属于同一时间线节点，44dp 目标不需要精确点击小 chevron。
3. 长结果不会把工具标题顶出屏幕，结果区可局部滚动；时间线主滚动仍能查看前后思考与最终回答。
4. 执行“清理缓存并同步”后，再展开同一历史工具仍能看到安全参数和结果；缺少单工具时间戳时不显示伪造耗时。
5. 服务端单测确认嵌套敏感字段、字符串内凭据和 argv 成对参数被隐藏，字符串/深度/项目数/UTF-8 字节数被裁剪；常见凭据形式不会进入手机参数投影。结果摘要沿用既有投影边界，不在本轮宣称通用 secret scanner。
6. 实时、应用重启和破坏性重建均无白屏、WebView render error、event sequence gap、4406 或协议反序列化错误。

### 自动化验证

- `.venv/bin/pytest -q tests/mobile_realtime/test_channel.py tests/mobile_realtime/test_gateway.py`：`40 passed`；覆盖实时工具事件、服务端 monotonic 耗时、最终参数、历史同步、非成功状态、常见敏感参数隐藏，以及真实 event encoder 的 UTF-8 帧预算。新增有界投影用例覆盖字符串、容器项数、深度和 40 个 emoji 工具调用的历史降载。
- `npm run typecheck && npm run lint`：通过；移动快照边界拒绝负数、非安全整数工具耗时和非对象参数。
- `ANDROID_HOME="$HOME/Android/Sdk" ./gradlew :app:testDebugUnitTest --tests com.akashic.mobile.data.local.StoredToolBlockTest --tests com.akashic.mobile.ui.web.MobileWebSnapshotTest`：通过；覆盖旧工具块兼容、安全参数/服务端耗时解码和 WebView 快照字段。
- 通过 release 签名注入构建 debug + androidTest APK；Pixel 7 上 `LocalDeliveryStoreTest` 为 `19/19`，再运行所有不依赖外部 pairing 参数的 Room migration、store 与 keystore instrumentation 为 `23/23`。另 2 条 `IsolatedGatewayDeviceTest` 需要显式 `pairingOfferBase64`、`historySessionId`，未把缺少外部参数的失败伪装成应用回归。
- 最终 `0.7.8 (17)` 重新执行 `clients/android/scripts/build-release.sh`：release unit、Lint、R8、assemble 和 APK v2 签名通过；66 个任务中 33 个真实执行。APK 为 8,281,638 bytes，SHA-256 `2d8b3f6b64eaa955a4034885153fb0070e0a5d8dab90949f2b3f4ad3bf05c45e`，不是复用修复前产物。
- kill-ai-slop 扫描 `frontend/chat/src` 为 36 个文件、9 组、52 个命中；新增两处命中均是工具名与参数值的语义等宽字体。本组没有新增渐变、玻璃拟态、发光点、悬浮详情卡或胶囊状态。

### 独立 Review

- 首轮发现并修复：非 ASCII 参数按字符而非 UTF-8 字节计预算、字段后缀和命令字符串可泄露凭据、完成事件没有覆盖最终参数、耗时取客户端收帧间隔、`blocked/denied` 被当成成功、忽略目录下 instrumentation fixture 仍使用旧协议、关闭详情时每个 token 都序列化参数、桌面长结果无局部滚动、无详情工具行仍暴露禁用 button 语义。
- 复核继续用真实反例发现：list/tuple 内 Authorization 和成对 `--api-key value` 仍可能泄露；40 个带 2000 emoji 描述的历史工具在删完参数后仍超帧；桌面旧历史缺少 `status` 时被错误标红。当前已统一字符串与 argv 脱敏、按参数后描述的顺序回收历史预算，并仅把明确的非成功状态映射为失败。
- 最终性能复核发现当前会话会在每个流式快照中重新序列化已完成参数；将实时参数从 48 KiB 收敛为与历史一致的 8 KiB，保留人类核对所需信息，同时避免多个工具调用把热快照推向 MiB 级。按需完整详情属于后续协议，不在首版引入第二套存储和拉取抽象。

### 真机证据

- Pixel 7 使用隔离 `docker/mobile-lab` 与签名 release APK；真实 `list_dir` 调用的收起态与历史展开态为 `/tmp/pixel7-cycle3-fixed-trace-open.png`、`/tmp/pixel7-cycle3-fixed-tool-expanded-history.png`。
- 真实 `shell` 执行 `sleep 3 && echo CYCLE3FINAL`：服务端完成事件给出 `duration_ms=3000+`，完成行显示 `完成 · 3s`，展开后显示 command 和结构化结果摘要；截图为 `/tmp/pixel7-cycle3-server-duration-trace.png`、`/tmp/pixel7-cycle3-server-duration-expanded.png`。
- 真实 `shell` 执行 8 秒期间，菱形、扳手、`运行中`和 chevron 同时点亮亮紫；截图为 `/tmp/pixel7-cycle3-tool-running-row.png`。
- 真机确认“清理缓存并同步”后，客户端重新发送 `session.list` 和 `history.get`；无 event gap、4406、协议反序列化或 FATAL。历史工具仍能展开 command 与 result，但按契约不伪造单工具耗时；截图为 `/tmp/pixel7-cycle3-final-resync-done.png`、`/tmp/pixel7-cycle3-final-resync-expanded.png`。
- 最终签名 `0.7.8` 无损覆盖安装到 Pixel 7，系统报告 versionCode 17 / versionName 0.7.8；手机内 `base.apk` 与发布产物 SHA-256 完全一致。最终工具展开截图为 `/tmp/pixel7-v0.7.8-tool-details-open.png`，logcat 无 FATAL、RenderProcessGone、event gap、4406 或协议校验错误。

### 已知边界

- 失败消息重试不能只在 UI 重发：当前 terminal `message.send.error` 会删除 outbox，附件也不再保留为草稿；在服务端提供 retryable/duplicate-safe 语义前不新增可能重复执行 Agent turn 的按钮。
- 历史工具链没有每次调用的起止时间，破坏性重建后省略单工具耗时；如果未来服务端正式持久化 call duration，再直接扩展同一投影字段，不从结果文本猜测。

## Cycle 4–6：可靠性、媒体与 Agent-native 扩展

状态：三批均已合入 `feature/im-phone`，在独立 Docker Mobile Lab 和 Pixel 7 上完成交叉回归；详细契约与逐项证据见：

- `docs/mobile-batches/2026-07-16-reliability.md`
- `docs/mobile-batches/2026-07-16-media.md`
- `docs/mobile-batches/2026-07-16-agent-native.md`

本轮形成的完整日用链路是：会话预览/时间/未读与阅读锚点、离线多消息 exactly-once 队列、通知精确消息导航和快捷回复、多附件同消息发送、后台大文件进度、图片整屏查看、后台 Agent 运行状态、显式确认通知，以及运行中插件目录与 KV Cache 试点看板。三批没有各造一套状态：会话活跃态来自真实 turn，附件继续由 Room transfer owner 持有，确认语义只由 `request_user_confirmation` 产生，插件数据只通过 `plugin.ui.call` 读取 canonical 插件 reader。

最终交叉门禁同时运行 Agent-native gate 与 media gate；Pixel 7 再验证插件返回栈和图片 history 共存。真机发现并修复两项自动构建未暴露的问题：缺少 `ACCESS_NETWORK_STATE` 导致 instrumentation 启动失败，以及 cursor 测试的 `Int`/`Long` 断言类型不一致。修复后对应真机测试均通过，应用和隔离网关日志无 FATAL、WebView render error、event gap 或协议反序列化错误。

### Cycle 6 所有权修正：Observe 自有移动看板

隔离环境暴露出一个真实架构错误：只安装 `status_commands` 时仍会注册 KV Cache 空看板，而 token 数据实际由 `observe` 写入。现已把导航、看板、样式、RPC 和 Turn 尾部输出 token 全部迁入 `observe`；核心仅保留插件资产注册、上下文 RPC 和通用消息 ID 关联。未启用 `observe` 时，插件目录不再出现 KV Cache，回答尾部也不会保留空插槽或伪造统计。

Pixel 7 真实回合验证得到 `model_output_tokens=51`，同一消息尾部显示“输出 51 tokens”；插件目录只有一个 Observe 提供的 KV Cache 入口，看板同步显示真实 92.6% 命中率。首次查库早于异步 writer 的竞态使用有限退避重试闭环，partial/unavailable usage 不显示为完整统计。

反向验收临时禁用 Observe 后，Pixel 7 的插件数量归零，KV Cache 入口和回答尾部 token 均消失；恢复插件后真实看板重新出现。最终 Agent-native、可靠性和媒体三条门禁全部通过，说明该能力确实由插件按启用状态动态提供。

## 2026-07-17 Material 3 发布收口

- 使用 Material 3、Better UI、Better Colors、Better Typography 和 Kill AI Slop 对新增移动界面做最终审阅，只调整本批新增表面，没有把既有聊天界面重写成另一套设计系统。
- 插件目录移除重复标题，保留“运行中 · N”状态和一个平面列表；插件自身提供 40dp 标识，颜色只表达当前可进入的运行能力。
- Observe 看板把“近期被动复用”“被动总览”“主动链路”组织成同一指标组；没有主动数据时使用中性 surface 并写明“暂无记录”，不再用紫色暗示并不存在的数据。
- Turn 尾注收敛为“输出 N tokens”，只在 Observe 提供真实 `model_output_tokens` 时出现；核心不创建空插槽，也不拥有 KV Cache 样式。
- 错误提示改为 Material snackbar：聊天页固定在输入栏之上，插件页固定在底部；连接错误可关闭，插件加载错误可重试，错误详情保持单行且不遮挡主要任务。
- Pixel 7 已验证 `OBSERVEOUTPUT` 历史 Turn 显示“输出 51 tokens”，插件目录与 KV 看板返回栈正常。截图为 `/tmp/pixel7-chat-token-tail.png`、`/tmp/pixel7-md3-plugin-directory-v3.png`、`/tmp/pixel7-md3-cache-v3.png`。
- 自动验证通过 `npm run typecheck`、`npm run lint`、Observe 的 2 项移动面板测试与 6 项插件测试、`./scripts/verify-mobile-agent-native.sh`、可靠性门禁和媒体门禁。
- `0.7.9 (18)` 已发布到 GitHub。签名 APK 为 8,306,626 bytes，SHA-256 `be39b2bcb62f4d5f25b8df3869d45953249c0b49dc75e4d71f04e7d055f070d6`；GitHub 资产摘要、本地产物和 Pixel 7 内 `base.apk` 三者一致。
- Pixel 7 无损安装签名 release 后系统报告 versionCode 18 / versionName 0.7.9，首次权限页和配对页均正常渲染，没有白屏或应用进程内的 FATAL、RenderProcessGone、event gap、协议校验错误。发布地址为 `https://github.com/kachofugetsu09/akashic-mobile-releases/releases/tag/v0.7.9`。

## 2026-07-17 插件 UI 热更新

正式环境首次安装新版 Observe 后，服务端 generation 已热加载，但已经完成同步的 Android 仍保留启动时取得的空插件目录。根因不是插件加载失败，而是客户端只在 `sync.completed` 或重建后请求 `plugin.ui.list`，插件快照提交没有对应的移动事件。

本轮补齐完整链路：

```text
插件 snapshot committed
          │ 移动 UI 目录摘要确实变化
          ▼
  plugin.ui.changed（connection control）
          ▼
Android 重新请求 list → asset → WebView 按 sha256 原位替换
```

- watcher 在每次 reconcile 尝试后调用目录刷新器，因为同一批次中较早插件可能已提交、较晚插件才失败；通知失败只重试通知，不会再次 reconcile 并重载全部插件。
- mobile channel 比较插件 ID、source revision 与资产 SHA-256，只有目录真实变化才向当前连接中明确声明支持热更新的设备并发发送有超时上限的非持久化控制帧；断线重连由既有首次目录同步取得最新内容。新版 Android 仍能消费并 ACK `0.7.10` 可能留下的 legacy durable event，升级不会形成重连循环。
- Android 若在旧目录或资产批次尚未结束时收到通知，会排队一次刷新，当前批次完整收束后再拉新目录，不清空进行中的请求，也不会产生未知 reply。
- 目录列出插件后、资产拉取前插件被移除时，`plugin_unavailable` 被视为目录已过期；客户端等待同批其余 reply 收束后只重拉一次目录，不显示伪错误也不触发断线重连。
- `0.7.10 (19)` 已发布，APK 为 8,306,622 bytes，SHA-256 `a02c4da4333ae9e135cf874609afb50c2a28611c550757d3eeab709b041397bd`。后续验收发现该版本尚未完成客户端 capability 订阅；本轮补上协商、旧服务端 fallback 与连接 epoch 隔离后，才把热更新视为可用能力。
- 线上核心已重启一次以加载新协议，Observe `5c7442f`、writer 与 mobile channel 均正常启动。完成 capability 修复后，支持热更新的客户端不再需要重启 runtime 或手机；旧 Android 客户端继续使用首次同步，不会收到 `plugin.ui.changed`。
- Pixel 7 使用隔离 Mobile Lab 验证最终契约：Observe 禁用时抽屉显示“插件 0”，运行中直接启用后原位变为“插件 1”，没有重启手机、Android 服务或 runtime。
- 同一 Pixel 7 连接在切换前后的 durable cursor 均为 `next_event_seq=10 / sent=9 / acknowledged=9`，`mobile_device_inbox` 保持为空；这证明 `plugin.ui.changed` 没有进入持久化序列，也没有产生 4406 或 event sequence gap。
- 隔离 tunnel 曾因本机透明代理路由中断返回 Cloudflare 1033/HTTP 530；重启 tunnel connector 后恢复。该故障发生在配对 WebSocket 建立前，与插件热更新协议无关，验收只在 tunnel 恢复并完成真实 WSS 配对后计入。

## 2026-07-17 大附件按需下载与抖动保护

- 隔离 Mobile Lab 的历史同步暴露一条真实 47,381,751-byte 文档：Android 会在建立历史投影时立即把所有服务端附件置为 `pending`，随后按分片完整下载。连续的 `attachment.download` 是正常断点传输，不是重试死循环，但它会无意义占用移动网络和消息同步时延。
- 新发现的小于 10 MiB 附件仍自动进入下载队列；达到 10 MiB 的附件保存为 `remote`，消息附件行原位显示文件大小与“尚未下载”，只有用户点击同一行右侧“下载”后才进入既有 `pending → downloading → cached` 断点链路。
- UI 没有增加外层卡片或新颜色：附件仍是原有 Material 3 行，主信息保持文件名，低强调文字承担大小/状态语义，唯一主色文字按钮表达显式下载动作；44dp 触控区域和原有进度条保持不变。
- connection-scoped 控制帧的 3 秒超时现只覆盖真正的 WebSocket 写入，不再覆盖等待同连接合法附件帧占用写锁的时间。正常在途帧可在 30 秒窗口内完成；超过窗口说明连接已无法释放发送权，或拿到锁后仍写超时，服务端才移除连接并以 4408 关闭。
- “下载 / 重试”原生入口允许 `pending`、`downloading` 和已经完成的 `cached` 重入，快速双击只复用同一下载队列，不会在后台协程抛异常；多附件的读屏名称包含文件名。
- 验证通过：gateway `23 passed`、Pyright 无错误、Web typecheck 与 ESLint、Android release unit/Lint/R8/assemble 和 v2 签名；Pixel 7 定向 instrumentation 为 `1/1`，其中边界用例确认 `10 MiB - 1 byte` 自动排队、`10 MiB` 保持 remote，缓存对账不改变 remote，连续点按只复用同一下载队列。
- Pixel 7 隔离 Mobile Lab 收到真实 11.0 MiB 历史附件后原位显示“尚未下载”和“下载”，历史同步阶段未主动发下载命令；连续点按后只运行一条分片链，界面原位显示 32%→69%，最终经 SHA-256 校验变为“已下载”并出现分享操作。截图为 `/tmp/pixel7-release-reconnected.png`、`/tmp/pixel7-large-download-progress2.png`、`/tmp/pixel7-large-download-finished2.png`、`/tmp/pixel7-large-download-complete.png`；日志无 FATAL、AndroidRuntime 或 RenderProcessGone。
- 签名 `0.7.11 (20)` 已发布到私有 GitHub Releases，APK 为 8,306,622 bytes，SHA-256 `1dbe8ad21b9a171a53b0a44ff673773108422dca2f951743a8090d4621f76b91`；远端资产摘要、本地产物和 Pixel 7 验收包一致。发布地址为 `https://github.com/kachofugetsu09/akashic-mobile-releases/releases/tag/v0.7.11`。

## 2026-07-17 工具详情结构化复制

- 工具详情的“参数 / 结果 / 错误”标题行附着一个 Material 3 文字动作，不增加详情卡、弹窗或浮动菜单；参数复制为保留真实类型的格式化 JSON，继续排除只用于摘要的顶层 `description`，结果复制与屏幕显示使用同一字符串。
- 复制复用既有 Android `AkashicNative.copyText` 桥，不增加协议、Room 字段、依赖或第二套剪贴板实现；shared WebChat 的 `ChatMessageView` 只增加可选回调，没有承担 Android 所有权。
- 动作使用既有 primary token 和 state layer，触控高度为 48dp；成功在原位切换为“已复制”和 check 图标，1.6 秒后恢复。流式结果变化后，只有当前文本仍等于剪贴板版本才保持成功状态，避免把旧结果标为已复制。
- 折叠详情增加 `inert`，复制按钮不会留在键盘焦点树或无障碍树；焦点轮廓、`prefers-reduced-motion` 和可访问名称保持完整。
- `npm run typecheck`、`npm run lint`、15 项 mobile web state 测试、`npm run build:mobile-web`、`git diff --check` 和 `clients/android/scripts/build-release.sh` 全部通过；release unit、Lint、R8、assemble 与 v2 签名校验通过。
- Kill AI Slop 前后均为 38 个文件、9 组、57 个机械命中，差异只有新增 CSS 造成的既有命中行号移动；没有新增渐变、玻璃拟态、发光点、状态卡或胶囊堆叠。
- Pixel 7 在隔离 Mobile Lab 真实发送 `use shell tool to run printf toolcopyresult and report`，服务端记录新的 `message.send`；真机时间线按顺序流式出现 thinking、`shell 完成 · 9ms` 和最终 `toolcopyresult`。证据为 `/tmp/pixel7-send-enter2.png`、`/tmp/pixel7-tool-copy-final-collapsed.png`。
- 最终 review 修复了三个真实问题：折叠交互元素仍可聚焦、流式结果更新后成功状态可能过期、44dp 未达到本项目 Material 3 48dp 基线。修复后的签名 APK 已无损安装到 Pixel 7，应用 PID 日志没有 FATAL、RenderProcessGone、event gap 或协议校验错误。
- release WebView 未开放调试，ADB 的 TAB 焦点会越出 WebView 并误启动系统应用，因此本轮不把自动化焦点副作用伪装成复制点击验收；剪贴板桥本身已由消息复制闭环验证，本组仍保留一次人工点击“参数 / 结果 → 粘贴比对”作为下个真机交互驱动的首个验收样例。完整批次记录见 `docs/mobile-batches/2026-07-17-tool-detail-actions.md`。

## 2026-07-17 消息选择与批量复制

- 长按任意已完成的用户或 Agent 消息进入排他的消息选择模式；后续点击同一消息取消，点击其他消息追加。流式临时消息不参与选择，快照删除、消息重新进入流式态或切换会话时会按稳定消息 ID 对账并清理选择。
- 选择态用 Material 3 contextual top app bar 替换常态顶栏，显示“已选择 N 条”；单选且属于当前发送会话时提供引用，多选只保留复制。Android 返回键先退出选择，不离开应用。
- 批量复制按真实对话顺序输出“角色 · 日期时间 + 正文 + 附件名”，单条复制保持原文，不强加 transcript 标签；复用既有 Android 剪贴板与触觉桥，不增加协议、Room 字段或第二套复制实现。
- merged mobile 历史虽然全量可读，但 `RealtimeSession` 明确只允许引用当前发送会话内的消息。界面能力判断现与该既有 owner 对齐，不再为旧 mobile 会话展示一个发送层必然拒绝的引用动作；没有放宽服务端校验。
- 选择态把整条消息作为 checkbox 语义平面，消息内部复制、引用、展开和下载动作进入 `inert`，不会继续抢焦点或读屏；长按使用 touch-first 事件，鼠标/触控笔路径捕获 pointer，移动超过 9px 立即取消。
- 视觉上没有新增消息卡片或弹窗：选中态是覆盖消息语义区的 11% primary state layer，顶部使用 `surface-container-high`；主蓝表达选择和动作，紫色继续只表达 Agent thinking/tool 过程。
- `npm run typecheck`、`npm run lint`、20 项 mobile web state 测试、`git diff --check` 和 `clients/android/scripts/build-release.sh` 通过；release unit、Lint、R8、assemble 与 v2 签名验证成功，最终验收 APK SHA-256 为 `885366a6…cf22`。
- Pixel 7 已用最终签名 APK 无损安装并验证：单选截图 `/tmp/pixel7-selection-final-one.png`；单选引用进入 composer 的截图 `/tmp/pixel7-selection-final-reply3.png`；双选截图 `/tmp/pixel7-selection-final-two3.png`；Android 剪贴板预览显示按顺序复制的两条消息，截图 `/tmp/pixel7-selection-final-copy.png`；返回键恢复常态输入区，截图 `/tmp/pixel7-selection-final-back2.png`。对应 logcat 无 FATAL、RenderProcessGone 或 event sequence gap。
- Kill AI Slop 扫描为 38 个文件、10 组、58 个机械命中；相对本组实施前只多出 `-webkit-touch-callout: none` 被“left-border callout”规则按字符串误报，实际没有新增 callout、渐变、玻璃拟态、发光点、卡片墙或胶囊堆叠。完整批次记录见 `docs/mobile-batches/2026-07-17-message-selection.md`。

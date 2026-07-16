# Akashic Mobile Agent IM 实施日志

本日志按语义完整的功能组记录设计、验收、实现、真实证据和提交。长期约束见 [mobile-agent-im-original-request.md](mobile-agent-im-original-request.md)。

## 总体状态

| 领域 | 当前基线 | 下一闭环 |
|---|---|---|
| 消息基本功 | 已有真实时间、日期分组、双向引用、复制、搜索、跳转和未读；缺失败重试 | 失败重试与乱序合并 |
| 实时 Agent | 有流式回答、思考/工具时间线和停止 | 工具参数/结果/错误/耗时详情 |
| 媒体 | 有上传、进度、缓存、预览、下载和分享基础链路 | GIF/meme、重试与大文件体验 |
| 网络 | 有认证、resume、durable inbox、连接状态 | 抖动场景矩阵和用户可恢复动作 |
| 会话 | 有 mobile 全量同步、抽屉、切换、新建和当前会话搜索 | 失效解释与会话内阅读位置 |
| 扩展 | 已有受控 `plugin.ui.*` 和渲染插槽 | KVCache 移动 Dashboard 试点 |
| 质量 | 有 Android 测试和隔离 Gateway | 每组固定 Pixel 7 真机闭环 |

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

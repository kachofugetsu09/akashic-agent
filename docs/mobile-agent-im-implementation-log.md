# Akashic Mobile Agent IM 实施日志

本日志按语义完整的功能组记录设计、验收、实现、真实证据和提交。长期约束见 [mobile-agent-im-original-request.md](mobile-agent-im-original-request.md)。

## 总体状态

| 领域 | 当前基线 | 下一闭环 |
|---|---|---|
| 消息基本功 | 已有真实时间、日期分组、双向引用和独立复制；缺搜索、跳转、未读与失败重试 | 搜索 + 跳转 + 未读锚点 |
| 实时 Agent | 有流式回答、思考/工具时间线和停止 | 工具参数/结果/错误/耗时详情 |
| 媒体 | 有上传、进度、缓存、预览、下载和分享基础链路 | GIF/meme、重试与大文件体验 |
| 网络 | 有认证、resume、durable inbox、连接状态 | 抖动场景矩阵和用户可恢复动作 |
| 会话 | 有 mobile 全量同步、抽屉、切换和新建 | 搜索、未读锚点、失效解释 |
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

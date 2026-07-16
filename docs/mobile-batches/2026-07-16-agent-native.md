# Agent-native 移动端批次：后台任务与插件看板

## 范围

本批只实现 C3、C7、C8、C9、C10。C4 不在范围内：会话执行中的停止按钮已经由输入区承担，再增加一个入口会重复表达同一动作。

| 能力 | 本批结果 | 语义边界 |
| --- | --- | --- |
| C3 后台任务状态 | 顶栏显示运行任务数；抽屉在对应会话旁显示紫色运行点 | 复用 `TurnStopCoordinator` 的真实活跃 turn，不新增第二套任务状态 |
| C7 完成与确认通知 | 完成通知与等待确认通知使用不同标题和动作；点击回到原会话 | runtime 的 `request_user_confirmation` 工具显式产生确认态，不从问号或文案猜测 |
| C8 插件入口 | 抽屉提供一个紧凑“插件”入口；只列出当前运行且声明移动看板的插件 | 插件目录和完整看板是两级页面，看板左上角始终可返回 |
| C9 KV Cache 试点 | `observe` 插件提供真实 KV Cache 总览、被动/主动命中率、turn 明细与输出 token | 数据生产、查询和界面由同一插件拥有；未启用 `observe` 时不注册入口 |
| C10 门禁 | 提供一条自动验证命令和两轮 Pixel 7 验收步骤 | 自动测试不代替通知落点、触觉和窄屏布局的真机观察 |

## 信息与调用路径

```text
┌─ Mobile realtime state ─────────────────────────────────────┐
│ active turn set ──→ snapshot ──→ 顶栏计数 / 抽屉会话状态   │
│ message.final ────→ attention ─→ Android 系统通知          │
└─────────────────────────────────────────────────────────────┘

┌─ Plugin mobile UI ──────────────────────────────────────────┐
│ 抽屉“插件” → 运行中看板目录 → 全屏看板                     │
│                                  │                          │
│                                  └→ plugin.ui.call          │
│                                      └→ KV Cache reader     │
└─────────────────────────────────────────────────────────────┘
```

## UI 决策

### Better UI

| 之前 | 之后 |
| --- | --- |
| 插件能力容易膨胀成抽屉里的入口墙 | 抽屉只保留一个紧凑目的地，插件选择放进独立目录 |
| 活跃任务仅在当前会话的停止按钮可见 | 顶栏显示总数，抽屉把状态附着在真实会话上 |
| 工具看板容易被做成等权卡片墙 | 总览使用一个统一指标组，turn 明细保持平面列表，点击后原位展开 |

所有触控目标至少 44dp；插件目录使用 state layer 表达按下态，不靠阴影或粗边框制造层级。

### Better Colors

颜色承担数据域，而不是装饰：钴蓝表示当前窗口，青色表示被动链路，亮紫表示主动链路。红色只在真实错误或命中率低于 50% 时出现。其余结构依靠中性色、留白和排版完成。

### Better Typography

总览数字使用 tabular numerals；标题、指标、辅助信息形成明确三级字号。会话和 turn 的长文本都在单行入口处截断，展开后才显示完整参数，避免窄屏被不可控内容撑开。

### Kill AI Slop 约束

本批没有新增气氛渐变、发光状态点、彩色左边框卡片、图标彩色底座或装饰胶囊。KV Cache 命中率圆环使用 `conic-gradient`，它直接编码真实比例，不属于气氛渐变。

## 协议与数据契约

移动快照协议从 v2 升至 v3，会话增加必需字段 `isRunning`。这是协议边界的结构变化，旧快照不会被静默当作新结构使用。

等待确认通知由 runtime 工具显式声明。模型只有在任务必须等待用户授权、选择或确认时调用 `request_user_confirmation`；Reasoner 持有该状态并写入最终事件：

```json
{
  "metadata": {
    "mobile_attention": "confirmation"
  }
}
```

缺少该字段表示普通完成；未知枚举或非对象 `metadata` 会在持久化事件和推进 cursor 前明确失败。入站消息中的同名 metadata 会被移除，不能由客户端伪造确认通知。当前实现不根据问号、按钮文案或模型措辞猜测“等待确认”。

KV Cache 看板只展示 `observe` 数据库的真实字段：跟踪 turn 数、prompt token、hit、miss、hit rate、source、session、用户预览和时间。助手回答尾部只在存在真实 `model_usage.output_tokens` 时显示“输出 N tokens”。移动端核心只注册插件资产并转发上下文 RPC；看板、样式、查询和 Turn 插入组件都由 `observe` 自己提供。

## 自动门禁

核心仓库执行：

```bash
./scripts/verify-mobile-agent-native.sh
```

门禁依次运行 TypeScript 类型检查、ESLint、Web surface history 测试、移动 Web 构建、插件 UI 通道测试，以及带全局锁、单 worker 的 Android 单元测试和 instrumentation APK 编译。

`observe` 插件仓库执行：

```bash
AKASHIC_AGENT_ROOT=/mnt/data/coding/akasic-agent \
  /mnt/data/coding/akasic-agent/.venv/bin/pytest -q
node --check mobile_panel.js
```

## Pixel 7 分轮验收

### 第一轮：C3 / C7

1. 在会话 A 发起持续时间足够切换页面的真实任务。
2. 切换到会话 B，确认顶栏仍显示“运行 1”，抽屉中只有会话 A 显示“Agent 正在运行”。
3. 熄屏等待任务完成，确认系统通知标题是“Akashic 已完成”。
4. 点击通知，确认应用打开并落到会话 A，而不是当前会话 B。
5. 发起一个必须明确授权才能继续的请求，确认 Agent 调用 `request_user_confirmation`，系统通知显示“Akashic 等待确认”。

### 第二轮：C8 / C9

1. 安装包含状态插件移动资源的隔离运行环境，进入抽屉。
2. 确认抽屉只有一个紧凑“插件”入口，且数量等于实际加载的移动看板数。
3. 进入插件目录，确认只列运行中且支持移动看板的插件。
4. 打开 KV Cache，看板左上角返回箭头必须始终可见。
5. 对照桌面 Dashboard，核对总 turn、hit、miss、被动/主动命中率和一条 turn 明细。
6. 点击 turn 行，确认参数在原位置展开；再次点击收起，列表位置不应跳变。
7. 在看板按 Android 系统返回：第一次回到插件目录，第二次回到聊天；Activity 不应退出，聊天输入和会话状态保持不变。

每轮先修复真实失败并重新跑自动门禁，再进入下一轮。正式 workspace 与 Pixel 7 的安装、ADB 操作由主联调流程统一执行。

## 2026-07-17 集成与真机结果

- `./scripts/verify-mobile-agent-native.sh` 完整通过：Web 类型检查、ESLint、15 项移动状态测试、生产构建、8 项服务端/协议定向测试，以及 Android JVM 与 androidTest APK 构建均成功。
- `status_commands` canonical 仓库的 `main` 已包含 `8dba7f1`、`cf74658`，6 项插件测试、生产代码 Pyright、JS 语法和 compileall 通过；随后只安装到 Docker Mobile Lab 的 `/sandbox/home/.akashic-plugin`，没有写正式插件缓存。
- Docker Mobile Lab 使用独立 `/sandbox/workspace` 和插件目录。Agent 重建后旧 chat-proxy/tunnel 仍占用旧 network namespace，导致 healthcheck 失败；把三个服务作为一组重建后恢复 healthy，正式实例未被操作。
- Pixel 7 无损覆盖安装 `0.7.8-debug (17)`。抽屉显示“插件 1”，目录只列出运行中的 `KV Cache`，真实 `plugin.ui.call` 返回 0 轮空数据；看板没有使用假指标。
- Android 系统返回实测为 `KV Cache 看板 → 插件目录 → 聊天 → Launcher`；聊天状态保留，logcat 没有 FATAL 或 WebView error。截图为 `/tmp/pixel7-kvcache-dashboard.png`、`/tmp/pixel7-back-plugin-directory.png`、`/tmp/pixel7-back-chat.png`。
- 真实长 turn 在顶栏显示亮紫“运行 1”和停止动作；Agent 确实调用 `request_user_confirmation`，后台通知显示“Akashic 等待确认”及“查看并确认”，点击后落到对应最终消息。截图为 `/tmp/pixel7-confirm-actually-sent.png`、`/tmp/pixel7-confirmation-background-real.png`、`/tmp/pixel7-confirmation-deeplink.png`。
- 真机 `LocalDeliveryStoreCursorTest` 首轮暴露断言把 Room 的 `Long(4)` 与 `Int(4)` 比较；改为 `4L` 后重编 androidTest APK，Pixel 7 复跑 `1/1` 通过。该测试确认无效 confirmation metadata 不会推进 cursor。

## 2026-07-17 插件所有权修正

- 首轮隔离验收出现“KV Cache 入口存在但没有数据”：Docker Mobile Lab 只安装了 `status_commands`，没有安装真正写入 `observe/observe.db` 的 `observe`。空看板证明此前把读取界面挂在命令插件上的所有权不成立。
- `status_commands` 已移除 Dashboard、移动模块和全部视觉资产，只保留 `/memorystatus`、`/kvcache` 命令；`observe` 现在拥有移动导航、KV Cache 看板、数据 RPC 和 `turn.after_answer` 组件。
- 核心只在通用 `TurnCommitted` 事件里暴露已持久化的助手消息 ID，使插件能把历史消息稳定关联到真实 Turn；没有加入任何 KV Cache 专用协议或样式。
- Turn 尾部不展示输入量、命中率或会话累计，只显示真实聚合的本轮模型输出 token；缺少 provider usage 时不伪造估算值，也不渲染占位。
- `status_commands` 所有权修正已发布为 `e424b3e`；`observe` 的界面迁移与竞态修复已发布为 `b6fb879`、`c3d952c`。隔离插件缓存通过 `plugin-install` 重装，没有修改正式插件缓存。
- Pixel 7 真实发送一轮对话后，`observe.db` 记录 `assistant_message_id=mobile:...:47`、`model_output_tokens=51`，手机同一回答尾部显示“输出 51 tokens”；数据库与 UI 一致。插件目录只列 1 个由 `observe` 注册的 KV Cache 看板，看板显示同一轮 92.6% 命中率和 29,696 / 32,071 token。截图为 `/tmp/pixel7-output-token-result.png`、`/tmp/pixel7-observe-plugin-directory.png`、`/tmp/pixel7-observe-dashboard-real.png`。
- 独立复核发现并修复四项发布前问题：KV 专用颜色仍在核心、异步 writer 首次查询竞态、partial usage 被当成完整统计、空 `turn_id` 唯一索引丢遥测。修复后核心已无 KV 专用 token；插件使用 scoped OKLCH token；RPC 按 0/100/300/700ms 有限重试；只展示 `coverage=exact`；旧唯一索引在迁移时删除。
- 反向验收临时禁用隔离环境的 `observe` 后，服务端日志明确记录插件被 manifest 禁用；Pixel 7 抽屉显示“插件 0”，KV Cache 入口和回答尾部 token 同时消失。恢复插件并重启隔离服务后，入口与真实数据恢复，证明界面没有残留在核心或 `status_commands`。
- 最终自动门禁通过 `./scripts/verify-mobile-agent-native.sh`、`clients/android/scripts/verify-reliability-gate.sh` 和 `clients/android/scripts/media-gate.sh`；Pixel 7 最新调试包未出现 FATAL、WebView render error、event gap 或 `plugin.ui` 错误。

## Material 3 收口与 0.7.9 验收

```text
插件目录                  Observe 看板                 回答尾部
┌─────────────────┐      ┌─────────────────────┐     ┌──────────────┐
│ 运行中 · 1       │      │ 近期被动复用  92.6% │     │ 最终回答正文  │
│ [KV] KV Cache  › │  →   │ 被动总览      1 轮  │  →  │ 输出 51 tokens│
└─────────────────┘      │ 主动链路      暂无记录│     └──────────────┘
                         └─────────────────────┘
```

- 插件目录是运行能力选择器，不再重复显示页面标题，也不使用等权卡片墙。
- 看板指标属于一个语义组：蓝色表示当前缓存窗口，青色表示被动链路，紫色只在真实主动数据存在时出现；空主动链路回到中性 surface。
- 形状按组件职责分级：snackbar 4dp、列表标识 12dp、指标组 20dp、触控胶囊使用 full shape；层级主要依靠 state layer、排版和留白，不增加装饰阴影。
- 新增界面统一使用 500/700 的可用字重、12px 以上辅助文字和 tabular numerals；没有为了视觉统一批量改写既有聊天排版。
- Pixel 7 在隔离 Mobile Lab 中验证插件目录、看板返回栈、真实空态与 `OBSERVEOUTPUT` 的 51-token 尾注。最终发布门禁包含 Web 类型检查、ESLint、15 项状态测试、Android 单元与 androidTest 构建、可靠性门禁及媒体门禁。
- 签名 `0.7.9 (18)` 已发布。APK 为 8,306,626 bytes，SHA-256 `be39b2bcb62f4d5f25b8df3869d45953249c0b49dc75e4d71f04e7d055f070d6`；GitHub 资产、本地产物和 Pixel 7 内安装包哈希一致。首次权限与配对界面启动正常，应用进程日志没有 FATAL、WebView render loss、event gap 或协议反序列化错误。

## 0.7.11 插件 UI 热更新契约

- 插件 watcher 在每次 reconcile 尝试后读取当前移动 UI 目录；即使批次后段失败，前段已提交的插件变化仍会被发现。只有 `(plugin_id, source_revision, asset_sha256)` 集合变化时才发送 `plugin.ui.changed`。
- 该通知只作为 connection-scoped control 发给通过 `plugin.ui.list {hot_updates:true}` 明确订阅的当前连接，不写 durable inbox。离线设备重连后先走既有目录同步，因此无需补发断线期间的变化；通知本身不携带模块正文，Android 仍通过 `plugin.ui.list` 和 `plugin.ui.asset` 边界取回并校验内容。
- Android 同一时刻只拥有一个目录/资产批次。热更新与旧批次重叠时只记录一次 queued refresh，旧批次完成后立刻请求最新目录，避免清空 pending map 后收到旧 reply 导致协议失败。
- 目录与资产请求之间插件被移除时，`plugin_unavailable` 只让当前批次失效并排队重拉目录，不断开 WebSocket；watcher 的通知重试也与插件 reconcile 分离，不会因移动通知失败每秒重载全部插件。
- WebView 已有的资产签名队列负责原位切换 JS/CSS；新资产全部解析成功后才替换旧 definitions 和 stylesheet，加载失败仍保留旧界面并显示可重试错误。
- `0.7.10 (19)` 只包含初版事件处理，尚未完成 capability 订阅；后续修复版才启用可靠热更新。新客户端连接旧服务端时会在首个目录请求被拒后自动退回旧 payload。服务端部署本功能时需要一次核心重启；此后单独安装、升级、禁用或移除带移动 UI 的插件均不要求 runtime 或手机重启。

### Pixel 7 热更新验收

```text
Observe disabled       connection control       Observe enabled
┌──────────────┐      plugin.ui.changed         ┌──────────────┐
│ 插件       0 │ ─────────────────────────────→ │ 插件       1 │
└──────────────┘     list → asset → replace     └──────────────┘
          durable cursor 与 inbox 全程不变
```

- 隔离 Mobile Lab 的 Observe 初始处于禁用状态；Pixel 7 完成真实 WSS 配对后，抽屉显示“插件 0”。
- 运行中执行 `plugin-enable observe@mobile-lab`，不重启 runtime、Android 服务或 Activity；6 秒内同一抽屉原位显示“插件 1”。
- 设备 `3cdf264b0b144af3844fb94aeb8b3818` 在切换前后均为 `next_event_seq=10 / sent_event_seq=9 / acknowledged_event_seq=9`，持久化 inbox 为空，因此该能力没有占用 durable event 序列。
- 截图证据为 `/tmp/hot-control-disabled.png`、`/tmp/hot-control-enabled.png`；Python 定向测试 `180 passed`，Pyright 无错误，Android `testDebugUnitTest` 与 `assembleDebug` 串行构建通过。

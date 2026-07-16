# 消息选择与批量动作

## 任务边界

本批补齐成熟 IM 的“选择消息”基本功，但只实现已有所有权语义能够完整闭环的动作：单条引用、单条/批量复制。删除和转发没有现成服务端协议、持久化所有权与多端一致性保证，因此不放置假按钮，也不顺手扩张核心协议。

```text
长按已完成消息
       │
       ▼
┌─ contextual top app bar ──────────────┐
│  ×   已选择 1 条          引用   复制 │
├───────────────────────────────────────┤
│  当前消息语义区：primary state layer  │
│  其他消息：点击追加 / 再点取消         │
└───────────────────────────────────────┘
       │ 多选
       ▼
┌─ contextual top app bar ──────────────┐
│  ×   已选择 2 条                 复制 │
└───────────────────────────────────────┘
```

## 参考与复用

- ExtraGram 的 `ChatActivity` 使用排他的 action mode、selected set、选中计数和随能力变化的动作集合。本批复用这套结构关系，不迁移 Telegram 的删除、转发或复杂权限矩阵。
- 复制继续走 `AkashicNative.copyText`，引用继续走既有 composer reply 与 `reply_to` 链路；选择本身只属于 WebView 瞬时视图状态。
- `RealtimeSession.enqueueMessage` 已拥有引用领域约束：目标必须存在、角色合法并与当前发送会话相同。merged mobile 历史只改变可见范围，不改变引用所有权。

## Better UI

| Before | After |
|---|---|
| 每条消息只能逐个点击复制或滑动引用 | 长按进入排他选择模式，点击追加或取消；顶栏原位变成上下文动作栏 |
| 批量动作容易被做成底部菜单或消息卡片 | 动作只出现在顶栏，选中状态覆盖原消息语义区，不增加弹窗、菜单卡或 checkbox 图标列 |
| 选择态仍可能触发消息内部动作 | 内容子树进入 `inert`，整条消息成为唯一 checkbox 交互面；composer 隐藏、自动跟随暂停 |
| 长按可能与滚动、文字选择、滑动引用竞争 | touch-first 长按 420ms；移动超过 9px 取消，触控笔/鼠标捕获 pointer，触发后抑制合成 click |

## Better Colors

| 语义 | Token / 混合 | 理由 |
|---|---|---|
| 上下文顶栏 | `--m-surface-container-high` + `--m-on-surface` | 表达当前页面模式变化，不伪装成警告或新内容卡 |
| 可执行动作 | `--m-primary` | 与既有引用、复制和导航动作一致 |
| 已选消息 | `color-mix(in oklch, --m-primary 11%, transparent)` | Material state layer；颜色表示选中状态，不成为永久主题块 |
| Agent 过程 | 保持既有紫色 trace token | 选择和 thinking/tool 不争夺同一颜色语义 |

没有新增 raw hex/rgb/hsl 色值；状态直接复用现有 OKLCH Material tokens。

## Better Typography

| Before | After |
|---|---|
| 顶栏没有批量状态 | “已选择 N 条”使用 16px、650 字重和 tabular figures，数字变化不跳动 |
| 单条复制与多条复制格式相同会丢失上下文 | 单条保留原正文；多条增加简短角色和本地日期时间，正文排版不被重写 |
| 动作只能靠图标猜测 | 每个动作保留完整可访问名称，顶栏计数使用 `aria-live` |

## 状态与所有权

```text
snapshot.messages ──稳定 ID──▶ selectedMessageIds（WebView 瞬时状态）
       │                             │
       │ 消失 / streaming / 切会话  ├─ 单条同会话 ─▶ 既有 reply composer
       └──────────对账清理───────────└─ 可复制集合 ─▶ 原生 clipboard
```

- 流式临时消息不允许进入选择，避免 canonical ID 迁移时选择漂移。
- Android back handler 先查询 selection ref；退出选择后才执行图片 history、插件 surface 或 Activity 返回。
- 消息内部按钮在选择态不可聚焦，外层 checkbox 仍支持键盘 Enter/Space 和读屏状态。
- 选择动作不写服务端、不写 Room；复制和引用分别复用既有边界。

## 验收闭环

### 自动化

- `npm run typecheck`：通过。
- `npm run lint`：通过。
- `npm run test:mobile-web-state`：20/20，通过选择顺序、流式排除、快照对账、单条/多条复制格式和同会话引用能力测试。
- `clients/android/scripts/build-release.sh`：release unit、Lint、R8、assemble、v2 签名与证书校验通过。
- `git diff --check`：通过。

### Pixel 7

1. 长按用户消息，原生文字选择柄不出现，顶部显示“已选择 1 条”，消息整行出现低强度主色 state layer。
2. 单选点击引用，选择态退出，键盘打开，composer 紧贴显示“回复 你 / cycle2alpha”：`/tmp/pixel7-selection-final-reply3.png`。
3. 再次进入选择并点击 Agent 回答，顶部变为“已选择 2 条”，引用动作自动消失：`/tmp/pixel7-selection-final-two3.png`。
4. 点击复制，Android 剪贴板预览显示“你 · 昨天 11:02 / cycle2alpha”和后续 Akashic 消息，顺序与屏幕一致：`/tmp/pixel7-selection-final-copy.png`。
5. 选择态按 Android 返回键，只退出选择并恢复输入区：`/tmp/pixel7-selection-final-back2.png`。
6. 应用日志没有 FATAL、RenderProcessGone、协议校验错误或 event sequence gap。

### 独立 Review

首轮指出两项重要风险：选择态内部按钮仍在焦点树；鼠标/触控笔长按离开元素仍可能触发。另有两项次要风险：长按后继续左滑可能误引用；复制格式缺少纯测试。最终实现分别用 `inert` 内容子树、pointer capture、实时 disabled ref 和独立 formatter 测试修复。最终只读复核确认四项均已闭环，`mobileMessageCanReply` 也与 `RealtimeSession` 的权威同会话校验一致；没有 blocker、important 或 minor 发现。

## Kill AI Slop 复核

扫描结果从 38 文件 / 9 组 / 57 命中变为 38 文件 / 10 组 / 58 命中。唯一新增命中是 `-webkit-touch-callout: none` 被扫描器字符串规则误认为 left-border callout；该属性实际用于阻止 Android 长按弹出原生文字选择，不是视觉 callout。本批没有新增渐变、玻璃拟态、发光状态点、圆角卡片墙、彩色图标方块或胶囊堆叠。

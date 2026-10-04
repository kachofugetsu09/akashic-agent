# Akashic 前端设计合同（design.md）

本文件是所有前端工作的唯一设计入口：修改 `frontend/**/src`、插件 web UI（`frontend/plugins/`）或 `plugins/*/web_module.*` 前必读。它规定可观察的几何、token 和反模式；品牌理由与迁移历史见 [0043](../docs/decisions/0043-paper-brand-tokens-replace-material-visual-semantics.md) 与[纸张品牌系统](../docs/design/akashic-paper-brand-system.md)，窄屏行为细节见[共享 Web 窄屏阅读设计](../docs/design/web-narrow-reading.md)。规则冲突时以本文件的几何合同为准，以决策记录的理由为准。

写法约定：每条规则必须可以观察或测量（"输入文字左缘 == 消息正文左缘"），不写主观词（"干净""高级感"）。评审中重复出现的视觉纠正，落成这里的一条规则、一个 token 或一个脚本检查，而不是逐处手调。

## 1. 品牌原则：克制的纸感

页面是一张连续纸面。纸感只由四件事承担：排版、留白、墨色层级、细规则线。

- 允许：纸面层级 `--ak-paper-*`、墨色层级 `--ak-ink-*`、规则线 `--ak-rule-*`、阅读/技术字体对比、充足且一致的留白。
- 禁止用装饰冒充纸感：位图噪点、`feTurbulence`、随机纹理图、持续动画、拟物阴影。去掉所有装饰后，阅读层级和交互状态必须仍然完整。
- 克制优先：不为了"更像纸"增加任何元素。一个分割线、一个底色层级，只有在它回答"这里是什么结构"时才存在。拿不准就不加。

## 2. Token 合同

组件只消费语义 token，不写裸色值、裸字号、裸阴影。

| 轴 | 前缀 | 用途 |
|---|---|---|
| 纸面 | `--ak-paper-*` | canvas / editing / quiet / sheet / raised |
| 墨色 | `--ak-ink-*` | primary / secondary / muted |
| 规则线 | `--ak-rule-*` | subtle / default / strong / focus |
| 字体 | `--ak-type-*` | reading / technical |
| 状态 | `--ak-color-status-*` / `--ak-sys-color-*` | success / warning / error / trace / info |
| 动效 | `--ak-sys-motion-*` / `--ak-sys-duration-*` | 曲线与时长 |

布局几何同样只有一个 owner：

| 值 | owner | 消费者 |
|---|---|---|
| 阅读列宽 | `.chat-main` 的 `--chat-column` | 消息正文、composer、回到底部按钮 |
| 阅读列内边距 | `.chat-main` 的 `--chat-dock-inset` | 消息区 padding、窄屏 composer inset、短屏 margin |
| 用户气泡上限 | `.chat-main` 的 `--chat-bubble-max` | `.user-bubble` |
| 输入卡文字 inset | `.chat-main` 的 `--composer-text-inset` | composer padding、stats 行 |
| 侧栏留白 | `--chat-rail-inline` | 侧栏所有行的左竖线 |

新组件不得新增 `--md-sys-*` 直接依赖（旧 namespace 是迁移期 transport alias）；不得为设想中的组件提前发布 token；同一几何值出现在两处时，先收进上表的某个 owner。

## 3. 布局与对齐合同

### 3.1 阅读竖线

宽屏（>820px）下阅读列偏左 1/4 留白。下面几条竖线必须重合，误差 0：

- 消息正文左缘 == composer 输入文字左缘。实现方式：composer 卡片以阅读列为中心、两侧各宽出 `--composer-text-inset`，使 1px 边框 + 15px padding 之后文字正好落在阅读竖线上。stats 行用 `padding-inline-start: var(--composer-text-inset)` 跟随同一条线。
- 用户气泡右缘 == 阅读列右缘。
- "回到底部"按钮对阅读列居中，不是对容器几何居中。

窄屏（≤820px）下屏幕内边距即阅读边界：composer 卡片左右缘 == 消息正文左右缘 == `--chat-dock-inset`，卡片内部文字有自己的 inset（卡片是独立编辑面，不要求卡内文字与正文同线）。

### 3.2 侧栏一条左竖线

侧栏所有行的图标/文字左缘落在 `calc(var(--chat-rail-inline) + 10px)` 一条线上：工具行、搜索框、分组标题、目的地、会话行、底部动作。嵌套会话的缩进从这条线起算。

### 3.3 断点只改 token

断点（820px / 36rem / 30rem 高）只允许重定义 `.chat-main` 上的变量值，不允许在 media query 里重写消费方的 padding/margin/width 硬编码。新增断点值前先确认现有三个不够用。确实需要逐断点覆盖几何（如窄屏 composer inset）时，覆盖规则必须写在被覆盖规则之后——同特异性下 CSS 靠源码顺序生效，media query 不提供额外优先级。

### 3.4 safe-area 全覆盖

凡是贴屏幕边缘的条形区域（顶部 band、会话标题行、抽屉、composer、底部动作），水平 inset 一律写成 `max(设计值, env(safe-area-inset-*, 0px))`。不允许只处理 composer 而漏掉标题行或抽屉。

### 3.5 光学对齐惯例

- 图标与文字同行时用 flex `align-items: center`，不用 `line-height` 数值法凑居中。
- 非对称 Lucide 图标沿用现有的亚像素 `translate` 补偿体系（0.5px 级，见 `styles.css` 末尾"光学对齐"区）；新图标按同一方式补，不用 margin 凑。
- 触控目标：窄屏可点区域 ≥44px；桌面行内图标按钮 40px。hit area 的视觉中心必须与图形中心重合。
- 滚动容器不得私自添加 `scrollbar-gutter`、`padding-inline` 或额外的 `%` 基准——阅读列的百分比基准必须与 composer 相同（这是曾导致两侧竖线错位 7.5px 的真实事故）。

### 3.6 次要操作显现合同

消息操作等次要操作默认 `opacity: 0`，父消息容器 `:hover` / `:focus-within` 时在 `--ak-sys-duration-short` 内显现；时间戳等阅读信息常驻。触控断点（≤820px 或 `hover: none`）hover 不存在，次要操作常驻可见。透明状态不得把按钮移出 Tab 序（禁用 `visibility: hidden` / `display: none`），键盘 Tab 到达时经 `focus-within` 必然可见。

### 3.7 遥测收纳合同

composer 统计行活动期只显示静态文案（不逐秒更新可见文本），结束只显示一条终态摘要，并在 `--ak-sys-duration-short` 淡出后移出（`prefers-reduced-motion` 下直接隐藏）。`aria-live` 不得挂在逐秒轮询更新的元素上：live 区域只在活动状态切换时更新一次性文案，逐秒数据放 `aria-hidden="true"` 呈现层。

### 3.8 单行菜单即删除

动作入口菜单只含一个动作时，删除菜单层，触发钮直接执行该动作（如 composer 附件钮点击直接唤起文件选择器）；菜单项 ≥2 时才允许菜单存在。判断依据是可观察的菜单项数量，不是"将来可能会加项"的设想。

### 3.9 浮层单面板合同

弹出选择器默认单面板直达：首屏即完整可选项，跨来源分组平铺，不要求先选来源再选项；来源筛选只作为次级手段（如来源数 ≥3 才出现的小筛选行）。次级设置（如思考强度）在同一面板内联展开收起，不换屏、不设返回键。确需两级才能承载的内容，先证明单面板无法表达。

### 3.10 侧栏宽度拖拽合同

宽屏（>820px）侧栏右缘提供 8px 透明拖拽热区，hover/drag/focus 显 1px `var(--ak-rule-strong)` 竖向强调线。拖拽只写 `--chat-rail-width` 一个值，写在被消费的 grid 容器（`.chat-shell-body`）inline style 上，范围钳制 15–26rem；双击复位（移除 inline 覆盖，回到 CSS clamp 默认）。拖拽柄同时是键盘 separator（`role="separator"`、`aria-orientation="vertical"`、左右方向键步进 0.5rem）。偏好持久化 localStorage（`akashic.chat.rail-width`），首次渲染前同步读取。窄屏（≤820px）与抽屉内不渲染拖拽柄。

### 3.11 会话排序投影合同

会话排序（最近活动 / 最近创建 / 标题）是纯客户端展示投影：只重排"最近会话"区与项目内会话子列表，不改变服务端权威顺序，置顶区始终跟随服务端顺序。时间倒序中缺失时间戳的行稳定排到末尾。选中项持久化 localStorage（`akashic.chat.session-sort`），入口是"最近会话"分组标题行右侧的幽灵图标钮，复用行菜单浮层与键盘模式。

### 3.12 破坏性行内操作合同

会话删除是就地两步确认，不用模态框打断列表浏览：行尾 ✕ 与 ⋯ 并排、共享同一套 hover/focus-within 显现合同；第一次激活在原位变成 error 墨色的"确认删除"药丸，3 秒未确认自动还原；行菜单的"删除"项（error 墨色、danger 修饰类）不直接执行，只亮起药丸并把焦点送过去。提交成功的删除是软删：行从目录投影消失，导航区出现一行可撤销的状态条（含撤销与关闭），撤销窗口内 POST undelete 可恢复；窗口结束 UI 不再提供恢复入口，由服务端与决策 0086 兜底。已删会话被直接打开时消息只读展示并给出恢复入口，不允许向其发送新消息。

### 3.13 过程区折叠与日界合同

同一回复里连续 ≥3 个已完成工具行折叠为一行"已执行 N 个工具 · Xs"摘要行（复用工具行栅格与 chevron），展开后逐行恢复原行与可点详情；运行中的工具永不参与折叠，保持逐行可见。思考、正文、部件项打断连续性，不跨行合并。日历日变化处在消息行之间插入居中日界（发丝线 + 日期，同年只标月日、跨年带年）；当天第一条消息上方也立界，分隔不进入消息行的锚点与 hover 区域。

## 4. 排版

- 阅读正文：LXGW WenKai GB Screen，正文 ≥16px，三行以上正文行高 ≥1.4。
- 代码、时间、运行身份、短技术标签：JetBrains Mono（`--ak-type-technical`）。
- 同一界面最多两种字体；不为单个组件引入第三种字体。
- 字号优先使用 `--chat-type-*` 别名；新增字号别名需要至少两个真实消费者。

## 5. 窄屏阅读

- 一次展示当前任务；目录、列表、详情逐层进入，返回动作明确。
- 320px 宽度不发生横向溢出；代码块和宽表在自己的区域内横向滚动，普通正文按可用宽度换行。
- 短屏（高 ≤30rem）让当前内容滚动，标题和操作不抢占全部高度。
- 验收入口：`npm run check:narrow-ui -- --url <服务> --output <私有目录>`，覆盖 320～1365px。

## 6. 命名的反模式

评审中见到以下模式直接打回，并优先用本文件的规则或 token 修复：

- **纹理纸**：用噪点、纹理图或动画假装纸感。纸感只能来自排版、留白、墨色和规则线。
- **双 owner 宽度**：同一个几何值（列宽、inset、缩进）写在两个文件或两条规则里，改一处就漂移。
- **断点绕行**：media query 里硬编码一个已有 token 的值，使变量在窄屏变成谎言。
- **层叠地雷**：组件上的 Tailwind 工具类靠"非 layered CSS 优先级更高"被静默压掉。覆盖关系必须显式，死类直接删。
- **居中幻觉**：`left: 50%` 对整个容器居中，而内容列是偏置的——浮动物必须对阅读列居中。
- **vw 假设**：用 `calc(100vw - Npx)` 猜测页面留白；容器链上任何一级 padding 都会让它失效。
- **装饰性 elevation**：用阴影堆叠层级。层级用 `--ak-paper-*` 纸面表达，阴影只留给临时浮层。
- **颜色借道**：因为"当前颜色恰好相同"而借用 border 或 status token 当背景/文字色。

## 7. 改动验收

前端改动交付前：

```bash
npm run typecheck
npm run build:chat        # 改 chat 时
npm run build:web-plugins # 改插件 UI 时
```

布局、断点或 token 变化另需在固定 Chromium 里做截图前后对比（`scripts/webui-performance/serve-desktop-fixture.mjs` 提供确定性桌面 fixture），窄屏行为变化跑 `check:narrow-ui`。截图保留在本地私有目录，不进 PR。

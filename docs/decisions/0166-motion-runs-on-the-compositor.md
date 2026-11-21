# 0166 动画只落在合成线程：白名单静态检查，运行时探针补漏

状态：accepted / implemented for review

## 背景

在 120Hz 手机上，侧栏项目展开还算顺，收起不顺。看代码发现三处：

- 收起只有 120ms，用的是开头快的 standard 曲线，单帧要推进 25%。
- 高度靠 `grid-template-rows` 过渡，每帧都要在主线程重排。
- 收起那一刻正是 React 提交和 `inert` 重算的时候。

Chrome trace 还显示两件静态读代码看不出的事：

- SVG 箭头上的独立 `rotate` 属性无法合成，原因码是 `SVGTargetHasIndependentTransformProperty`。
- 布局动画会把同批启动、作用在它子孙上的 `opacity`、`translate` 也拖回主线程。

这两件事放在 CPU 降速 4 倍、主线程有负载的条件下尤其明显：高度类动画掉帧，同样的展开收起改用 transform 后一帧不掉。

全仓库扫下来，静态检查报出 85 处不能合成的动画属性，分布在 66 条声明里。现成的 stylelint 插件（stylelint-high-performance-animation）走黑名单，漏掉了 `grid-template-rows`，也查不到脚本里写的 `style.transition`。

## 决定

1. 动效合同写进 design.md §2.3：
   - 只许对 `transform`、`translate`、`scale`、`rotate`、`opacity` 做动画。
   - 颜色类属性只重绘、面积小，允许保留；`display`、`visibility` 的离散切换允许保留。
   - 其余属性一律不许过渡，包括布局类、`box-shadow`、`filter`、`stroke-dash*`、`background-position`、自定义属性和 `all`。
2. `scripts/check_motion.mjs` 走白名单，接入 CI 的 frontend job。检查范围：
   - 入库的 CSS 过渡、关键帧；
   - 脚本里的 `style.transition` 和 `element.animate()`；
   - motion 组件的 `initial`、`animate`、`exit`；
   - Tailwind 的 `transition-all` 和 `transition-[…]`。

   对选择器末段像图标的独立变换属性给出提示。
3. 例外要在声明前写 `motion-exempt: 理由`。本次给现存的 66 条声明全部写上注释：要迁移的写"待迁移"，后续 PR 改写后删掉注释；这一轮不改的写明原因，列在下面的已知限制里。
4. 新增 `--ak-sys-duration-open/close`（240/200ms）和 `--ak-sys-motion-open/close` 两条曲线，供会移动的展开、收起使用。120Hz 下每帧推进不超过约 12%；收起曲线慢起步，让主线程最忙的头几帧走得最少。减少动态效果时两个时长都归零。
5. `scripts/motion-probe.mjs` 用于验收，不进 CI：在真实页面上跑交互剧本，从 trace 读 `compositeFailed` 与 `FrameSequenceTrackerV3`，补静态检查认不出的元素类型、同批拖累。它不是 required check（TST-008），真实设备上的流畅度另用 Perfetto FrameTimeline 核对。

## 已知限制

这一轮保留在主线程的动画，各处的 `motion-exempt` 注释里写了原因：

- 桌面侧栏面板展开时的几何形变（top/left/width/height）：合成写法要让内容逐帧反向缩放。
- 输入卡随内容长高：合成写法要把外框拆成两片。回到底部按钮跟着输入卡一起移动，留到和输入卡一起改。
- 长文本折叠（用户消息、工具结果预览）的高度过渡：FLIP 需要盖板，下方消息也要有不透明底色。
- 窄屏阅读时顶栏收起，吸顶元素的 `top` 跟着过渡：合成写法要逐个吸顶元素做 FLIP。
- 低频或面积很小的几处：
  - 会话内查找条、选中框、插件降级时的插图高度、阅读器筛选栏与大纲面板（motion 库逐帧驱动高度）；
  - 设置页插件清单展开、首跑引导进度条、Computer 光标标签；
  - 删除提示的倒计时描边、复制成功的打勾描线；
  - 侧栏标题跑马灯：只在桌面悬停时出现；
  - 侧栏滚动渐隐遮罩：滚动本身在合成线程，只是遮罩边缘逐帧重画。

检查的边界：

- 静态检查只看本仓库。外置插件、内联 style 属性、运行时拼出来的样式串不在范围内。
- 图标判断靠命名约定，可能漏报或误报。
- 静态检查和探针都看不见滚动驱动的动画（scroll-timeline）。
- Tailwind 的 `transition`、`transition-colors` 这类固定组合不检查。
- motion 组件的 `x`、`y`、`scale` 由库在主线程逐帧拼成 transform，按变换放行；它们其实不受主线程卡顿保护，新代码优先用 CSS 或 `element.animate()`。

## 补充：消息区展开区不做高度动画

消息区的工具步骤、过程组、思考行，开合不做高度动画，点下去当帧就到位，展开时内层只淡入 120ms。做法对标 Claude Desktop：它的工具行用 Collapsible，并且显式关掉面板动画，只转箭头。

这里没有沿用侧栏那套 FLIP，原因有两个：

- 对话可以很长，展开处下方往往有几十条消息。FLIP 要测量并移动下方全部内容，高度过渡则每帧重排整段对话。两种都把成本放在了点击那一帧。首版 FLIP（#2156）上线后，长对话里点开工具步骤明显发粘，已撤回（#2158）。
- 实测中，点击到下一帧的主要开销是样式重算和重绘，不是布局。另一个罪魁是 `.web-message-anchor:has(.shared-message-actions button.copied)`：主体是整条消息，参数里有后代选择器，消息里任何节点增删都会让整条消息的子树重新算样式。复制高亮已改用锚点上的 `data-copied`。

测量条件：长对话，10 轮，每轮 600 行输出；400×860；CPU 降速 4 倍。展开一个工具步骤的结果如下：

| 指标 | 改前 | 改后 |
| --- | --- | --- |
| 点击到下一帧 | 144ms | 64–88ms |
| 样式重算 | 202ms | 37ms |
| 重绘 | 143ms | 26ms |

新代码不要在会挂载或卸载节点的大容器上写带后代参数的 `:has()`。

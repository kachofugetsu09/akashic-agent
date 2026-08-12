# WebUI 交互性能与组件边界优化设计

- 状态：Web 阶段已实施；Android 真机阶段未开始
- 关联条款：WEBUI-001～WEBUI-007
- 关联决策：[0018](../decisions/0018-chat-webui-has-one-source-and-two-adapters.md)、[0023](../decisions/0023-akashic-tokens-own-material-3-semantics.md)
- 上游设计：[共享对话 WebUI](shared-chat-webui.md)

## 1. 问题与用户意图

桌面 WebUI 已解决流式消息唤醒 App root 和 100 条富历史首载长任务，但入口组件仍同时拥有 session、transport、composer、导航和多个 surface 的编排。部分交互状态与网络请求混在视图组件中，导致重复请求、难以独立测试和修改一处时扩大回归范围。

本轮目标是在不改变消息、会话、插件、模型或移动原生 owner 的前提下，逐个覆盖 Web 端真实交互。每项优化都固定前后指标、能力断言、独立提交和回滚点；不能以文件变短或引入状态库替代用户可观察证据。

## 2. 当前边界与目标结构

```text
┌────────────────────────────────────────────┐
│ Web shell                                  │
│ URL surface / navigation / error boundary │
└──────────────────┬─────────────────────────┘
                   ▼
┌────────────────────────────────────────────┐
│ Headless controllers                       │
│ session · transport · runtime · settings   │
│ model selection · composer                 │
└──────────────┬─────────────────┬───────────┘
               ▼                 ▼
┌──────────────────────┐  ┌──────────────────┐
│ Presentation         │  │ Explicit adapters │
│ native controls      │  │ WebSocket / HTTP  │
│ focus / live regions │  │ Android bridge    │
└──────────────────────┘  └──────────────────┘
```

共享层只拥有中立消息合同、纯投影和可复用展示。桌面 WebSocket 与 Android bridge 保持两个明确 adapter；不把平台生命周期统一成一个隐式全局 store。

## 3. 交互覆盖矩阵

| 交互域 | 当前 owner / 风险 | 完成判据 | 状态 |
|---|---|---|---|
| 流式 thinking/answer/terminal | per-message projection；不得回到 App root | 跟随当前 `main` 透传语义；补丁立即落到单行，terminal 提交 root；真实 600 delta 无 long task | 已完成 `75d16247`，并与 `eed3c7ec` 对账 |
| 历史加载、滚动、回复、复制 | desktop conversation；富历史同步工作曾产生 195ms long task | 100 rich 无 long task；锚点、查找、回复和复制可用 | 已完成 `1bed6261` |
| 知识与运行 | 单组件混合 API、选择 effect 和视图；切 tab 重复详情请求 | controller/data/view 分离；每次 tab 切换恰好一个详情请求；键盘 tab、复制反馈可用 | 已完成 `02797ca4` |
| 会话导航与切换 | `main.tsx` 生成全部导航模型和请求动作 | 导航展示与 session controller 分离；快速切换不提交 stale history | 已完成 `aa0443d5` |
| 模型与思考强度 | picker 同时拥有领域选择、focus 和 popover | 纯选择规则可测；完整方向键/Escape/焦点恢复；无 O(n²) 查找 | 已完成 `4e2d1b67` |
| 编辑器、附件、发送、停止 | `main.tsx` 与 PromptInput context 共同拥有提交条件 | 提交状态单 owner；IME、拖放、附件 ready、send/stop E2E 全覆盖 | 已完成 `de4ad36e` |
| 发送可见性与断线停止 | 旧 history 可覆盖乐观消息；socket 卡在 connecting 时发送和停止都无超时 | 提交先取消旧 history；连接 10 秒显式失败；停止可撤销尚未送达的发送并恢复输入 | 已完成（本 PR） |
| 手机配对 | Dialog 内混合轮询、批准、关闭状态 | transport hook 与步骤视图分离；取消会中止请求并恢复焦点 | 已完成 `ca4dda2d` |
| 设置连接与认证 | settings 表单集中在单文件；多类异步状态共用视图 | provider adapter、表单 state、credential flow 分离；错误聚焦与 live status 可用 | 已完成 `a69ca91b` |
| 记忆设置 | 保存、向量验证和表单状态共用组件 | adapter/controller/view 分离；错误聚焦、取消和保存 E2E 可用 | 已完成 `df753f6f` |
| 错误恢复与空状态 | 入口 lazy chunk 失败会越过 Suspense 形成空白页 | 错误能被感知、重试不重复提交、懒加载失败有边界 | 已完成 `949ee9d8` |
| 响应式、键盘与缩放 | `≤820px` 隐藏全部导航且无替代入口 | 320px reflow、窄屏导航、reduced motion、焦点恢复验证 | 已完成 `c8f25ab2` |
| 桌面 HTTP 数据边界 | `main.tsx` 同时定义 fetch、外部 payload 校验和消息投影 | transport 校验与纯投影分层；格式错误 fail-loud；全部交互 E2E 保持 | 已完成 `7648f504` |
| 桌面 WebSocket 边界 | 入口同时解析协议、归并 turn 和管理 socket 发送 | frame schema、turn controller 和发送生命周期可独立测试；流式 E2E 不退化 | 已完成 `a556e9c2` |
| 桌面自动滚动 | 尾消息外部订阅混在入口且 Hooks 依赖不完整 | 单独组件只订阅尾 identity/revision；上滚锁和用户消息规则不变；lint 零 warning | 已完成 `f33320cd` |
| Chat 模块依赖图 | Settings 与 Memory 数据模块互相反向依赖；入口边界只能人工检查 | 全源码局部依赖零循环；桌面/移动入口保持依赖根；门禁随交互测试执行 | 已完成 `c1aecb06` |
| 桌面入口与应用边界 | `main.tsx` 同时拥有启动、路由、状态编排和完整视图 | 入口只启动和选 surface；产品应用独立；禁止入口吸收 state/effect/transport | 已完成 `b65ae862` |
| 桌面 controller / view | 产品应用组件仍混合全部请求副作用和 JSX 页面树 | headless controller 只编排状态与副作用；view 不接触 HTTP/WebSocket；App 只组合二者 | 已完成 `b1d48861` |
| 流式滚动逃逸与返回 | 算法测试覆盖 escape，但真实按钮无可访问名称且被编辑器遮挡 | 五轮真实流式保持上滚；命名按钮可见、可点击并准确回到底部 | 已完成 `2431a1f5` |
| 自动化可访问性 | 键盘断言存在，但无统一 WCAG 浏览器扫描；附件按钮无名称，配对文本对比度不足 | axe 扫描 6 个正式 surface；WCAG 2 A/AA 零违规；不进入生产 bundle | 已完成 `929da5ff` |

Showcase 只用于展示候选，不计入产品交互完成状态；正式 Chat、Settings、Runtime 和 Pairing 才是验收对象。

## 4. 性能与能力证据

每个交互提交至少记录：

1. 相同 fixture、浏览器版本、采样次数和 source commit。
2. duration、long task、frame gap、layout shift、heap 或请求数中与该交互直接相关的指标。
3. 鼠标和键盘主路径、accessible name/state、焦点关闭/恢复和错误路径。
4. 受影响的 unit/contract tests、构建预算和 change-impact Gate。
5. 不可严格横比的指标必须说明原因，不把噪声写成收益。

## 5. 失败、取消与回滚

- HTTP、WebSocket 与 bridge 在 adapter 边界验证；内部 controller 信任已验证类型，失败显式进入 error state。
- 新请求替代旧请求时 abort 旧 owner；被取消的结果不得提交到新 surface 或 session。
- lazy chunk 加载失败交给相邻 Error Boundary，不提供假内容或静默 fallback。
- 每项优化开始前创建 Git bundle；提交可以独立 revert。性能报告只作证据，不自动提升 baseline 或放宽预算。
- 本设计不写正式 Akashic workspace，不发布移动 WebUI，不修改 Android release pointer。

## 6. 当前已测基线

| source | 交互 | 指标 |
|---|---|---|
| `75d16247` | desktop history 100 rich | P75 870.9ms；long task max 195ms |
| `1bed6261` | desktop history 100 rich | P75 127.7ms；long task max 0ms |
| `1bed6261` | runtime tab switch | 三轮 P75 83.4ms；每次 2 个详情请求，其中一个为旧 key |
| `02797ca4` | runtime tab switch | 三轮 P75 77.1ms；每次 1 个详情请求；long task、layout shift 为 0 |
| `02797ca4` | desktop session switch | 三轮 P75 95.9ms；重复点击再发 1 次 history + 1 次 model 请求 |
| `aa0443d5` | desktop session switch | 三轮 P75 96.0ms；重复点击请求为 0；long task、layout shift 为 0 |
| `aa0443d5` | model picker 关闭态（48 模型） | 49 个隐藏 option；503 个 DOM 元素 |
| `4e2d1b67` | model picker 关闭态（48 模型） | 0 个隐藏 option；129 个 DOM 元素；方向键/Home/End/Escape 通过 |
| `4e2d1b67` | composer 240 字 + 附件 + 双击停止 | 输入 P75 422.5ms；2 个 `turn.stop` |
| `de4ad36e` | composer 240 字 + 附件 + 双击停止 | 输入 P75 399.2ms；1 个 `turn.stop`；1 次上传/1 次发送 |
| `de4ad36e` | 手机配对 | 初始 JS 267,839B gzip；取消不 abort 在途请求；配对 P75 840.1ms；heap P75 11.06MB |
| `ca4dda2d` | 手机配对 | 初始 JS 253,509B gzip；取消 abort 1 个在途请求；配对 P75 835.5ms；heap P75 9.84MB |
| `ca4dda2d` | 设置连接输入、发现模型、Codex 登录（48 连接） | 输入 P75 249.1ms；发现 2 请求；登录 2 请求；heap P75 8.57MB |
| `a69ca91b` | 设置连接输入、发现模型、Codex 登录（48 连接） | 输入 P75 182.8ms；发现 1 请求；登录 1 请求；heap P75 8.32MB |
| `a69ca91b` | 记忆与向量模型同步双击 | 向量验证 2 请求；记忆保存 2 请求；关闭焦点未恢复 |
| `df753f6f` | 记忆与向量模型同步双击 | 向量验证 1 请求；记忆保存 1 请求；错误与关闭焦点均正确 |
| `df753f6f` | 320px 窄屏导航 | sidebar `display:none`；会话、Runtime、配对与新聊天无入口 |
| `c8f25ab2` | 320px 窄屏导航 | 同源 modal drawer；6 个 surface 横向溢出均 0；焦点恢复通过 |
| `c8f25ab2` | Settings lazy chunk 加载失败 | 无顶层 Error Boundary；页面空白且无恢复动作 |
| `949ee9d8` | Settings lazy chunk 加载失败 | 错误可见、重载动作可见、重载恢复均为 1/1 |

Runtime 三轮同机对比：tab 切换 P75 降低 7.6%，详情请求减少 50%；初始详情 ready P75 从 776ms 降到 751ms。JS heap P75 从 9.91MB 降到 9.85MB，差异较小，不单独归因为有效收益。after 报告为 `artifacts/webui-performance/browser-2026-08-12T12-13-48.909Z.json`，SHA-256 为 `2ee9cd5a766f51393286553af1b53399bf19cd332e81fcccb9971f897c32fcf6`。

会话切换三轮同机对比：一次真实切换的 P75 在 95.9ms 与 96.0ms 之间，无可归因的时延收益；已选会话重复点击从两个网络请求降为 0，model 请求也会 abort 被更新的 owner，避免旧选择覆盖新会话。baseline 报告为 `artifacts/webui-performance/browser-2026-08-12T12-27-06.947Z.json`（SHA-256 `e3a890811c37271d00e115e550a70978e00892f763288414a6e43714c5a10f5f`），after 为 `artifacts/webui-performance/browser-2026-08-12T12-31-42.283Z.json`（SHA-256 `b89c48893ef3f429578f218124a7e8513e1209f460f0b2ab748a85360b42fdea`）。

模型选择器三轮同机对比：48 模型时关闭态 DOM 从 503 降到 129（-74.4%），隐藏 option 从 49 降到 0；打开 P75 93.2ms 与 96.0ms 属测量波动，不声称提速。分组投影保留全局 index，消除 render 期逐项 `findIndex`。baseline 报告为 `artifacts/webui-performance/browser-2026-08-12T12-43-42.969Z.json`（SHA-256 `76ca6319ebeaaf19387c2a64c6b90809ae2598e6a10e468bafac457f13872994`），after 为 `artifacts/webui-performance/browser-2026-08-12T12-46-03.525Z.json`（SHA-256 `94cfb9a40b9deacb7f2fa6c257f84ef0330b3513bbcb930175497a8072e7cc51`）。

编辑器三轮同机对比：240 字逐键输入 P75 从 422.5ms 到 399.2ms（-5.5%），该数字包含 Playwright 逐键调度，只作方向性证据。确定性收益是输入 state 从 App root 下沉到 `DesktopComposer`，同步双击停止从 2 个 frame 降到 1 个；附件仍恰好 1 次上传并出现在唯一 `message.send`。baseline 报告为 `artifacts/webui-performance/browser-2026-08-12T12-58-19.141Z.json`（SHA-256 `74c9b3b44dcf1fa9f33cc7330ab30f02568ecb570ef59a0969d3aac1067f6f32`），after 为 `artifacts/webui-performance/browser-2026-08-12T13-02-03.613Z.json`（SHA-256 `0954442941b6132c3faa21de1da2a477eb3667f44d4bd8beda805e33c8fd4091`）。

手机配对使用同一 Chromium 150 和延迟 300ms 的真实 HTTP fixture 对比。初始 JS 减少 14,330B gzip（-5.35%），配对代码在首屏资源中为 0、打开后按需加载 1 个 chunk；关闭会 abort 恰好 1 个已发出的创建请求，完成后焦点恢复保持 100%。配对闭环 P75 840.1ms 到 835.5ms、最大长任务和布局偏移均为 0，不声称时延提速；heap P75 从 11.06MB 到 9.84MB（-11.1%）只作同机方向性证据。baseline 为三轮 `artifacts/webui-performance/browser-2026-08-12T13-12-47.826Z.json`（SHA-256 `ea99066d38d2b9398d489e219e5a937d205bd1a7873d73e01a54d061e174d7d4`），after 为五轮 `artifacts/webui-performance/browser-2026-08-12T13-23-36.877Z.json`（SHA-256 `1bf25991e9cdaed622a17c50e9909dfb3badcb55bf2b37e7ccbcc788964ed93d`）。

设置连接与认证使用同一 Chromium 150、48 个连接和真实延迟 HTTP fixture 对比。连接名称逐键输入 P75 从 249.1ms 到 182.8ms（-26.6%），原因是表单 state 从 52 张连接卡片的页面根下沉到 dialog controller；发现模型同步双击从 2 个请求降到 1 个，Codex 登录同步双击也从 2 个请求降到 1 个。首屏 ready P75 925ms 到 924ms、长任务和布局偏移均为 0；heap P75 从 8.57MB 到 8.32MB（-2.9%）只作方向性证据。Radix 统一拥有 modal inert、Tab 环绕、Escape、标签 ID，条件挂载场景显式恢复焦点到打开者。baseline 为三轮 `artifacts/webui-performance/browser-2026-08-12T13-37-28.981Z.json`（SHA-256 `09223818c767fd127dcf6ed1937b70026fa796cab31e5e8e289af1c9f08e10cb`），after 为五轮 `artifacts/webui-performance/browser-2026-08-12T13-52-57.913Z.json`（SHA-256 `ea0f04aac91155cabb6d5219d581d627b5987655eceeec8cdda06ee572da2131`）。

记忆设置使用同一 Chromium 150 和延迟 150ms 的真实 HTTP fixture 对比。同步双击向量验证从 2 个外部请求降到 1 个，记忆保存也从 2 个降到 1 个；缺少向量模型时仍聚焦“添加向量模型”，弹窗成功关闭后的焦点恢复从失败变为通过。记忆选择、持久化 mutation、向量凭据表单和 HTTP adapter 现在是四个明确边界，API Key 不进入记忆展示 owner。baseline 为三轮 `artifacts/webui-performance/browser-2026-08-12T13-59-56.672Z.json`（SHA-256 `789d2ed6e6bc2581f723d0a0d36173226fc8455892a00b32da09e9fdf167668a`），after 为五轮 `artifacts/webui-performance/browser-2026-08-12T14-06-00.874Z.json`（SHA-256 `6cf229ec101a3a36845285a6ece5f33684c81a34a18ca366d542a81595607314`）。

窄屏基线的生产 CSS 在 `≤820px` 直接隐藏唯一 `DesktopSidebar`，没有可操作入口，因此旧版无法完成会话选择、Runtime、配对或新聊天的端到端场景。优化后新增仅窄屏可见的 modal drawer，复用同一 `DesktopSidebar`，不复制导航模型；五轮 320×800、`prefers-reduced-motion: reduce` 下 Chat、模型选择器、配对、Settings、设置 dialog、Runtime 的页面级横向溢出均为 0，关闭导航焦点恢复 100%，Runtime 3 个 tab 可见。代价为桌面首屏 JS 253,526B 到 253,806B gzip（+280B），CSS 16,493B 到 16,626B gzip（+133B）。after 报告为 `artifacts/webui-performance/browser-2026-08-12T14-22-40.585Z.json`（SHA-256 `bc1b4fb41a1aec8facac3f901cdea62acda3d0f4797087f6ac6eb62ecc8afe3e`）。

入口错误恢复在生产构建上主动中断首个 Settings lazy chunk；旧版没有顶层 Error Boundary，会越过 Suspense 形成空白页。优化后五轮均捕获 1 次真实 chunk 失败、显示 1 个 `role=alert` 与重载动作，并在重载后恢复到完整“模型连接”页面；消息渲染的局部边界也提供同一 fail-loud 恢复动作。代价为桌面首屏 JS 253,806B 到 253,966B gzip（+160B），CSS 16,626B 到 16,754B gzip（+128B）。after 报告为 `artifacts/webui-performance/browser-2026-08-12T14-31-47.229Z.json`（SHA-256 `f378107f4fab91457e12ef29abcc5fe17b5a6bbd19c9c21f666a3b7c457fe334`）。

桌面 HTTP 数据边界从入口拆为 `web-chat-data.ts` 的外部响应校验和 `web-chat-message-data.ts` 的纯消息投影；内部展示不再接触未验证的 session、message、model、shell 或 upload payload，入口从 1,343 行降到 985 行。新增测试覆盖正常响应、缺字段、半条 reply、无效 JSON、上传和历史 tool/media 投影。五轮完整浏览器交互对比没有可归因的性能变化：history P75 133.7ms 到 127.4ms，session switch P75 90.3ms 到 92.2ms，600 delta stream P75 1,429.8ms 到 1,449.4ms，全部 long task 和 layout shift 仍为 0；send/stop/upload、重复会话请求、模型键盘操作、配对取消、设置认证、记忆保存、320px reflow 和 lazy recovery 的确定性计数全部保持通过。首屏 JS 253,966B 到 253,965B gzip（-1B），CSS 保持 16,754B gzip。baseline 为 `artifacts/webui-performance/browser-2026-08-12T14-31-47.229Z.json`，after 为 `artifacts/webui-performance/browser-2026-08-12T14-48-14.088Z.json`（SHA-256 `c4f8a879e566b377008c5a586640a57d8499f276c9522a822ef3c6d8ecfb3f1f`）。

桌面 WebSocket 边界进一步抽到 `web-chat-transport.ts`：外部 frame schema、trace lane、session 隔离、thinking/tool/answer/terminal reducer 和等待 open/abort 的发送生命周期由单一 adapter 拥有，React 入口只注入状态提交与 reload 回调。入口从 991 行降到 694 行；新增 4 组测试覆盖 malformed/unknown frame、完整 turn、foreign session、message push 即时提交、连接等待和 abort。五轮真实浏览器对比中 600 delta stream P75 1,449.4ms 到 1,413.6ms，history P75 127.4ms 到 126.9ms，session switch P75 92.2ms 到 90.8ms；这些小差异只作为无退化证据，不宣称提速。全部 long task、layout shift 仍为 0，发送、中止和上传继续各 1 次。首屏 JS 增加 53B gzip，CSS 不变。baseline 为 `artifacts/webui-performance/browser-2026-08-12T14-48-14.088Z.json`，after 为 `artifacts/webui-performance/browser-2026-08-12T14-59-24.634Z.json`（SHA-256 `2f16ddd293f63c4e556e685b7323232c79f0ded35a585312609354d4d8060377`）。

自动滚动现在由 `desktop-auto-scroll.tsx` 独立拥有，只按最后一条消息 identity 订阅流式 store，并用尾消息 role、正文/过程 revision 和消息数量触发滚动；用户主动上滚时仍不抢回底部，新用户消息仍忽略旧 escape 锁主动到底。两条 `react-hooks/exhaustive-deps` warning 清零，`npm run lint -- --max-warnings 0` 通过。五轮浏览器对比中 history P75 126.9ms 到 135.1ms、600 delta stream P75 1,413.6ms 到 1,429.5ms，属于同机波动；两者 long task、layout shift 仍为 0，最大 frame gap 保持 16.8ms。首屏 JS 从 254,018B 降到 253,897B gzip（-121B），CSS 不变。baseline 为 `artifacts/webui-performance/browser-2026-08-12T14-59-24.634Z.json`，after 为 `artifacts/webui-performance/browser-2026-08-12T15-07-29.056Z.json`（SHA-256 `e39ca98bea220417f030a02f4f72a8a4a0d554543bccbffa8798bca98aeb1396`）。

Chat 源码依赖图审计覆盖 103 个 TypeScript/TSX 模块：改动前由 `settings-data.ts` 的 Memory 类型反向引用与 `memory-settings-data.ts` 的 HTTP helper 引用形成 1 条循环；改动后通用 transport/error 映射由 `settings-http.ts` 拥有，依赖环降到 0。`module-boundaries.test.mjs` 会扫描全部本地静态和动态 import，持续阻止循环以及其他模块反向依赖 `main.tsx`/`mobile-native.tsx`。五轮浏览器对比中 history P75 135.1ms 到 133.5ms、600 delta stream P75 1,429.5ms 到 1,440.8ms，属于测量波动；所有 long task、layout shift 仍为 0，frame gap 最大 16.8ms，完整交互计数保持通过。after 为 `artifacts/webui-performance/browser-2026-08-12T15-18-13.629Z.json`（SHA-256 `4586bb3b50710575b6f28195eb64233f70cef7fd037968f31d0afd007340d2b2`）。

桌面入口从 643 行降到 69 行，只拥有 theme/bootstrap、surface 选择、顶层 Suspense/Error Boundary 和 `createRoot`；产品状态与页面组合进入 `DesktopChatApp`，架构测试禁止入口重新引入 `useState`、`useEffect`、WebSocket 或 HTTP。五轮真实浏览器对比中 history P75 133.5ms 到 144.2ms、session switch 102.3ms 到 97.2ms、600 delta stream 1,440.8ms 到 1,432.4ms，属于测量波动；所有 long task、layout shift 仍为 0，最大 frame gap 16.8ms，send/stop/upload 仍各 1 次，配对取消、设置认证、焦点和窄屏场景保持通过。首屏 JS 为 254,013B gzip，较上轮 253,897B 增加 116B，仍通过 416,768B 预算。after 为 `artifacts/webui-performance/browser-2026-08-12T15-27-06.829Z.json`（SHA-256 `39c555b46e4bcc806fec302bc6b3f7c234d2337f454c51c92270df1f91516a19`）。

桌面产品应用进一步成为三层显式边界：13 行 `DesktopChatApp` 只组合 hook 与 view，`useDesktopChatController` 独占状态、请求、socket 和取消生命周期，137 行 `DesktopChatView` 只消费已建立的 controller 合同并组合导航、历史、编辑器、错误与 lazy surface。架构测试禁止 App 重新吸收 state/effect/transport/view 细节，整个 Chat 模块仍保持零循环。五轮浏览器对比中 history P75 144.2ms 到 127.4ms、session switch 97.2ms 到 90.8ms、600 delta stream 1,432.4ms 到 1,440.6ms，均作无退化证据；long task、layout shift 为 0，最大 frame gap 16.8ms，send/stop/upload 各 1 次，其他完整交互场景保持通过。after 为 `artifacts/webui-performance/browser-2026-08-12T15-49-17.351Z.json`（SHA-256 `8a0c5af58f0efdde06b4fb3025874dfd440f2b985dabfe1f60825268f044ac6f`）。

流式滚动场景现在先模拟用户向上滚轮逃逸，再输入 600 个 delta，断言页面不抢回底部；随后通过具名“滚动到底部”按钮恢复。首轮真实点击暴露按钮虽渲染但被底部编辑器遮挡，样式已将其提升到编辑器上方。五轮中 `streamPreservedScrollEscape`、`scrollReturnAvailable`、`scrollReturnReachedBottom` 均为 1；600 delta P75 为 1,451.1ms，long task、layout shift 为 0，最大 frame gap 16.8ms。history P75 127.4ms 到 161.9ms、session switch 90.8ms 到 101.1ms，没有对应代码热路径变化，记录为同机波动而不宣称退化或收益。after 为 `artifacts/webui-performance/browser-2026-08-12T16-04-12.518Z.json`（SHA-256 `2b2153e4ea5b44d22d15b1e0678d32eddc7c90cb6bbd35e6ad326b74d33cac7c`）。

浏览器门禁引入仅开发期的 `axe-core`，逐轮扫描 Chat、模型选择器、手机配对、Settings、连接弹窗和 Runtime 六个正式 surface 的 WCAG 2 A/AA 规则。首轮发现 1 个 critical 无名称附件菜单按钮和配对流程 3 个 serious 文本对比度节点；修复后五轮均为 6 个 surface、0 个违规。history P75 为 133.7ms、session switch 93.3ms、600 delta stream 1,440.5ms，stream long task 与 layout shift 为 0，最大 frame gap 16.8ms。`axe-core` 不进入生产 import 或 bundle。after 为 `artifacts/webui-performance/browser-2026-08-12T16-22-54.284Z.json`（SHA-256 `b502eeb4718c5e441589f819e1601efeef165281960032b8ebb4ad9394dd9eee`）。

## 8. 最新 `main` 流式语义对账

本分支验收完成后，`main` 的 `eed3c7ec`（#379）删除客户端 grapheme 队列、token bucket 和 rolling 1 秒 ledger，改为每个服务端补丁到达即发布权威 target。合并时保留该上游语义，没有把旧 pacing 实现带回 PR；本分支只继续保证活动 assistant 尾行通过 `StreamProjectionStore` 单独通知，普通历史、用户消息和 terminal 仍提交 React root。

对账后的五轮 Chromium 150 报告为 `artifacts/webui-performance/browser-2026-08-12T16-49-39.541Z.json`（SHA-256 `d5cae0d7a4b0db6e8cc9b565b910c23a584f685a1e524b334118f99ee6e48cbc`）：600 delta stream P75 为 1,304.8ms，long task 与 layout shift 为 0，最大 frame gap 16.8ms；上滚逃逸、返回按钮可用和准确回到底部均为 1。历史 P75 为 133.0ms、session switch P75 为 91.1ms；6 个可访问性 surface 仍为 0 violation。生产构建的桌面首屏 JS/CSS 为 250,573B/16,771B gzip，移动 Web 为 164,829B/19,025B gzip，均通过预算。

用户现场暴露了发送后只出现终止按钮、用户消息不可见且终止无响应的组合故障。根因包含两个独立竞态：上一个 terminal 触发的 history reconciliation 可以在新提交后覆盖乐观用户消息；WebSocket 长期停在 `CONNECTING` 时，发送与停止都等待同一条永不打开的连接。修复后新提交会先取消旧 history owner，连接等待有 10 秒显式超时；用户在消息尚未送达时点击停止会取消发送、释放死连接、撤销乐观行并恢复原输入，消息已经送达时仍发送唯一 `turn.stop`。最终提交上的五轮 Chromium 故障注入中，500ms 延迟 history 到达后用户消息可见为 5/5；永久 connecting socket 下停止恢复输入为 5/5、残留乐观行为 0/5；正常发送、上传与 `turn.stop` 仍各恰好一次，输入 P75 288.7ms，long task 与 layout shift 为 0，最大 frame gap 16.8ms。报告为 `artifacts/webui-performance/browser-2026-08-12T17-20-19.496Z.json`（SHA-256 `6dd23f1c8961ecfd8fc8973b36229fb8a97b4e9cb3e1d0839af2668f34ee4933`）。

`projectneed.md:190` 与 `shared-chat-webui.md:73-76` 仍描述已被 #379 删除的客户端 pacing 合同；本 PR 不借组件重构改写长期产品合同。该文档漂移已在 PR 阻塞项中显式列出，需维护者决定是勘误合同为服务端节奏 owner，还是恢复客户端 pacing，不能把两套语义同时宣称为已验证。

## 9. 阶段结论

Web 阶段的产品交互矩阵已全部实施。最终结构审计覆盖 106 个 TypeScript/TSX 模块，循环依赖为 0；桌面入口为 69 行 bootstrap，产品 App 为 13 行组合层，controller 与 view 各自拥有副作用和展示。与最新 `main` 对账后，桌面专项回归为 32/32，共享 Web 状态回归为 125/125；消息、行级流式投影、插件 slot、terminal、响应式、可访问性和错误恢复保持通过。桌面与移动 Web 构建、lint 零 warning、typecheck 和构建预算均通过；公开 change-impact Gate 在最终 merge commit 后重新生成。

本阶段没有执行 Pixel 7 WebView、Android Macrobenchmark 或 Perfetto，也没有修改或发布 Android 包、移动 WebUI release pointer 或正式 workspace。真机启动、Room → DTO → JSON → WebView 链路与 Android adapter 拆分属于后续移动端阶段，不能用本轮 Chromium 结果替代。

## 10. React 组织依据

- [Sharing State Between Components](https://react.dev/learn/sharing-state-between-components)：每个独立状态保持单一 owner，需协同的交互由最近公共父层控制。
- [Reusing Logic with Custom Hooks](https://react.dev/learn/reusing-logic-with-custom-hooks)：Hook 抽取有语义的有状态逻辑，不复制状态本身。
- [You Might Not Need an Effect](https://react.dev/learn/you-might-not-need-an-effect)：可从 props/state 派生的展示值在 render 期计算，不用 Effect 再同步一份。

后续移动端工作继续使用同一能力矩阵，但单独记录 Android 真机指标，不修改本阶段 Web 基线来掩盖平台差异。

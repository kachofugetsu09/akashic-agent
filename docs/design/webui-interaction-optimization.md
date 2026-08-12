# WebUI 交互性能与组件边界优化设计

- 状态：实施中
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
| 流式 thinking/answer/terminal | per-message projection；不得回到 App root | 400g/s P95、600g/s cap、EGC、terminal、reduced motion 全通过 | 已完成 `75d16247` |
| 历史加载、滚动、回复、复制 | desktop conversation；富历史同步工作曾产生 195ms long task | 100 rich 无 long task；锚点、查找、回复和复制可用 | 已完成 `1bed6261` |
| 知识与运行 | 单组件混合 API、选择 effect 和视图；切 tab 重复详情请求 | controller/data/view 分离；每次 tab 切换恰好一个详情请求；键盘 tab、复制反馈可用 | 已完成 `02797ca4` |
| 会话导航与切换 | `main.tsx` 生成全部导航模型和请求动作 | 导航展示与 session controller 分离；快速切换不提交 stale history | 已完成 `aa0443d5` |
| 模型与思考强度 | picker 同时拥有领域选择、focus 和 popover | 纯选择规则可测；完整方向键/Escape/焦点恢复；无 O(n²) 查找 | 已完成 `4e2d1b67` |
| 编辑器、附件、发送、停止 | `main.tsx` 与 PromptInput context 共同拥有提交条件 | 提交状态单 owner；IME、拖放、附件 ready、send/stop E2E 全覆盖 | 已完成（本提交） |
| 手机配对 | Dialog 内混合轮询、批准、关闭状态 | transport hook 与步骤视图分离；取消会中止请求并恢复焦点 | 已完成（本提交） |
| 设置连接与认证 | settings 表单集中在单文件；多类异步状态共用视图 | provider adapter、表单 state、credential flow 分离；错误聚焦与 live status 可用 | 已完成（本提交） |
| 记忆设置 | 保存、向量验证和表单状态共用组件 | adapter/controller/view 分离；错误聚焦、取消和保存 E2E 可用 | 待实施 |
| 错误恢复与空状态 | 多 surface 各自处理 loading/error | 错误能被感知、重试不重复提交、懒加载失败有边界 | 待实施 |
| 响应式、键盘与缩放 | 桌面/窄屏路径分散在 CSS 与组件 | 320px/200% reflow、键盘全流程、reduced motion、focus ring 验证 | 待实施 |

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
| 本提交 | 设置连接输入、发现模型、Codex 登录（48 连接） | 输入 P75 182.8ms；发现 1 请求；登录 1 请求；heap P75 8.32MB |

Runtime 三轮同机对比：tab 切换 P75 降低 7.6%，详情请求减少 50%；初始详情 ready P75 从 776ms 降到 751ms。JS heap P75 从 9.91MB 降到 9.85MB，差异较小，不单独归因为有效收益。after 报告为 `artifacts/webui-performance/browser-2026-08-12T12-13-48.909Z.json`，SHA-256 为 `2ee9cd5a766f51393286553af1b53399bf19cd332e81fcccb9971f897c32fcf6`。

会话切换三轮同机对比：一次真实切换的 P75 在 95.9ms 与 96.0ms 之间，无可归因的时延收益；已选会话重复点击从两个网络请求降为 0，model 请求也会 abort 被更新的 owner，避免旧选择覆盖新会话。baseline 报告为 `artifacts/webui-performance/browser-2026-08-12T12-27-06.947Z.json`（SHA-256 `e3a890811c37271d00e115e550a70978e00892f763288414a6e43714c5a10f5f`），after 为 `artifacts/webui-performance/browser-2026-08-12T12-31-42.283Z.json`（SHA-256 `b89c48893ef3f429578f218124a7e8513e1209f460f0b2ab748a85360b42fdea`）。

模型选择器三轮同机对比：48 模型时关闭态 DOM 从 503 降到 129（-74.4%），隐藏 option 从 49 降到 0；打开 P75 93.2ms 与 96.0ms 属测量波动，不声称提速。分组投影保留全局 index，消除 render 期逐项 `findIndex`。baseline 报告为 `artifacts/webui-performance/browser-2026-08-12T12-43-42.969Z.json`（SHA-256 `76ca6319ebeaaf19387c2a64c6b90809ae2598e6a10e468bafac457f13872994`），after 为 `artifacts/webui-performance/browser-2026-08-12T12-46-03.525Z.json`（SHA-256 `94cfb9a40b9deacb7f2fa6c257f84ef0330b3513bbcb930175497a8072e7cc51`）。

编辑器三轮同机对比：240 字逐键输入 P75 从 422.5ms 到 399.2ms（-5.5%），该数字包含 Playwright 逐键调度，只作方向性证据。确定性收益是输入 state 从 App root 下沉到 `DesktopComposer`，同步双击停止从 2 个 frame 降到 1 个；附件仍恰好 1 次上传并出现在唯一 `message.send`。baseline 报告为 `artifacts/webui-performance/browser-2026-08-12T12-58-19.141Z.json`（SHA-256 `74c9b3b44dcf1fa9f33cc7330ab30f02568ecb570ef59a0969d3aac1067f6f32`），after 为 `artifacts/webui-performance/browser-2026-08-12T13-02-03.613Z.json`（SHA-256 `0954442941b6132c3faa21de1da2a477eb3667f44d4bd8beda805e33c8fd4091`）。

手机配对使用同一 Chromium 150 和延迟 300ms 的真实 HTTP fixture 对比。初始 JS 减少 14,330B gzip（-5.35%），配对代码在首屏资源中为 0、打开后按需加载 1 个 chunk；关闭会 abort 恰好 1 个已发出的创建请求，完成后焦点恢复保持 100%。配对闭环 P75 840.1ms 到 835.5ms、最大长任务和布局偏移均为 0，不声称时延提速；heap P75 从 11.06MB 到 9.84MB（-11.1%）只作同机方向性证据。baseline 为三轮 `artifacts/webui-performance/browser-2026-08-12T13-12-47.826Z.json`（SHA-256 `ea99066d38d2b9398d489e219e5a937d205bd1a7873d73e01a54d061e174d7d4`），after 为五轮 `artifacts/webui-performance/browser-2026-08-12T13-23-36.877Z.json`（SHA-256 `1bf25991e9cdaed622a17c50e9909dfb3badcb55bf2b37e7ccbcc788964ed93d`）。

设置连接与认证使用同一 Chromium 150、48 个连接和真实延迟 HTTP fixture 对比。连接名称逐键输入 P75 从 249.1ms 到 182.8ms（-26.6%），原因是表单 state 从 52 张连接卡片的页面根下沉到 dialog controller；发现模型同步双击从 2 个请求降到 1 个，Codex 登录同步双击也从 2 个请求降到 1 个。首屏 ready P75 925ms 到 924ms、长任务和布局偏移均为 0；heap P75 从 8.57MB 到 8.32MB（-2.9%）只作方向性证据。Radix 统一拥有 modal inert、Tab 环绕、Escape、标签 ID，条件挂载场景显式恢复焦点到打开者。baseline 为三轮 `artifacts/webui-performance/browser-2026-08-12T13-37-28.981Z.json`（SHA-256 `09223818c767fd127dcf6ed1937b70026fa796cab31e5e8e289af1c9f08e10cb`），after 为五轮 `artifacts/webui-performance/browser-2026-08-12T13-52-57.913Z.json`（SHA-256 `ea0f04aac91155cabb6d5219d581d627b5987655eceeec8cdda06ee572da2131`）。

## 7. React 组织依据

- [Sharing State Between Components](https://react.dev/learn/sharing-state-between-components)：每个独立状态保持单一 owner，需协同的交互由最近公共父层控制。
- [Reusing Logic with Custom Hooks](https://react.dev/learn/reusing-logic-with-custom-hooks)：Hook 抽取有语义的有状态逻辑，不复制状态本身。
- [You Might Not Need an Effect](https://react.dev/learn/you-might-not-need-an-effect)：可从 props/state 派生的展示值在 render 期计算，不用 Effect 再同步一份。

后续每个独立提交在本节追加同口径 after 结果；阶段结束后状态改为“已实施”，未完成项才进入 `NOW.md`。

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
| 知识与运行 | 单组件混合 API、选择 effect 和视图；切 tab 重复详情请求 | controller/data/view 分离；每次 tab 切换恰好一个详情请求；键盘 tab、复制反馈可用 | 已完成（本提交） |
| 会话导航与切换 | `main.tsx` 生成全部导航模型和请求动作 | 导航展示与 session controller 分离；快速切换不提交 stale history | 待实施 |
| 模型与思考强度 | picker 同时拥有领域选择、focus 和 popover | 纯选择规则可测；完整方向键/Escape/焦点恢复；无 O(n²) 查找 | 待实施 |
| 编辑器、附件、发送、停止 | `main.tsx` 与 PromptInput context 共同拥有提交条件 | 提交状态单 owner；IME、拖放、附件 ready、send/stop E2E 全覆盖 | 待实施 |
| 手机配对 | Dialog 内混合轮询、批准、关闭状态 | transport hook 与步骤视图分离；取消会中止请求并恢复焦点 | 待实施 |
| 设置、认证、记忆 | settings 表单集中在单文件；多类异步状态共用视图 | provider adapter、表单 state、credential flow 分离；错误聚焦与 live status 可用 | 待实施 |
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
| 本提交 | runtime tab switch | 三轮 P75 77.1ms；每次 1 个详情请求；long task、layout shift 为 0 |

Runtime 三轮同机对比：tab 切换 P75 降低 7.6%，详情请求减少 50%；初始详情 ready P75 从 776ms 降到 751ms。JS heap P75 从 9.91MB 降到 9.85MB，差异较小，不单独归因为有效收益。after 报告为 `artifacts/webui-performance/browser-2026-08-12T12-13-48.909Z.json`，SHA-256 为 `2ee9cd5a766f51393286553af1b53399bf19cd332e81fcccb9971f897c32fcf6`。

后续每个独立提交在本节追加同口径 after 结果；阶段结束后状态改为“已实施”，未完成项才进入 `NOW.md`。

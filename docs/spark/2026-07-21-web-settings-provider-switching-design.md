# Web 设置中心与 Provider 切换设计

日期：2026-07-21
状态：首期已实现；legacy 配置迁移留待独立 commit

## 1. 目标

在 `127.0.0.1:6321` 提供一个随 Supervisor 生命周期常驻的 Web 设置中心。它同时承担首次初始化和长期配置维护，允许用户配置、保留并切换以下认证方式：

- OpenCode Go API Key；
- Codex Auth；
- OpenAI-compatible API Key，包括预设服务商和自定义端点。

设置中心必须在 Gateway 没有配置、启动失败、重启或回滚时继续可用。切换 Provider 时先验证候选配置，等待现有 turn 自然结束，再原子写入并重启 Gateway；失败时恢复原配置和凭据。

## 2. 非目标

- 不做远程管理面板，不监听非 loopback 地址；
- 不在本轮支持 OpenCode Go 的 Messages 或 Responses 模型；
- 不为每个模型构建协议探测、计费探测或永久兼容性数据库；
- 不重写现有 Dashboard、Chat UI 或 Gateway runtime；
- 不建立第二套配置格式、Provider runtime 或重启机制。

## 3. 用户流程

### 3.1 首次启动

```text
启动 Supervisor
      │
      ├── 配置有效 ──→ 启动 Gateway ──→ 6321 显示运行状态
      │
      └── 无配置/旧配置 ──→ 只启动 6321
                                  │
                     选择认证 → 验证 → 选择模型
                                  │
                     验证配置 → 保存 → 启动 Gateway
```

首次引导只要求完成主模型。频道、记忆、主动任务和高级模型角色保留到设置中心后续页面，不阻塞首次对话。

新用户状态必须独立验收：

- 没有 `config.toml` 或可运行 Gateway 时，Supervisor 仍只启动 6321；
- 用户取消、关闭或刷新未提交的引导时，不写配置或 API Key；
- 验证通过并确认应用后，才创建 named runtime 和主 runtime 指针；
- 首次 Gateway 未 ready 时恢复到“尚未配置”，保留诊断和操作备份，不留下看似成功的半配置；
- Codex 设备授权已经成功但尚未选择模型时，已取得的 Codex 凭据可以保留，页面继续显示“已登录”；其他表单草稿不持久化。

旧式 inline `[llm.main]` 的迁移不属于本阶段。旧配置进入“需要修复”，迁移器将在独立 commit 中实现和验证。

老用户状态必须独立验收：

- 现有 named runtimes、未知配置字段、注释、文件权限和非模型配置保持不变；
- API Key 直接保存在对应 named runtime 中，切换回来时由服务端复用，前端不回显；
- 已有 Codex 凭据直接复用，不强制重新登录；
- 切换 Provider 只改变主 runtime 指针，非当前 Provider 的 runtime 和凭据继续保留；
- 当前配置无法解析或引用的凭据缺失时进入“需要修复”，不自动覆盖原文件，也不启动一个伪成功 Gateway；
- 切换或新 Gateway 启动失败时恢复配置，并重新拉起切换前的 Gateway。

### 3.2 长期切换

Provider runtime 配置可以并存。切换只改变 `[llm].main` 指向，不删除原 runtime。

```text
Providers 中配置连接
        │
Models 中选择 Provider + 模型
        │
验证候选配置和真实最小请求
        │
停止接收新 turn，等待已接收 turn 完成
        │
备份并原子写入 → 启动新 Gateway → 等待 readiness
        │
    ┌───┴───┐
   成功     失败
    │        │
切换完成   恢复配置并启动旧 Gateway
```

切换不会中断既有 turn；页面等待 Gateway 返回最终成功或失败。本阶段不提供取消切换。

## 4. 配置模型

复用现有 `ModelRuntimeConfig` 和 `[llm.runtimes]`，不新增平行的 Provider 配置树。API Key 直接保存到稳定 runtime；Codex OAuth 继续通过 `auth` 引用 CredentialStore。

```toml
[llm]
main = "opencode_go_main"

[llm.runtimes.opencode_go_main]
provider = "opencode-go"
api_key = "<saved locally; never returned by settings API>"
model = "kimi-k2.5"
base_url = "https://opencode.ai/zen/go/v1"
context_window = 262144
max_output_tokens = 8192
input_modalities = ["text"]

[llm.runtimes.codex_main]
provider = "codex"
auth = "codex_default"
model = "gpt-5.4"
context_window = 272000
max_output_tokens = 8192
input_modalities = ["text", "image"]
```

预设 runtime ID 由认证类型稳定生成；自定义兼容端点使用用户可读 slug，冲突时追加短 ID。切换 OpenCode Go 模型只更新 `opencode_go_main`，不会创建无界 runtime。

## 5. 模型目录、协议与能力

实现采用与 Hermes 相同的简单策略：

1. OpenCode Go 的授权可见模型来自其 `/models`；
2. OpenCode Go 已知 `minimax-` 和 `qwen` 家族属于 Messages，当前 UI 隐藏；其余型号默认走 Chat Completions；
3. 保存前执行一次真实最小 Chat Completions 请求，协议不兼容时不修改当前运行时；
4. OpenCode Go 的流式请求显式要求 usage；模型返回缓存明细时写入 Observe，未返回时保持未知；
5. 本阶段不接入 models.dev 或离线模型能力库，上下文窗口和最大输出由用户确认。

这使新增 Chat Completions 型号在 `/models` 或 `models.dev` 出现后无需发布新代码。OpenCode 若新增另一种非 Chat 协议家族，它可能先出现在候选目录，但会在应用前明确失败；后续只需更新短小的协议排除表，而不是新增 Provider 实现。

上下文和多模态按以下顺序解析：

```text
Codex 目录可信元数据
        ↓
自动回填能力字段

OpenCode Go / 普通兼容端点
        ↓
安全默认值 + 页面明确覆盖
```

Codex 继续使用其目录返回的 `context_window`、`max_context_window` 和 `input_modalities`。OpenCode Go v1 保守限制为文本输入，直到目录元数据和真实传输都证明图片输入可用。普通 OpenAI-compatible 端点允许高级覆盖。普通用户只看到“自动检测”；高级页才展示来源和覆盖字段。

## 6. 进程与所有权

```text
Supervisor
├── SettingsServer · 127.0.0.1:6321
│   ├── SetupService
│   ├── CandidateValidator
│   └── Operation event stream
├── RuntimeController
│   ├── Gateway admission/drain control
│   ├── generation + readiness
│   └── rollback owner
└── Gateway child
    └── ConversationRuntime
```

Supervisor 仍是一个 workspace 的唯一进程 owner。SettingsServer 作为 Supervisor 拥有的常驻服务运行，不能自行 spawn、kill 或重启 Gateway；所有生命周期请求进入 RuntimeController 串行状态机。

Gateway 和 Supervisor 复用继承的私有 commit fd，并以 `SIGUSR2` 发起设置重启，不暴露到公共 HTTP。ConversationRuntime 是 turn 准入和排空不变量的 owner：停止接收新 turn 后等待已经持久化的 task 结束，不取消它们。

## 7. 应用事务与回滚

一次应用操作只能有一个 owner，状态为：

```text
editing → validating → applying → draining → restarting → ready
                              ↘ rollback → restarting → ready
```

应用顺序：

1. 在内存中构造候选 TOML；
2. 校验 HTTP 输入 schema、Provider 领域规则、TOML 和完整 Config；
3. 使用候选配置执行模型目录请求和最小模型请求；
4. 串行化设置事务并创建带 operation ID 的配置备份；
5. 原子保存配置；已运行 Gateway 仍使用内存中的旧配置完成既有 turn；
6. 请求 Gateway 停止新 turn、自然排空并退出；
7. 启动新 Gateway，并用 boot ID 验证 readiness；
8. 成功后提交操作；失败则恢复配置备份并重新启动旧 Gateway。

每次设置事务创建唯一命名的 `config.toml.<operation>.bak`。Codex device login 是独立认证动作，不与模型切换事务混写。

## 8. HTTP 与敏感信息

首期接口：

- `GET /api/settings/state`：初始化、配置和 Runtime 状态；
- `POST /api/settings/models`：读取 Codex 或 OpenCode Go 模型目录；
- `POST /api/settings/apply`：验证并提交候选配置；
- `POST /api/settings/codex-login`、`GET /api/settings/codex-login/{id}`：设备授权状态；
- 本阶段不提供凭据删除 API；API Key 随 runtime 配置管理，Codex OAuth 仍由现有认证存储拥有。

敏感信息规则：

- 已保存 secret 永不通过 GET 或错误响应回传；
- 前端仅显示 `configured`、认证类型、来源和更新时间，不显示前缀、后四位或指纹；
- 替换凭据时使用空密码框，提交、取消、跳转后立即清空 React 状态和 DOM 值；
- 不使用 URL query、localStorage、sessionStorage、IndexedDB 或导航状态保存 secret；
- Codex access/refresh token 仅在服务端；浏览器只看到 verification URI、临时代码和状态；
- HTTP 响应使用 `Cache-Control: no-store`、`Referrer-Policy: no-referrer`、CSP、同源和 CSRF 防护；
- 访问日志不记录 body，结构化日志统一脱敏 authorization、cookie、key 和 token 字段；
- SSE 只发送 operation ID、阶段、时间和安全错误码，不发送上游原始 body。

本机用户可以在提交前通过浏览器开发工具看到自己正在输入的 secret；系统保证的是已保存 secret 不会被重新暴露。

## 9. 前端

技术栈保持现有工程：React 19、Vite、Tailwind、shadcn/ui + Radix 和 Lucide。设置页复用 Chat bundle，不新建第二套工程、组件库或样式系统。

首次引导和长期设置共用一个单页壳层：左侧选择 Provider，右侧完成认证、模型和能力字段，底部执行真实验证并启动或切换。运行遥测继续由现有 Dashboard 承担，不在本阶段复制 Runtime 页面。

视觉为克制的本地控制台。颜色使用 OKLCH 语义 token；状态不依靠颜色单独表达。保持同心圆角、克制阴影、足够的交互区域和轻量按压反馈，并尊重 reduced-motion。排版使用小型语义字号体系，正文控制在 60–75 字符行宽。

## 10. 验证

最小测试集合：

- 配置：多个 runtime 并存；仅切换 `[llm].main`；API Key 不出现在状态响应；
- 凭据：API Key 内联保存与复用、配置权限、Codex OAuth 复用；
- Catalog：OpenCode `/models`、Messages 家族过滤、新未知型号默认 Chat；
- HTTP：所有状态 schema、CSRF、no-store、日志脱敏、响应不含 secret；
- Runtime：无配置只启动 6321、排空不取消 turn、readiness 成功、失败回滚；
- 前端：首次引导、三种认证、Provider 切换、凭据不回显、刷新/离页清空草稿、可访问键盘操作；
- 新用户浏览器场景：完全空 HOME/workspace、取消引导零写入、首次成功启动、首次 readiness 失败后无半配置；
- 老用户浏览器场景：既有 named runtimes 无损读取、Codex 登录复用、跨 Provider 往返切换、失败回滚；legacy 迁移留给独立 commit；
- 构建：frontend typecheck、lint 和 settings bundle build；
- 真实边界：使用用户授权的 OpenCode Go 凭据完成目录、最小 chat、配置应用、Gateway readiness 和 Observe 检查；真实日志不得包含 secret。

## 11. 实现顺序

1. 修正 OpenCode Go 为 Hermes 式目录、协议默认与 models.dev 能力解析；
2. 抽出可复用的主 runtime patch 和候选验证服务；
3. 实现配置事务和恢复证据；
4. 扩展 Supervisor/ConversationRuntime 私有排空与回滚状态机；
5. 实现 SettingsServer HTTP 边界；
6. 实现 settings 前端与敏感信息处理；
7. 补集成、真实 Provider 和 Observe 验证；
8. Review、文档对账并提交独立 commit。

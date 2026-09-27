# 内置插件引导验收记录

状态：实现及运行验收完成，独立评审修复后复核中。

## 隔离与恢复

- 基线：`d99dd2b4b4ae2fdb146d9265554262cb39fdcdb5`。
- 分支：`codex/builtin-plugin-onboarding`；实现 worktree：`/mnt/data/coding/akasic-agent-onboarding-design-20260928`。
- 源码恢复包：`/mnt/data/coding/akasic-agent-design-backups/onboarding-implementation-base-d99dd2b4.tar`。
- 真实 Supervisor 场景：`/tmp/onboarding-formal-20260928`，独立 workspace、plugin home 与 config，46 个内置插件通过正式安装链安装。
- CDP：独立 CloakBrowser profile `/tmp/onboarding-cdp-20260928`，端口 19222。没有连接用户日常 profile 或正式 workspace。

## 已验证的用户行为

- 模型连接添加、探测聊天模型、默认模型选择；向量模型填写维度、真实 HTTP 校验、保存默认绑定。
- 四个渠道插件的开关、配置换代、请求回执；错误保留草稿、放弃离开确认与 Escape。
- Telegram 无效 token 返回可理解错误，正确本地探测后保存，token 不回填；留空保留已有凭证。
- 桌面和 390px 移动视口，设置弹窗、键盘关闭、无水平溢出、无页面 JavaScript 异常。

## 证据边界

模型与 Telegram 探测使用本地协议服务；证明浏览器、插件验证、正式输入、换代与回执链路，不证明真实账号鉴权、外部模型质量或 Telegram/QQ 消息送达。本次没有部署生产、没有发送外部消息。

## 依赖、卸载与恢复

- 正式运行图目录：models L0(0)；QQ sender、Telegram sender L1(1)；QQ channel、Telegram channel L1(2)；Akasha L2(6)；Wake 未选 sender 时 L4(8)。选定 sender 后计入该真实边。
- 关闭 Akasha 后 Wake blocked，页面只有下一步，无开关；Wake `enabled` 仍为 null。完整七项流程到达完成摘要。
- 开启 Akasha、完成真实 Web 对话后，Wake 能选择已有 Akashic 会话并开启。聊天回复来自本地 SSE 协议服务。
- 卸载已选 Akashic sender 后，已开启 Wake 变为 blocked，仍可下一步，原选择保持 true；重装 sender 后恢复 ready。
- 卸载 Wake 后目录由七项变六项，其余项可用。卸载 onboarding 后 Akasha 独立设置页仍可用；重启后已关闭选择仍保留。
- 在临时 Wake 数据库通过 owner API 写入一条明确标记的场景历史，未开启时能读取，启用、sender 撤回/恢复后记录仍在。
- 旧 TOML 离线转换保留恢复原件、原 enabled，固定配置只含自身 CredentialRef；旧 Akasha/Wake 缺省仍为 true。
- 缺 Core 配置的全新发行组合 workspace 能重建配置、进入欢迎弹窗和模型页；未配置模型时下一步与后续步骤禁用。发行初始化仍必须提供 context provider 绑定和 Web 入口配置，Core 不猜测插件身份或代为安装。

## 浏览器检查与可复现边界

使用实际 CloakBrowser Chromium 146，经 Playwright `connectOverCDP(http://127.0.0.1:19222)` 连接独立进程。正式 Supervisor 18238 使用真实插件安装、归档、配置 API、selection 与回执；新发行组合首跑 18239 使用另一套 workspace/plugin home。

- `/tmp/onboarding-cdp-flow.log`：完整七项选择与 blocked 摘要。
- `/tmp/onboarding-cdp-ergonomics.log`：错误草稿、Escape、密钥、390px 布局；axe WCAG 2 A/AA 与 2.1 AA 无违规，页面 JS errors 为 0，scrollWidth = clientWidth = 390。
- `/tmp/onboarding-cdp-chat-wake.log`：真实持久对话回复与 Wake 目标选择。
- `/tmp/onboarding-cdp-after-restart.log`：引导卸载后独立设置和重启恢复。
- `/tmp/onboarding-cdp-fresh.log`：缺配置首跑欢迎弹窗、无跳过、axe 无违规。
- `/tmp/onboarding-cdp-theme.log`：深色表单 axe 无违规，移动引导无水平溢出。
- `/tmp/onboarding-cdp-retry.log`：仅浏览器拦截注入 failed 回执，验证原草稿可直接重试。此项是前端失败路径证据，不冒充真实宿主失败恢复证据。

截图：`/tmp/onboarding-complete.png`、`/tmp/onboarding-mobile.png`、`/tmp/onboarding-settings-mobile.png`、`/tmp/onboarding-wake-enabled.png`、`/tmp/onboarding-sender-blocked.png`、`/tmp/onboarding-fresh.png`。

## 最终检查与独立评审

- `pytest -q tests`：44 passed；无新增镜像实现的单元测试。
- 主工程、tests 与新增插件路径 pyright：0 errors。
- plugin_boundary：R1/R2/R3 均 0；yoyo migration 检查通过。
- Control schema 与 Host Bridge 协议生成物 check 通过。共享 venv 缺 grpcio-tools 元数据，使用固定版本隔离环境运行 Host Bridge check，没有修改共享环境。
- `npm run typecheck`、完整前端构建、`git diff --check` 通过。
- 独立内置 subagent：`gpt-6-sol`，`xhigh`，只读审查 `6b38687f`。两项 finding 为 sender 缺席的 blocked 判断、failed 回执后的原样重试；均已修复并用上述场景验证。后续首跑和表单刷新修复交回同一 reviewer 复核。

尚未验证：真实 Codex/OpenCode 登录、外部模型与 Telegram/QQ 账号、真实外部送达、Android 设备、生产发布。远端 CI 状态在 PR 中单列。

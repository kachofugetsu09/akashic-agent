# 内置插件引导验收记录

> QQ 运行支持已按 [0080](../decisions/0080-retire-qq-runtime-support.md) 退役；下文 QQ 实现、拓扑与验收描述只保留历史证据。

状态：实现、运行验收及独立概念 Gate 已完成。

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
- 最终重启及正式重装后，9 个配置/VEDA 文件哈希未变，已有 Wake 历史逐条保持，SQLite integrity_check 为 ok。
- 旧 TOML 离线转换保留恢复原件、原 enabled，固定配置只含自身 CredentialRef；旧 Akasha/Wake 缺省仍为 true。
- 缺 Core 配置的全新发行组合 workspace 能重建配置、进入欢迎弹窗和模型页；未配置模型时下一步与后续步骤禁用。发行初始化仍必须提供 context provider 绑定和 Web 入口配置，Core 不猜测插件身份或代为安装。

## 浏览器检查与可复现边界

使用实际 CloakBrowser Chromium 146，经 Playwright `connectOverCDP(http://127.0.0.1:19222)` 连接独立进程。正式 Supervisor 18238 使用真实插件安装、归档、配置 API、selection 与回执；新发行组合首跑 18239 使用另一套 workspace/plugin home。

- `/tmp/onboarding-cdp-flow.log`：完整七项选择与 blocked 摘要。
- `/tmp/onboarding-cdp-ergonomics.log`：错误草稿、Escape、密钥、390px 布局；axe WCAG 2 A/AA 与 2.1 AA 无违规，页面 JS errors 为 0，scrollWidth = clientWidth = 390。
- `/tmp/onboarding-cdp-chat-wake.log`：真实持久对话回复与 Wake 目标选择。
- `/tmp/onboarding-cdp-after-restart.log`：引导卸载后独立设置和重启恢复。
- `/tmp/onboarding-cdp-fresh.log`：缺配置首跑欢迎弹窗、无跳过、axe 无违规。
- `/tmp/onboarding-cdp-draft-race.log`：延迟真实 GET 响应期间编辑，返回后草稿与保存按钮保持。
- `/tmp/onboarding-cdp-theme.log`：深色表单 axe 无违规，移动引导无水平溢出。
- `/tmp/onboarding-cdp-retry.log`：仅浏览器拦截注入 failed 回执，验证原草稿可直接重试。此项是前端失败路径证据，不冒充真实宿主失败恢复证据。

截图：`/tmp/onboarding-complete.png`、`/tmp/onboarding-mobile.png`、`/tmp/onboarding-settings-mobile.png`、`/tmp/onboarding-wake-enabled.png`、`/tmp/onboarding-sender-blocked.png`、`/tmp/onboarding-fresh.png`。

## 最终检查与独立评审

- `pytest -q tests`：44 passed；无新增镜像实现的单元测试。
- 主工程、tests 与新增插件路径 pyright：0 errors。
- plugin_boundary：R1/R2/R3 均 0；yoyo migration 检查通过。
- Control schema 与 Host Bridge 协议生成物 check 通过。共享 venv 缺 grpcio-tools 元数据，使用固定版本隔离环境运行 Host Bridge check，没有修改共享环境。
- `npm run typecheck`、完整前端构建、完整 PR `git diff --check` 通过。
- 固定提交 `470f4fd8` 的发行构建通过，输出 `/tmp/onboarding-distribution-470f4fd8/distribution.json`；后续修改仅修正表单草稿竞态及文档。发行构建不等于生产安装验收。
- 独立内置 subagent：`gpt-6-sol`，`xhigh`，只读审查 `6b38687f`。两项 finding 为 sender 缺席的 blocked 判断、failed 回执后的原样重试；均已修复并用上述场景验证。后续复核发现自动刷新响应覆盖新草稿的竞态；响应应用前检查编辑状态和请求序号后，用确定性 CDP 延迟真实响应验证。最终复核 `2b357d017e1908ab8e4f28dc6bbeaf772010f45b`：PASS，剩余 must-fix 0；本记录提交仅补充验收结论。

尚未验证：真实 Codex/OpenCode 登录、外部模型与 Telegram/QQ 账号、真实外部送达、Android 设备、生产发布。远端 CI 状态在 PR 中单列。

## 配置回执启动恢复补充

真实 Manager、固定输入、selection 与 SQLite 场景复现了提交 CAS 后、写 active 回执前退出的窗口。旧提交 `a79e9f21` 在重启后输入已选中、generation 已就绪、启动任务已完成，回执却保持 selected；同一场景在修复后返回 active。修复复用宿主任务的 done 状态，不新增恢复 owner、不重试未提交请求，也不改写业务数据。

## Issue #801 弹窗离开与撤回回归

独立分支 `codex/onboarding-801-dialog-lifecycle`，基线 `efb36cd596e1bcd3ac35e4c435401bd752c13878`，隔离根目录 `/mnt/data/akashic-onboarding-fixes-20260928/`。模型草稿、dialog 和请求生命周期修复不改业务数据库 schema，也不删除已有模型、连接或认证凭证。

真实 Chromium 146 / CDP 首轮场景覆盖 clean Escape、dirty Escape 拒绝保留字段和密码显隐、X 拒绝与显式放弃、真实 hash / back / forward、busy 检测离开阻止与失败保留草稿。原生 modal 的 Tab 可能短暂进入浏览器界面，随后回到表单；不能把单次 BODY 当成旧版隐藏 top layer 的连续焦点陷阱。没有增加自定义 Tab 拦截。最终发行包的浏览器复跑结果、SHA 与完整检查在关联 PR 单列。

确定性生命周期场景加载真实前端代码，使用受控协议回执，不调用实际 OAuth、不写业务状态：

| 场景 | main 旧代码 | 修复后 |
|---|---|---|
| OpenCode finishAuth 晚到，模块已撤回 | 继续发 sync_models | 只保留已提交请求和原 cancel_auth |
| 向量 add_model 晚到，模块已撤回 | 继续发 set_default | 不发尚未提交的默认绑定 |
| 已登记 auth 的非 BFCache pagehide | 不取消、继续同步 | 取消 attempt，不继续同步 |
| BFCache pagehide | 保留上下文 | 保留上下文，不提前取消 |

本机证据 `issue801-browser.json`、`issue801-lifecycle.json`；未上传凭据。生命周期回执是明确的前端协议场景，不能当成真实账号 OAuth 登录验收。真实 Codex/OpenCode 登录、读屏、Android 与生产发布未验证。

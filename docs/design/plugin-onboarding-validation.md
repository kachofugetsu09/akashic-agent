# 内置插件引导验收记录

状态：实现已完成，独立评审与最终运行验收进行中。

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

## 最终检查与独立评审

最终结果在评审修复后补齐；不以静态全绿代替浏览器和运行图验收。

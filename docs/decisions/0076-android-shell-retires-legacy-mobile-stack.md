# 0076 · Android Shell 取代旧 Mobile 协议与 OTA

- 状态：accepted；本 PR 实现服务端与 Web 源码清理，正式部署和旧数据物理删除不在本次范围。
- 日期：2026-09-28
- 决定者：维护者

维护者已将移动端切换到独立的 [akashic-android-shell](https://github.com/kachofugetsu09/akashic-android-shell)。它加载服务端 Web 聊天页，以服务端通知 SSE 消费已提交的 Message。旧 `akashic-mobile` 的 WebSocket 实时协议、QR 配对、设备密钥、原生 JS bridge、独立 WebUI bundle、Stable/Preview OTA 与客户端协议 pin 不再是产品合同。

```text
┌─────────────────┐  HTTP Web 页面   ┌────────────────────────┐
│ Android Shell   │ ───────────────▶ │ akashic_clients Web    │
│ 地址/通知/进度  │ ◀──── SSE ────── │ Session 已提交消息投影 │
└─────────────────┘                  └────────────────────────┘
```

本仓库只维护 Web 聊天页、`/api/shell/state` 和 `/api/chat/notifications/stream` 等 Shell 实际消费的接口，以及其背后的 `akashic_clients` Channel。响应式页面和 Web 插件界面仍有真实消费者；插件界面的公开契约改为 `PLUGIN_UI` / `register_plugin_ui`，不再声称原生桥接或移动专属传输。原生平台权限、后台连接、通知和本地进度由 Shell 仓库拥有。旧客户端不能再连接本次删除的协议，升级需改用 Shell。

旧 Mobile 协议、独立 WebUI 和 OTA 的专属决策、设计和协议快照从当前源码树删除。本决定保留新的 Shell 分工；Session/Message 权威身份、插件查询结果校验等跨客户端不变量继续由现行通用合同和实际 Web 消费者拥有。先提交再通知的语义继续有效。仍使用旧 `register_mobile` / `core.mobile_ui.v1` 的外部插件必须迁移后再与此版本组合；不以同名别名维持退役接口。

这次不删除正式 workspace 的旧设备密钥、receipt、mobile-webui generation、客户端缓存或已提交消息，也不清理旧应用的数据。插件配置迁移仅删除 `mobile_realtime` 配置键，先在同目录保存原文件的逐字节备份；Mobile-only 配置拒绝自动转换为 Web 配置。正常消息仍只追加，Shell 的通知进度只在手机本地前移。旧运行数据的物理减少必须有独立清单、恢复点、owner 和执行前后完整性检查。

回退代码可使用本 PR 的父提交和迁移配置备份；旧协议在新版本中不维持兼容层。验收分别核对 Web 对话/插件界面、Shell state 与通知接口、配置迁移、构建和静态边界。真实设备、正式部署和旧数据清理需另行取证。

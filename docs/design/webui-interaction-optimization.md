# WebUI 交互性能与组件边界

- 状态：Web Chat 已实施；Android Shell 真机验收独立进行。
- 关联：[Android Shell 与通知合同](android-shell-experiment.md)、[纸张品牌系统](akashic-paper-brand-system.md)。

## 边界

`frontend/chat/src` 是浏览器与 Android Shell 加载的同一对话页面。Session、Message、Turn 和插件状态由服务端原 owner 持有；页面只保留当前导航、输入、滚动和渲染状态。窄屏布局属于同一页面，不另建移动入口或发布产物。

```text
┌──────────────────────────────┐
│ Web Chat：导航 / 输入 / 展示   │
└──────────────┬───────────────┘
               │ 现有 Web API
               ▼
┌──────────────────────────────┐
│ Session + Message + 插件事实  │
└──────────────────────────────┘
```

性能验收按用户操作记录从触发到可用的耗时、请求数、长任务、布局位移和失败恢复。聊天打开以输入框可接收并保留文字为完成点；切换会话以目标近期消息显示为完成点。流式正文只按当前消息的最新 target 更新，terminal 立即提交；不能用延迟动画掩盖服务端已收到的正文。

浏览器性能夹具位于 `scripts/webui-performance/`，仅覆盖 Web 页面。它的构建预算和 Chromium 结果不能替代 Android Shell 的 WebView、后台通知、断线重放或系统生命周期验收；真机结果应绑定设备、APK、服务端源码与测试场景。

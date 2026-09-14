# 普通 UI provider 交接

基线：`d39569f926b73e1831b1139bf992317da125eee9`。
工作分支：`codex/plugin-simplify-ui-provider`。
恢复点：工作树根目录 `.ui-provider-before-d39569f9.tar`，不随提交发布。

## 实现与边界

- 普通 `ui` 插件提供 `ui.v1` 注册服务和既有 `core.web_ui.v1` 浏览器读取服务。
  profile 显式选择，贡献插件显式 inject；Manager 不补装 provider。
- Core 的五个 UI 特殊导出和 snapshot 的三个 Web/UI 字段已删除。
  JS/CSS、合同、Dashboard 模块与路由校验由 provider 拥有。
- 注册身份来自实际 Context，Web 路径、Dashboard loader 源码和返回模块均校验代码制品归属。
  Dashboard 使用原包的延迟 loader，避免第二个 Python namespace 破坏领域类身份。
- 目录由 provider 的 `SNAPSHOT_SEALING` listener 封存。宿主校验服务的实际 Root token；
  请求能力由公开 `Context.require_runtime_owner` 验证，拒绝跨 Root 借用。
- 注册 Effect 拥有 Dashboard 资源；失败保留关闭句柄，成功后才注销。
  返回列表有坏项时也保留其中实际资源；不重放初始化。没有新增代码/资产复制或持久状态减少。
- HTTP/WebSocket 路由前缀、身份头、鉴权和 stale 响应沿用现有宿主逻辑。
  直接绕过宿主调用 Dashboard 的无租约/错误 Root 情况，现在传播公开 Context 的
  `RuntimeError` / `PermissionError`，不再依赖 Core generation 专用错误判断；仍 fail-closed。

## 剩余调用与集成假设

- Manager 仍保留既有 `bind_dashboard_preparer`、`_dashboard_preparer` 和
  `_dashboard_validation_releaser`，用于 Workload 启动后的宿主准备及晋升前隔离资源释放。
  实际 UI 工作委托所选 Root 的服务；这些不是模块导出解释器或新增通用 callback registry。
  完全删除这组时序接口需要协调宿主生命周期，未在本层伪装完成。
- `PluginDashboardHost` / `SnapshotDashboardMiddleware` 是域消费者；
  browser client 仍消费既有 `core.web_ui.v1`，没有协议字符串兼容转换。
- Mobile UI Slots 不在本层范围内，snapshot 的 Mobile 字段保留。
- Context 及 Manager Root 构造/清理职责未改；封存后的 Root freeze 调用由另一 worker 整合。
- Akasha/Wake 仍按基线编辑 `message_plugin.py`；父分支整合入口 rename 时应将注册改动带到新路径。
  静态 manifest 凭据链不在写集内；新 provider 的 manifest 仅声明自身普通入口。
- 安装 fixture 中真正装载 UI 贡献插件的组合已显式加入 `ui`。
  发布脚本按 manifest 自动发现新包，无新增隐藏选择逻辑。

## 实际验证

只进行了源码/配置/调用点文本检查和 `git diff --check`。
未执行测试、Gate、CI、build、lint、产品代码或 AST。
新增测试覆盖固定代码字节、跨 Root/跨制品、合同冲突、目录封存、实际 Effect 关闭重试，
已有热更新路由用例增加同包类身份断言；这些用例均未运行。

## 完整写集

- `agent/plugin_composition/ui.py`
- `agent/plugins/composable.py`
- `agent/plugins/dashboard_host.py`
- `agent/plugins/generation.py`
- `agent/plugins/manager.py`
- `agent/plugins/snapshot.py`
- `agent/plugins/web_ui.py`（删除）
- `docker/host-runtime/profiles/default.json`
- `docs/design/plugin-v3-capabilities.md`
- `docs/design/plugin-v3-final-migration-map.md`
- `docs/design/plugin-v3-package-contributions-task-contract.md`
- `docs/design/ui-provider-handoff.md`
- `docs/design/web-ui-plugin-composition.md`
- `plugin_boundary.toml`
- `plugins/akasha/message_plugin.py`
- `plugins/codex/plugin.py`
- `plugins/computer/plugin.py`
- `plugins/conversation_ui/plugin.py`
- `plugins/models/plugin.py`
- `plugins/openai_compatible/plugin.py`
- `plugins/opencode_go/plugin.py`
- `plugins/runtime_ui/plugin.py`
- `plugins/shell_ui/plugin.py`
- `plugins/ui/__init__.py`
- `plugins/ui/akashic.plugin.toml`
- `plugins/ui/dashboard.py`
- `plugins/ui/plugin.py`
- `plugins/ui/web.py`
- `plugins/wake/message_plugin.py`
- `plugins/workbench_ui/plugin.py`
- `tests/fixtures/formal_plugins.py`
- `tests/test_akasha_message_plugin.py`
- `tests/test_channel_input.py`
- `tests/test_computer_driver_plugin.py`
- `tests/test_core_messages.py`
- `tests/test_dashboard_api.py`
- `tests/test_installed_reply_delivery.py`
- `tests/test_message_control.py`
- `tests/test_message_plugin_dashboards.py`
- `tests/test_model_execution.py`
- `tests/test_plugin_business_validation.py`
- `tests/test_plugin_hot_reload.py`
- `tests/test_proactive_feedback_emotion_interop.py`
- `tests/test_reply_follow.py`
- `tests/test_reply_program.py`
- `tests/test_standard_tools.py`
- `tests/test_ui_provider.py`
- `tests/test_wake_messages.py`

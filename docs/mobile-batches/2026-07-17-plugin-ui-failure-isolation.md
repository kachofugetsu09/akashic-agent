# 插件移动 UI 失败隔离

## 问题

Pixel7 在 Fitbit 面板请求一个当前分钟的睡眠片段时，插件投影抛出 `ValueError`。异常越过移动插件边界、命令通道和 ASGI WebSocket，导致整个 IM 连接断开；手机需要重新完成 proof、resume、目录和历史同步。

```text
plugin.ui.call
└── 插件异常
    ├── 修复前：逃逸到 ASGI → WebSocket EOF → 全量重连
    └── 修复后：plugin_failed reply → 面板原位错误 → IM 连接保持
```

## 所有权

- 插件输入错误继续使用 `MobileUiRpcInvalidRequest`，映射为 `plugin_invalid_request`。
- 超时继续由 `PluginMobileUiProvider` 拥有，映射为 `plugin_timeout`。
- 插件代码异常及返回契约错误由插件调用边界统一记录并转换为 `MobileUiRpcExecutionError`，移动通道持久化为 `plugin_failed`。
- `CancelledError` 不被捕获，宿主关闭和任务取消仍按生命周期传播。
- 客户端只获得插件和方法标识，不泄漏内部异常细节；完整 traceback 保留在服务端日志。

## 验收

```bash
/mnt/data/coding/akasic-agent/.venv/bin/python -m pytest -q \
  tests/test_plugin_mobile_ui.py tests/mobile_realtime/test_channel.py \
  tests/mobile_realtime/test_gateway.py -k 'plugin_ui or mobile_ui_rpc'
/mnt/data/coding/akasic-agent/.venv/bin/pyright \
  agent/plugins/mobile_ui.py infra/mobile_realtime/channel.py
```

自动化覆盖插件抛错、非对象返回、非字符串键、不可 JSON 编码和超限响应；同一失败命令重放不重复调用插件。Gateway 级测试随后发送 `ping`，证明错误回复后仍使用原 connection epoch。

Pixel7 端到端通过：在隔离 Mobile Lab 暂停 Fitbit monitor 后打开“健康状态”，请求 `01KXQ1XW3YZ53BC62H1R7ZFN21` 完成为 `plugin.ui.call.error / plugin_failed`，面板原位显示错误和“重试”。恢复 monitor 后点击重试，请求 `01KXQ1ZZMM0C43PDE2B9EDBGDP` 完成为 `plugin.ui.call.ok`，健康总览原位恢复。

两次请求均由 `SocketCandidateId(generation=14, ordinal=0)` 以 `connection_epoch=276` 发出；中间没有 `device.proof`、`resume`、EOF 或目录/历史重同步。截图为 `/tmp/pixel7-fitbit-plugin-failure-isolated-final.png` 和 `/tmp/pixel7-fitbit-plugin-recovered-same-connection-final.png`，数据库与 Android wire 日志摘录为 `/tmp/pixel7-plugin-ui-failure-e2e.txt`。

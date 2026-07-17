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

- 插件输入错误继续使用 `MobileUiRpcInvalidRequest`，映射为 `plugin_invalid_request`；面向手机的文案由宿主重建，插件不能注入超长文本或非法 Unicode 破坏回复帧。
- 超时继续由 `PluginMobileUiProvider` 拥有，映射为 `plugin_timeout`。
- 插件代码异常及返回契约错误由插件调用边界统一记录并转换为 `MobileUiRpcExecutionError`，移动通道持久化为 `plugin_failed`。
- `CancelledError` 不被捕获，宿主关闭和任务取消仍按生命周期传播。
- 客户端只获得插件和方法标识，不泄漏内部异常细节；完整 traceback 保留在服务端日志。

## 验收

```bash
/mnt/data/coding/akasic-agent/.venv/bin/python -m pytest -q \
  tests/test_plugin_mobile_ui.py tests/mobile_realtime/test_channel.py
/mnt/data/coding/akasic-agent/.venv/bin/pyright \
  agent/plugins/mobile_ui.py infra/mobile_realtime/channel.py
```

Pixel7 在隔离 Mobile Lab 中完成端到端验证：连接保持 generation 1、epoch 275 时，临时移走隔离 Fitbit 令牌触发真实 401。手机收到 `plugin_failed` 并原位显示可重试错误，服务端只记录插件边界 traceback；Android 没有重新发送 `device.proof`、`resume` 或重拉历史。恢复令牌后在同一 WebSocket 点按重试即返回真实健康数据，generation 与 epoch 均未变化。截图为 `/tmp/pixel7-plugin-isolation-error.png`、`/tmp/pixel7-plugin-isolation-retry-success.png`。

验证结束后令牌已恢复；正式 workspace 全程未读写。

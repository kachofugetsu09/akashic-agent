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
  tests/test_plugin_mobile_ui.py tests/mobile_realtime/test_channel.py
/mnt/data/coding/akasic-agent/.venv/bin/pyright \
  agent/plugins/mobile_ui.py infra/mobile_realtime/channel.py
```

Pixel7 端到端标准：构造插件执行异常后，应收到可重试错误，连接 epoch 不变化，后续 `plugin.ui.call` 能在同一 WebSocket 上成功；不得出现 ASGI traceback、EOF 或重新拉取全部历史。

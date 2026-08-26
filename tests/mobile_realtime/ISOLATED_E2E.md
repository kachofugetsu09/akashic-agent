# MobileRealtime 隔离端到端验收

这套验收不读取 `~/.akashic`，不复用线上配置，也不启动 `main.py`。Gateway 身份、SQLite、会话、上传和出站媒体都写入测试指定的临时根目录。

```text
┌──────────────┐    adb reverse / TLS WebSocket    ┌──────────────────┐
│ Android debug│ ─────────────────────────────────▶│ isolated Gateway │
│ Room + cache │ ◀─────────────────────────────────│ fixed reply bus  │
└──────────────┘                                    └────────┬─────────┘
                                                              │
                          ┌───────────────────────────────────┼────────────┐
                          │ mobile.db   workspace/sessions.db │ attachments│
                          └──────────────── isolated root ────┴────────────┘
```

## 无设备自动验收

```bash
uv run pytest -q tests/mobile_realtime/test_isolated_e2e.py
```

断言包括：

- Web 与 Mobile 通过真实公开协议往返写入同一个 `akashic:*` Session；
- 定时结果同时投影到两个 UI，并经过 durable ledger 只追加一条 Session Message；
- 同一页 `history.get` 连续同步两次，canonical message identity 不增加；
- 客户端只处理到 `turn.started` 后断线，新 connection epoch 从 `last_ack` 补回 `turn.started` 和 `message.final`；
- 重放不会再次触发 Agent 入站；
- 固定 GIF 通过真实 `attachment.download` 二进制帧下载，内容与 SHA-256 一致；
- `mobile.db`、`sessions.db` 和附件目录全部位于 pytest 的 `tmp_path`。

## Android 真机或模拟器验收

只有主线程明确允许使用设备后再执行以下步骤。

终端一启动隔离 Gateway；显式 `--root` 便于验收后检查数据库：

```bash
rm -rf /tmp/akashic-mobile-device-e2e
uv run python -m tests_scenarios.mobile_isolated_gateway \
  --root /tmp/akashic-mobile-device-e2e \
  --port 16323
```

脚本会输出 `pairing_qr`、`isolated_root` 和对应的 `adb reverse` 命令。终端二连接目标设备：

```bash
adb devices
adb reverse tcp:16323 tcp:16323
adb install clients/android/app/build/outputs/apk/debug/app-debug.apk
```

打开输出的 `pairing-offer.png`，用 Android 客户端扫码。脚本只会自动批准本进程创建的单次 pairing，不接受其他 Gateway 或 pairing ID。
独立 Pilot 包建议通过 `-PakashicDebugApplicationIdSuffix=.webuipilot` 构建，避免覆盖正式包或普通 debug 包。

要复现“WebSocket 已连但应用协议停滞”，可在首次配对后注入一次性故障：

```bash
uv run python -m tests_scenarios.mobile_isolated_gateway \
  --root /tmp/akashic-mobile-device-e2e \
  --port 16323 \
  --fault-mode stall_before_challenge

uv run python -m tests_scenarios.mobile_isolated_gateway \
  --root /tmp/akashic-mobile-device-e2e \
  --port 16323 \
  --fault-mode stall_after_auth
```

前者不发送 `server.challenge`，后者在 `auth.accepted` 后不产生同步进展。两种模式都只触发一次，方便确认 Android 超时后自动重连并恢复 READY。

要验证全新 Core 没有任何 Session 时仍能进入 READY 并创建第一条会话：

```bash
uv run python -m tests_scenarios.mobile_isolated_gateway \
  --root /tmp/akashic-mobile-empty-e2e \
  --port 16324 \
  --empty-history
```

设备 Gate 使用 `IsolatedGatewayDeviceTest#freshEmptyCoreBecomesReadyAndCreatesFirstSession`；它断言初始选择为空、Core 返回 `akashic:<32hex>`，随后客户端选中并加载该 Session 的模型目录。

设备侧验收顺序：

1. 首次连接后出现 `akashic:00000000000070008000000000000001`，其中两条历史消息各出现一次。
2. 退出再进入该 session，确认第二次 history sync 不产生重复消息。
3. 发送任意消息，确认思考文字逐段生长、`inspect_shared_webui` 工具从运行中变为完成，随后 Markdown 标题、列表和代码块逐段生长；结束时没有跳回、重复或闪烁，并确认 GIF 到达且可打开。
4. 回复流式进行时移除端口转发，再恢复端口转发：

   ```bash
   adb reverse --remove tcp:16323
   adb reverse tcp:16323 tcp:16323
   ```

5. 确认状态依次表达“断开/重连/连接正常”，最终回复不丢失、不重复。
6. 结束后检查隔离目录；不得出现任何指向真实 workspace 的符号链接：

   ```bash
   find /tmp/akashic-mobile-device-e2e -type l -print
   find /tmp/akashic-mobile-device-e2e -maxdepth 3 -type f -print
   ```

停止 Gateway 后，显式删除隔离根目录：

```bash
rm -rf /tmp/akashic-mobile-device-e2e
```

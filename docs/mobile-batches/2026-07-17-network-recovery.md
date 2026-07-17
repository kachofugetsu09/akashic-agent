# Android 弱网恢复批次

## 目标

连接处于“已建 WebSocket、但应用协议没有继续前进”的状态时，客户端不能无限等待；网络从不可用恢复时，也不应继续等待旧退避计时。

```text
┌──────────┐  10s 无 challenge  ┌──────────┐
│ CONNECT  │ ──────────────────▶│ reconnect│
└────┬─────┘                    └──────────┘
     │ challenge
     ▼
┌──────────┐  10s 无 auth       ┌──────────┐
│  AUTH    │ ──────────────────▶│ reconnect│
└────┬─────┘                    └──────────┘
     │ auth.accepted
     ▼
┌──────────┐  20s 无同步进展     ┌──────────┐
│  SYNC    │ ──────────────────▶│ reconnect│
└────┬─────┘                    └──────────┘
     │ 任一有效同步帧会重置 20s
     ▼
┌──────────┐
│  READY   │ 取消阶段截止时间
└──────────┘
```

## 已实现

- 每个连接 generation 只持有一个应用层阶段截止时间；进入下一阶段时替换，READY、关闭、重配对或新 generation 时取消。
- challenge、认证分别使用 10 秒截止时间；同步使用 20 秒无进展截止时间，`sync.completed`、`session.list`、`history.page` 和 reply page 都会续期。
- LAN/Tunnel 竞速保持阶段单调：迟到候选不能把 `DEVICE_PROOF`、`SYNCING` 或 `READY` 降回 `SERVER_CHALLENGE`，旧 generation 帧直接忽略。
- 任何通过 candidate、epoch 和协议处理的有效认证帧都会刷新同步 idle deadline；处理失败的帧不会续命。即使 replay 持续超过 20 秒，只要每段有效进展间隔未达到 20 秒就不会误超时。
- 网络 outage 与恢复由 generation 绑定的一次性 latch 持有。无论 OkHttp failure 先到还是 Connectivity 恢复先到，都只消费一次立即重连；重复网络回调和旧 generation failure 不会再创建连接。
- 超时仍复用原有 durable outbox 与 reconnect 流程，不创建第二套重放队列。

## 隔离故障注入

隔离 Gateway 新增两个一次性模式。它们都先允许扫码配对，随后只阻塞一次；客户端超时重连后，下一条连接恢复正常。

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

- `stall_before_challenge`：TCP/TLS/WebSocket 已连接，但服务端不发送 `server.challenge`。
- `stall_after_auth`：完成 challenge、proof 和 `auth.accepted`，但不处理 `resume`、不发送同步帧。

控制台出现 `fault_triggered=...` 后，手机应先显示重连语义，并在对应 10/20 秒截止时间后自动恢复到“连接正常”。历史消息和 outbox 消息不得重复。

## 自动验证

```bash
python -m pytest -q \
  tests/mobile_realtime/test_isolated_gateway_faults.py

cd clients/android
./gradlew :app:testDebugUnitTest \
  --tests com.akashic.mobile.data.realtime.ConnectionRecoveryPolicyTest \
  -x :app:buildMobileWeb --no-daemon --max-workers=2
```

本批次不连接正式 Gateway，不读取正式 workspace，也不写线上数据库。

自动门禁只证明 deadline/latch 的纯状态机规则、隔离 Gateway 故障控制器和既有服务端恢复语义；它不声称替代 Android callback、真实计时器与 OkHttp 生命周期联调。

## Pixel7 真机责任

主线程使用 Pixel7 与上述 fault Gateway 对账以下日志和界面结果：

1. `stall_before_challenge` 和 `stall_after_auth` 各只触发一次，手机分别在 10 秒和 20 秒 idle deadline 后恢复 READY。
2. 分别制造“OkHttp failure 先到”和“Connectivity 恢复先到”，每次 outage 只能看到一个新 generation，不能继续等待旧的最长 30 秒退避。
3. 配对确认页断线后，新 generation 仍有 challenge/auth deadline；旧 generation 的确认等待不屏蔽新连接。
4. LAN/Tunnel loser 的迟到 `onOpen` 不消费恢复 latch，不改变活动 endpoint 或连接阶段。
5. 恢复前留在 durable outbox 的消息最终只发送一次，历史和最终回复均不重复。

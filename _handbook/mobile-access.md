# Akashic Mobile 接入手册

Akashic Mobile 是一个通过独立实时网关连接 Akashic Agent 的 Android 应用。本手册面向维护者和自动化 agent，使用 Cloudflare Tunnel 发布移动实时网关，并保留 Akashic 自己的扫码、确认码和设备密钥认证。

## 1. 接入结构

Akashic Agent 同时提供两个端口。`6322` 承载本机 Web Chat 和配对管理页面；`6323` 承载手机使用的 WSS 实时协议与同源 HTTPS 插件查询。Cloudflare 只转发 `6323`。

```text
┌────────────┐  本机 HTTP   ┌────────────────────────┐
│ 维护者浏览器 │ ──────────▶ │ Web Chat :6322         │
└────────────┘              │ 生成二维码、批准设备      │
                            └───────────┬────────────┘
                                        │ 一次性配对状态
                                        ▼
┌────────────┐  WSS   ┌───────────────┐  Tunnel  ┌─────────────┐  本机 HTTPS  ┌──────────────────┐
│ Android    │ ◀────▶ │ Cloudflare    │ ◀──────▶ │ cloudflared │ ──────────▶ │ Mobile :6323     │
│ 客户端      │        │ Edge          │           │ 出站连接     │             │ challenge / 认证 │
└────────────┘        └───────────────┘           └─────────────┘             └──────────────────┘
```

Cloudflare Tunnel 由本机的 `cloudflared` 主动向 Cloudflare 建立出站连接，不要求公网 IP，也不用在路由器上开放入站端口。Cloudflare 支持 WebSocket，客户端仍使用标准 `wss://` 地址。

扫码数据包含服务端应用身份、可用 endpoint、一次性 secret 和有效期。手机连到 `6323` 后先收到 `server.challenge`，再提交配对声明。电脑显示同一组六位确认码并批准后，服务端记录设备公钥。后续连接使用 challenge 和设备签名完成认证。

## 2. 准备条件

- 一台持续运行 Akashic Agent 的 Linux 主机。
- 一个已经接入 Cloudflare DNS 的域名，例如 `example.com`。
- 可用且已解锁的 Linux Secret Service。移动网关只支持 `secret_service` 保存主密钥；服务不可用或 collection 被锁定时会明确启动失败。
- 一台能安装当前 Akashic Mobile APK 的 Android 手机。
- 当前 Akashic Mobile APK：<https://github.com/kachofugetsu09/akashic-mobile/releases/latest>。
- 当前版 `cloudflared`。安装方式以 [Cloudflare 下载页](https://developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/downloads/) 为准。

先确认 Agent 可以正常启动，并且本机 Web Chat 能打开：

```bash
uv run python main.py
```

```text
http://127.0.0.1:6322
```

## 3. 启用移动实时网关

修改配置前先保留恢复点：

```bash
cp --preserve=all --no-clobber config.toml config.toml.before-mobile
```

在 `config.toml` 中保留 loopback Web Chat，并加入移动网关配置。把示例域名换成自己的 Cloudflare 域名。

```toml
[channels.chat]
enabled = true
host = "127.0.0.1"
port = 6322
channel_name = "web"

[mobile_realtime]
enabled = true
host = "0.0.0.0"
port = 6323
database = "data/mobile_realtime.db"
lan_hostname = "akashic.local"
public_url = "wss://mobile.example.com/ws"
max_attachment_mb = 50
inbox_retention_days = 7

[mobile_realtime.key_encryption]
provider = "secret_service"
master_key_namespace = "akasic/mobile-realtime"
keyset_manifest = "data/mobile/keys/current.json"
```

`public_url` 必须使用 `wss`，路径必须是 `/ws`，不能携带用户名、密码、query 或 fragment。`channels.chat.host` 必须保持 loopback；配置加载器会拒绝把配对管理页面暴露到局域网。

优雅停止旧进程，再重新启动：

```bash
./scripts/stop-runtime.sh
uv run python main.py
```

确认端口已经监听：

```bash
ss -ltnp | grep ':6323'
```

移动网关根路径没有普通网页，下面的 `404` 只证明本机 TLS 监听器已经响应：

```bash
curl -k -sS -o /dev/null -w '%{http_code}\n' https://127.0.0.1:6323/
# 预期：404
```

## 4. 创建 Cloudflare Tunnel

下面使用 Cloudflare Dashboard 创建 remotely-managed tunnel。Dashboard 的具体入口会调整，最新步骤见 [Create a tunnel](https://developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/get-started/create-remote-tunnel/)。

1. 登录 Cloudflare Dashboard，进入 `Networking → Tunnels`。
2. 选择 `Create a tunnel`，类型选择 `Cloudflared`。
3. 给 tunnel 起一个能识别来源主机的名字，例如 `akashic-home`。
4. 选择主机系统，按页面提示安装 `cloudflared`。
5. 保存页面给出的 tunnel token。token 可以启动该 tunnel，应按密钥管理，不能写入 Git、聊天记录或普通日志。

### 4.1 保存 token

`cloudflared 2025.4.0` 及更新版本支持 `--token-file`。下面的命令从权限受限的文件读取 token，进程参数不会包含 token：

```bash
install -d -m 700 "$HOME/.config/cloudflared"
install -m 600 /dev/null "$HOME/.config/cloudflared/akashic-mobile.token"
read -rsp 'Cloudflare Tunnel token: ' AKASHIC_TUNNEL_TOKEN
printf '%s' "$AKASHIC_TUNNEL_TOKEN" > "$HOME/.config/cloudflared/akashic-mobile.token"
unset AKASHIC_TUNNEL_TOKEN
printf '\n'
```

先以前台方式验证 connector：

```bash
cloudflared tunnel --protocol auto run \
  --token-file "$HOME/.config/cloudflared/akashic-mobile.token"
```

Dashboard 中的 tunnel 状态应变为 `Healthy`。Cloudflare 把 `auto` 作为默认协议；网络允许时可使用 QUIC，连接异常时会按自身策略选择可用传输。

### 4.2 添加 Published application route

打开 tunnel 的 `Routes` 页面，添加 `Published application`：

| 字段 | 值 |
|---|---|
| Hostname | `mobile.example.com` |
| Service URL | `https://127.0.0.1:6323` |
| Path | 留空 |

移动网关的 origin 使用 Akashic 生成的自签名 LAN 证书。在该 route 的 `Additional application settings → TLS` 中打开 `No TLS Verify`，让同机 `cloudflared` 可以连接这个 origin。这个开关只作用于 `cloudflared → 127.0.0.1:6323`；手机到 Cloudflare 仍使用公开域名的正常 TLS 证书，Akashic 还会核对扫码得到的服务端应用身份。

不要把 Service URL 写成 `http://127.0.0.1:6323`。移动网关只接受 TLS，协议写错通常会得到 `502 Bad Gateway` 或 TLS 握手错误。

不要给该 hostname 添加要求浏览器登录的 Cloudflare Access 策略。当前 Android 客户端不处理 Access 登录页；访问控制由 Akashic 的一次性配对和设备签名负责。

### 4.3 作为用户服务运行

前台验证成功后可以创建 `~/.config/systemd/user/akashic-mobile-tunnel.service`。先运行 `command -v cloudflared`；可执行文件不在 `/usr/bin/cloudflared` 时，修改下面的 `ExecStart`。

```ini
[Unit]
Description=Akashic Mobile Cloudflare Tunnel
After=network-online.target
Wants=network-online.target

[Service]
ExecStart=/usr/bin/cloudflared tunnel --protocol auto run --token-file %h/.config/cloudflared/akashic-mobile.token
Restart=always
RestartSec=5

[Install]
WantedBy=default.target
```

加载并启动：

```bash
systemctl --user daemon-reload
systemctl --user enable --now akashic-mobile-tunnel.service
systemctl --user status akashic-mobile-tunnel.service
```

用户服务默认在用户登录后运行。需要在未登录时也启动，请让系统管理员按主机策略为该用户启用 linger。

## 5. 验证公开 WSS

公开域名根路径返回 `404` 只能证明 HTTP 请求到达移动网关。下面的检查会建立真实 WebSocket，并要求第一帧是 Akashic 的 `server.challenge`：

```bash
uv run python - <<'PY'
import asyncio
import json

import websockets


async def main() -> None:
    async with websockets.connect(
        "wss://mobile.example.com/ws",
        open_timeout=10,
    ) as websocket:
        frame = json.loads(await asyncio.wait_for(websocket.recv(), timeout=10))
        assert frame["type"] == "server.challenge", frame
        print("mobile realtime ok:", frame["type"])


asyncio.run(main())
PY
```

只有这条检查通过，才能说明 DNS、Cloudflare Edge、Tunnel、origin TLS、`/ws` 路由和移动网关共同可用。

## 6. 安装并配对手机

1. 从 [Akashic Mobile Releases](https://github.com/kachofugetsu09/akashic-mobile/releases/latest) 下载 APK 并安装。
2. 在电脑上打开 `http://127.0.0.1:6322`，点击“连接手机”。
3. 在 Android 客户端选择“扫描电脑”，扫描电脑页面上的二维码。
4. 手机和电脑会显示六位确认码。逐位核对，数字一致后在电脑上选择“确认并连接”。
5. 等待电脑显示“手机已连接”，再关闭配对窗口。

二维码和一次性 secret 有有效期。过期后关闭窗口并重新生成。确认码不同意味着当前连接没有通过预期的配对上下文，不要批准。

首次批准后，服务端保存设备公钥，手机把设备私钥放入 Android Keystore。应用重启、网络切换和正常覆盖升级会继续使用设备签名，不要求重新扫码。清除应用数据、卸载后重装、删除服务端移动数据库或撤销设备会移除这份连续性，需要重新配对。

自动化 agent 可以修改配置、启动进程、创建 tunnel、检查端口和验证 `server.challenge`。扫描二维码、核对确认码和批准新设备保留给人执行；agent 不应代替维护者批准一个无法当面核对的设备。

## 7. 排障顺序

按下面的顺序检查，每一步只回答一个问题：

```text
Akashic 是否 ready
        │
        ▼
本机 6323 是否监听并返回 TLS 404
        │
        ▼
Cloudflare Tunnel 是否 Healthy
        │
        ▼
公开 /ws 是否收到 server.challenge
        │
        ▼
手机是否完成配对或设备认证
```

| 现象 | 检查 |
|---|---|
| 启动时报 Secret Service 不可用或已锁定 | 解锁当前用户的 Secret Service，再启动 Akashic。不要改成明文密钥或删除 keyset 绕过错误。 |
| `6323` 没有监听 | 检查 `[mobile_realtime].enabled`、启动日志和配置校验错误。 |
| Tunnel 显示 `Inactive` 或 `Down` | 检查 `cloudflared` 进程、用户服务和 token 文件权限。 |
| 公开地址返回 `502` | 核对 Service URL 是 `https://127.0.0.1:6323`，端口已监听，并已为自签名 origin 打开 `No TLS Verify`。 |
| 根路径返回 `404` | 这是移动网关的正常 HTTP 结果；继续运行真实 WSS challenge 检查。 |
| WSS 没有收到 `server.challenge` | 检查 Cloudflare WebSockets 设置、重叠的 Worker route、WAF/Bot 规则和 `cloudflared` 日志。 |
| 手机一直显示重连 | 先执行公开 WSS 检查。公开检查失败时修 Tunnel；公开检查通过后再看手机时间、已保存设备和服务端撤销状态。 |
| 更新后要求重新扫码 | 检查 APK 签名是否和旧版一致，以及安装过程是否清除了应用数据。 |

Cloudflare 对 `502`、Tunnel 状态和 WebSocket 的解释见官方的 [Tunnel troubleshooting](https://developers.cloudflare.com/cloudflare-one/troubleshooting/tunnel/) 与 [WebSockets](https://developers.cloudflare.com/network/websockets/) 文档。

## 8. 安全边界

- 只把 `6323` 配到 Cloudflare；`6321` 设置中心和 `6322` Web Chat 留在 loopback。
- Tunnel token 只放在权限为 `0600` 的文件或专用 secret store 中。怀疑泄露时在 Cloudflare Dashboard 轮换 token，并重启 connector。
- `mobile_realtime.db`、`data/mobile/keys/` 和 Secret Service 中的 master key 共同维持服务端身份与已配对设备。备份和恢复必须成组处理。
- 不要删除 `current.json`、加密 key blob 或移动数据库来处理启动错误。数据库有身份而 keyset 丢失时，runtime 会按设计拒绝启动。
- 公开 WSS 可被互联网访问，业务数据仍要求一次性配对、人工确认和设备签名。持续查看 Tunnel 健康状态，并及时更新 `cloudflared`。

## 9. 官方资料

- [Cloudflare Tunnel 原理](https://developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/)
- [Dashboard 创建 remotely-managed tunnel](https://developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/get-started/create-remote-tunnel/)
- [Published application routes](https://developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/routing-to-tunnel/)
- [Origin TLS 参数](https://developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/configure-tunnels/origin-parameters/)
- [`cloudflared tunnel run` 参数](https://developers.cloudflare.com/tunnel/advanced/run-parameters/)
- [Cloudflare WebSocket 支持](https://developers.cloudflare.com/network/websockets/)

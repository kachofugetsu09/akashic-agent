# Mobile Lab

隔离的 Android 被动链路测试环境，不读取或写入正式 workspace，也不启动 Telegram、QQ
和 proactive。源仓库与正式 `config.toml` 只读挂载；容器只把生成配置、数据库、密钥环和
插件状态写入忽略版本控制的 `docker/debug/profiles/mobile-lab/`。
生成的 `config.toml` 会复制 provider 凭据并以 `0600` 保存；该 profile 只用于本机测试，
不能提交或共享。

```text
┌─ Pixel Android
│  └─ WSS mobile-lab.wangyuanzhe28.site
├─ 独立 Cloudflare Tunnel 容器
│  └─ https://127.0.0.1:16323
└─ Mobile Lab Agent 网络命名空间（独立 Docker bridge）
   ├─ WebChat / QR   127.0.0.1:16322
   ├─ Mobile WSS     0.0.0.0:16323
   ├─ chat-proxy     16322 → 127.0.0.1:6322
   ├─ workspace      /sandbox/workspace
   ├─ plugin home    /sandbox/home/.akashic-plugin
   └─ Secret Service /sandbox/home/.local/share/keyrings
```

首次创建独立 Tunnel 并配置 DNS：

```bash
cloudflared tunnel create akashic-mobile-lab
cloudflared tunnel route dns akashic-mobile-lab mobile-lab.example.com
```

把真实域名作为 `AKASHIC_MOBILE_LAB_PUBLIC_URL` 启动；本机局域网地址会自动探测：

```bash
AKASHIC_MOBILE_LAB_PUBLIC_URL=wss://mobile-lab.example.com/ws \
  docker/mobile-lab/start.sh
```

打开 <http://127.0.0.1:16322> 生成二维码。停止环境：

```bash
docker/mobile-lab/stop.sh
```

# 使用手册

安装见 [README](../README.md)。这里是装好之后会用到的操作。

## 数据在哪里

| 安装方式 | 数据位置 |
| --- | --- |
| Docker | 数据卷 `<目录名>_data`。目录是 `akashic` 时，卷名就是 `akashic_data` |
| 不用 Docker | `~/.akashic` |

配置、会话、插件和记忆都在这里。各类数据的具体位置和恢复方法，见[持久状态地图](./design/persistence-state-map.md)。

**不要运行 `docker compose down -v`**，它会删除数据卷。

## 更新

### Docker

先停止服务，备份数据，再下载新版 `compose.yaml` 并启动：

```bash
docker compose stop
docker run --rm -v akashic_data:/data -v "$PWD":/backup alpine tar czf /backup/akashic-data.tgz -C /data .
curl -fsSLO https://github.com/kachofugetsu09/akashic-agent/releases/latest/download/compose.yaml
docker compose up -d --wait
```

备份命令里的 `akashic_data` 要换成你的数据卷名（见上表）。

### 不用 Docker

先按 Ctrl+C 停止，再切换到新版本，然后重新运行：

```bash
git fetch --tags
git checkout <新版本号，例如 v0.3.0>
./start
```

### 插件

功能插件单独更新，不需要更新整个程序。可以直接对 Akashic 说“检查一下插件更新”，也可以在“功能设置”→“插件”里操作。不用 Docker 时用这两种方式即可；Docker 的命令行用法：

```bash
docker compose exec akashic python main.py pkg list --remote   # 看哪些可以升级
docker compose exec akashic python main.py pkg upgrade         # 全部升级
```

## 换端口

默认端口是 2236。被其他程序占用时：

- **Docker**：在 `compose.yaml` 同目录创建 `.env` 文件，写入 `AKASHIC_PORT=2237`，然后重新启动。
- **不用 Docker**：运行 `./start --port 2237`。

## 登录

网页可以执行 Shell 命令和读写文件，所以带有登录门，只有你能进。用户名是 `akashic`。

- **一次性登录链接**：在运行 Akashic 的机器上执行下面的命令，会得到一条 2 分钟内有效、只能用一次的登录链接。
  ```bash
  ./start --login                                       # 不用 Docker（在 akashic-agent 目录里）
  docker compose exec akashic python main.py web-login  # Docker（在 akashic 目录里）
  ```
- **记住设备**：默认 90 天内免登录。在别人的电脑上登录时，请取消勾选“记住这台设备”。
- **忘记密码**：用上面的一次性登录链接进入即可。Docker 也可以运行 `docker compose exec akashic python main.py web-passwd`，密码会重置为临时密码，所有设备都会退出登录。还可以在 Telegram 或 QQ 里让 Akashic 帮你重置。

SDK、CLI 等程序调用走本机 unix socket，不经过登录门。

## 从其他设备访问

默认只能在运行 Akashic 的那台机器上打开（`http://localhost:2236`）。

想在其他设备上通过域名访问，可以用 Cloudflare Tunnel、nginx、Caddy 等工具，把域名转发到这台机器的 `localhost:2236`。需要注意：

- 反向代理要把原始 Host 转给 Akashic，例如 nginx 写 `proxy_set_header Host $host;`。否则网页的写操作和实时连接会被同源检查拒绝。
- 公网访问请使用 HTTPS。

## 记忆

Akashic 的记忆怎么记、怎么查、怎么改，见[记忆手册](../_handbook/memory-markdown.md)。

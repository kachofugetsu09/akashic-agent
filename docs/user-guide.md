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

先停止服务，备份数据，再下载新版 `compose.yaml` 和 `chromium-seccomp.json` 并启动：

```bash
docker compose stop
docker run --rm -v akashic_data:/data -v "$PWD":/backup alpine tar czf /backup/akashic-data.tgz -C /data .
curl -fsSLO https://github.com/kachofugetsu09/akashic-agent/releases/latest/download/compose.yaml
curl -fsSLO https://github.com/kachofugetsu09/akashic-agent/releases/latest/download/chromium-seccomp.json
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
docker compose exec akashic python main.py pkg upgrade         # 全部升级（官方与各私有源）
docker compose exec akashic python main.py pkg upgrade --source release   # 只升级官方插件
docker compose exec akashic python main.py pkg upgrade --source fleet     # 只升级私有源 fleet
docker compose exec akashic python main.py pkg upgrade calendar           # 只升级一个
```

每个包只从它原来的来源升级，不会换到别的来源。某个私有源连不上时，只有它自己的包报错，其余照常升级。

### 私有源

和 [akashic-plugins](https://github.com/kachofugetsu09/akashic-plugins) 同样结构的仓库都可以当插件来源，可以登记多个，类似 pacman 的仓库。

1. 仓库结构：`plugins/<插件目录>/`（每个目录一个插件，插件仓库也可以用 git submodule 放进来），加上根目录的 `groups.toml`（每个插件归入一个组，格式见官方仓库）。
2. 发布：插件的前端已经构建好并提交在仓库里时，用官方仓库的 `scripts/publish_source.py --root . --bare <临时目录> --remote <仓库地址>`，生成 `index` 分支和每个插件一条 `release/<插件名>` 分支。可以放进 GitHub Actions，推送后自动发布。
3. 登记：

```bash
python main.py pkg source add fleet https://github.com/me/akashic-fleet.git --token <只读 token>   # 公开仓库不用 --token
python main.py pkg source list
python main.py pkg groups --source fleet
python main.py pkg install fleet/calendar      # 也可以只写 calendar：按来源顺序取第一个有它的
python main.py pkg source remove fleet         # 已装的包保留，只是不再随这个来源更新
```

- 来源名称就是插件 ID 的后缀（`calendar@fleet`），装过包以后不要改名。
- 私有仓库用 GitHub fine-grained token，只给这一个仓库的 Contents 只读权限。token 单独存放在插件目录的 `source-tokens/` 下，只在拉取时临时交给 git，不写进任何仓库配置。
- 同名包按来源顺序查找：官方源永远排第一，私有源按登记顺序。
- 已装的插件想换到另一个来源（比如私有源里的插件进了官方源）：`python main.py pkg move calendar@fleet --source release`。插件 ID 会变成 `calendar@release`，数据目录和插件自己存的状态一起迁过去；新来源装不上时旧数据不动，可以从原来源重装。来源名与原 ID 后缀相同时（例如 `pkg move calendar@github --source github`）ID 不变，只是改从这个来源重装，适合修正 `pkg list` 里来源显示成 `git:...` 的包。

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

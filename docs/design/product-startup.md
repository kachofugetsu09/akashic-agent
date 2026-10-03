# Product startup

Status: implementation; local acceptance is recorded in the pull request.

`./start` owns product preparation. It reports terminal progress, builds one committed distribution, installs its default profile through the formal installer, and starts that distribution's Supervisor. Core does not discover or install business plugins.

```text
┌───────────────────────────────────────┐
│ Terminal: product preparation         │
│ dependencies → distribution → install │
└──────────────────┬────────────────────┘
                   │ start Supervisor, wait for readiness
┌──────────────────▼────────────────────┐
│ Supervisor owns port 2236              │
│ Web shell + plugin gateway ready       │
└──────────────────┬────────────────────┘
                   │ print URL and open browser
┌──────────────────▼────────────────────┐
│ existing onboarding → Chat             │
└───────────────────────────────────────┘
```

Preparation status is process-local terminal output. The launcher waits for `chatReady` and a successful Web bootstrap before opening the browser. `--no-browser` keeps URL output for headless machines and Agents. Failures return a nonzero exit code with the log path; fix the cause and run the same command again. Ctrl+C or SIGTERM stops the child process group before releasing the state lock. The launcher remains attached while Supervisor runs. Compose shows the same progress in its foreground output or Docker Desktop logs; it never tries to open a browser inside the container.

## State and versions

- `--state` contains config.toml, workspace, plugin-home, startup.json and per-launch logs.
- `startup.json` identifies the launcher-owned first installation and its software commit. It does not hold plugin choices or business progress. It remains the historical first-install marker, not the currently deployed revision; the current distribution/runtime identity is authoritative. A new distribution composes its built-in code before startup; required Core/built-in Yoyo migrations finish before selection publication and runtime startup; external migrations are excluded.
- `workspace/runtime/distribution-install.json` remains the formal install receipt. Later starts retain this receipt unchanged as legacy source evidence, compose current distribution sources with selected external inputs, and preserve disabled or uninstalled choices. See [0082](../decisions/0082-distribution-owned-plugin-composition.md).
- A nonempty installation without the launch marker is not adopted automatically. Use its original entry or a separate empty state directory.
- Native build cache is scoped to the checkout and committed revision. Failed staging directories and logs remain for inspection; completed environments are not moved because virtual environments contain absolute paths.
- Source builds reject tracked local edits. They do not silently build an older HEAD while claiming to run those edits.

The standalone Compose image builds a fetched Git commit in a builder stage. It contains the Core archive and independent plugin bundles, with no business plugin source in Core. Default tools run inside the container; the Docker socket and host filesystem are not exposed. Plugins needing external Workload infrastructure require that infrastructure to be explicitly configured; the default profile does not start Computer.

仓库 Compose 构建明确指定的完整源码提交。Release 附件是独立 Compose，不包含 build context 或额外环境文件；镜像固定到已发布的多平台 manifest digest。两者复用同一份端口、healthcheck 和数据卷配置。发行工作流分别构建并实际启动 AMD64、ARM64 镜像，验证首次安装和保留数据卷重建后，组合这两个已验证镜像。匿名下载和发行附件启动验收通过后才发布 release。程序和发行制品归 root 所有，运行用户只读；运行数据写入 data 卷。发行不会部署 hua-home 或修改已有实例。

首次推送 GHCR 包后，包 owner 需要在 GitHub 的 package settings 中将该发行包设为 Public。工作流使用空 Docker config 验证匿名拉取；若公开访问未就绪，保留构建与验收证据，不创建面向使用者的 release。

## Development entry

For direct checkout development, prepare Python and Node dependencies, build the Web assets, and explicitly install the desired plugin distribution once. The product launcher is not a watcher for dirty business-plugin source.

```bash
uv venv
uv pip install -r requirements.txt -e sdk/python
npm ci
npm run build
uv run python main.py init
# Build and install a committed distribution into the chosen workspace/plugin home.
uv run python main.py
```

The full build/install API is `scripts/build_plugin_distribution.py --help` and `scripts/install_plugin_distribution.py --help`. Use explicit config, workspace and plugins-home paths for development isolation. Product starts and [operator deployment](operator-deployment.md) use the same distribution composition policy. Migration approval, backup and actual runtime readiness remain separate from the complete selection commit.

## Acceptance

Use isolated HOME, state, plugin home, cache, browser profile and Compose project. Verify the distribution commit, default profile, formal install receipt, selected runtime and actual Web bootstrap modules. Exercise cold start, warm start, failed preparation and rerun, port collision, stop/restart, and persistence. An HTTP 200 or a running container alone does not prove that the plugins loaded. Without model credentials, the browser must show onboarding and allow opening conversation and settings. Runtime UID must be nonzero; Core, dependencies and distribution sources must not be runtime-writable. Retained-volume recreation must preserve configuration, the install receipt and plugin selection.

维护者可用 Python 标准库运行 `python3 scripts/standalone_compose_smoke.py --image <本地镜像> --revision <完整 commit>`。它创建独立 project 和空数据卷，验证普通用户读取制品、网页与插件模块、重建容器后的配置和选择保留；不连接模型。日志留在输出目录，结束后只删除本次 project 的测试卷。

## Updating standalone Compose

源码构建先取得并切换目标已推送提交，将 `AKASHIC_REVISION` 重新设为 `git rev-parse HEAD`，再运行 `docker compose up -d --build --wait`。发行版使用者先停止服务并备份实例数据卷，在同一目录替换新发行版的 Compose，运行 `docker compose up -d --wait`。保持 project 和数据卷不变。旧镜像只恢复代码；已经发生的数据迁移需要升级前的匹配数据备份。不要使用 `down -v` 升级。

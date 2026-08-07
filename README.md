[![欢迎加入交流群](https://img.shields.io/badge/QQ%E4%BA%A4%E6%B5%81%E7%BE%A4-%E6%AC%A2%E8%BF%8E%E5%8A%A0%E5%85%A5-2ea44f?style=for-the-badge)](./COMMUNICATION.md)

# akashic Agent

一个**会主动找你**的 AI 伙伴——不只是被动回答问题，还能根据你订阅的信息源主动判断"现在该不该发消息、发什么"，在空闲时自主执行后台任务。

---

## 先装常用插件

如果你想让自己的 Akashic 具备和作者差不多的扩展能力，先看社区插件组织：

- <https://github.com/orgs/akashic-plugins/repositories>

很多能力现在都不是写死在主仓里，而是做成独立插件仓库，例如：

- `steam-mcp`
- `feed-mcp`
- `huayue-skills`

如果 Akashic 已经在运行，你通常可以直接像聊天一样让它安装：

```text
帮我安装这个插件试试看：
https://github.com/akashic-plugins/steam-mcp
```

或者更自然一点：

```text
steam mcp 我想用插件方式加载，你帮我把这个插件装一下看看能不能用：
https://github.com/akashic-plugins/steam-mcp
```

Akashic 理想上的动作应该是：

```text
┌─ 安装插件
│  ├─ 识别 GitHub 插件仓库
│  ├─ 执行 plugin-install
│  ├─ 检查 manifest.toml 与 plugin.py
│  └─ Runtime 自动发现并原子发布新快照
└─ 不重启，下一次执行使用新代际
```

安装、升级、启停、源码和 `config.local.toml` 修改都会自动热重载。正在执行的请求保持旧代际，新请求统一使用新代际；候选验证失败时继续保留旧版本。

想看完整机制，直接看 [插件系统 Handbook](./_handbook/plugins-tutorial.md)。

---

## Quickstart

需要 Python 3.12。

```bash
git clone <this-repo>
cd akashic-agent
uv venv && uv pip install -r requirements.txt
```

没有 uv？先 `pip install uv`。

**1. 启动 Akashic Web**

```bash
uv run python main.py
```

Supervisor 会始终提供唯一的本机 Web 入口：<http://127.0.0.1:2236>。访问后直接进入
Chat；没有模型配置时，Chat 会保留完整界面并引导进入“模型与认证”。

第一次运行不需要先创建 `config.toml`。打开设置中心，选择一种认证方式：

| 认证方式 | 适用场景 |
|---|---|
| API Key | 任意 OpenAI Chat Completions 兼容端点 |
| OpenCode Go | 粘贴 OpenCode Go Key，或复用本机已有的 OpenCode Go 登录 |
| Codex Auth | 复用本机 Codex 登录，未登录时按页面提示完成设备授权 |

```text
打开 2236 Chat
   │
   ├── 点击“连接模型”
   ├── 选择 Provider 与认证
   ├── 读取或填写模型
   ├── 发送最小真实请求验证
   └── 保存配置 → 同一页面自动恢复对话
```

API Key 会直接写入本机 `config.toml`，文件权限为 `0600`；设置 API 和页面不会回显
已经保存的密钥。切换 Provider 时，旧 runtime 会保留，切回来无需重新输入密钥。

OpenCode Go 会动态读取订阅当前提供的模型，隐藏已知走 Messages API 的型号，其余型号
默认按 Chat Completions 验证。因此新增 Chat Completions 型号通常不需要更新 Akashic。

**2. 可选：使用终端初始化或手动配置**

仍然可以使用原有命令：

```bash
uv run python main.py setup    # 交互向导
uv run python main.py init     # 非交互，CI/自动化用
```

当前主模型配置使用 named runtime。手动配置的最小示例：

```toml
[runtime]
workspace = "~/.akashic/workspace"

[llm]
main = "deepseek_main"

[llm.runtimes.deepseek_main]
provider = "deepseek"
model = "deepseek-v4-flash"     # 主模型：推理强、速度快、价格低
api_key = "sk-..."
base_url = "https://api.deepseek.com/v1"
enable_thinking = true          # 开启 reasoning
context_window = 128000
effective_context_percent = 0.9
compaction_trigger_percent = 0.74
max_output_tokens = 8192
input_modalities = ["text"]

[llm.runtimes.qwen_fast]
provider = "qwen"
model = "qwen-flash"            # 轻量模型：memory gate / query rewrite / HyDE
api_key = "sk-..."
base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
context_window = 128000
max_output_tokens = 4096
input_modalities = ["text"]

[memory]
enabled = true
engine = ""                     # 记忆引擎，留空 = default_memory 插件

[memory.embedding]
model = "text-embedding-v3"     # 向量模型
api_key = "sk-..."
base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"

[channels.telegram]
token = "123456:ABC..."
allow_from = ["your_username"]

[channels.chat]
enabled = true
channel_name = "web"
```

当前代码形状是新的迁移原点。启动时 Yoyo 只读取 `migrations/yoyo/`，并在
`<workspace>/migrations.sqlite3` 记录已成功执行的迁移；它不依赖 Git 历史、分支或版本号。
旧 Git cursor 时代的脚本保留为历史源码，但不会注册或自动执行，也不承诺接管旧格式。
原点迁移只删除退役的 `config.toml.migration-{cursor,lock,backups}` companion state，
不修改配置与业务数据。

新增迁移前请阅读 [Yoyo 迁移维护手册](./docs/design/git-migration-authoring.md)。已注册脚本
只追加不修改；修正错误时新增 migration ID。

`workspace` 默认是 `~/.akashic/workspace`。临时切换隔离环境时传
`--workspace PATH`；它的优先级高于 `AKASHIC_WORKSPACE` 和 `config.toml`。

**个人推荐**：主模型使用 DeepSeek，轻量和视觉任务使用 Qwen。通信渠道推荐
Telegram；只想先本机试用时，打开 2236 绑定模型后即可直接对话。

**3. 运行与安全切换**

无参数启动会先进入内置 supervisor，再由它启动正式 gateway。这样核心代码或主配置
确需完整重载时，Agent 可以通过当轮 `tool_search` 解锁 `agent_restart`，并在回复持久化、
送达和私有提交证据全部完成后安全拉起下一代进程。需要让调试器直接附着未托管 gateway
时，显式运行 `uv run python main.py gateway`；该模式不会注册自重启工具。

在 2236 的“模型与认证”切换 Provider、模型或默认角色时，Gateway 会原子发布新模型代际，不停止接收新
turn，也不重启进程。已经开始的执行继续使用旧代，下一个真正开始的执行使用新代；候选
配置或真实请求校验失败时保持原配置和当前代际。

从终端或 supervisor 切换到 PyCharm 前，先优雅停止当前 workspace 的 runtime：

```bash
./scripts/stop-runtime.sh
```

脚本遵循 `--workspace`、`AKASHIC_WORKSPACE`、`config.toml` 的 workspace
优先级，优先停止 supervisor，并等待 runtime 真正释放实例锁。它不会删除锁文件，
也不会在超时后自动强制终止进程。PyCharm 仍直接运行 `main.py`，默认同样进入
supervisor；需要直接调试 child 时把程序参数设为 `gateway`。也可以把
`scripts/stop-runtime.sh` 配置为 Run Configuration 的 Before Launch external tool。

如果配置了 Telegram / QQ，也可以直接给 bot 发一条消息开始对话。

---

## 用 Android 手机接入

Akashic Mobile 是一个通过独立实时网关连接 Akashic Agent 的 Android 客户端。远程接入推荐使用 Cloudflare Tunnel：Web Chat 和模型设置继续留在本机 `127.0.0.1:2236`，Tunnel 只转发由 Akashic 设备认证保护的 `6323` 端口。

```text
1. 在 config.toml 启用 [mobile_realtime]
2. 用 Cloudflare Tunnel 把一个公共域名转到 https://127.0.0.1:6323
3. 在本机 Web Chat 点击“连接手机”，用 Akashic Mobile 扫描二维码
4. 两端核对六位确认码，在电脑上批准设备
```

- Android 安装包：<https://github.com/kachofugetsu09/akashic-mobile/releases/latest>
- 配置、Cloudflare、验证与排障：[移动端接入手册](./_handbook/mobile-access.md)

首次配对成功后，手机会保存设备密钥，正常升级应用或重连无需再次扫码。

### 把前端改动更新到移动端

Android 的对话界面与 Web Chat 共用 `frontend/chat/src`。只修改 React、CSS 或插件插槽时，
不需要重新打包 APK；服务端把构建结果发布成不可变 WebUI generation，支持 OTA 的客户端会
下载、校验并切换到所选频道。只有原生壳、Native Bridge 协议或最低原生 build 发生变化时
才需要发布新的 APK。

先从发布仓读取当前服务身份，并为指针和可达资源创建恢复点：

```bash
AKASHIC_WEBUI_SERVER_ID="$(sqlite3 -readonly \
  ~/.akashic/workspace/mobile-webui/publication.sqlite3 \
  "SELECT value FROM webui_meta WHERE key = 'server_id'")"

.venv/bin/python scripts/publish-mobile-webui.py backup \
  --workspace ~/.akashic/workspace \
  --server-id "$AKASHIC_WEBUI_SERVER_ID" \
  --destination ~/.akashic/backups/mobile-webui-"$(date +%Y%m%d-%H%M%S)"
```

开发中的 dirty 前端只能发布到 Preview，适合在配置为 Preview 频道的真机上验收：

```bash
.venv/bin/python scripts/publish-mobile-webui.py publish \
  --source-repository "$PWD" \
  --workspace ~/.akashic/workspace \
  --server-id "$AKASHIC_WEBUI_SERVER_ID" \
  --allow-dirty \
  --actor local-preview
```

合并后切到最新且干净的 `main`，再从确定的 commit 发布 Stable；普通设备随后会通过 OTA
取得该 generation：

```bash
git checkout main
git pull --ff-only origin main
test -z "$(git status --porcelain)"

AKASHIC_WEBUI_SOURCE_COMMIT="$(git rev-parse HEAD)"
.venv/bin/python scripts/publish-mobile-webui.py publish \
  --source-repository "$PWD" \
  --workspace ~/.akashic/workspace \
  --server-id "$AKASHIC_WEBUI_SERVER_ID" \
  --source-commit "$AKASHIC_WEBUI_SOURCE_COMMIT" \
  --stable \
  --actor local-stable
```

用 `publish-mobile-webui.py inspect` 核对 Stable/Preview 的 generation、协议窗口和
`minimum_native_build`。发布只更新 WebUI 发布仓，不会改写会话、记忆或插件数据。

---

## 系统全景

```
你的消息 → [被动回复] ──→ agent loop ──→ 回复
                │
                ├── 记忆系统 ─── 每轮注入长期记忆 + 对话后 consolidation
                │
                └── 插件系统 ─── 拦截命令、注入协议、阻断工具、挂载新工具...

[主动推送] ──→ 定期轮询 ──→ 三路数据 (alert/content/context) ──→ LLM 决策 ──→ 推送或跳过
                │
                └── [Drift] ──→ 没东西推时执行后台任务 (SKILL.md)
```

| 想看什么 | 文档 |
|---------|------|
| 怎么首次配置或切换 Provider | 启动后访问 `http://127.0.0.1:2236/settings`，支持 API Key、OpenCode Go 和 Codex Auth |
| 怎么打开本机 Web Chat | 启动后访问 `http://127.0.0.1:2236`；没有模型时页面会直接引导配置 |
| 怎么用 Android 手机远程连接 | [移动端接入手册](./_handbook/mobile-access.md) |
| 怎么让 agent 主动推送消息、怎么配数据源 | [_handbook/proactive-guide.md](./_handbook/proactive-guide.md) |
| 怎么写后台任务让 agent 空闲时自动干活 | [_handbook/drift-guide.md](./_handbook/drift-guide.md) |
| MEMORY.md / SELF.md / consolidation / 记忆怎么流转 | [_handbook/memory-markdown.md](./_handbook/memory-markdown.md) |
| 怎么写插件介入生命周期、注册工具 | [_handbook/plugins-tutorial.md](./_handbook/plugins-tutorial.md) |

---

## 被动回复

收到消息 → 记忆检索 → 工具调用 → 流式回复。每轮经过 6 个 Phase（BeforeTurn → BeforeReasoning → PromptRender → Reasoner → AfterReasoning → AfterTurn）。

插件有 **4 种介入方式**：PhaseModule 链（7 个 Phase 方法 + slot 依赖声明）、EventBus 装饰器（9 种事件）、`@on_tool_pre`（工具拦截）、`@tool`（注册工具）。见 [插件系统](./_handbook/plugins-tutorial.md)。

## 主动推送（Proactive）

Agent 根据电量模型自适应调整轮询频率——你刚聊完时不烦你（8 分钟一次），半天没动静就加速到 1 分钟一次。每轮拉取三路 MCP 数据：

- **alert** — 高优先级告警，直接透传
- **content** — 内容流，逐条 LLM 评分分类
- **context** — 背景上下文，概率注入做 fallback

见 [Proactive 配置指南](./_handbook/proactive-guide.md)。

## 记忆系统

对话通过 **consolidation** 自动提取为结构化事实：HISTORY.md（时间线事件） + PENDING.md（待归档缓冲） + RECENT_CONTEXT.md（近期上下文摘要）。**Optimizer** 定时将 PENDING 归档到 MEMORY.md——中间隔一层是为了保护 prompt cache（MEMORY.md 全文注入 system prompt，高频修改会破坏缓存）。同时 `memory2.db`（向量层）提供语义检索。

见 [记忆系统](./_handbook/memory-markdown.md)。

## Drift 空闲任务

没内容可推时 agent 不空转——执行你写的 `SKILL.md`（分步操作指南），比如审计长期记忆是否准确、补用户画像、自我诊断。

见 [Drift 指南](./_handbook/drift-guide.md)。

---

## 其他命令

```bash
uv run python main.py exec --new --final-only "总结最近上下文"
uv run python main.py app-server --stdio # 父进程托管 JSON-RPC app-server
uv run python main.py dashboard # 单独运行 Dashboard 调试入口
# 正式 Supervisor 只提供 http://127.0.0.1:2236，根页面是统一壳层并默认选中 Chat
uv run python main.py --help    # 查看全部子命令

pytest tests/
akashic_RUN_SCENARIOS=1 pytest -c pytest-scenarios.ini tests_scenarios/
```

## 工作区

所有运行时数据都在 `[runtime].workspace` 指定的目录下。默认值是
`~/.akashic/workspace`；可设置 `AKASHIC_WORKSPACE`，也可以为单条命令传入
`--workspace /absolute/path`。优先级为 `--workspace`、`AKASHIC_WORKSPACE`、
`config.toml`。不同测试环境使用不同目录，不共享会话、记忆、附件或插件数据。
插件代码缓存和启停清单默认仍位于 `$HOME/.akashic-plugin`；需要完整隔离插件安装状态时，
额外设置 `AKASHIC_PLUGIN_HOME=/absolute/test/plugin-home`。

从旧版升级时，第一次重启前显式复制旧插件数据；命令保留旧目录，目标已存在时拒绝覆盖：

```bash
uv run python scripts/migrate_plugin_data.py \
  --workspace "$HOME/.akashic/workspace" \
  --plugins-home "$HOME/.akashic-plugin"
```

程序化客户端连接 workspace 下的 `akashic.sock`，先完成 JSON-RPC
`initialize`/`initialized`，再使用 `thread/start`、`turn/start`、`turn/read` 和
`turn/interrupt`。Python SDK 位于 `sdk/python/`；旧 TUI 和无 request id 的 IPC payload
已删除，不提供兼容 fallback。

完整配置、协议和回滚说明见[程序化控制面迁移指南](./_handbook/programmatic-control-migration.md)。

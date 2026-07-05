---
name: plugin-system
description: 说明并执行 Akashic 插件系统的安装、加载、启停、配置、MCP、skill、lifecycle 与 registry。用户提到 插件, plugin, marketplace, 安装插件, GitHub 插件, 启用插件, 禁用插件, 更新插件, 卸载插件, mcp, skill, lifecycle, registry, 插件仓库, 插件目录 时使用。
when_to_use: 用户在问 Akashic 插件怎么装、怎么加载、怎么停、放哪、数据在哪、MCP/skill/lifecycle 怎么接入，或直接要求安装/启用/禁用/排查某个插件时。
metadata: {"akashic": {"always": false}}
---

# Akashic 插件系统说明

这个 skill 不只是解释，还要尽量直接完成插件管理动作。

先看这两个地方：

1. 全局注册表：`~/.akashic-plugin/registry.json`
2. 运行配置：`config.toml` 里的 `config.plugins.*`

## 当前插件目录

```text
~/.akashic-plugin/
├─ cache/<marketplace>/<plugin>/<version>/
├─ data/<plugin>-<marketplace>/
└─ registry.json
```

- `cache/` 放插件代码包
- `data/` 放插件状态数据
- `registry.json` 是当前插件系统总表

## registry.json 看什么

每个插件条目至少会有这些字段：

- `plugin_id`
- `source_type`：`builtin` 或 `installed`
- `plugin_root`
- `data_dir`
- `enabled`
- `local_disabled`
- `active`
- `capabilities.lifecycle`
- `capabilities.skills`
- `capabilities.mcp`
- `skills`
- `drift_skills`
- `mcp_servers`

判断插件状态时优先这样看：

1. `enabled=false`：配置上被禁用
2. `local_disabled=true`：插件目录里有本地禁用标记
3. `active=false`：当前进程没真正加载成功

## 你要怎么行动

用户不是来听架构课的。只要请求足够明确，优先直接执行。

### 安装插件

如果用户给了 GitHub 仓库链接或本地仓库路径：

1. 先确认这是一个插件仓库，至少要有 `.aka-plugin/plugin.json`
2. 执行：

```bash
python main.py plugin-install --source <repo_or_github_url> --marketplace github
```

如果是本地仓库路径，也可以直接作为 `--source`。

安装后必须做这些检查：

1. 读 `~/.akashic-plugin/registry.json`
2. 确认有对应 `plugin_id`
3. 确认 `plugin_root`、`skills`、`mcp_servers` 是否符合预期
4. 明确告诉用户“默认需要重启主进程后生效”

### 启用插件

如果用户要启用插件：

1. 先从 `registry.json` 找到准确 `plugin_id`
2. 修改 `config.plugins.<plugin_id>.enabled = true`
3. 如果用户只想开某个能力，就改：
   - `capabilities.lifecycle.enabled`
   - `capabilities.skills.enabled`
   - `capabilities.mcp.enabled`
4. 告知重启后生效

### 禁用插件

如果用户要禁用插件：

1. 先定位 `plugin_id`
2. 修改 `config.plugins.<plugin_id>.enabled = false`
3. 如果只是禁用单个 MCP server，就改：
   - `config.plugins.<plugin_id>.mcp_servers.<server_name>.enabled = false`
4. 告知重启后生效

### 排查插件

如果用户说插件没生效，按这个顺序查：

1. `~/.akashic-plugin/registry.json`
2. `config.toml`
3. `~/.akashic/workspace/skills`
4. `~/.akashic/workspace/mcp_servers.json`
5. 运行日志

优先直接跑：

```bash
python main.py plugin-doctor <plugin_id>
```

不要先猜，先看 registry。

## 安装方式

Git 仓库插件：

```bash
python main.py plugin-install --source <git_url_or_local_path> --marketplace <name>
```

常用可选参数：

- `--ref <branch_or_tag_or_commit>`
- `--sparse path1,path2`

安装完成后只代表代码和 registry 就位，不代表当前运行中的进程已经拿到新插件。

## 加载方式

Akashic 现在的插件加载点在进程启动时。

启动流程：

```text
启动
├─ PluginManager 扫描 builtin plugins
├─ PluginManager 扫描 ~/.akashic-plugin/cache
├─ 按 config.plugins.* 判断 enabled
├─ 加载 lifecycle
├─ 同步 plugin skill -> workspace/skills 软链接
├─ 启动 plugin MCP servers
└─ 写回 ~/.akashic-plugin/registry.json
```

现在的真实约束：

- lifecycle 不支持热重载
- skill 不支持热重载
- MCP 也按启动流程接入
- 新装插件、改配置、改 lifecycle/skill/MCP 声明后，默认要重启主进程

## 启用与禁用

主开关：

```toml
[plugins."<plugin_id>"]
enabled = true
```

能力级开关：

```toml
[plugins."<plugin_id>".capabilities.lifecycle]
enabled = true

[plugins."<plugin_id>".capabilities.skills]
enabled = true

[plugins."<plugin_id>".capabilities.mcp]
enabled = true
```

单个 MCP server 开关：

```toml
[plugins."<plugin_id>".mcp_servers."<server_name>"]
enabled = false
```

如果仓库里已经有旧式 workspace MCP 或旧 skill 链路，要顺手检查是否冲突。

## skill 接入

- 插件自己的 `skills/` 会被软链接到 `~/.akashic/workspace/skills/`
- `.aka-plugin` 普通 skill 用裸名
- drift skill 仍保留带前缀名字空间

先查 `registry.json` 的 `skills` / `drift_skills`，再去 `workspace/skills` 看是否已经出现对应目录。

## MCP 接入

- 插件在 `.aka-plugin/plugin.json` 里声明 `paths.mcp_servers`
- 启动时会合入 MCP runtime
- 如果插件 MCP 需要独立数据目录，会通过 `AKA_PLUGIN_DATA_DIR` 注入

先查 `registry.json` 的 `mcp_servers`，再查运行日志里是否连接成功。

## lifecycle 接入

- `.aka-plugin/plugin.json` 里声明 `akashic.lifecycle.entry` 和 `class`
- 启动时才会真正 import 并绑定到生命周期

如果 `capabilities.lifecycle=true` 但 `active=false`，优先检查导入失败、配置无效、初始化失败。

## 排查顺序

```text
插件不生效
├─ 先看 ~/.akashic-plugin/registry.json
│  ├─ 没条目 -> 没安装或没被发现
│  ├─ enabled=false -> 配置禁用了
│  ├─ active=false -> 加载失败
│  └─ capabilities.*=false -> 某能力被关了
├─ 再看 config.toml
├─ 再看 workspace/skills 是否有软链接
├─ 再看 MCP 日志是否连上
└─ 最后重启主进程
```

## 执行规则

当用户要管理插件时，默认按这个顺序做：

1. 先读 `~/.akashic-plugin/registry.json`
2. 确认目标 `plugin_id`
3. 优先执行 `python main.py plugin-doctor <plugin_id>`
4. 如果 doctor 还不能解释清楚，再继续看软链接、MCP、日志
5. 如果是安装请求，直接执行 `python main.py plugin-install ...`
6. 如果是启停请求，直接修改 `config.plugins.<plugin_id>`
7. 如果发现旧 `workspace/mcp_servers.json` 里有同名 server，提醒冲突或协助迁移
8. 明确告知“当前插件系统默认需要重启后生效”

## 不要这样做

- 不要只讲概念，不落到具体文件和命令
- 不要看到 GitHub 链接后只说“可以安装”，要真的去装
- 不要跳过 `registry.json` 校验
- 不要隐瞒“需要重启”这个事实

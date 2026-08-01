---
name: plugin-system
description: 说明并执行 Akashic 插件安装、加载、启停、配置、插件内 MCP、skill、生命周期与 manifest 管理。
when_to_use: 用户询问或要求处理 Akashic 插件、marketplace、插件自带 MCP、skill、插件配置、安装、更新、启用、禁用或排障时。独立本地 MCP server 使用 manage-workspace-mcp。
metadata: {"akashic": {"always": false}}
---

# Akashic 插件系统

优先直接完成明确的插件管理请求，并在修改后验证。

独立 binary、脚本或本地项目需要作为 MCP 常驻时，加载 `manage-workspace-mcp`；
不要为它创建插件，也不要修改主 `config.toml`。

## 事实来源

```text
┌─ ~/.akashic-plugin/manifest.toml
│  └─ 全局安装清单与 enabled
├─ ~/.akashic-plugin/cache/<marketplace>/<plugin>/<version>/plugin.py
│  └─ 插件能力声明与代码
└─ <workspace>/plugin-data/<plugin>-<marketplace>/config.local.toml
   └─ 插件配置和持久状态
```

不要查找或创建 `registry.json`、`.aka-plugin/plugin.json`、`manifest.yaml` 或插件级 `mcp/servers.json`。

## 安装

仓库根目录必须有 `plugin.py`，其中声明 `Plugin` 子类、`name` 与 `version`。

```bash
python main.py plugin-install --source <repo_or_url> --marketplace github
```

安装后检查 manifest、cache、data，并运行：

```bash
python main.py plugin-doctor <name>@github
```

## 更新已有插件

不要直接修改 `~/.akashic-plugin/cache`。先修改插件的可编辑源码仓库；个人插件通常位于 `/mnt/data/coding/akashic-plugin/<plugin-name>`。如果只知道已安装插件而找不到源码仓库，先确认其 Git remote 或向用户询问。

`plugin-install` 即使接收本地仓库路径，也会执行 `git clone`，只安装已提交的 Git HEAD，不会复制工作区里的未提交文件。必须先提交；需要从 GitHub 更新时，还必须先推送，再使用 GitHub source 安装。

```text
┌─ 修改插件源码仓库
├─ 运行插件自身测试
├─ 提交并推送源码
├─ 用原 source 与 marketplace 再次执行 plugin-install
├─ watcher 自动准备并发布新代际
├─ 运行 plugin-doctor 检查结构与配置
└─ 发起一次新请求验证真实行为
```

重新执行 `plugin-install` 即更新：它替换 cache 中的已安装版本，但保留 data 与配置。运行中的 watcher 会自动热重载，不要重启 Agent。

如果用户指定把现有项目中的 skill 收入某个插件，应复制或适配到该插件源码的 `skills/<skill-name>/`，再走上述更新流程；不要改写原项目，也不要先落到 workspace。若 skill 依赖外部项目或 CLI，把它安装到用户指定目录或稳定的数据目录，禁止让 wrapper、符号链接或服务依赖 `/tmp`。

## 完成判定

工具调用成功只代表命令退出码为零，不代表更新目标成立。汇报完成前必须从最终状态重新验收：

```text
┌─ 源码仓库
│  ├─ 目标文件存在
│  ├─ 预期改动已 commit
│  └─ 要求发布时，远端已包含该 commit
├─ 安装缓存
│  └─ 目标能力的具体文件或声明确实存在
├─ Runtime
│  ├─ candidate 已通过 Gate，snapshot 已发布
│  └─ 新请求能实际使用目标能力
└─ 外部依赖
   ├─ CLI 从稳定目录运行
   └─ 用户要求常驻服务时，health 返回健康
```

`plugin-doctor` 只证明插件结构、根目录与声明可加载，不证明某个具体 skill 已安装，也不能代替真实行为验证。不要用中途检查、手动改 cache 后的结果或 doctor healthy 推断最终成功。

删除类操作使用对应章节的 absence oracle，不套用上面的“目标文件存在”判定。

## 启用与禁用

使用管理命令修改 `manifest.toml` 对应条目的 `enabled`。运行中的 watcher 会自动完成启停，不需要重启进程。

```bash
python main.py plugin-disable demo@github
python main.py plugin-enable demo@github
```

## 卸载

```bash
python main.py plugin-uninstall demo@github
```

执行前记录对应的 manifest 条目、cache 路径，以及 plugin-data 是否存在。卸载命令返回“卸载已安排”或 operation ID，只表示请求已受理，不表示已经完成。

当前 turn 可能仍持有包含该插件的 runtime snapshot lease。此时 `enabled = false` 且 cache 仍存在表示正在排空，不是失败，也不是完成。不要在同一 turn 反复等待或再次执行 `plugin-uninstall`；先明确报告“卸载已安排，正在排空”，在后续 turn 从最终状态重新验收。

只有同时满足以下条件，才能报告“卸载完成”：

```text
┌─ ~/.akashic-plugin/manifest.toml 不再包含该 plugin ID
├─ ~/.akashic-plugin/cache/<marketplace>/<plugin>/ 不存在
└─ 卸载前已存在的 <workspace>/plugin-data/<plugin>-<marketplace>/ 仍存在
```

验收命令必须让任一条件不满足时返回非零退出码；不要用末尾的 `echo` 或无条件成功命令掩盖失败。后续 turn 仍停在排空状态或 operation 明确失败时，检查 operation 与 runtime 日志；不要手动删除 cache、data、配置、Token、数据库或模型。

## 配置

读取插件 `plugin.py` 的 `ConfigModel`，再编辑对应数据目录的 `config.local.toml`。不要把插件配置写回主 `config.toml`。

## 能力排查

```text
┌─ skills
│  └─ 检查 skill_roots() 与 drift_skill_roots()
├─ MCP
│  └─ 检查 mcp_servers()、入口、requirements.txt 与 .venv
├─ proactive
│  └─ 检查 proactive_sources() 与插件配置的 enabled
└─ lifecycle
   └─ 检查 initialize()、terminate() 和运行日志
```

插件能力全部由代码声明，公共 runtime 不应出现具体插件名或业务路径特判。

## 热重载验证

`plugin-install` 成功只代表文件就绪；运行时发布由 watcher 异步完成，本 turn 开始时绑定的 runtime snapshot 不含新能力是正常现象。不要用当前 turn 的 `tool_search` 结果、`exec --new` 新进程或反复重试安装来判断成败。

安装或修改 manifest.toml/config.local.toml 后，等待 watcher 完成一轮扫描（≤10 秒），然后查询 reload journal 最新事务的终态：

```bash
sqlite3 <workspace>/runtime/plugin-reloads.sqlite3 \
  "SELECT phase, error FROM reload_transactions WHERE plugin_id='<name>@<marketplace>' ORDER BY started_at DESC LIMIT 1;"
```

按终态收尾并停止：

```text
┌─ complete → 候选已通过 Gate、snapshot 已发布
│  报告“安装成功，热重载已发布”，停止；下一轮新消息即可使用新能力
├─ aborted  → 读取 error 与 manager 日志中“候选验证失败”的 gate 行
│  报告失败原因，停止；重试或修复留到后续 turn
└─ 仍处于 preparing/validating/committed → 再等 ≤5 秒重查一次
   仍不推进则报告“热重载未推进”，停止
```

同一 turn 内 journal 已到 `complete` 或 `aborted` 后，不再重复执行安装或验证命令；`tool_search` 只用于下一轮确认新工具可见，`exec --new` 是启动全新进程，不能验证热重载发布。

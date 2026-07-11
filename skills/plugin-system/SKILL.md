---
name: plugin-system
description: 说明并执行 Akashic 插件安装、加载、启停、配置、MCP、skill、生命周期与 manifest 管理。
when_to_use: 用户询问或要求处理 Akashic 插件、marketplace、MCP、skill、插件配置、安装、更新、启用、禁用或排障时。
metadata: {"akashic": {"always": false}}
---

# Akashic 插件系统

优先直接完成明确的插件管理请求，并在修改后验证。

## 事实来源

```text
┌─ ~/.akashic-plugin/manifest.toml
│  └─ 全局安装清单与 enabled
├─ ~/.akashic-plugin/cache/<marketplace>/<plugin>/<version>/plugin.py
│  └─ 插件能力声明与代码
└─ ~/.akashic-plugin/data/<plugin>-<marketplace>/config.local.toml
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

卸载删除 manifest 条目与 cache，始终保留插件 data。不要手动删除 data、配置、Token、数据库或模型。

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

```text
┌─ plugin-install 更新 cache，或修改 manifest.toml/config.local.toml
├─ 等待 watcher 生成候选代际
├─ 检查 candidate 与 snapshot Gate 日志
├─ 发起一次新请求验证新能力
└─ 确认旧 MCP、任务、Channel 与服务已排空
```

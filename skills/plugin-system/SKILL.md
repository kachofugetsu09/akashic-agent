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
python main.py plugin-doctor --plugin <name>@github
```

## 启用与禁用

只修改 `manifest.toml` 对应条目的 `enabled`。启停后提醒用户重启运行进程。

```toml
[plugins."demo@github"]
enabled = false
```

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

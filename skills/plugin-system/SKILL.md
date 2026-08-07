---
name: plugin-system
description: 说明并执行 Akashic 插件安装、加载、配置、插件内 MCP、Skill、生命周期、卸载与 turn 边界更新。
when_to_use: 用户询问或要求处理 Akashic 插件、marketplace、插件自带 MCP、Skill、插件配置、安装、更新、卸载或排障时。独立本地 MCP server 使用 manage-workspace-mcp。
metadata: {"akashic": {"always": false}}
---

# Akashic 插件系统

优先直接完成明确的插件请求。创建或改写源码、加入 Skill/MCP、递归验证候选时，先加载 `develop-akashic-plugin`。用户要管理不属于插件的独立本地 MCP server 时，加载 `manage-workspace-mcp`。

## 事实来源

```text
┌─ ~/.akashic-plugin/manifest.toml
│  └─ 全局安装清单
├─ ~/.akashic-plugin/cache/<marketplace>/<plugin>/.artifacts/
│  └─ 不可变 installed code 与 stable/latest 内部 pointer
├─ <workspace>/plugin-data/<plugin>-<marketplace>/
│  └─ 插件配置和持久状态
└─ <workspace>/runtime/plugin-reloads.sqlite3
   └─ Core reload 与恢复证据
```

不要查找或创建 `registry.json`、`.aka-plugin/plugin.json`、`manifest.yaml` 或插件级 `mcp/servers.json`。不要直接编辑 cache、pointer、manifest 或正式 plugin-data。

## Agent 可用动作

Agent 只使用：

```text
plugin-install    安装或更新本 turn 的候选
plugin-uninstall  登记本 turn 结束后的卸载
plugin-revert     撤销本 turn 最近一次尚未提交的操作
```

不要调用 status、promote、discard、enable、disable 或手工 restart。它们不是 Agent 更新流程的一部分；stable/latest、排空、提交和恢复由 Core 管理。

## 安装与更新

仓库根必须有 `plugin.py`，Plugin 子类声明 `name` 和 `version`。只安装已提交 Git HEAD；远程 source 必须先 push 对应 commit。

```bash
python main.py plugin-install --source <repo_or_url> --marketplace github
```

命令必须从 active Agent turn 的 Shell 发起。成功返回表示：候选已准备；当前父 turn 仍使用原版本；本 turn 的 attached programmatic child 自动使用候选；通过后正常结束父 turn，Core 自动切换，下一 turn 生效。

```text
修改 canonical source → source tests → commit/push → plugin-install
       → attached child 真实行为验证
       ├─ pass → 正常结束父 turn → Core 自动切换
       └─ fail → plugin-revert → 修复后递归
```

`plugin-doctor` 只是人工诊断，不能证明新行为正确，也不能代替 child trace。

## 卸载与撤销

```bash
python main.py plugin-uninstall demo@github
python main.py plugin-revert
```

卸载成功返回表示意图已绑定当前 turn，不表示代码已经删除。告诉用户：当前 turn 可以完成；本轮结束后 Core 自动停止 endpoint、移除能力和 installed code；plugin-data 保留；下一 turn 不再加载。

`plugin-revert` 只撤销同一 turn 最近一次未提交 install/uninstall，不能跨 turn 回滚历史版本。不要反复卸载、轮询、手改 manifest/cache 或删除 plugin-data。

下一用户 turn 会收到 Core 的自然语言运行事实。卸载完成必须满足 manifest entry 和 cache 已移除、原 plugin-data 仍存在；清理失败时说明残留路径和错误，不能假报完整成功。

## 独占 endpoint

固定 listener 的 managed service 必须声明 `ManagedServiceSpec.validation_port_env`，服务进程和同插件 MCP 必须读取同名环境变量。Core 会复制 plugin-data 到隔离验证目录、分配临时 loopback 端口并验证 readiness。忽略变量、缺少声明、端口冲突或 readiness 失败都必须暴露。

Channel 的 candidate 不接管正式 bot token/webhook/long-poll ownership。父 turn 结束后由 Core 统一执行：

```text
old Channel.stop → managed service switch → new Channel.start
        └──────── 任一步失败：恢复并验证 old generation
```

`stop()` 返回即承诺新 ingress 已停止、在途工作已收束且 ownership 已释放；`start()` 返回即承诺 ownership 已取得并 ready。

## 配置与排障

读取插件 `ConfigModel`，配置只写对应 plugin-data，不写主 `config.toml`。缺失依赖、导入失败、配置错误、command 失败和数据损坏必须 fail-loud。

```text
┌─ Skill      检查 skill_roots()/drift_skill_roots() 与真实触发轨迹
├─ MCP        检查 mcp_servers()、入口、依赖、候选 endpoint env
├─ service    检查 process identity、listener、readiness 和恢复
├─ Channel    检查 ingress/ownership 的 stop/start 证据
└─ rollout    检查 child terminal/tool trace 与 reload journal
```

插件能力全部由通用代码声明，Core 不应出现具体插件名或业务路径特判。

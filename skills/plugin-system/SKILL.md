---
name: plugin-system
description: 说明并执行 Akashic v3 插件安装、加载、配置、插件内 MCP、Skill、生命周期、卸载与 turn 边界更新。
when_to_use: 用户询问或要求处理 Akashic 插件、marketplace、插件自带 MCP、Skill、插件配置、安装、更新、卸载或排障时。所有 MCP server 都必须作为 v3 插件声明并经 Core generation host 发布。
metadata: {"akashic": {"always": false}}
---

# Akashic 插件系统

优先完成明确的插件请求。创建或改写 source、加入 Skill/MCP、安装候选或递归验证时，先加载 `develop-akashic-plugin`；这里只负责安装链、状态边界和 turn 结果。

## 事实来源与边界

```text
┌─ ~/.akashic-plugin/manifest.toml
│  └─ 全局安装清单与启用状态
├─ ~/.akashic-plugin/cache/<marketplace>/<plugin>/.artifacts/
│  └─ 不可变 installed artifact 与 generation pointer
├─ <workspace>/plugin-data/<plugin>-<marketplace>/
│  └─ 插件配置与持久状态
└─ <workspace>/runtime/plugin-reloads.sqlite3
   └─ reload、candidate、恢复与 turn rollout 证据
```

外部 source 的 canonical 根必须包含 `akashic.plugin.toml` 和其 `entrypoint` 指向的 v3 module。manifest 的 `schema_version=1`、`api_version=3`、`name`、`version`、`entrypoint` 在 import 前校验；module 还必须导出同值的 `api_version/name/version` 与精确 `apply(ctx, config)`。Skill/MCP 通过 source 的静态 root 和 typed service 进入 generation catalog。

不要查找或创建 `registry.json`、`.aka-plugin/plugin.json`、`manifest.yaml`、插件级 `mcp/servers.json` 或 workspace 手工 Skill owner。不要直接编辑 cache、pointer、全局 manifest、workspace Skill 软链接或正式 plugin-data。

主动信息源也只组合普通服务：`TIMERS` 驱动来源私有 poll，离散事实提交 `CONTENT`，当前状态保留在插件私有 cache，候选行动进入 `DRIFT`，完整推理由 `BACKGROUND_JOBS` 创建普通 Turn。不要创建 proactive catalog、私有 lifecycle family 或 MCP 聚合桥。

## Agent 可用动作

```text
plugin-install    安装或更新本 turn 的候选
plugin-uninstall  登记本 turn 结束后的卸载
plugin-revert     撤销本 turn 最近一次尚未提交的操作
```

不要手工编辑 manifest/cache/pointer，不要手工切换 generation、重启 Gateway 或用第二个进程绕过 turn 边界。stable、candidate、lease、排空、提交、恢复、能力投影和服务切换由 Core 拥有。

## 安装与更新

只从已提交 Git HEAD 安装；远程 source 必须先 push 对应 commit：

```bash
python main.py plugin-install \
  --source <repo_or_url> \
  --marketplace <marketplace>
```

正常链路：

```text
source test → commit/push → plugin-install
       → attached child 的真实行为 oracle
       ├─ pass → 正常结束父 turn → Core 自动提交 → 下一 turn 生效
       └─ fail → plugin-revert → 修复 source 后递归
```

安装成功只表示候选准备并绑定当前父 turn；父 turn 仍使用旧 stable，attached child 自动使用候选。子 turn 不指定 runtime、不 detach、不直接选择 candidate。结果必须保存 candidate identity、reload transaction、child identity、tool trace 和 terminal；只看命令返回或 catalog 不足以证明功能。

## Skill、MCP 与服务检查

```text
┌─ Skill
│  └─ source root、SKILL.md、references、catalog source、真实触发轨迹
├─ MCP
│  └─ manifest command、requirements、required tools、candidate read-only tools、endpoint env
├─ managed service
│  └─ process identity、port_env、readiness、退出与隔离 plugin-data
├─ Channel
│  └─ descriptor、credential paths、stop/start ownership 与恢复
└─ rollout
   └─ child terminal、tool items、reload journal、turn 后 generation
```

固定 listener 必须在 manifest 的 `[[processes]]` 中声明 `port_env`、`formal_port`、`readiness_path` 和超时；module 的 typed `ManagedProcessDefinition` 必须与其一致，服务进程和同插件 MCP 必须真正读取注入端口。候选验证使用隔离端点和数据副本，写型 Tool/MCP 仅在事务、dry-run、隔离目标或明确授权下执行。

Channel candidate 不复制正式 token、webhook 或 long-poll ownership。父 turn 结束后的顺序是：

```text
old Channel.stop → managed service switch → new Channel.start
       └─ 任一步失败：恢复并验证 old generation
```

`stop()` 返回必须证明 ingress、在途工作和 ownership 已收束；`start()` 返回必须证明新 generation ready。验证结果不能把 endpoint 试运行写成正式外部切换。

## 卸载与撤销

```bash
python main.py plugin-uninstall demo@github
python main.py plugin-revert
```

卸载成功只表示意图已登记当前 turn；当前 turn 可完成，结束后 Core 停止 endpoint、移除能力投影和 installed code，并保留 `<workspace>/plugin-data/<plugin>-<marketplace>/`。下一 turn 核对 manifest entry、artifact/cache、process/socket 清理和 plugin-data 仍在；清理失败必须报告实际残留和错误。

`plugin-revert` 只撤销同一 turn 最近一次尚未提交的 install/uninstall，不能跨 turn 回滚已发布版本。不要反复安装、轮询、删除 plugin-data 或借手工文件修改伪造恢复。

## 配置与排障

插件 `Config` 是 module namespace 中可选的 typed 配置模型；配置只写对应 plugin-data，不改主 `config.toml`。缺少依赖、导入失败、manifest/module 不一致、配置错误、命令失败、readiness 失败和数据损坏必须 fail-loud。

只有以下情况才进入 runtime diagnostics：子 turn 长时间 queued、超时、terminal 错误、candidate identity 不一致、cleanup 残留或行为 oracle 缺层。按 reload journal、SessionDB、tool items、process/readiness 和 write set 逐层定位，不重复安装同一 source revision。

完成时至少能独立证明：source commit 可回源；manifest/module、source tests 和 readiness 通过；attached child 使用目标 generation 并实际执行 Skill/Tool/MCP；父 turn 正常结束；下一 turn 的 Core 事实已提交或明确报告恢复失败；正式 SessionDB、memory、plugin-data 和未授权外部效果没有被候选验证改写。

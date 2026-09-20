---
name: plugin-system
description: 说明并执行 Akashic 插件安装、组合、配置、资源归属及完整 snapshot 换代。
when_to_use: 用户询问或要求处理 Akashic 插件、marketplace、插件自带 MCP、Skill、插件配置、安装、更新、卸载或排障时。
metadata: {"akashic": {"always": false}}
---

# Akashic 插件系统

优先完成明确的插件请求。创建或改写 source、加入 Skill/MCP 或准备候选时，先加载 `develop-akashic-plugin`；这里只负责安装链、组合边界和更新结果。历史父 Turn rollout 文档不能覆盖当前合同。

## 事实来源与边界

```text
┌─ ~/.akashic-plugin/manifest.toml
│  └─ 全局安装清单与启用状态
├─ ~/.akashic-plugin/cache/<marketplace>/<plugin>/.artifacts/
│  └─ 不可变 installed artifact 与 generation pointer
├─ <workspace>/plugin-data/<plugin>-<marketplace>/
│  └─ 插件配置与持久状态
├─ <workspace>/runtime/plugin-stable.json
│  └─ 最后成功提交的完整组合；启动只恢复这份选择
└─ <workspace>/runtime/plugin-reloads.sqlite3
   └─ 更新请求、候选、发布与恢复证据
```

外部 source 根必须包含普通文件 `plugin.py`。loader 在 import 前只读取顶层 `name`、`version`、`api_version` 字面量，要求 API 为整数 3；模块提供 `apply(ctx)`。Python 安装输入来自固定制品内实际存在的 `requirements.txt`，不从插件专用 TOML 声明身份、入口或环境。`akashic.plugin.toml` 不再读取或解释，不创建该文件；历史制品中的旧字节不自动删除。Skill、MCP 和其他资源由插件代码向实际 provider 注册。

这不影响 installer 的 `<plugins_home>/manifest.toml` 安装选择，也不删除用户业务 `config.toml`。候选从独立空数据目录开始；删除旧 `validation.exclude_data_paths` 不授权复制正式数据。验证材料由调用程序与数据 owner 显式准备，不能把正式浏览器 profile、登录资料、凭据、环境或资源句柄带入候选。

不要查找或创建 `registry.json`、`.aka-plugin/plugin.json`、`manifest.yaml`、插件级 `mcp/servers.json` 或 workspace 手工 Skill owner。不要直接编辑 cache、pointer、全局 manifest、workspace Skill 软链接或正式 plugin-data。

主动信息源也只组合普通服务：`TIMERS` 驱动来源私有 poll，离散事实提交 `CONTENT`，当前状态保留在插件私有 cache，候选行动进入 `DRIFT`，完整推理由 `BACKGROUND_JOBS` 创建普通 Turn。不要创建 proactive catalog、私有 lifecycle family 或 MCP 聚合桥。

## Agent 可用动作

```text
plugin_install       安装并固定候选，返回 update_id
plugin_latest run    对该 update_id 发起普通 latest 调用
plugin_latest status 读取同一次调用的过程和更新状态
plugin_latest revert 撤销该更新尚未提交的晋升授权
```

以上是工具名与 action，不是 shell 命令。不要手工编辑 manifest/cache/pointer，不要手工切换 generation 或重启 Gateway 绕过排空。底座拥有组合、固定 snapshot、lease 与唯一提交；能力目录、资源启停和业务数据由实际 provider 拥有。

## 安装与更新

只从已提交 Git HEAD 安装；远程 source 必须先 push 对应 commit。使用 `plugin_install` 的实际 schema，指定 source、marketplace 以及普通调用的任务要求；不要编造参数或用 shell 安装代替更新来源的请求记录。

正常链路：

```text
┌─ commit/push → plugin_install → 固定候选
├─ plugin_latest run → 普通 Task / Message / 工具调用
│  ├─ 正常 complete，未 revert → 请求晋升
│  └─ 异常、取消或结果未知 → 不晋升，不自动重跑
└─ 关闭候选 → 排空旧组合 → 新正式实例就绪 → 提交 stable
```

安装成功只证明候选已准备。更新关联的普通调用固定到这次候选，不跟随后来变化的 latest；其他普通调用不会因读取 latest 而获得晋升权。不要求 `passed/reason` JSON，也没有后台业务裁判。发起者通过同一 update_id 查看过程、结果和发布状态，提交前可以 revert。

调用正常结束只是请求晋升，不等于 stable 已提交。底座仍须等实际租约归还、资源关闭与新正式实例就绪；状态和原通知 owner 报告最终结果。未知调用只能查询原证据，不以重试重新执行。不要把候选的空数据运行宣称为正式历史数据兼容证明。

## Skill、MCP 与服务检查

```text
┌─ Skill
│  └─ source root、SKILL.md、references、catalog source、真实触发轨迹
├─ MCP
│  └─ 代码注册的 command、实际 requirements、required tools、candidate read-only tools、endpoint env
├─ managed service
│  └─ process identity、port_env、readiness、退出与隔离 plugin-data
├─ Channel
│  └─ channels provider、实际 factory、凭据归属与关闭回执
└─ 更新
   └─ exact candidate、普通调用结果、reload journal、stable 选择
```

固定 listener 由插件代码向进程或 Workload provider 注册；服务进程和同插件 MCP 必须真正使用 provider 返回的隔离端点，不维护第二份 TOML 资源声明。候选使用宿主授予的候选权限和独立数据目录，不能借用正式容器或改成正式凭据 factory。写型 Tool/MCP 仅在事务、dry-run、隔离目标或明确授权下执行。

Channel candidate 不复制正式 token、webhook 或 long-poll ownership。完整换代顺序是：

```text
关闭来源接纳 → 有限排空 → 依赖者先关闭 → 新正式 Root 初始化
                                      └─ stable 提交后才开放来源
```

资源关闭返回必须证明 ingress、在途工作和 ownership 已收束；失败句柄留给原 owner。失败恢复也创建新的物理实例，不复活旧 Root。提交已确认或结果不确定时，不能恢复旧内存指针来冒充外部效果回滚。

## 卸载与撤销

```bash
python main.py plugin-uninstall demo@github
```

卸载是明确的 runtime 管理操作：先停用并排空，再移除 installed code 和安装记录，保留 `<workspace>/plugin-data/<plugin>-<marketplace>/`。只有实际回执才能证明卸载完成；清理失败必须报告实际残留和错误。

`plugin_latest` 的 `revert` 只撤销对应更新尚未提交的授权，不撤销已发布版本或插件数据。撤销授权与清理完成是两件事；清理失败须保留原 owner 和诊断。不要反复安装、删除 plugin-data 或借手工修改伪造恢复。

## 配置与排障

插件自行解析 `ctx.config`，Core 不寻找 `Config` 导出。固定配置与私有 `CredentialRef` 继续经过显式配置及升级入口；旧配置和备份不能因 TOML 策略退役而自动删除。不要改主 `config.toml`。缺少依赖、导入失败、代码身份不一致、配置错误、命令失败、readiness 失败和数据损坏必须 fail-loud。

调用长期无结果、超时、candidate identity 不一致或 cleanup 残留时，沿原 update、普通 Message/ToolResult、reload journal 和实际资源 owner 定位，不重复安装同一 source revision。按用户要求拉长查询间隔，不忙轮询。

交付时区分源码与安装身份、编译合法性、真实普通调用结果、资源关闭、stable 提交及恢复证据。明确哪些验证未执行；静态阅读和 PR 不证明真实模型、Skill/Tool/MCP、发布或故障恢复成功。用户禁止测试或 Gate 时遵守，不为了填写完成结论擅自运行。正式 SessionDB、memory、plugin-data 和外部效果始终受各自授权边界约束。

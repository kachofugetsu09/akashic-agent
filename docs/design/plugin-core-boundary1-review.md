# Issue 1179 · 边界①审查入口

状态：实现已交付为 stacked Draft PR，等待维护者审查。本文固定边界①的范围，
不宣布 #1179 完成；未来的验收要求仍由 Issue 与 ADR-0102 拥有。

## 停点与调用链

维护者要求做到“入口与主要业务服务”后停下；Models 类型、错误语义和公共合同
完整收口后，不再继续叠加执行环境、声明式组合或 Ledger 迁移。
主要业务服务收口是 [#1235](https://github.com/kachofugetsu09/akashic-agent/pull/1235)，
`c4ff08d3`；评审修正依次叠在 [#1236](https://github.com/kachofugetsu09/akashic-agent/pull/1236)
之后：[合同选择与重启状态 #1238](https://github.com/kachofugetsu09/akashic-agent/pull/1238)、
[命令来源 #1239](https://github.com/kachofugetsu09/akashic-agent/pull/1239) 与调试入口修正。
这些修正仍属于边界①，原 checkout 未改动。

```text
┌──────────────────────────────┐
│ Core：宿主、组合与进程生命周期 │
└──────────────┬───────────────┘
               │ Context / Effect / generation
     ┌─────────┼────────────────┐
     ▼         ▼                ▼
┌────────┐ ┌──────────┐ ┌─────────────────┐
│Gateway │ │UI / Timer│ │MCP / 业务 provider│
└───┬────┘ └──────────┘ └─────────────────┘
    │ Message → Sources / Reply / Delivery
    ▼
┌────────────────────────────────────────┐
│ Models：请求、响应、失败、投影与 driver 合同│
└──────────────────┬─────────────────────┘
                   ▼
┌────────────────────────────────────────┐
│ 现有 Core 账本与执行环境：留待边界②、③  │
└────────────────────────────────────────┘
```

真实服务与公共合同由提供方拥有，兄弟消费者只依赖公开合同。
公共类型来自实际安装的 owner 制品；禁用实现不使已安装 API 消失。
没有 checkout 补接口、中央转发壳或自动选择 provider。

## 审查顺序

每层 base 指向前置层，请按依赖顺序审查，不能只按 PR 编号排序。
尤其 #1220 是 #1218 的前置。所有实现层均控制在 500 行新增以内，删除不抵新增。

| 主题 | PR 入口 | 重点 |
|---|---|---|
| CI、死代码与公共合同加载基础 | [#1182](https://github.com/kachofugetsu09/akashic-agent/pull/1182)、[#1180](https://github.com/kachofugetsu09/akashic-agent/pull/1180) → [#1185](https://github.com/kachofugetsu09/akashic-agent/pull/1185) | 旧存储只退役代码；API 类型身份与安装源码范围 |
| 旧 SessionStore 删除 | [#1181](https://github.com/kachofugetsu09/akashic-agent/pull/1181) | 单独审查大批代码删除；没有删表、减少消息或更改 Session 持久事实 |
| Timer 与 MCP | [#1186](https://github.com/kachofugetsu09/akashic-agent/pull/1186)、[#1187](https://github.com/kachofugetsu09/akashic-agent/pull/1187) | 等待、诊断、取消与物理资源清理的 owner |
| 其他业务合同 | [#1188](https://github.com/kachofugetsu09/akashic-agent/pull/1188) → [#1197](https://github.com/kachofugetsu09/akashic-agent/pull/1197)、[#1204](https://github.com/kachofugetsu09/akashic-agent/pull/1204) → [#1207](https://github.com/kachofugetsu09/akashic-agent/pull/1207)、[#1222](https://github.com/kachofugetsu09/akashic-agent/pull/1222) → [#1225](https://github.com/kachofugetsu09/akashic-agent/pull/1225) | Protocol、冻结值、ServiceKey 是否位于真实提供方；持久 binding 不被改名重写 |
| UI 与 Web Shell | [#1199](https://github.com/kachofugetsu09/akashic-agent/pull/1199)、[#1200](https://github.com/kachofugetsu09/akashic-agent/pull/1200)、[#1212](https://github.com/kachofugetsu09/akashic-agent/pull/1212) → [#1214](https://github.com/kachofugetsu09/akashic-agent/pull/1214) | UI 查询、线程、监听器与资产归插件；Shell 按实际端点转发 |
| 宿主窄端口与 Gateway | [#1201](https://github.com/kachofugetsu09/akashic-agent/pull/1201) → [#1203](https://github.com/kachofugetsu09/akashic-agent/pull/1203)、[#1208](https://github.com/kachofugetsu09/akashic-agent/pull/1208)、[#1209](https://github.com/kachofugetsu09/akashic-agent/pull/1209)、[#1215](https://github.com/kachofugetsu09/akashic-agent/pull/1215) → [#1221](https://github.com/kachofugetsu09/akashic-agent/pull/1221) | 控制 RPC、stdio、监听、帧写出回执与进程停止；端点发布失败不能假回滚 |
| HostExecution 已有部分 | [#1210](https://github.com/kachofugetsu09/akashic-agent/pull/1210)、[#1211](https://github.com/kachofugetsu09/akashic-agent/pull/1211) | Controller 与监控已迁出，进程/文件系统/Bridge 完整迁移仍未完成 |
| Models 完整公共合同 | [#1226](https://github.com/kachofugetsu09/akashic-agent/pull/1226) → [#1235](https://github.com/kachofugetsu09/akashic-agent/pull/1235) | 请求与响应冻结、失败分类、发送证据、空间恢复、取消和 driver 关闭；#1234 迁出 Tools 的模型消费接口 |

## 验证入口与实际边界

收口清理移除了旧 EventBus/SessionStore 的无调用残留、Gateway 未接入方法表的参数模型、
Delivery/MCP 的无消费者接口和两个纯再导出模块。`DashboardContext` 仍是同一个
`RequestContext` 类型；MCP 关闭失败仍保留原 Effect，由所属 Fiber/Root 重试排空。
历史 binding 名由 Commands 的合同声明，读取不改原 descriptor、hash 或消息。

边界门的 `--base` 现在同时拒绝公开模块清单和 Core 字面 key 声明的增长，
不能靠扩大清单或登记角色放行。工具库仍计入 Core 源码；模块内部新增 API 仍需 owner
评审。这是当前边界的防回退检查，不代表最终 R7～R11 或九项 Core provider 验收已完成。

栈顶 Core 与本次相关 Models、Tools、ReplyProgram、四种 driver Pyright 零错误。
插件边界 R1/R2/R3 零债务，Yoyo 与 diff 检查通过。
[保留清单](../refactor/orthogonality-test-baseline.md)中仍存在的节点实际运行
40 项通过；已退役 EventBus 等代码的既有删除按对应 PR 审查，没有新增单元测试。

| 行为 | 可复跑入口 | 验证内容 |
|---|---|---|
| 模型记账与取消 | `scripts/check_model_ledger_io.py` | 真实 SQLite/HTTP；22 场景、18 POST；共享 key、发送前取消、结算失败、丢失提交 ACK、Root 排空 |
| 历史命令 binding | `scripts/binding_upgrade_scenario.py` | 原 Core 生成选择、真实 journal 迁移、当前合同恢复一次；完整原 Message/binding/owner 记录不变，无旧运行 key |
| MCP 资源生命周期 | `scripts/mcp_plugin_scenario.py` | 合同依赖显式安装且实现停用；真实 stdio 握手/调用、换代、重启、可选消费者保持与子进程排空 |
| driver 流进展 | `scripts/check_stream_progress.py` | 44 场景；原取消类型、进展与失败证据 |
| 原生 Gateway | `scripts/gateway_cli_scenario.py` | 默认与自定义 Unix/TCP、stdio 冷启动/EOF/超长输入、安装更新、卸载、历史保存、禁用实现的 API 制品 |
| 公共合同与重启 | `scripts/plugin_contract_scenario.py` | 同名已安装覆盖、停用 API、持久 restart_required、继续管理与新 OS 进程加载；14 项 |
| journal 升级 | `scripts/deployment_migration_scenario.py --contract-restart` | 真实 Yoyo、唯一原生备份、原字段/行序/事件保留、完整性与重复执行；普通入口的旧 config 断言不在此结果内 |
| 命令来源 | `scripts/plugin_command_scenario.py` | 空选择的真实 Dashboard/Controller 监听、选择保持、管理恢复提示、业务不回退；7+5 项 |
| 完整调试入口 | `python docker/debug/entrypoint_scenario.py --image <debug-image>` | 用 `docker/debug/Dockerfile` 构建镜像后执行 init/run、原生 RPC、exec、重启与 stdio EOF；Core 配置原字节、Gateway 设置与完整消息保持 |
| 进程停止 | `scripts/process_shutdown_scenario.py` | 停止接纳、失败退出、EOF、失败后重启、既有消息完整保留 |
| 前端统计调用 | `PYTHONPATH=. python docker/debug/orthology_model_stats.py` | 实际 TypeScript fetch/parser → Shell → 可选能力 → 调用账；404/503 与原记录保存 |

每张 PR 的完整 CI 与插件边界结果以对应最新 head 为准。
调试入口镜像验证只覆盖上述隔离路径；正式部署、真实设备和外部插件完整验收仍独立。

## 尚未完成与下一边界

边界②：完整迁出进程、文件系统与 Bridge；迁出 Onboarding 实现和剩余非账本合同；
完成 base/headless/minimal 声明式组合及旧默认开关退役。

边界③：Ledger 独占权威库与事务、可信调用者身份、入站和崩溃恢复；派生库复制分离；
逐插件禁用矩阵、第二 provider、最终 Core 白名单、全部静态门与文档对账。
前端 slot 化仍为独立验收线，不以此阻塞后端阶段。

当前 `agent/plugin_contracts/` 仍有 configuration、message、tools、turn_effects、ui；
其中 `ui.messages.v1`、`ui.plugin.v1` 的声明仍在 Core，留待边界②迁出；
#1213 收口的是注册与查询合同。消息、账本与部分执行合同仍在 Core。这是明确的剩余范围，不能视为最终 Core 已清空。
更大范围插件类型扫描曾发现 Akasha、Channels、MCP、StableView、Telegram 的既有类型问题；
本次只声明实际检查范围零错误，最终验收前还需处理这些问题。

## 状态与恢复

没有操作正式 workspace，没有删除、重建、迁移或减少业务数据，没有合并或部署。
隔离场景在临时 workspace 验证正常追加、状态提交与恢复，并比较既有完整记录。
消息正文、身份、顺序和未知外部效果语义未获得新的减少或重发权限。

源码恢复点是各层已发布 PR 的 base SHA，评审者可从对应 PR 获取。
边界①审查前为 [d5c1e945](https://github.com/kachofugetsu09/akashic-agent/commit/d5c1e9455dcf0bfcffc3ded190d833e4aac696d6)，
合同修正后为 [4ef5779d](https://github.com/kachofugetsu09/akashic-agent/commit/4ef5779dbc7bcd71612ed81801737a22fb29b57b)，
命令修正后为 [0f78c60e](https://github.com/kachofugetsu09/akashic-agent/commit/0f78c60e13a81cbb9ee5219c6a3e5960ba09c752)。
后续修正用普通提交，不改写已发布历史；运行数据仍由自身 owner 的恢复协议负责。

Gateway 配置退休是单向迁移。回滚到仍读取 `[app_server]` 的旧镜像前，先停实例、
另存当前配置，并核对 `config.toml.before-gateway-config-migration.bak`；恢复旧表时
保留迁移后的其他合法修改，不盲目用旧整文件覆盖当前配置。再升级时还须由 Gateway
迁移 owner 核对固定输入并显式修复已完成的配置退休回执；只换镜像不会重跑已经
完成的 step。保留 Yoyo 账本、插件输入和备份，不删除整个账本或重放无关迁移。

# 插件组合的实现边界

状态：#1179 边界②、③已实现并交付审查；未合并、未部署。
分层决定沿用 [0102](../decisions/0102-plugin-owned-services-and-public-contracts.md)，
逐层证据与重大决定见 [审查入口](plugin-core-boundary23-review.md)。

```text
┌──────────────────────────────────────────────────────┐
│ 进程外壳：启动/停止、工作区锁、Supervisor、通用 Web 代理 │
├──────────────────────────────────────────────────────┤
│ 插件宿主：安装源码、selection、journal、配置、迁移回执   │
├──────────────────────────────────────────────────────┤
│ 组合内核：Context、Fiber、Effect、Task、scope、能力登记  │
└────────────────────────┬─────────────────────────────┘
                         │ 可信 Context 身份 + 公开能力
          ┌──────────────┼────────────────────┐
          ▼              ▼                    ▼
┌────────────────┐ ┌───────────────┐ ┌────────────────────┐
│ HostExecution  │ │ Ledger        │ │ Gateway / UI       │
│ 进程/文件/Bridge│ │ 消息/事务/交接 │ │ 协议/查询/监听资源   │
└────────────────┘ └───────┬───────┘ └────────────────────┘
                          │ 窄读写合同
          ┌───────────────┴─────────────────────────────┐
          │ Models、Reply、Delivery、Sources、Tools、记忆 │
          │ Timer、Scheduler、Onboarding 等普通插件       │
          └─────────────────────────────────────────────┘
```

## Core 的固定范围

Core 声明恰好九个 key：`core.host_info`、`core.restart_gate.v1`、
`core.runtime_catalog.v1`、`core.plugin_config.v1`、`core.plugin_updates`、
`core.credentials`、`host.execution.v1`、`core.tasks`、`executor`。
其中执行授权仍由内核持有；实际进程、文件与 Bridge 后端由 HostExecution 拥有。
HTTP 只保留 `external_default` / `local_service` 传输策略与共享连接资源。

Core 不声明业务合同、不打开业务库、不持有模型、消息或渠道服务对象。
宿主 reload journal、Yoyo 回执、完整备份仍可使用 SQLite；这些例外不能成为
新的业务表 owner。只有 `bootstrap/web_shell.py` 使用 ASGI 框架，按已登记端点转发。
静态边界及其分析限制见 [R7～R11](plugin-core-static-boundary.md)。

## 合同和生命周期

公开合同由提供方 `plugins/<name>/contract.py` 拥有。真实安装源码提供公共类型，
实现是否启用不影响已安装 API 的可读性；没有 checkout 回退或中央再导出。
公共类型与纯编码函数保持进程身份，合同变化要求重启；实现与资源随 generation 排空。
私有耐久 binding 的 key 仍由业务 owner 解释原记录，不改写旧 service 名或摘要。

硬依赖缺失只挂起相应 Fiber，恢复时按原组合规则重新 apply。请求期借用的消费者
可跨 provider 替换而不重新 apply。能力缺席、排空失败和未知外部效果均显式返回，
不能用空数据或内存回滚伪装成功。第二实现的覆盖范围见 [替换场景](plugin-second-providers.md)。

## 权威状态和派生状态

Ledger 拥有 `sessions.db`、附件和入站交接，业务 owner 通过窄接口在同库事务提交。
源消息正常只追加；展示、上下文、索引维护没有正文减少权限。可信调用者身份由内核
登记的 Context 给出，插件不从 Root 私有字段猜测 owner。

向量使用 `sessions-derived.db`；Yoyo 只复制旧表，保留源表与原数据。缺少派生向量时，
Akasha 按原模型空间补算后读取原图；空间变化或模型不可用明确失败，不自动改写图。
已开始且无可查询回执的非幂等发送记录失败与“可能已送达”，不宣称成功，也不自动重发。
具体增、改、减与恢复见 [状态地图](persistence-state-map.md)。

## 声明式组合

```text
base.toml → headless/minimal 整行覆盖 → workspace/bundle.patch.toml
                                          │ 发布已验证输入
                                          ▼
                                plugin-stable.json
```

base 启用 49 个内置插件；headless 启用 42 个；minimal 为零插件。
已提交 selection 是唯一运行输入，制品目录只说明哪些代码可用。
workspace patch 拥有启停意图；既有 `config.input.json` 由配置事务独占，bundle config
只初始化新数据目录，不在重启时覆盖用户配置。旧全局 enabled/disabled_builtin 通过
可恢复的 Yoyo 计划转交，正常路径不再读取旧全局清单。

前端区域 owner slot 化仍是 #1179 P5 的独立工作线。正式环境迁移、外部 Fleet
发布与部署、真实账号和设备验收不由本次 Draft PR 自动完成。

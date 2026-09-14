# 插件整体换代重构

- 状态：已授权实施，尚未完成
- 基线：`origin/main@91b09ecd4f774f00d2d7a61a5d9ea3ad3878a339`
- 目标及取舍：[0071](../decisions/0071-plugin-composition-and-whole-runtime-updates.md)
- 范围：本仓库插件、底座、安装加载链、相关测试与文档；外部插件源码迁移、部署和正式数据迁移不在本次范围。

## 当前事实与目标

基线中的 `manager.py` 同时装配服务、解释静态贡献、复制验证数据、启动各类资源与发布 snapshot。
`static_manifest.py` 与 Python 重复描述运行请求，snapshot 拼接能力目录和 Overlay，
部分关闭失败路径丢失资源 owner。这些是删除目标，不是要长期保留的兼容合同。

当前分层代码已采用单参数入口和完整候选 Root，删除 Overlay 与重复运行描述；
资源关闭失败保留实际句柄，未交接 Root 同样保留模块和数据依赖。
资产由普通 `assets` provider 注册，直接读取固定代码制品，不再由 Core 复制第二份目录。
入口统一为 `plugin.py`，身份只声明一次；配置输入与凭据引用固定，业务字段归插件解释。
Web/Dashboard、Mobile UI 已迁入显式 `ui` provider，命令目录与执行归显式 `commands` provider。
删除 `is_active`、`ServiceView`、`static_active`
及旁路依赖列表；功能启用分支留在 `apply(ctx)`，选入组合的硬依赖仍必须满足。
旧 `ToolRegistry` 在实际启动链没有构造者；只有测试向 Manager 注入它，复制 MCP facade
并写入另一份 snapshot 目录。这条路径及其搜索后端已删除，实际工具仍由 `plugins/tools`
与 MCP 的调用 scope 拥有。MCP 只读工具目录尚待对应 provider 提供，当前继续明确报告
`mcp_catalog_unavailable`，不把未打开的服务伪装成空工具成功。
候选、正式及失败恢复现从同一组固定组件归档分别创建全新的模块、Scope 和 generation；
snapshot 保存实际挂载的实例，禁止跨 snapshot 共用物理 Root 或 generation。
关闭候选后才开始正式换代；旧组合排空并实际释放后再创建新正式组合。
旧 payload 替换、候选 clone 以及正式/候选目录和身份来回切换已删除。
恢复也是一次真实新 Root 构建，关闭失败仍由原 Root 或 Store 保存 owner，不能隐式重试。
Dashboard 从该实例的验证环境归属读取限制，不再比较两份不一致的数据路径猜测环境。
这仍未完成整体重构：Manager 仍拥有业务验证、逐类运行宿主和发布特例，
snapshot 仍枚举其他能力，持久选择仍使用旧更新协议。不能把局部删除视作整体换代已经完成。

余下收敛顺序如下；并行实现只用于互不争夺 owner 的切片：

1. 收拢普通入口、固定配置输入和凭据授权，删除剩余插件专用 TOML。
2. 各能力改由实际 provider 注册、冻结和关闭，移除 Core 静态贡献及目录解释。
3. 固定发布后的依赖绑定；候选与正式组合使用不同实例，删除目录和身份来回切换。
4. 将业务验证及数据准备交还调用程序和数据 owner，保留候选隔离。
5. 合并为完整组合的暂停接纳、有限排空、逆序关闭、初始化与唯一 stable 提交；
   删除逐插件发布参与者、可推导状态和旧恢复分支。
6. 对账仓库消费者与显式升级入口，完成固定提交的累计只读审查；运行证据另行授权。

底座继续解释依赖及服务选择，发布后的绑定固定。不增加通用业务规格、兼容层或恢复入口。

## Root 绑定冻结边界

`RuntimeSnapshotCompiler.compile` 成功返回前调用 `CompositionRoot.freeze()`。
正式、候选和验证 Root 共用此出口；`SNAPSHOT_SEALING` 仍在编译前完成插件注册。
冻结不可逆，之后 mount、inject、provide 明确报 `COMPOSITION_FROZEN`，
更换组合必须创建新 Root。服务内部目录、连接重试、Effect/Task 和健康诊断仍归插件；
没有生产调用方的 Fiber restart 与挂载/状态/退出观察者协议已删除。
`RUNTIME_STARTED` 可以取得这些资源，健康变化不会重新装配 Fiber。
冻结后单独关闭 provider Effect 或 FiberHandle 也明确拒绝；只有整个 Root 已进入
`UNLOADING` 时才能移除绑定和挂载节点。普通资源 Effect 仍可独立关闭。
整个 Root 退出时保留原绑定直到消费者关闭成功，再移除 provider；失败句柄仍归原 owner
供退出重试，不激活待定消费者或重绑旧工作。

```text
┌───────────────────────────────┐
│ 初始依赖装配 → SNAPSHOT_SEALING │
└───────────────┬───────────────┘
                ▼
┌───────────────────────────────┐
│ 编译成功 → freeze → 启动 / 退出 │
└───────────────────────────────┘
```

本步提供固定绑定原语；初始化仍保留 pending、provider epoch 和依赖协调，
整体发布与旧增量分支的删除仍按下述分层合同继续。新增回归尚未运行。

## 分层合同

| 层 | 改动 | 独立验收 |
|---|---|---|
| 01 | 确认职责、启停与整体提交合同 | 用户决定、持久状态保护和文档一致 |
| 02 | Scope/Effect 关闭责任及确定性依赖装配 | 取消、重入、关闭失败与依赖退出顺序 |
| 03 | 代码入口和插件内部配置，删除静态 TOML 协议 | 仓库插件实际安装、加载、配置错误传播 |
| 04 | provider 自主管理能力，代码注册全部贡献 | Workload/MCP/Channel/UI/Tool 切片与替换 |
| 05 | 全量 Root、统一接纳和完整 stable 提交 | 更新、revert、排空、初始化失败及强杀恢复 |
| 06 | 删除旧路径并对账全部消费者 | 累计静态审查、独立概念评审与文档；运行验证待另行授权 |

每层以上一层为 base 发布 Draft PR，提供相邻 diff 和只读审查结果。未经运行验证不部署半套协议。
临时接线只服务已列迁移步骤，最终删除，不长期维护双轨。

## 权限与恢复

`change_type: refactor`，`semantic_delta: breaking`。变化限于插件装配协议、更新暂停、资源退出与提交；
业务结果和数据保留协议不变。Core 拥有组合选择、Scope 和发布，plugin/provider 拥有领域及实际资源。
实施只写独立 Git worktree 和一次性测试目录，不打开正式 workspace、发送真实消息、部署或合并 PR。
Git archive/备份分支保留基线，每层 commit 是下一层恢复点。消息、附件、plugin-data、binding 和
历史归档不得自动减少。旧格式升级必须有明确输入、备份及完整性检查，不能藏进加载路径。

## 验证与集成

复用行为边界测试，补关闭失败、启动顺序、候选隔离和提交前后强杀回归；不保护内部枚举或计数。
用户明确要求本轮重构不运行 Gate 或 CI，只交付 PR；本轮也不执行测试，不能宣称行为已通过验证。
提交使用 `[skip ci]` 避免 PR 自动触发工作流，不修改仓库共享 CI 配置。上述行为验收保留为后续验证清单。
独立概念评审使用用户指定的 Agent Bridge Devin `swe-2-high`，审查固定 commit，主 agent 核验结论。

0046 曾避免不一致的数据复制。删除 Overlay 不授权全 workspace 复制；调用程序和数据 owner
准备一致候选数据，底座不猜格式。调用者 Scope 持有使用句柄，provider 保留物理资源状态；
消费者释放前保留清理所需 provider。配置格式归插件，但装配输入必须固定；动态凭据不写入代码归档。

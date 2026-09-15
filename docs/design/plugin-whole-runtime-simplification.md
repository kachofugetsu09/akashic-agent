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
与 MCP 的调用 scope 拥有。普通 `mcp` provider 在每次实际 open 后提供工具目录，并在候选
route 上执行只读限制；没有打开会话的 runtime catalog 继续明确报告 `mcp_catalog_unavailable`，
不把未打开的服务伪装成空工具成功。
MCP、Workload、托管进程已移入[普通资源 provider](plugin-resource-providers.md)，
Snapshot 不再保存这三类目录，Manager 不再为它们执行第二次启动。
候选、正式及失败恢复现从同一组固定组件归档分别创建全新的模块、Scope 和 generation；
snapshot 保存实际挂载的实例，禁止跨 snapshot 共用物理 Root 或 generation。
关闭候选后才开始正式换代；旧组合排空并实际释放后再创建新正式组合。
旧 payload 替换、候选 clone 以及正式/候选目录和身份来回切换已删除。
恢复也是一次真实新 Root 构建，关闭失败仍由原 Root 或 Store 保存 owner，不能隐式重试。
Dashboard 从该实例的验证环境归属读取限制，不再比较两份不一致的数据路径猜测环境。
运行目录已按[当前 snapshot 查询投影](plugin-active-projections.md)收敛：
删除重复 loaded/active metadata/Scope 索引和 PluginContributions；未交接 Root、待发布候选、
Store 和清理失败的原 owner 仍保留。
Channel 已迁入显式 `channels` provider，Snapshot 不再枚举其目录，Manager 不再解释
Channel factory、凭据或适配器启停。底座只提供来源接纳及窄输入端口，输入队列仍归实际宿主。
插件专用 TOML 与正式业务数据复制链已删除，候选从独立空目录开始；
安装清单及用户业务配置不是本次删除对象。旧 Snapshot 晋升 API、固定发布参数和
重复验证身份已删除，唯一 stable 仍由完整 selection 提交。
安装不再自动启动裁判，也不等待父 Turn terminal；更新来源显式发起固定 latest 的普通调用，
正常完成且未撤销才请求晋升。详见 [latest 普通调用](plugin-latest-programmatic.md)。
隔离输入已有独立 owner；普通调用返回实际 Task 句柄，过程和结束后的原消息均可查询。
通用丢弃沿原安装更新结算，关闭成功回执支持取消或文件恢复失败后的精确重试。
隔离宿主直接持有 Root、SnapshotStore、Task、Process、Bus 和数据连接，不再嵌套完整 Manager，
装配路径与正式组合共用。卸载依赖实际完整 Root 关闭，删除重复逐代等待和镜像 generation
租约计数；唯一计数归 snapshot，失败 owner 不由 cache 删除接管。
这些是源码实现状态，不是整体验收：真实模型接线、最终累计审查及运行证据仍未完成。

真实模型路径的静态核对发现一项集成缺口：`models.apply` 从当前 workspace 的
`model-registry.sqlite3` 加载连接、模型和角色，空库只初始化 schema，不从 `ctx.config`
导入它们。因此普通 latest 回复会在模型执行入口报“尚未配置 default 聊天模型”，
不能以替身测试或可读的 Input 证明真实调用可用。维护者已确认默认沿用已有模型设置
和凭据，由模型插件负责接续，不要求独立账号。可信插件合同见 PLG-001 和 0071 勘误；
本轮不为补足此缺口恢复 Core 业务库复制或跨 Root 借用旧模型执行实现。

实施顺序如下；各项当前状态以上文为准，并行实现只用于互不争夺 owner 的切片：

1. 收拢普通入口、固定配置输入和凭据授权，删除剩余插件专用 TOML。
2. 各能力改由实际 provider 注册、冻结和关闭，移除 Core 静态贡献及目录解释。
3. 固定发布后的依赖绑定；候选与正式组合使用不同实例，删除目录和身份来回切换。
4. 将业务验证及数据准备交还调用程序和数据 owner，保留候选隔离。
5. 合并为完整组合的暂停接纳、有限排空、逆序关闭、初始化与唯一 stable 提交；
   删除逐插件发布参与者、可推导状态和旧恢复分支。
6. 对账仓库消费者与显式升级入口，完成固定提交的累计只读审查；运行证据另行授权。

底座继续解释依赖及服务选择，发布后的绑定固定。不增加通用业务规格、兼容层或恢复入口。

装配错误只沿原异常、Root incident 与所属更新记录报告，不再另存 Manager `GateResult`。
原 Gate 缓存仅覆盖 snapshot 编译的一个失败分支，导入、配置和拓扑失败已直接抛出，
不能用其缺失推断插件没有尝试初始化。验收脚本读取真实启动错误；整组失败时未取得
实例的插件是否执行过 apply 保持未知。Root 清理失败仍保留实际 owner 和原异常。

候选编译不再读取 `runtime/deliveries/settlements.sqlite` 的历史目标来决定服务是否可移除。
原 `_preflight_durable_delivery_targets` 是 Manager 唯一的业务库消费者，`forward_targets`
只有该检查调用；两者及只保护这项旧检查的测试已删除。旧投递存储、恢复实现、迁移和未知效果
回执保持原位，没有自动结算、重放或删除。实际依赖仍由本次组合检查；已有业务记录的解释与
恢复归实际接管它的 owner，不把历史目标字符串升级为 Core 的永久插件依赖。

启动后的文件 watcher 首次成功扫描只建立基线，不自动 reconcile 关机期间遗留的候选。
后续文件变化或明确的手动唤醒才请求处理；当前源码或安装元数据损坏时报告扫描错误，
不改变已从 stable 归档恢复的组合。

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

## 完整选择的持久格式与 Manager 消费者

`agent/plugins/selection.py` 的 `PluginSelection(workspace)` 拥有唯一可变文件
`runtime/plugin-stable.json`。Manager 的 boot、整组替换和显式恢复现消费该选择；
App 已移除 boot 时的安装清单改写；watcher 首次扫描只建立基线，不自动选择未晋升代码。

指针 v1 为 `{"version": 1, "root_ref": null}` 或指向 SHA-256 记录的同形对象。
null 只由显式 `initialize()` 创建，表示尚无成功提交；它与已提交的空组件集合不同。
缺失、未知版本、非法结构、链接或损坏记录明确报错，不在读取中初始化或猜测迁移。

完整记录复用 `PluginArchive.save_descriptor()`，仍在 `runtime/plugin-archives/<hash>.json`：

```json
{"version": 1, "components": ["<component archive_ref>"], "previous": "<previous committed root_ref>"}
```

首次记录的 `previous` 是 null。组件已有代码、配置和环境引用，本层不再保存身份、配置、
能力或路径副本，也不限制 workspace 搬迁；实际 venv 的位置约束仍由 PythonEnvironment
owner 处理，本层没有修改它。`components` 必须由 wholeRoot 构造方提供完整正式输入，
建议按 plugin_id 固定顺序；存储层只检查引用结构、重复和 descriptor 存在，不判断依赖
或把任意 binding 子集推断成完整组合。恢复代码与配置前仍须由各输入 owner 验证组件内容。

最小调用合同：

- `initialize()`：只新建，不覆盖已有 null、已提交选择或未知文件；调用者证明它属于
  新 workspace 或已批准的显式升级。存储层不扫描业务数据。
- `read() -> str | None`：只读指针并验证当前选择记录，不打开环境、不启动插件、不自动续跑候选。
- `commit(components: tuple[str, ...], *, expected_ref: str | None) -> str`：核对基线，
  追加完整记录，再替换唯一指针并返回新 ref。`previous` 固定本次基线，所以 A→B→A
  的末次 ref 不等于首次 ref；过期 expected_ref 明确报 `SelectionConflictError`。

initialize 和 commit 的调用者必须持续持有 workspace 单 writer 锁；expected_ref 不是
跨进程锁的替代品。候选授权、未 revert、正式实例已就绪但接纳仍关闭，均由提交调用者保证。
候选沿用 ReloadJournal 事件保存本次基线及完整 refs，不建立另一份可变 stable。

记录先 fsync 发布；指针采用同目录临时文件 fsync → 原子 replace → 父目录 fsync。
初始化改用无覆盖 hardlink 发布，防止覆盖未知指针，并同步新 runtime 目录的父目录。
`SelectionWriteError` 保留原始异常及 `operation/target_ref/outcome/observed_ref/observation_error`：
发布尝试前失败为 `unchanged`；发布尝试后失败为 `uncertain`，即便读到新 ref 也不能宣称
已经耐久提交。观察失败不能当成 null。调用方必须保持接纳关闭并报告不确定结果，不能简单
恢复旧指针或重开旧实例。记录写入失败也可能留下未引用记录，但不改变 stable。

`init_workspace()` 只给本次 `mkdir` 新建的目录接线：先调用现有
`initialize_empty_workspace()` 建立 migration baseline，再持 workspace 锁初始化选择。
既有空目录、旧 baseline、旧安装或中途失败留下的目录都不自动补写 pointer；缺失时报告
需要后续显式升级。baseline 与 pointer 尚不是一个原子事务：两者之间死亡会留下无 pointer
的已建目录，再次普通 init 不猜测恢复。现有初始化函数未改动，也没有新增历史扫描或迁移。

不可变记录只增加；每次完整提交仅原位替换 pointer，旧记录保留为 previous 链或未引用证据。
失败的临时文件保留用于诊断，没有自动扫描清理、GC 或制品删除。恢复材料是 pointer 与
整个 archive 及各输入 owner 的环境/数据；业务数据不随代码选择回滚。
本层只编写 storage/init 与 Manager 消费者边界测试并做静态复查，未运行测试或 runtime 验收。

### 完整 Root 的读取、提交与故障

`PluginManager.load_all()` 在扫描、资源启动和 journal 结算之前读取 selection。
缺失或坏格式直接失败；非 null 使用完整 record.components 调用
`_replace_formal_root(components, expected_ref=ref)`，由 `_compile_topology_snapshot`
与 `_archived_generations(..., sources={})` 创建真实新实例。
它不按当前源码、enabled、latest、安装 cache 或可变配置重选插件。
空 tuple 是已经选中的空组合；只有显式 null 从安装 stable desired 固定初始整组。
安装输入的临时导入仍有 Root owner，但在正式构建及 durable commit 前关闭；
非 null boot 不创建临时输入 generation。

```text
┌──────────────────┐
│ read 唯一 selection│
└────────┬─────────┘
         ├─ null ── 固定安装输入，关闭临时 owner
         └─ ref ─── 读取完整归档 components
                          │
                          ▼
        fresh Root → closed 初始化 → CAS commit → 开放接纳
```

`_build_and_publish_root` 完成 Composition/Channel 启动及 exact closed scope 内的
`RUNTIME_STARTING`、`RUNTIME_STARTED` 后才运行同步提交回调。生命周期回调必须完成
初始化，不能等待尚未开放的外部接纳。回调检查原 publisher 的取消请求及候选授权；
首次 null 或完整 refs 改变时调用 `commit(expected_ref=...)`。恢复同一选择不新增记录。
候选和正式实例使用相同完整 refs，但分别构建，候选先实际关闭。

本次 `SnapshotTransaction.selection_result` 保存真实结果：成功 ref、类型化写入错误，
或尚未得到结果。调用同步 commit 前先保留 uncertain 结果；调用若中断而未返回，
不能假定写入没有发生。成功或 uncertain 后，SnapshotStore 保留新 closed owner，
Manager 不回滚旧内存选择、安装指针或端点，不自动重建旧组合。
普通异常与 `unchanged` 才允许清理新 Root，并从旧完整 refs 重建；清理失败保留原 owner。
确认成功后的资源故障可走显式恢复，恢复输入仍重新读取 selection；uncertain 必须先
显式关闭并重新启动，从磁盘结果恢复，不能把当前可读 ref 当成已确认刷盘。
Store 中的 closed current 仅持有物理资源，不能作为第二份 durable authority。

ReloadJournal 的 preparing 事件保存 `base_selection_ref`，同一次候选事件保存完整
components；记录不复制插件身份、配置或环境内容。boot 沿唯一 selection 的 previous
链检查精确转换，能处理 stable 已提交而 journal 尚未更新的崩溃窗口，不能只比较源码版本。
旧 boot 清理仍使用 `_cleanup_boot_processes` 的真实回执；原有 supervised/旧 boot ID
前置条件保留。本层不把安装指针对齐或每类能力检查当成运行恢复依据。
有证据的提交结算为 recovered，未提交候选结算为 aborted，均不续跑候选。
历史记录缺少完整转换证据时保持原 phase，只追加未知诊断，等待显式处理。
未确认提交的安装更新保留 armed 和错误说明，不能谎称安装文件已经 rolled_back；
安装 owner 的显式回退仍拥有安装文件副作用。Manager 的 promote/drop 不再写 per-plugin pointers。
业务数据和归档不随上述结算删除或回滚。

### App 接线与操作许可

- `CoreRuntime.start` 不再扫描可变安装或写安装清单；安装清单由显式安装入口维护。
- watcher 不在启动时自动 reconcile，首次扫描只建立基线。
- App 在 load_all 之前绑定现有 endpoint callback，闭接纳初始化不再依赖启动之后的接线。
  该 callback 在当前 App 中无实际端点操作，待 Channel provider 收敛时连同中央参与者删除。
- 统一 operation owner/deadline 已接到同步 commit 前；先撤销外层操作许可，
  内层清理等待不能恢复许可。成功写入立即记录选中的实际 snapshot，之后取消不回滚。
  安装线程仍保留到结束；成功或 uncertain 的选择阻止安装回退冒充运行恢复。

Manager 聚焦测试不代表累计 App 启动验收已通过。
旧测试中隐式初始化 Manager 的批量适配另行处理，本层新增测试均显式初始化。

### 存量 workspace 的显式空选择入口

`scripts/upgrade_plugin_selection.py` 只建立新协议的明确 null，不转换旧组合，
不从历史 binding 猜测完整 Root，也不表示当前安装代码已经验证或晋升。
它与 Manager 消费者和 App 接线一起发布，不单独发布 primitive PR。

停用目标 workspace 的宿主及安装 writer 后，在项目环境中明确执行：

```sh
python -m scripts.upgrade_plugin_selection \
  --workspace /path/to/workspace \
  --plugins-home /path/to/plugin-home \
  --backup-dir /path/to/recovery/before-stable-init
```

三个路径均须明确指定；workspace、plugin-home 和备份父目录须已存在。
备份目录必须全新且位于两份运行目录之外，拒绝复用或覆盖。命令先取得既有
WorkspaceInstanceLock，再取得 PluginPublicationLock，持锁完成备份及初始化。
任何已有 stable（包括 null、损坏文件、未知格式或链接）均拒绝，没有 force。

恢复点仅包含 `plugin-home/manifest.toml`、`cache/<marketplace>/<plugin>/.pointers.json`
和 `workspace/runtime/plugin-reloads.sqlite3`。清单和指针按原字节备份；journal 用只读
SQLite backup 纳入已提交 WAL，并检查副本完整性，不修改源 journal 阶段。
`recovery.json` 记录原路径、缺失项、备份校验摘要及原 stable 缺失事实；它只是恢复清单，
不是另一份运行选择。文件及目录同步完成后才调用 `PluginSelection.initialize()`。
不读取消息库、binding、代码归档或 plugin-data；不安装、导入插件或复制业务数据。

命令成功只说明 null 已初始化。下一次 Manager 启动才从操作者明确的
安装选择固定整组代码和配置，构造 whole Root，并在真正 ready 后首次 commit。
未晋升候选不会因此获授权；旧 journal、安装指针和业务数据保持不变。

备份失败不调用 initialize；初始化失败保留完整恢复点并传播原类型化失败，CLI 输出
`outcome/observed_ref/observation_error`，不因观察到 null 而把不确定刷盘说成成功。
恢复时先停宿主并核对指针：只有仍为本次新建 null、且尚未发生首次提交时，操作者才可
显式撤销该新增文件；已有非空选择不得据此恢复点自动回退。旧元数据未被本命令改写，
不要无条件覆盖它们；备份不是业务数据的回滚材料。中途失败的恢复目录保留，不自动清理。
本层只编写边界测试，未执行 CLI、测试或运行验收。

## Manager 操作与截止

公开加载、启动、候选准备、晋升、撤销和清理重试使用同一 Manager 操作 owner。
普通占用返回 `OperationBusyError`，不排队；内部步骤直接调用私有实现，不重新取得 owner。
操作只保存实际 task、绝对 deadline、永久 revoked 和实际 committed 回执。
`POST_PUBLISH_TIMEOUT_SECONDS` 现在约束整个入口，从接纳开始计时，内部步骤不重置预算。
底层资源锁与 Root/Effect 关闭任务继续保护各自实际资源，不再承担 Manager 更新授权。

```text
┌────────────────────────────────────────┐
│ 接纳操作 → 固定 deadline → 准备 / 切换 │
└─────────────────┬──────────────────────┘
                  ▼
┌────────────────────────────────────────┐
│ 同步检查 owner → 同步提交 → 保存回执   │
└────────────────────────────────────────┘
       取消 / 超时 │
                  ▼
┌────────────────────────────────────────┐
│ 永久撤销许可；实际任务退出前保留 owner  │
└────────────────────────────────────────┘
```

`terminate_all` 先同步设置停止标记、关闭已有接纳并撤销当前许可，再串接原任务与全量关闭。
并发 terminate 加入同一关闭任务；观察者超时或取消不重复取消 Root/Effect。
失败结束后的显式 terminate 才重试失败 owner，停止标记不会因迟到清理而清除。
持有同一 Manager 的工作 lease 时，同步更新与 terminate 在产生副作用前拒绝自等待。
`start_update_publication` 同步接纳无 lease 的后台操作；返回表示已接纳，实际结果仍查询原更新 journal。

公开观察使用带定时器的 Future；它的 `cancel()` 同步撤销许可，不等调用者下一次调度。
实际关闭任务用 `asyncio.wait` 加入，不传递重复取消。调用方退出时，操作任务仍持有正在关闭的资源或不可取消的线程，
其他更新不能越过它。关键清理等待不会恢复提交许可。旧资源尚未开始释放的 drain 失败可以恢复原接纳；
进入 release 后，被撤销的操作保持维护状态，必须经显式恢复真实重建，不能后台自动提交或重开。
业务验证的准备和回收也经过 owner；验证程序本身仍属于调用者，验证 scope 未退出时普通更新报 busy。

durable stable 实现直接在同步写入前调用 `manager._check_operation_commit()`，
从检查到同步提交之间不得 await；写入确认成功后，把返回 owner 的 `committed` 设为选中的实际 snapshot。
当前接线保存实际 snapshot。提交后取消保留该事实并报告运行恢复未完成，不伪称 stable 已回滚。
本层没有新增耐久记录、资源阶段表或数据清理协议。

同步 import、fsync 和阻塞事件循环的插件不能被异步 deadline 强制中断，硬截止仍归外部宿主。
线程准备超时后，其任务及目录仍保留到线程真正结束；不提前删除正在写入的目录。
新增 Event 协调回归覆盖吞取消、并发关闭、失败重试、drain/release 区别与迟到成功；尚未执行。

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

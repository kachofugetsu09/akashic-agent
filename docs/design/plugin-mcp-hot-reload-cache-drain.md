# 插件 MCP 热重载缓存与排空事故解决设计

- 状态：implemented / macOS arm64、Python 3.14 本地事故回归通过；Linux CI 待远端验证
- 事故环境：macOS arm64、Python 3.14、`bangumi-mcp 0.2.1 -> 0.3.0`
- 核对基线：`53231756`
- 关联决策：[0008](../decisions/0008-plugin-runtime-publishes-only-committed-snapshots.md)、[0024](../decisions/0024-plugin-self-validation-uses-stable-and-latest.md)
- 关联条款：PLG-003、PLG-006、PLG-012、PLG-013
- 上位设计：[插件递归自验证运行时设计](recursive-plugin-self-validation.md)
- 状态合同：[持久化状态地图 10.2.1](persistence-state-map.md#1021-0024-stablelatest-实现)

## 1. 结论

这次故障不是 TLS 配置错误。旧 MCP 进程启动后保存了旧虚拟环境中的绝对路径，插件更新却先删除了该进程仍在使用的 cache 目录。`certifi` 在后续 HTTPS 调用中读取原路径时，才暴露为 CA bundle 不存在：

```text
Could not find a suitable TLS CA certificate bundle, invalid path:
~/.akashic-plugin/cache/github/bangumi/0.2.1/mcp/.venv/lib/python3.14/site-packages/certifi/cacert.pem
```

正确解决方式不是让在途 turn 强制切换到新 MCP。PLG-003 要求一个 turn 从 admission 到结束绑定同一 runtime snapshot；中途换工具进程会让同一 turn 看见两代代码、工具 schema 和 plugin-data writer。当前设计选择：

1. 每个 source revision 安装到独立、不可变的 artifact，更新不覆盖旧目录。
2. 新代先作为 `latest` 准备 MCP 并完成行为验证，普通 turn 继续使用 `stable`。
3. promote 后不再给旧 snapshot 新 lease；已有 lease 继续使用旧 MCP。
4. 最后一个旧 lease 释放后，runtime 逆序关闭旧 generation scope 和 MCP client。
5. 已发布 artifact 不随 watcher 排空自动删除；当前只在显式卸载完成 runtime drain 后物理删除。尚未发布的新 artifact 可以由创建它的安装事务回滚。

因此，更新期间短暂存在两个 MCP 进程是预期行为。事故判据不是“进程一度并存”，而是旧进程失去 owner、旧 lease 归零后仍不退出，或旧进程退出前它依赖的 artifact 被删除。

## 2. 用户可见目标

更新一个已经在 Gateway 中使用的 MCP 插件后：

- 当前在途 turn 仍能完成旧代 MCP 工具调用，不出现指向已删除 `.venv`、CA bundle、脚本或依赖文件的错误。
- 显式 `runtime=latest` 验证只调用新代 MCP；验证失败不影响 stable 和旧 MCP。
- promote 后的新 turn 使用新代；旧 turn 不被中断，也不被强制换代。
- 全部旧 lease 释放后，旧 MCP 进程退出且不再持有 runtime 资源。
- 不需要重启 Gateway 来恢复工具调用。

这里的“当前会话”需要按生命周期拆开：同一 session 的后续 turn 会重新取得当时的 stable；只有已经 admission 的在途 turn 继续绑定旧 snapshot。设计不把整个 session 永久钉在旧代。

## 3. 已确认事实与未知边界

### 3.1 已确认的旧实现失效链

基线 `040a6c02` 的安装器以版本号作为可见 cache 路径。更新事务把全部旧版本目录移入临时 backup，再把新目录移动到版本路径；manifest 写入成功后，`_CacheActivation.finalize()` 立即递归删除 backup。

watcher 和 runtime snapshot 另有异步生命周期。已经启动的 `McpClient` 仍持有旧命令、`cwd`、环境和 Python 运行时派生出的绝对路径，安装器却没有旧 snapshot lease 信息。因此旧进程还活着时，其 `.venv` 已经可能被删除。

```text
旧 Turn T / Snapshot S1                 plugin-install                 watcher/runtime
        │                                    │                              │
        │ 启动 MCP G1，路径指向 artifact A1  │                              │
        │                                    ├─ 移走 A1                     │
        │                                    ├─ 发布 A2                     │
        │                                    └─ 删除 backup/A1              │
        │                                                                   ├─ 启动 G2
        └─ 再次调用 G1
             └─ 读取 A1/.venv/.../cacert.pem
                  └─ 路径不存在，工具调用失败
```

TLS 只是最稳定的触发器。任何由旧进程延迟读取的解释器文件、动态库、证书、模板或插件资源都可能以同样方式失败。

### 3.2 当前 `53231756` 已实现的机制

- `agent/plugins/install.py` 将安装结果写入 `.artifacts/<version>-<source-revision-prefix>/`；同版本不同 commit 不再覆盖同一路径。
- 同一插件目录的 `.pointers.json` 原子保存 `stable/latest`，`_CacheActivation.finalize()` 不再删除旧 artifact。
- `plugin-install` 由当前 Gateway runtime owner 执行，并等待 candidate 达到 `latest_ready`。
- `RuntimeSnapshotStore` 分别提供 stable 和 latest lease；promote 只切换 pointer，并把旧 snapshot 标为 retired。
- `PluginManager._on_snapshot_drained()` 只在 snapshot 没有 lease 后处理旧 generation；`PluginScope.aclose()` 逆序执行 cleanup。
- `McpGenerationHost.prepare()` 把每个 `McpClient.disconnect()` 登记到 generation scope；scope 排空会关闭旧 MCP catalog 和进程。

### 3.3 事故专项证据

`tests/test_plugin_runtime_control.py::test_installed_mcp_update_keeps_old_artifact_until_lease_drains` 已把原来分散的机制接成一个边界回归：

- 已安装插件的 MCP 进程正在被旧 turn 使用；
- 同一插件安装新 commit，并准备第二个 MCP 进程；
- 插件版本号保持 `1.0.0`，source revision 和 artifact identity 发生变化；
- 旧 MCP 在更新后继续读取旧 artifact 内 `mcp/.venv/lib/pythonX.Y/site-packages/certifi/cacert.pem`，并用 `ssl.create_default_context(cafile=...)` 完成真实证书加载；
- stable/latest 分别返回旧、新 generation 与 PID；
- promote、旧 lease 释放、旧 PID 退出、旧 artifact 保留和显式卸载删除按顺序发生。

`tests/test_plugin_runtime_control.py::test_mcp_hot_reload_oracle_rejects_deleted_old_ca_bundle` 在 `latest_ready` 后主动删除旧 CA 文件，证明同一个调用会稳定产生 `McpToolExecutionError`，而不是静默切到新 MCP。当前本地组合证据已经覆盖原事故；Linux CI 结果仍由远端流水线确认。

## 4. 目标时序

定义：

- `A1/A2`：磁盘上的不可变安装 artifact。
- `G1/G2`：插件 runtime generation，各自拥有 MCP client/process。
- `S1/S2`：完整 runtime snapshot。
- `T`：更新发生时已经持有 S1 lease 的 turn。
- `V`：显式选择 latest 的验证 turn。

```text
canonical source        install/runtime owner       stable Turn T        latest Turn V
      │                         │                         │                    │
      │ commit revision R2      │                         │                    │
      ├────────────────────────►│ 创建不可变 A2          │ 持有 S1/G1 lease   │
      │                         │ 准备 G2 MCP             │                    │
      │                         │ 发布 latest=S2          │                    │
      │                         │ stable 仍为 S1          │                    │
      │                         │                         ├─ 调用 G1           │
      │                         │                         │  读取 A1/.venv     │
      │                         │                         │  成功               │
      │                         │                         │                    ├─ 租用 S2
      │                         │                         │                    └─ 调用 G2
      │                         │◄───────────────────────────────────────────── 验证通过
      │                         │ promote: stable=S2      │
      │                         │ retire S1，停止新 lease │
      │                         │                         └─ T 结束，释放 S1
      │                         │ 旧 lease 总数归零
      │                         ├─ terminate G1
      │                         ├─ disconnect 旧 MCP
      │                         └─ 保留 A1；不自动 GC
```

如果 V 验证失败，runtime 执行 discard：`latest` 恢复指向 `stable=S1`，等待 S2 的验证 lease 释放后关闭 G2。G1、S1 和 A1 全程不受影响。

## 5. 状态增改减合同

| 对象 | 正常增加 | 原位更新或逻辑失效 | 物理减少条件 | owner 与恢复证据 |
|---|---|---|---|---|
| 不可变 artifact | 每个新的 source revision 增加独立目录；同版本新 commit 也增加新目录 | 内容不得原位更新；pointer 改变不等于 artifact 改写 | 同一安装事务可在恢复旧 pointer 后删除自己创建且未发布的新 artifact；已发布 artifact 当前只随显式卸载在 runtime drain 后删除。自动 GC 在能证明 stable、latest、lease、rollback 和 recovery source set 均无引用前不得增加 | install/uninstall owner；source revision、artifact 路径、`.pointers.json`、manifest |
| `.pointers.json` | 首次安装建立 stable/latest pair | 单 writer 原子替换完整 pair；promote/discard 只改变引用 | 不用删除 pointer 表达版本淘汰；仅插件显式卸载随整个 cache 移除 | runtime install/control owner；前后 pointer identity、journal phase |
| runtime snapshot | install/readiness 创建 candidate snapshot | `validating -> committed -> retired/aborted`；retire 是停止新 lease 的逻辑失效 | lease 归零且 drain callback 成功后从内存保留集移除 | `RuntimeSnapshotStore`；snapshot ID、state、lease count、drain failure |
| plugin generation | snapshot 准备时创建，并取得自己的 scope | retire 后不再服务新 turn；已有 lease 保持 generation 可达 | 没有其他 snapshot/lease 引用后执行 `terminate()` 和 `scope.aclose()` | `PluginManager`；generation ID、draining registry、scope cleanup failures |
| MCP process/client | generation readiness 启动并登记 client | 进程不跨 generation 复用；candidate discard 或旧 generation retire 后进入排空 | generation 没有 lease 后由 scope owner 调用 `disconnect()`；清理失败必须保留失败证据，不得报告 drained | `McpGenerationHost` + `PluginScope`；PID/process handle、catalog identity、cleanup report |
| plugin-data | 插件按自己的已批准 schema 增加 | 由插件 data owner 定义 | 普通更新、discard 和卸载都不得级联删除；永久删除需要名称不同的用户操作及恢复点 | 插件 data owner；workspace 备份和领域完整性检查 |

本事故的关键事件顺序是：最后一个引用 generation 的 snapshot lease 先释放，随后 generation scope 开始清理并终止 MCP process；该 generation 依赖的 artifact 必须在整个进程可调用区间内保持可读。已发布 artifact 的删除还要晚于 runtime drain 完成。

## 6. Owner 与权限边界

安装器只拥有 source clone、artifact staging、pointer 文件和 manifest 写入。它不知道进程内有哪些 turn lease，因此不能在普通更新完成时推断某个旧 artifact 已经无人使用。

Runtime snapshot owner 持有 lease set，负责决定 generation 何时进入 retired、何时完成 drain。MCP host 只管理 generation 内的 client/catalog，并通过 scope 暴露窄 cleanup；它不能删除安装 cache。

显式卸载是当前唯一同时拥有“停止新 admission、等待旧 generation 排空、删除 cache 与 manifest entry”的操作。watcher 只发现 revision 并触发候选发布，不获得独立删除权限。

## 7. 失败、取消和重启语义

### 7.1 安装或 readiness 失败

新 artifact 未完整发布时恢复原 pointer；已经完整创建但未晋升的 artifact 可以保留为恢复或诊断证据。stable pointer、旧 generation、旧 MCP 和 plugin-data 不变。失败结果必须指出 candidate phase，不能把“cache 已写入”报告成“新 runtime 可用”。

### 7.2 latest 验证失败

先停止 S2 接收新 latest lease，再等待已有验证 lease 归零，随后关闭 G2。`latest=stable=S1` 之后，普通 turn 继续使用 G1。snapshot pointer 回滚不声称撤销候选已经产生的远端副作用；写型 MCP 验证仍需 dry-run、隔离目标或用户明确授权。

### 7.3 promote 时旧 turn 尚未结束

promote 不等待调用者自己持有的 S1 lease，也不取消 T。新 turn 取得 S2；T 继续使用 S1/G1。只有全部 S1 lease 释放后才关闭 G1。这避免 PLG-012 已记录的“turn 等待自身 lease”环形等待。

### 7.4 调用方取消或 cleanup 失败

取消不能截断 generation cleanup。scope 继续逆序尝试全部 cleanup，并聚合失败。旧 MCP 未确认退出时保留 process ownership 和结构化失败；不得从 draining registry 提前移除后报告完成。

### 7.5 Gateway 重启

启动先从 durable pointer 重建 stable，再恢复或拒绝未决 latest。无法按精确 source revision 重建 stable 时 fail-loud；不能静默切到磁盘上更新的版本。重启后的进程重新建立 runtime handle，但不改变 artifact 保留合同。

### 7.6 同版本更新

版本号不是 artifact identity。`0.2.1` 的两个 commit 必须落到两个 `<version>-<revision-prefix>` 目录；只要 source revision 不同，就不得复用或覆盖 stable 当前目录。

## 8. 兼容与迁移

旧 cache 可能仍是 `cache/<marketplace>/<plugin>/<version>/` 单目录布局。source resolver 可以把一个旧可见版本作为初始 stable，但首次 staged 更新必须把新 revision 写入 `.artifacts/` 并建立 pointer pair；不得为了统一布局先删除或搬空当前 stable。

本设计只覆盖插件 `mcp_servers()` 安装链。`mcp/servers/*.toml` 与 `WorkspaceMcpWatcher` 是待迁移兼容 owner，不应从它们反推插件 cache 删除协议，也不在本事故修复中扩展第二套安装机制。

## 9. 已有证据

| 合同 | 当前直接证据 | 仍缺什么 |
|---|---|---|
| staged update 保留旧 artifact | `tests/test_plugin_install.py::test_install_can_stage_one_latest_without_changing_stable` | 没有让旧 MCP 在更新后读取 artifact 文件 |
| 同版本更新不覆盖旧目录 | `tests/test_plugin_install.py::test_default_update_keeps_immediate_stable_compatibility` | immediate compatibility 路径不是完整 runtime staged/promotion 时序 |
| MCP readiness 与 candidate cleanup | `tests/test_plugin_hot_reload.py::test_candidate_mcp_catalog_uses_stable_public_names_and_closes`、`test_candidate_mcp_readiness_failure_closes_process` | 没有安装 cache 与旧 turn lease 的组合 |
| stable/latest 与旧 lease drain | `tests/test_plugin_hot_reload.py::test_runtime_snapshot_latest_requires_explicit_selector_and_promotion`、`test_runtime_snapshot_discard_keeps_stable_and_waits_for_latest_lease` | callback 级证明，没有旧 MCP PID/artifact oracle |
| install 返回 latest 可租用 | `tests/test_plugin_runtime_control.py::test_runtime_install_waits_until_latest_is_leasable` | 没有 TLS/文件保留和旧进程退出时序 |
| 已安装 MCP 更新的完整事故时序 | `tests/test_plugin_runtime_control.py::test_installed_mcp_update_keeps_old_artifact_until_lease_drains` | Linux CI 尚未在本地执行 |
| 提前删除旧 CA 的事故 mutant | `tests/test_plugin_runtime_control.py::test_mcp_hot_reload_oracle_rejects_deleted_old_ca_bundle` | 无；该用例固定证明 oracle 会命中原错误 |

事故专项回归与原有测试共同组成完整证据。旧 turn 的 snapshot 绑定另由 `tests/test_plugin_hot_reload.py::test_tool_schema_search_and_execute_share_snapshot_generation` 证明；新回归直接持有同一种 snapshot lease，把 MCP 进程、artifact 和删除边界从模型行为中隔离出来。

## 10. 事故专项回归实现

实现的主边界测试是：

```text
tests/test_plugin_runtime_control.py::
test_installed_mcp_update_keeps_old_artifact_until_lease_drains
```

实际文件位于 `tests/test_plugin_runtime_control.py`。测试使用一次性 workspace、plugin home、Git source 和本地 MCP stdio 子进程，不访问真实 Bangumi 或公网。夹具通过与 MCP 声明相同的 `python` 命令解析实际子进程版本，再写入对应的 `mcp/.venv/lib/pythonX.Y/site-packages/certifi/cacert.pem`；server 的 `probe` 工具在每次调用时才读取该文件，并通过 `ssl.create_default_context(cafile=...)` 解析完整 certifi bundle。这直接命中原错误的文件读取边界，也不假设测试进程与 `PATH` 中的 Python 版本相同，不引入 socket、DNS 或外部证书时钟。

测试显式持有 `RuntimeSnapshotLease`，避免模型或调度时序影响进程生命周期 oracle。AgentLoop 的 turn 级 snapshot 绑定由现有生产链测试独立覆盖。

验收步骤：

1. 安装并发布 revision R1，取得 `A1/G1/S1`；启动 turn T 并确认 G1 PID 存活。
2. T 持有 S1 lease 时，提交并安装 R2；版本可提升，也必须覆盖“版本号不变、commit 改变”的分支。
3. 等待 `latest_ready`，证明 A1 和 A2 同时存在，G1 和 G2 是不同 PID。
4. 通过 S1 lease 调用 G1 的 `probe`；断言读取路径位于 A1，CA bundle 可解析且证书数大于零。
5. 通过 latest lease 调用 G2；断言结果来自 G2/A2，而不是 G1。
6. promote 后取得普通 stable lease；断言它与已验证 S2 是同一 snapshot。
7. T 未释放前断言 G1 仍存活；释放 T 后等待 drain，断言 G1 PID 退出且 G2 继续存活。
8. 断言 A1 在 G1 退出后仍存在；只有执行显式卸载并等待 operation drained 后，插件 cache 才消失。
9. 删除 A1 的 CA bundle；断言旧调用稳定失败，同时 latest 调用仍来自 G2。这是原事故 mutant。
10. G2 handshake/readiness 失败和 G1 disconnect/drain 失败分别复用现有候选 cleanup、discard retry 与 drain failure 测试。

测试 oracle 必须从进程、文件和工具 trace 边界观察，不只断言内存 pointer：

```text
before/after pointers
+ artifact path existence
+ old/new PID liveness
+ tool item generation identity
+ CA bundle parse result
+ lease/drain terminal
+ cleanup failure report
```

## 11. 完成标准

当前完成状态：

1. [完成] 第 10 节回归在 macOS arm64 / Python 3.14 通过；Linux CI 待远端执行。
2. [完成] mutant 在 `latest_ready` 后删除 A1 的 CA bundle，稳定命中旧 MCP 的延迟文件读取。
3. [完成] 正向测试在旧 lease 释放前后分别断言 PID 存活与退出；跳过 generation cleanup 会使 oracle 失败。
4. [完成] candidate readiness 失败、promote、discard、取消和 cleanup failure 由本文件证据表及相邻测试覆盖。
5. [局部完成] 本地 `git diff --check`、Black、Pyright、4 个事故文件测试，以及 pytest Python 3.14 / `PATH` Python 3.9 的 2 个错版本专项回归通过；此前 30 个定向/相邻测试、157 个插件 generation 场景、42 个递归自验证场景，以及 13 个 MCP 生命周期/卸载排空场景也已通过，后者另有 4 个平台条件跳过。change-impact Gate 已生成影响计划，但本机缺少 Docker CLI，未在容器中执行公开场景，不能记为通过。
6. [完成] 文档、PLG-003/006/012/013、0024 与持久化状态地图对 artifact 删除条件的描述一致。

## 12. 回滚点与非目标

本次只新增事故解决设计与测试，不改变 runtime、cache 或 workspace 生产实现。回滚时删除本设计并将测试文件恢复到引入前基线，不需要迁移或恢复任何生产状态。

本次不做以下扩展：

- 不让在途 turn 切换 snapshot 或 MCP client。
- 不为 artifact 增加后台 TTL/数量型 GC。
- 不把 Gateway 重启作为正常更新步骤。
- 不修改外部插件 canonical source，也不直接编辑 `~/.akashic-plugin/cache/`。
- 不改变 plugin-data、会话消息或长期记忆的保留语义。

# Manager 当前实例投影

状态：有界实现，只有静态复查，未运行行为验证。基线：`3f9f7461`。
依据：[ADR 0071](../decisions/0071-plugin-composition-and-whole-runtime-updates.md)。

## 当前实例的唯一来源

`PluginManager._active_generations` 是 `current_snapshot.generations` 的只读查询投影。
没有 current snapshot 时返回空映射，不保存第二份字典。`generation()`、变更准备、
禁用和 Channel 初始化的现有读取都沿此投影；候选仍由 latest snapshot 拥有。
验证 child 在安装自己的 snapshot 后自然得到自己的实例，不单独赋值 active 目录。
该投影表示 Store 当前选中的物理组合，不表示接纳已开放或 durable 提交已确认。
停止和失败恢复继续由原 operation、Store 与 stable 协议负责。

`_loaded/loaded_count` 没有生产消费者；`_active_plugins/ActivePluginInfo/active_plugins`
及 `_registry_active` 只形成无生产消费者的 metadata 视图。`PluginContributions`
及 collector 仅给该视图提供 manifest，均删除。插件静态身份与固定配置输入保持原归属。
每个 Root 只注册自己的固定模块 namespace，延迟导入继续读取同一归档代码。
不再发布可变 stable import alias，也不保存别名目录或在退出时猜测别名是否已被新代接管。
跨插件组合使用声明的服务依赖；没有仓库内插件依赖旧别名入口。

## Scope 与失败归属

两个生产 `PluginScope` 构造点分别是 `_load_one` 的临时输入导入和
`_archived_generations` 的完整 Root 实例构造。两者均先登记 building Root，
先登记模块 namespace 清理，再构造 Scope 并立即登记其清理回调，最后才导入插件。
回调本身持有实际 Scope，因此 import 失败且尚无 generation 时也不会失去 owner。

```text
┌─────────────────────────────────────────┐
│ building Root → 模块回调 → Scope 回调 → 导入 │
└───────────────────┬─────────────────────┘
                    ▼
┌─────────────────────────────────────────┐
│ 编译 → Store 接管实际 snapshot / Root     │
└───────────────────┬─────────────────────┘
                    ▼
┌─────────────────────────────────────────┐
│ 逆序关闭 → Scope 成功 → 模块释放 → owner 解除 │
└─────────────────────────────────────────┘
```

`CompositionRoot._dispose` 仅在回调成功后移除回调；Scope 失败阻止更早取得的
模块依赖被释放。`_close_building_root` 仅在 Root 成功退出后解除 building owner。
成功发布通过 `_begin_snapshot_publication` 先交给 Store，再解除 building owner；
验证 child 同样先 install 再解除。Store 的 drain 回调先关闭 Root，再处置 generation。
失败 snapshot 仍由 Store 保留，snapshot 外失败 generation 仍由 draining 集合保留。

所以删除 `_scopes` 以及 terminate 的旁路 Scope/module sweep。terminate 继续先关闭
building Roots，再关闭 Store，并显式重试 prepared/draining owner；失败就保留责任，
不能绕过 Root 单独扫除 Scope 或模块。`_prepared_generations`、`_draining_generations`、
`_building_roots`、lease count、Channel、stable 和 operation 的协议不变。
Scope 关闭不删除插件数据，本次没有新增持久数据减少行为或清理协议。

## 静态证据与交接

调整真实候选/正式实例、验证 child、部分导入和关闭失败回归，断言实际 Scope、模块、
Root 与 snapshot 的归属，不再断言已删除的目录。只允许 `git diff --check`，
未执行 tests、Gate、CI、build、lint、AST 或 runtime。

另一 writer 拥有的 `tests/test_ui_provider.py` 在基线仍导入并构造
`PluginContributions`；集成时需删除该 import 和 `contributions=` 参数。
本切片不改该文件，也不为它保留第二协议。整体权威设计与索引由协调器负责对账。

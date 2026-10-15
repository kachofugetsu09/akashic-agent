# 0121 · 中断的插件安装按磁盘结算

- 状态：proposed / implemented for review
- 日期：2026-10-15
- 关联：PLG-013、ADR-0094、ADR-0109；修订[状态地图 10.2.2](../design/persistence-state-map.md#1022-单图安装与停止期指定回退)

## 背景

安装的可见提交点是 `.pointers.json` 的一次原子替换。修订前，进程如果在 `plugin_updates`
写成 armed 之后、提交之前死亡，启动、doctor 和下一次安装都会拒绝运行，直到维护者停机
运行 `rollback_plugin_install.py`。另外，指针与启停选择已经写完、`plugin-stable.json`
还没换代时死亡，之后每次启动都会继续运行旧 selection：

- 安装记录是 committed，状态却一直是 unknown；
- 卸载已写入停用选择，但插件每次启动仍在运行。

## 决定

外置插件的磁盘意图只有两份：

- 当前制品，即指针；
- 启停选择，即 workspace patch。

`plugin-stable.json` 是上次应用的投影。

1. **孤立 armed 安装**，即没有 runtime 记录的 armed 行，在启动和每次安装开始时，
   在发布锁内结算：
   - 指针仍是原值：提交点之前中断，行写成 rolled_back，磁盘不变；
   - 指针已是候选：提交点之后中断，补写同一安装的启用选择，行写成 committed；
   - 其他值：fail loud，不覆盖。
2. **已提交的 selection**，启动时只对外置输入按磁盘修正：
   - 已停用的移除；
   - 代码目录与当前指针不同的，换成当前制品；
   - 已启用但未选中的，加入；
   - 以上改动一次 CAS 提交，并把指向当前制品、尚未绑定输入的安装记录接到新输入。
3. 新制品源码无法编译时，沿用首次启动规则：只留 source diagnostic，跳过该插件，不恢复旧版本。
4. 删除停机显式回退脚本及其专用 journal 入口。想回到旧版本时，显式安装旧版本，这是一次普通安装。

```text
提交点前死亡 ─► 启动 A，记录 rolled_back
提交点后死亡 ─► 启动 B，记录 committed，状态随 B 的 Fiber 投影
```

## 理由与代价

指针替换是原子的，所以结算只需比较这一个文件的确切值，不需要猜测。
内置插件和发行来源不在本决定内，仍按原有 selection 启动。

物理清理不会续跑：卸载在删除 cache 前中断的，插件停用但 cache 仍在，再执行一次卸载即可完成。
提交点前留下的未引用制品也保持原样，留给后续制品保留策略处理。

## 验收

`scripts/plugin_update_crash_e2e.py` 在真实子进程的各阶段 SIGKILL，再用两个新进程
依次启动，覆盖：

- 构建后、armed 后、指针后崩溃（离线与在线两条路径）；
- 在线安装返回后、selection 换代前崩溃；
- armed 后不经启动直接重试安装；
- 卸载写入停用后崩溃。

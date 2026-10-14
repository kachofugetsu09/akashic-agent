# 运行诊断

默认组合包含本插件，记录近期内存、任务数、GC、事件循环延迟、插件操作与测量。
采样和只读服务不需要 UI、Loki、Grafana 或联网。
最小组合停用本插件；headless 组合保留采样，消费者可以独立使用公共读接口。

## 采样与边界

- 每 5 秒采集当前 Python 进程 RSS、匿名内存、Swap、任务数和事件循环延迟；
  每 30 秒读取 PSS。内存来自 Linux `/proc`，cgroup v2 单独记录总量、匿名内存、
  文件缓存、Swap、上限和 OOM kill 次数。缺席或权限拒绝显示不可用及原因，不填零。
- 非 Linux 环境仍提供 GC、任务、延迟、操作和 Python 剖析，OS 内存指标明确不支持。
- GC 次数、回收对象数和回调耗时从本次诊断实例开始累加；不强制 GC、不遍历对象堆。
- 操作沿用 `ctx.diagnostics.operation/measure` 的日志，保留插件、generation、Fiber、
  父子 operation 和已有 Session/Turn 身份；不保存日志正文、异常正文、参数或对象值。
  INFO 被关闭时，普通操作记录不可用，样本通过 `operations_enabled` 明确说明。
  本插件自己的 Dashboard 轮询不进入事件窗口，避免查看诊断挤走目标操作。
- 720 个样本和 2048 条事件分别保存在有界内存记录中。消费者不持有生产者回调，
  慢读者只会漏读自身的诊断数据，不阻塞业务。插件换代或进程退出清空记录。

cgroup 不一定专属于 Akashic；本地运行时可能包含同组其他程序。进程和 cgroup
不能相加。同期操作只帮助定位，不能用并发操作前后的 RSS 差值计算插件独占内存。

## 只读扩展接口

其他插件从 `plugins.runtime_diagnostics.contract` 导入 `DIAGNOSTICS`，声明依赖后
调用 `read(after=cursor)`。不获得剖析控制、业务 repository、文件写入或发送权限。
返回值是独立 JSON 值，`schema` 为 `akashic.diagnostics.v1`：

| 字段 | 合同 |
|---|---|
| `stream_id` | 本次记录实例身份；变化时丢弃旧游标，从 `after=0` 重读 |
| `through_seq` | 本响应已读取的全局游标，供下次读取使用 |
| `samples` / `events` | 各自按 seq 排序，只含 `seq > after` 的保留记录；两数组共用 seq |
| `seq` / `timestamp` / `kind` | 单调递增序号、Unix 秒时间和 `sample` / `operation` 类型 |
| `capacity` | 各数组的最大条数 |
| `evicted` | 各数组从本实例启动起淘汰的总条数 |
| `gap` | 对应数组曾淘汰游标之后的记录；本次返回不完整，不假装已连续送达 |
| `reset_required` | 游标超过当前序号，消费者必须重新从零读取 |
| `pid` / `release_commit` | 当前进程与发行身份；开发运行的 commit 可为空 |

数值字段带单位后缀；`*_bytes` 为字节，`loop_lag_ms` 为毫秒，`gc_seconds` 为秒。
PSS 只出现在实际读取它的样本，不把上次值冒充新采样。所有空缺指标保留状态，不能
由消费方归零。测量事件沿用 `measurement`、`measurement_value`、`measurement_unit`。
同 schema 内可增加可选字段；删除字段或改变含义必须升级 schema。调用方忽略未知字段。

将来的外部适配器在自己的生命周期拉取、聚合和发送，自己处理重试、容量与配置。
当前不创建 exporter 插件、外部端点或后台发送。普通业务插件继续只调用已有诊断 facade；
精确 Session/Turn/operation 身份不直接变成指标标签。

本插件不创建数据库、配置或诊断文件。临时窗口溢出淘汰与关闭清空只减少诊断副本。
消息、附件、模型配置与其他 plugin-data 都没有读写入口，无需数据迁移。

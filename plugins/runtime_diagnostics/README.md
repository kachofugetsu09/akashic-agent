# 运行诊断

默认组合包含本插件，供 SRE 读取近期内存、任务数、GC、事件循环延迟、插件操作与测量。
不提供前端页面或顶部导航，不需要 Loki、Grafana 或联网。无 UI 时采样和只读服务仍工作。
最小组合停用本插件；headless 组合保留采样，消费者可以独立使用公共读接口。

## 采样与边界

- 每 5 秒采集当前 Python 进程 RSS、匿名内存、Swap、任务数和事件循环延迟；
  每 30 秒读取 PSS。内存来自 Linux `/proc`，cgroup v2 单独记录总量、匿名内存、
  文件缓存、Swap、上限和 OOM kill 次数。缺席或权限拒绝显示不可用及原因，不填零。
- 非 Linux 环境仍提供 GC、任务、延迟、操作和 Python 剖析，OS 内存指标明确不支持。
- GC 次数、回收对象数和回调耗时从本次诊断实例开始累加；不强制 GC、不遍历对象堆。
- 操作沿用 `ctx.diagnostics.operation/measure` 的日志，保留插件、generation、Fiber、
  父子 operation 和已有 Session/Turn 身份；不保存日志正文、异常正文、参数或对象值。
  INFO 被关闭时，样本的 `operations_enabled=false` 明确标记普通操作记录不可用。
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

## HTTP 与剖析

HTTP 复用现有 UI 后端监听器，以无前端模块的 Dashboard 注册。SRE 脚本或外部适配器
直接请求下列路径，不发送 `x-akashic-web-*` 模块头；访问控制仍由部署入口负责。
接口拒绝带 `Sec-Fetch-Site` 的浏览器请求，包括跨站表单，避免移除模块后开放浏览器控制。
这些接口不供普通网页调用；该请求检查不替代部署入口的身份认证。未启用 UI 后端时，通过上述 `DIAGNOSTICS` 服务读取。

- `GET /api/dashboard/runtime-diagnostics?after=0`：上述记录和独立的 `profile` 状态。
- `GET /api/dashboard/runtime-diagnostics/export`：下载完整保留窗口和最近一轮剖析的 JSON。
- `POST /api/dashboard/runtime-diagnostics/profile`，正文 `{"seconds":30}`：手动开始
  10～120 秒的 Python 分配追踪。重复开启或已存在其他 tracer 返回 409，非法输入返回 422。
- `POST /api/dashboard/runtime-diagnostics/profile/stop`：提前结束并汇总。未运行时只返回状态。

追踪默认关闭。开启后记录新分配的 5 层 Python 调用栈，结束时保留仍存活分配的前
30 组位置、大小、数量，以及追踪内存当前值、峰值和 tracer 自身内存。不是对象持有者
分析，不包含启动追踪前的分配，也不保证覆盖原生扩展。记录的是期间峰值数值，
最终调用栈对应结束快照，不能把二者混称为峰值的分配栈。

剖析额外占用 CPU 和内存；定时停止依赖进程仍能调度。汇总在 worker 线程执行，
取消会等物理工作结束后停止本插件的 tracer。失败明确显示 `failed`，取消显示
`cancelled`。卸载取消定时器并清理回调，不停止其他 owner 已开启的追踪。

本插件不创建数据库、配置或诊断文件。正常增长仅限上述临时窗口；溢出淘汰与关闭
清空由本插件执行，只减少诊断副本。运维显式导出的文件由运维管理，本插件不自动删除。
消息、附件、模型配置与其他 plugin-data 都没有读写入口，无需数据迁移。

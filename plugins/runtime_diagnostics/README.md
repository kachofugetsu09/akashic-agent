# 运行诊断

默认组合包含本插件，供 SRE 读取近期内存、任务数、GC、事件循环延迟与停顿现场、插件操作与测量。
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
- 720 个样本、2048 条事件和 8 次停顿分别保存在有界内存记录中。消费者不持有生产者回调，
  慢读者只会漏读自身的诊断数据，不阻塞业务。插件换代或进程退出清空记录。

## 事件循环停顿现场

`loop_lag_ms` 只能在停顿结束后补记幅度。本插件另有一个看门狗线程，不依赖事件循环：
循环上每 0.5 秒写一次心跳，线程每 0.25 秒比较一次时间戳。心跳迟到超过 2 秒时，立即
用 `sys._current_frames()` 抓取全部 Python 线程的栈；同一次停顿在第 6、18 秒各补抓一次，
区分单个长调用与接连的多处阻塞。心跳恢复后写入一条 `kind=stall` 记录，带实际停顿时长，
并输出一行 WARNING 日志（循环线程与占用 CPU 线程的栈顶位置）。

2 秒避开正常回调、大堆完整 GC 与偶发同步文件 I/O 的噪声，又远早于 Web 守卫 10 秒超时，
阻塞代码仍在执行时就能拿到现场。栈只含与剖析相同脱敏规则的文件定位、行号和函数名，
不读取局部变量或对象值。空闲时看门狗只读两个时间戳，每 4 秒读一次各线程 CPU 时间作为基线。

| 字段 | 含义 |
|---|---|
| `started_at` / `duration_ms` | 停顿开始的 Unix 秒时间；心跳迟到时长，误差不超过心跳间隔 |
| `threshold_ms` | 触发阈值 |
| `captures[].elapsed_ms` | 抓取时停顿已持续的毫秒数 |
| `captures[].watchdog_delay_ms` | 看门狗自身被推迟的时间。大于阈值说明整个解释器被占住（通常是某线程持有 GIL 的 C 调用），栈在 GIL 释放的瞬间抓取 |
| `captures[].cpu_window_ms` | 计算线程 CPU 差值的窗口 |
| `threads[]` | `name`、`native_id`、`loop`（事件循环所在线程，排在最前）、`cpu_ms`（窗口内 CPU 毫秒，Linux 以外为空）、`hot`（占满停顿期一半以上 CPU 的那个线程） |
| `threads[].frames` | 由内向外最多 20 层 `文件:行 in 函数`，`frames_omitted` 为截掉的外层数；每次最多 48 个线程，`threads_omitted` 为其余数量 |

循环线程 `cpu_ms` 接近 0 表示它在等待系统调用或锁；`hot` 线程与循环线程不同则说明是
另一个线程占着 GIL。看门狗线程不出现在结果里。停顿未结束前进程退出则不留记录。

`duration_ms` 取恢复那一拍心跳的迟到量：迟到超过 1 秒的心跳另外留底，看门狗处理前
后续健康心跳不会把它覆盖成 0。停用或换代时先通知看门狗线程停止，再在循环上异步等它
真正退出，任务才结束；等待不阻塞事件循环，重复取消也不会提前返回。线程 2 秒仍未退出
（例如卡在 `/proc` 读取）时打一行 WARNING，继续等待，不当作卸载成功。

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
| `samples` / `events` / `stalls` | 各自按 seq 排序，只含 `seq > after` 的保留记录；三数组共用 seq |
| `seq` / `timestamp` / `kind` | 单调递增序号、Unix 秒时间和 `sample` / `operation` / `stall` 类型 |
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

剖析开始和结束时各读一次 glibc `mallinfo2()`，放在 `profile.allocator_start` 与
`profile.allocator_end`（结束值只在 `complete` 时有），区分「真在用」与「已释放未归还」：
`malloc_in_use_bytes`（arena 内在用，uordblks）+ `malloc_mmap_bytes`（大块 mmap，hblkhd）
是 C 层实际持有量；`malloc_free_bytes`（fordblks）是 arena 里已释放但未还给系统的空闲块，
`malloc_arena_bytes`（arena）是各 arena 向系统申请的总量，`malloc_top_releasable_bytes`
（keepcost）是主 arena 顶部可直接收缩的部分，`malloc_scan_ms` 是这次读取的耗时。
这些都是地址空间口径：`malloc_trim` 用 madvise 归还物理页后 `malloc_free_bytes` 不变，
物理是否驻留要看同期样本的 `anonymous_bytes` 与「在用 + mmap」的差。pymalloc 的小对象
arena 直接 mmap，不计入这些字段。musl、macOS 等没有该函数时为 `malloc_status=unsupported`，不填零。

这项读取会逐个 arena 加锁遍历空闲块，持锁期间业务线程在同一 arena 上的 malloc/free
都要排队；碎片严重时单次可达数百毫秒，释放 GIL 也挡不住这把锁，降频或换线程都不能
限制单次持锁时长。所以周期采样从不调用它，只在运维显式发起剖析时读这两次。

剖析额外占用 CPU 和内存；定时停止依赖进程仍能调度。汇总在 worker 线程执行，
取消会等物理工作结束后停止本插件的 tracer。失败明确显示 `failed`，取消显示
`cancelled`。卸载取消定时器并清理回调，不停止其他 owner 已开启的追踪。

本插件不创建数据库、配置或诊断文件。正常增长仅限上述临时窗口；溢出淘汰与关闭
清空由本插件执行，只减少诊断副本。运维显式导出的文件由运维管理，本插件不自动删除。
消息、附件、模型配置与其他 plugin-data 都没有读写入口，无需数据迁移。

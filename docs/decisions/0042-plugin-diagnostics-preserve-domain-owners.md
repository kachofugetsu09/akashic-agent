# 0042 · 插件诊断保留领域 owner

- 状态：accepted / implementing
- 日期：2026-08-26
- 关联条款：OBJ-002、PLG-003、PLG-006、PLG-014～PLG-015、ERR-001

## 背景

Core 可以看见插件的正式调用边界，却不知道 Akasha 检索候选数、Feed 条目数等插件内部含义；
插件知道自己的阶段和计数，却不应自行持有 Loki、Prometheus、Grafana 或任意日志字段。若 Core
逐个认识插件领域，新增插件会产生特判；若只让插件自行计时，插件卡死时又没有边界证据。

已有 `ObserveEventKey` 表达 Core 向插件分发已结算领域事实。它拥有 listener 顺序、失败隔离和
Fiber 生命周期，方向与诊断上报相反，不能复用为 telemetry 总线。

## 决定

```text
Core formal ingress
  └─ entrypoint operation：边界、时间、结果、plugin/generation/fiber identity
       └─ ctx.diagnostics
            ├─ operation(name)：插件拥有的内部阶段
            └─ measure(name, number, unit)：插件拥有的数值含义
```

Core 在 apply/cleanup lifecycle、typed event listener、task、Tool、Command、Background Job、MCP、
Channel factory/lifecycle/delivery/presentation、Dashboard module hook/HTTP 和 Mobile UI 等正式接入点
建立 operation。插件只能通过当前 Context 取得身份已绑定、不可直接构造 concrete 的
`PluginDiagnostics`；不能指定别的 plugin、generation 或 Fiber，也不能提交任意 label mapping。
这个限制定义受支持 API 的 owner，不把同进程 Python 插件伪装成恶意代码安全沙箱。
候选插件在获得正式 generation 前的 module import、`is_active` 和静态语义检查仍属于准入证据，
不伪装成已经进入 formal runtime 的 plugin operation。

operation 使用进程内 monotonic duration 和不透明 `operation_id`；嵌套 operation 记录
`parent_operation_id`。显式队列 handoff 使用同一 facade 的 `capture/resume`，只允许同一
plugin/generation/Fiber 恢复。取消、异常和成功沿同一个 terminal 合同记录，诊断出口故障不能
改变插件业务结果、listener 顺序、Incident 或清理语义。

第一阶段只输出字段白名单内的一行 JSON 日志。精确 Turn、operation 和 generation identity 是
structured metadata，不是 Prometheus label。只有经过真实运行观察后确认稳定、低基数的 operation、
outcome 和数值名称，才能由外围栈聚合为指标；Core 和插件都不直接连接 Grafana 或 Prometheus。

Akasha 是普通使用者：Core 不出现 Akasha 分支；Akasha 通过 `ctx.diagnostics` 上报检索阶段、候选和
completion 等数值。它已有的细粒度 milestone 暂时作为现有兼容日志保留，后续只能向同一 facade
收敛，不能建立第二个 Akasha telemetry owner。

## 影响

- Core 拥有边界计时、关联身份、字段白名单、脱敏和日志出口。
- 插件拥有内部阶段名和数值含义，但不拥有 exporter、标签策略或权威运行状态。
- `ctx.observe` 和领域 Observe bridge 不变；诊断不进入 Composition topology identity。
- 不新增 SessionDB、plugin-data 或 Akasha schema，日志不能成为业务权威事实。
- 本阶段不承诺 Prometheus 指标名；先用 Loki 证据收敛实际需要的低基数聚合。

## 验收

- 非 Akasha fixture 能产生自动 entrypoint、嵌套内部 operation 和数值 measurement，父子关系闭合。
- 成功、异常、取消和 Observe 失败隔离保持原合同，日志 sink 故障不改变业务结果。
- Tool、Command、Job、MCP、Channel、Dashboard/Mobile UI 使用同一 Core wrapper，不出现来源插件特判。
- Akasha 的 Prompt retrieval 和 queued post-commit 使用同一 facade，handoff 不丢父因果。
- 一次性 workspace E2E 能用同一 Turn 的 Core plugin boundary、Akasha 内部日志和 Provider milestone
  区分记忆检索与模型等待。

# V3 durable inbound 与 host boot identity 合同

- 状态：Core/Host 中立端口保留；旧客户端消费者已按 0076 退役
- 范围：`ChannelFactoryContext`、formal channel host、durable inbound port、MessageBus handoff
- 关联条款：RUN-003、AKC-001～AKC-003

## 1. Owner 与调用链

durable inbound 只描述“provider 输入已经有权威交接记录，等待当前 channel binding 接纳”的
传输事实。它不属于某个 channel 名称、插件 ID 或客户端业务协议。

```text
provider adapter
  -> ChannelFactoryContext.ingress / durable_inbound
  -> exact ChannelGenerationHost binding
  -> MessageBus narrow durable methods
  -> InboundHandoffStore
```

`ChannelGenerationHost` 拥有 generation、binding、lease 和 admission 生命周期；
`MessageBus` 拥有 reserve、queue、complete、retain 的原子顺序；`InboundHandoffStore` 只拥有
交接行及其附件引用。插件只能获得声明过的 `ChannelDurableInboundPort`，不能获得 MessageBus。

## 2. 当前公开合同

`agent/plugin_composition/channels.py` 的 `ChannelCapability.DURABLE_INBOUND` 必须和
`INBOUND`、`InboundIdentity.PROVIDER_MESSAGE_ID` 一起声明。Host 只在这个精确 binding 上提供：

```python
class ChannelDurableInboundPort(Protocol):
    async def reserve(self, raw: RawInbound) -> bool: ...
    async def defer(self, handoff_id: str) -> None: ...
    async def settle_rejected(
        self, *, session_key: str, provider_message_id: str
    ) -> None: ...
    def has_pending(
        self, *, session_key: str, provider_message_id: str
    ) -> bool: ...
    def pending_attachment_refs(
        self, *, session_key: str, provider_message_id: str
    ) -> tuple[AttachmentRef, ...] | None: ...
    async def recover(self, raw: RawInbound) -> bool: ...
```

对应的 `ChannelRuntimePorts` 字段是 `durable_inbound`。普通 `ingress` 不得传入
`session_key_override`；只有 durable marker、声明能力和非空 override 同时满足时才允许它。
恢复还必须通过当前 channel、generation、binding 和 admission 校验。

交接 metadata 使用中立字段：`durable_inbound`、`durable_handoff_id`、
`provider_message_id`、`durable_attachment_refs`。MessageBus 对外的窄方法是
`reserve_durable_inbound`、`defer_durable_inbound`、`settle_rejected_inbound`、
`has_pending_durable_inbound`、`pending_durable_attachment_refs` 和
`bind_durable_inbound_recoverer`。

`bind_durable_inbound_recoverer` 只由 `ChannelGenerationHost` 绑定一次。它按持久行的
`RawInbound.message.channel` 选择当前唯一打开的 durable binding，再调用该 binding 的
内部恢复路径；adapter 不得把自己的 `recover` 抢占 Bus 的全局恢复槽。

Bus 的 `settle_rejected_inbound`、`has_pending_durable_inbound` 和
`pending_durable_attachment_refs` 均必须接收 `channel=`；它们不接受只由
`session_key/provider_message_id` 组成的全局查询。

## 3. Host identity

`ChannelFactoryContext.boot_id` 是真实 host identity，不是 generation 或 Gateway identity。
`ChannelGenerationHost` 构造时可以接收现有 readiness 的 `boot_id`；未传入时只在该 host 实例
内生成一次 UUID。所有 generation 和 binding context 复用同一个值；不同 Host 不共享全局值。
Core 只传递该通用身份，不解释客户端名称或 storage。旧客户端 reset、inbox 与 receipt
只在历史恢复清单中保留；本合同不伪造 terminal 结果。

## 4. 持久化与恢复不变量

1. provider message identity 先经过 `RawInbound.message_id` 校验，且必须有中立
   `provider_message_id`；reserve 在附件发布和 Input 可见之前完成。`prepare_channel_input`
   只接管已有 reservation，不能隐式补写 handoff。
2. reservation 绑定 channel、session 和 provider identity。持久 dedupe key 使用
   `channel:session_key:provider_message_id`；旧行的历史 key 只在同 channel 的只读投影查找，
   不原位改写。
3. Input 追加成功后才允许 complete；删除 handoff 行失败时保留行、lease 和 retry owner。
4. cancel、prepare 失败、进程停止和旧 generation drain 都只能 retain 供重试/恢复，不能把内存
   callback 当成完成证据。
5. `InboundHandoffStore` 对旧 pending 行只在读出时把
   `mobile_v3_handoff/mobile_handoff_id/client_message_id/mobile_v3_attachment_refs` 投影为
   中立字段；不更新既有 SQLite 行、不修改 Message。新记录只写中立字段。

## 5. 当前消费者与历史恢复边界

旧 `akashic_clients/mobile_realtime/channel.py` 已按 [0076](../decisions/0076-android-shell-retires-legacy-mobile-stack.md) 删除。当前 `akashic_clients` 只运行 Web Channel，但仍声明 `DURABLE_INBOUND`；通用 Channel host 可将历史 `channel=akashic` handoff 交给这个 binding 恢复。Web 的正常新请求不创建旧 Mobile handoff，Core 也不为旧客户端建立兼容入口。

既有 handoff 行可能仍含旧字段。`session/inbound_store.py` 仅在读出时将其投影到中立 metadata，不更新原 SQLite 行，也不写新的旧字段。旧行的清理必须按[状态地图](persistence-state-map.md)逐项核对回执、附件与已提交 Message；删除源码不能充当业务结算证据。

## 6. 验收边界

- ordinary provider 能力替换时，Core 不按名称或插件 ID 分支；未声明 durable 能力的 binding
  对 marker、override 和 recovery 明确拒绝。
- 同一 Host 的 generation 替换保持相同 `boot_id`；新 Host 产生新身份。
- 旧 generation 的 lease、handoff、附件引用和 retry owner 在 drain 完成前保持有效；新代只
  接管被明确恢复的 pending row。
- 当前中立端口变更需运行 durable/message/channel 定向测试和类型检查；旧客户端 Gate 已退役，不再作为合并条件。

# 0016 · 渠道投递使用完整逻辑消息

- 状态：accepted
- 日期：2026-08-01
- 关联条款：OUT-001～OUT-003、RUN-001～RUN-003、MOB-001、MOB-005、SES-005～SES-006

## 背景

`message_push` 原先把正文、文件和图片交给三个独立 sender。这个调用顺序既不是消息提交协议，也无法表达正文已发而附件失败的部分送达。Mobile 只注册了正文 sender，因此实时主动事件丢失附件，而成功后写入历史的仍是原始 media；Core 还通过返回文案判断成功，可能把不完整投递追加为会话事实。

## 决定

1. Core 使用一个带类型附件的完整出站消息，并且每个 channel 只注册一个 delivery adapter。
2. adapter 返回结构化 receipt，至少区分 `success`、`partial` 和 `failed`；禁止解析人类可读字符串作为提交证据。
3. 被动回复和主动发送复用同一消息模型与 adapter。主动路径直接等待 adapter 的实际终态，不把 MessageBus 入队视作送达。
4. Mobile 把正文、附件描述符与 `delivery_id` 编码为一个 `message.proactive` durable event。附件记录和全部目标设备 inbox 行在同一数据库事务中提交。
5. 只有完整成功才追加主动 SessionDB 消息并运行成功副作用。receipt 提供 canonical media，使历史保存已提交稳定副本而不是调用方的临时路径。
6. SessionDB 继续保存字符串 media 路径；本决定不改变 wire protocol、Android schema 或现有客户端展示模型。

## 理由

逻辑消息是 Core 与渠道共同拥有的最小提交单位。完整输入与结构化终态让每个渠道可以按平台能力映射原生调用，同时不把平台拆包泄漏成 Core 的成功语义。Mobile 的单事件和单事务提交保证 durable 投影内部一致，且不要求客户端增加第二套协议。

## 影响

- channel 注册接口由多个 sender 收敛为一个 adapter。
- Telegram 等需要多个原生调用的渠道必须准确报告部分送达。
- Mobile 需要把附件登记与 durable inbox 提交合并到同一事务。
- 保留既有崩溃窗口：Mobile durable event 成功后、SessionDB 追加前崩溃时，手机可能独有该投影。
- 不新增附件自动清理、outbox、重试状态机或客户端迁移。

## 验收

- 一条含正文和附件的主动消息只调用一次 adapter。
- 任一必需部分失败时不追加主动历史，也不运行成功副作用。
- Mobile 事件包含正文、附件与同一 `delivery_id`，重启后可从 durable inbox 重放。
- Mobile 事务失败不留下附件记录或部分设备 inbox 行；候选文件清理不删除已提交引用。
- 历史页使用 receipt 返回的 canonical media。

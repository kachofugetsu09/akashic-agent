# 0002 · 上下文缩减是非破坏性投影

- 状态：accepted
- 日期：2026-07-16
- 关联条款：CTX-001～CTX-005、SES-002、SES-003、SES-005、CAP-001

## 背景

PR #111 的一次重构把“上下文超限后缩小模型窗口”解释成“从数据库删除窗口外历史”。实现新增了 `DELETE FROM messages`，同时清理 `message_embeddings`，普通测试也被改成期待旧消息消失。PR #124 撤销了这条破坏性路径，并从备份恢复可恢复数据。

这次事故暴露了一个命名和所有权问题：`history` 同时指完整持久历史、进程内 session 列表和发给模型的消息窗口；`trim` 没有说明裁切哪个对象。上下文模块还从完整 SessionManager 间接获得持久层写入和删除接口。

## 决定

上下文缩减定义为从只读权威历史生成临时模型投影：

```text
┌──────────────────────────┐
│ Persistent conversation │  完整、权威、可审计
└────────────┬─────────────┘
             │ read-only snapshot
             ▼
┌──────────────────────────┐
│ Runtime history view     │  进程内派生值，可重建
└────────────┬─────────────┘
             │ select / compact / render
             ▼
┌──────────────────────────┐
│ Prompt history           │  本次模型请求，可裁切
└──────────────────────────┘

Destructive session port ─────► 只供用户主动撤销或删除使用
```

Runtime history view 是进程内派生值。选中的 history window 变小时可以缩短 `session.messages`；只移除动态区块时不能改写它。预算不足只影响后两层。正常收发只向 `sessions.db/messages` 追加新行；完整会话历史、消息身份、序列和受保护派生索引保持不变。只有用户主动撤销消息或删除会话时，独立数据管理命令才可以减少持久对话，并携带权限、cascade、备份和审计语义。旧消息编辑采用原位 UPDATE 还是追加 revision，留待独立决策。

## 理由

模型窗口是计算资源边界，持久历史是用户数据边界。两者生命周期和 owner 不同。把它们放在同一个 `history` 容器和全功能 repository 后面，会让局部合理的“清掉窗口外数据”变成全局灾难。

只读快照和独立类型可以减少歧义；受保护 semantic test、SQLite write-set 和已知删除 mutant 负责在实现者仍然误解时阻止合并。

## 影响

- 新接口不得使用无修饰的 `history` 表示三种状态。
- `DefaultReasoner` 不接收 `SessionManager`。Prompt/context 路径只使用既有 prompt 输入、runtime-only mutator 或只读 snapshot；retry trace 随结果返回，不为它引入持久 writer。
- `semantic_delta: none` 的上下文重构不能修改数据库历史或降低 CTX-001 oracle。
- 对话归档、用户删除和索引重建另设明确入口，不能复用上下文预算函数。

## 验收

- 上下文超限重试前后，`messages` 和 `message_embeddings` 规范化快照一致。
- SQLite authorizer 没有观察到针对受保护表的 INSERT、DELETE 或 UPDATE 尝试。
- 单次请求的 prompt history 可以变短；runtime `session.messages` 只在 history window 变小时同步缩短；关闭并再次加载后完整历史仍然可见。
- 后续消息 seq 从裁切前最大值继续。
- 注入已知删除实现时，semantic gate 稳定失败。

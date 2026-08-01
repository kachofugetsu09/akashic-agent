# 0017 · 单一 Companion 的安全、容量与可恢复失败边界

- 状态：accepted
- 日期：2026-08-01
- supersedes：本仓库旧安全扫描文档中将 channel、device、session 当作独立授权主体的提议性表述
- 关联条款：SEC-001～SEC-010、OBJ-001～OBJ-003、STA-001～STA-003、ERR-001、SCH-001～SCH-002、PRO-002、CTRL-001～CTRL-002、TST-001～TST-006

## 背景

Akashic 服务一个人。Telegram、QQ、Mobile、Web Chat、设备和 session 是同一位用户与同一个 Agent 相遇的渠道，不构成多租户权限边界。此前安全扫描把认证、Origin、per-channel ACL、peer trust 和 device isolation 混入待办，既不符合产品模型，也会割裂全局记忆和跨渠道陪伴。既有 Mobile QR pairing、控制面握手、查询授权、设备撤销和实时协议仍然有效；本决定不削弱这些控制面机制。

## 决定

1. 所有已进入渠道的消息按服务对象本人处理；本轮不增加认证、Origin 或渠道/设备/session ACL。
2. runtime provenance 与模型参数分离。普通工具不接受原始 channel/chat/session；`message_push` 与 Schedule 只在语义需要时接受显式 target。
3. Peer 能力整体退役，不保留第二套主体或路由。
4. 只为外部边界、容量、持久化连续性和错误可观察性建立限制：Schedule 10、MCP material window 100、receipt 7 天/10,000 条/64 MiB、control replay 256 events/4 MiB/32 MiB/5 分钟。
5. `web_fetch` 可按单人本地使用需要访问 localhost、私网和内网 HTTP 服务；大响应仍转入 execution-owned 临时文件，合法大响应不是拒绝理由。MCP 坏 item quarantine；receipt、reservoir、replay 的物理减少各自遵守 owner 和恢复证据。
6. 可恢复失败不会结束 runtime，也不回滚已经提交的 turn 或外部效果；权威状态损坏和无法建立 owner 才 fail-loud。

## 理由

这组边界保护的是状态完整性、外部效果、资源上限和长时间运行连续性，而不是虚构多用户隔离。将渠道视为身份边界会错误阻止跨渠道 `message_push`、全局记忆召回和历史陪伴；将所有异常升级为 runtime 失败又会把单条坏数据、超限附件或 cleanup 失败放大成服务中断。

## 影响与迁移

- 旧安全扫描中的认证、Origin、per-channel scope、peer 信任和自动 correction 语义标记为 removed/non-goal，不进入 Gate。
- `schedules.json`、receipt、Wake reservoir、shell log 和 control replay 的 owner、保留和物理减少规则写入持久化状态地图。
- 具体实现分为 D1～D9，Gate 分为 G1～G9；合同提交不得和产品实现混合。
- Peer 遗留配置必须返回明确 unsupported/unknown capability；不得将旧配置静默解释为空能力。

## 验收

- 每个 G1～G9 场景包含 observes 和至少一个 known-wrong mutant。
- 合同 Gate 只锁定 owner、场景、状态和证据字段；G1～G9 只有在对应实现 head、focused tests、私有 gitlink/native tests（G9）和受保护状态观察全部存在后才能通过。
- Gate audit 拒绝未映射可执行文件、缺少 P0 mutant、未知状态 owner、过期 baseline 或合同/产品混改。
- semantic tests 验证失败分类、持久化 write set、临时 owner、replay fallback 和 runtime continuation。
- 受保护状态的删除、截断或清理必须能从 DB、文件、事件和诊断中观察并恢复。

# 0044 · Akashic Channel 使用 Web 与 Mobile 两个 Adapter

- 状态：accepted
- 日期：2026-08-26
- 关联条款：AKC-001～AKC-003、MOB-001～MOB-008、SES-001～SES-008、MIG-001～MIG-002
- extends：[0018](0018-chat-webui-has-one-source-and-two-adapters.md) 的 UI 源码与平台入口边界
- supersedes：[未来路线草案](../design/akashic-future-roadmap-issue-drafts.md) 第 5 节中与本决定冲突的 Canonical Session 提议

## 背景

Web 与 Mobile 当前分别注册 `web`、`mobile` Core Channel，因此相同的 Akashic 对话被分成
两个 Session 空间。两端的真实差异是认证、传输、本地投影和平台能力，不是 Session、
Message 或 Turn。

仅把两个 Channel 改成同名也不可行：Core catalog 要求 Channel name 唯一。需要一个 owner
只注册一次，再组合两个现有边界实现。

## 决定

1. Core 内建且只注册一个 `akashic` Channel；它不是插件。
2. Web 与 Mobile 是该 Channel 的两个 adapter，不再各自成为 Core Channel catalog entry。
3. 身份固定为 `channel = "akashic"`、bare `chat_id`、
   `session_key = "akashic:<chat_id>"`。Mobile 复用 Web 已有的 allocate-only `session.create`；
   持久 Session 继续由首次消息提交路径创建。
4. 两个 adapter 复用现有 Channel、Session、Message、Turn 和各自 transport。不得为本变更
   新增共同 Port、wire protocol、reducer、Session 生命周期或平台状态 owner。
5. 旧 `web:*` 与 `mobile:*` Session 按完整旧 key 一对一 rekey；历史 Message 身份及其真实
   引用一起迁移。不合并、无 alias、双读、双写或旧 APK 兼容。Akasha 复用现有 rebuild。

## 理由

这个结构只统一一个事实：Web 与 Mobile 面向同一个 Akashic Session 空间。认证、wire、
durable handoff、Room/cache、附件和通知仍可独立变化；Session 的存储和执行语义也不因入口
统一而变化。它同时满足一个 Channel name 只能有一个 Core owner 的现有限制。

## 影响

- App wiring 改为只注册一个组合 `AkashicChannel`。
- Web/Mobile 现有实现只注入共同 route，不建立新的客户端平台层。
- 两端入站使用 bare `chat_id` 作为 provider identity 与 recipient；device identity 只留在
  Mobile 认证和诊断边界。
- Mobile durable handoff 从 `channel == "mobile"` 特判收敛为读取既有 handoff marker/owner。
- Schedule/Wake/delivery 的真实目标引用随 Session rekey；target 形状和投递语义不变。
- 两个 adapter 都尝试实时投影；至少一端明确送达且其余端明确拒绝时，由共享历史补齐未在线端。
  任一端结果不明时整体仍为 `UNKNOWN`，不把未知外部效果结算成成功。
- Session 与 Message 新身份由完整旧 Session key 确定性生成；服务端 embedding、附件、reply
  与 compaction 引用一起迁移，不增加长期 mapping owner。Android 不生成身份，清除旧
  Session 状态后从 Core 全量同步。
- `[channels.chat].channel_name` 删除；内建 `akashic` 名称不再是配置事实。

## 验收

- Core catalog 只出现一个内建 `akashic` 对话 Channel。
- Web 与 Mobile 都能创建、列出、打开和继续同一批 `akashic:*` Session。
- 两端现有历史、实时、停止、模型、附件和 Mobile durable recovery 行为保持原合同。
- 迁移一对一更新 Session、Message 与已知引用，Akasha 使用现有固定输入路径重建。
- 旧 APK 明确失败，新 APK 完成自己的投影迁移后真实收发。

## 关联设计

- [Akashic Channel 与 Web/Mobile Adapter 规格](../design/akashic-channel-client-adapters.md)
- [移动协议交付按变更性质分阶段](0035-mobile-protocol-delivery-is-phased.md)

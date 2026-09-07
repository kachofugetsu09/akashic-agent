# 0059 · 插件附加信息使用普通 Message metadata

- 状态：accepted
- 日期：2026-09-07
- 关联条款：SES-001、SES-003～SES-006、SES-009
- 设计：[Message metadata](../design/0902-reviewed-v4.md#34-message-metadata-的实现与迁移)

## 背景

Message 插件栈最初用 ContentPart 保存 Citation 引用和 Meme 分类，使可选说明与公共正文占用相同的内容类型扩展口。维护者确认这些信息应像 extra 一样，直接保存为 Message 上的普通 JSON 字段。

## 决定与理由

Citation 的依据放入 metadata.citation；Meme 的类别放入 metadata.meme，实际图片仍为 body 中的通用附件。无需为附加标签建立新内容 schema，也不建立 MetadataPart、注册表或解释插件版本的 Core 服务。公共行为合同仍由 body 和原有能力承担，避免把运行依赖藏进可忽略字段。

同一 Message 的 metadata 与正文一起不可变提交。解释归各插件，保存与同步不依赖插件安装状态。JSON 内部可以按需要携带插件自己的版本，未知版本不可猜测解释。完整可观察合同由 SES-009 拥有。

## 影响、验收与恢复

本次增加消息字段、写入授权、Content 返回值与客户端传递路径，不批量搬迁已有内容类型。现有 TextProtocol 的 decoder 改为返回 `(spans, metadata)`，外部插件须按新合同迁移；仓库内 Citation/Meme 样例用于接口验收，不代替外部源码及正式安装验收。

验证消息重放冲突、跨 owner 写入拒绝、正文与附加信息原子回滚、插件缺席后的历史/同步，以及加列迁移对完整旧消息的保全。迁移备份与源码恢复分别处理，不用恢复旧数据库丢弃切换后新增消息。正式数据迁移与部署不由此决定授权。

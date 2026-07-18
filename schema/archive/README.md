# 移动协议历史快照

这里保存已经由外部客户端发布并固定引用的协议快照。文件一旦被客户端的
`protocol/source.json` 引用，只能新增后继快照，不能原地改写。

`mobile-realtime-v1-mobile-pr6.json` 是移动端 PR6 的最小协议面：它包含附件下载、
`command.list` 与当时已落地的基础实时语义，不提前包含后续 plugin UI 或 reply 能力。

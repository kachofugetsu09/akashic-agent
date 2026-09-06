# 0055 · Host Bridge 使用 typed Protobuf V2

- 状态：accepted
- 日期：2026-09-05
- Supersedes：[0032](0032-host-bridge-preserves-host-equivalent-execution.md) 的第 3 项 V1 wire 选择；其他决定不变。
- 关联条款：RUN-013～RUN-015、SH-001～SH-003
- 设计：[Host Bridge Protocol V2](../design/host-bridge-protocol-v2.md)

## 背景

V1 已使用异步 gRPC UDS 和长期 channel，但请求及结果经过 JSON `BytesValue` envelope，
Shell 输出另经 Base64。维护者允许为清晰的协议设计 breaking change，并要求 Terra xhigh
设计审查通过后在独立 worktree 实施。

## 决定

1. 全部现有 Bridge RPC 一次切换到 `akashic.host.v2` typed Protobuf；删除 V1 service、codec 和 fallback。
2. Shell 输出与文件图片通过 `bytes` 传输；文件操作使用四支 oneof，不引入通用动态 JSON 载荷。
3. 保留 unary RPC、既有 ShellProcessManager、boot lease、执行句柄和输出消费 owner。
   取消 RPC 只结束等待；操作可能已生效，不自动重发。`request_id` 只用于诊断，不承诺幂等。
4. Core 与 Bridge 按同 commit 在既有 release 事务中成对升级。V1/V2 混版拒绝提供服务。
   协议实现不授权发布、部署或正式 workspace 迁移。
5. 固定生成器，提交生成文件，CI 检查 source proto 与生成物一致。

## 理由与未选择方案

去掉 JSON/Base64 能减少复制、编码 CPU 和载荷体积，但不能据此承诺 shell 总耗时大幅下降。
长期 streaming 会引入输出订阅、背压和断线续读语义；当前等待式工具 API 没有证明收益，因此本次不加。
临时双协议服务会增加兼容义务，同 commit 成对发布已提供清晰的升级边界。

## 影响、验收和恢复

字段 presence、错误、取消和真实 UDS 验收由设计文档维护。软件恢复使用上一套成对 release；
恢复必须经过原有维护窗口、旧进程清理和 readiness 验证，不声称已经回滚命令、文件或正式数据。

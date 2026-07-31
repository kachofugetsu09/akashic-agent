# 0011 · Benchmark 隔离实例并发上限提高到六

- 状态：accepted
- 日期：2026-07-30
- 关联条款：TST-009、WSP-004、SH-001
- supersedes：[0010](0010-provider-default-output-and-benchmark-diagnostics.md)（仅并发上限）
- superseded by：无

## 背景

诊断控制面最初把并发硬限制为三个独立 attempt。首批 campaign 已证明每个 case 使用
独立 Docker project、workspace、Akasha、网络和 artifact，完成后容器停止但保留。
运行时观测显示三实例下 CPU、内存、磁盘和线上 owner 均有余量，维护者随后明确批准
把最大并发提高到六。

## 决定

Diagnostic campaign 的 `max_concurrent` 允许范围改为 1～6。六仍是 fail-closed
硬上限，不从主机资源自动推导，也不允许调用方越过。当前源码必须先完成一次隔离
smoke，smoke manifest 才能为同一 source digest 打开六并发 Gate；旧的三并发 Gate
不能授权新 campaign。

每个 case 的独立容器、workspace、HOME、Akasha、网络、凭据引用和 artifact 边界保持
不变。调度器的 semaphore 仍是唯一 slot owner；正式 workspace、线上 PID 或冻结源码
发生变化时，campaign 失败。

## 理由

并发六能缩短 89 题 discovery 扫描时间，同时仍保留显式硬上限和逐实例隔离。把授权
绑定到当前 source digest，可防止旧 smoke 为新控制面代码背书；拒绝自动资源探测修改
上限，可避免不同主机悄悄改变实验协议。

## 影响

- 正面影响：无成熟验证任务时可同时扫描六个未见 case。
- 行为语义：同一 campaign 最多可产生六个并行模型调用和 Docker runtime。
- 受保护状态：正式 workspace、线上进程、用户 checkout、凭据值和 case 间状态隔离。
- 回滚：回退本决定对应 commit 后，重新使用三并发源码与其 source-bound smoke Gate。

## 验收

- [ ] `max_concurrent=6` 被接受，`7` fail-loud。
- [ ] 新源码 smoke manifest 打开 `max_concurrent=6` Gate。
- [ ] 旧 `max_concurrent=3` Gate 不能授权新 campaign。
- [ ] 六并发 campaign 每题仍使用独立 Docker project，并通过线上与源码不变检查。
- [ ] 正式完整 89 题 eval 仍需维护者另行明确授权。

## 未决问题

- 六并发完整资源画像在本轮 diagnostic campaign 结束后补入实验 ledger；出现 provider
  限流或宿主资源压力时只暂停调度，不自动改变已冻结 campaign 语义。

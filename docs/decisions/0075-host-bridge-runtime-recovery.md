# 0075 · Host Bridge 运行期按故障范围恢复

- 状态：accepted
- 日期：2026-09-26
- 关联：RUN-013、RUN-015、SH-001～SH-003、ERR-001
- 补充：[0032](0032-host-bridge-preserves-host-equivalent-execution.md)、[0055](0055-host-bridge-uses-typed-protobuf.md)

## 背景

一次 Probe DEADLINE_EXCEEDED 会结束 Core 的关键监控任务，关闭已有聊天连接。
服务端旧实现让 Probe/Heartbeat 隐式创建 manager；客户端又把一次心跳失败永久记为租约失效。
这使短暂传输故障扩大为整个应用重启，同时缺乏安全的续期边界。
维护者授权本地实现与故障实验，PR 评审后才决定合并和部署。

## 决定

1. 启动 ClaimBoot、身份和权限检查保持严格；运行期仅 UNAVAILABLE/DEADLINE_EXCEEDED
   降级宿主执行并继续探测。身份、boot 和程序合同错误不按暂时失联吞掉。
2. 使用现有随机 manager_id。OpenManager 是唯一首次登记入口；续期、业务操作和关闭
   只能引用已存在的 manager。旧 manager 丢失后由原生命周期 owner 创建新客户端，旧句柄不迁移。
3. Exec、stdin 和文件操作每次只发送一次。传输失败可能已有外部效果，不自动重放。
4. 文件读写、编辑和目录遍历离开事件循环，并限制同时执行数量。RPC 取消后必须等实际线程
   结束再释放文件锁和 manager operation，不能把返回取消当作写入回滚。
5. AppRuntime 的监控任务独占短命连接状态；HTTP 只读投影和聊天提示消费同一状态。
   它与某个执行 manager 的存亡不同，不进入 Root/Fiber 健康项或持久化 readiness。

## 理由与影响

增加另一层 lease_id 不能解决执行 owner 丢失，已有 manager_id 足以表达一次生命周期。
恢复策略只重试可安全重复的探测/心跳，避免以便利性换取命令重复执行。
文件线程改善事件循环响应，但不能中断卡在内核里的磁盘调用；回收仍需等待物理完成。
长期失联可能使 lease 被回收，此时明确失败，不能把健康 Probe 当作旧命令仍存在的证明。

本变更修改同 release 私有协议，发布须 Core/Bridge 同 commit。回滚也须整对回滚。
会话、项目、记忆、附件保留语义和数据库 schema 不变。

## 验收

真实 UDS 验证超时/断线恢复、租约过期、boot fencing、响应丢失不重放和文件取消排空；
真实公开 Shell 代理验证状态查询；Chromium 验证实际 React 提示出现、恢复消失和未知状态。
详细命令和证据边界见[设计](../design/host-bridge-reliability.md)。

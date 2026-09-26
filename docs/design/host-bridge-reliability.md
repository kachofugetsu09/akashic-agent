# Host Bridge 运行期可靠性

状态：已实现并完成本地实验；用户授权 PR，合并和部署另行评审。

## 任务合同

- base：db215d872d56c35cff15129f458d89535ad0cad8；writer：Codex；独立 worktree。
- change_type：fix；semantic_delta：breaking（同 release 私有协议与 RUN-013 失败范围）。
- capability_owner：Core 运行监督、Bridge RPC/租约、ShellProcessManager execution 各自保留。
- consumer_scope：Web/Mobile 共用 Core 与所有 Host Bridge Shell/File 调用。
- runtime_patch：required；客户端无法区分 RPC 未达、租约回收或 host ownership 丢失。
- authoritative_state_owner：会话仍属 MessageLog；宿主 execution 仍属 ShellProcessManager；
  Bridge 独占 boot admission 与 manager lease；本次不迁移持久状态。
- client_only_alternative：不可行，浏览器重连不能阻止 Core 因 Probe 超时退出。
- concept_gate：required；最终 head 由独立 terra/xhigh 审查。
- 允许：源码、手册、本地临时 UDS、临时文件和受控子进程；不写正式 workspace，不调用生产控制面。
- 恢复点：Git base 与 /tmp/host-bridge-reliability-backup-20260926/baseline.tar。

## 行为与 owner

```text
Core 运行监督
  ├─ Probe：只验证身份与 boot，短暂传输失败报告降级后继续探测
  └─ manager：首次登记 → RPC/Heartbeat → 明确关闭或 lease 回收
       └─ ShellProcessManager：唯一 execution 和输出消费 owner
```

1. 启动 ClaimBoot 与身份核对仍必须成功。运行期只恢复 UNAVAILABLE / DEADLINE_EXCEEDED；
   权限、boot、版本、响应损坏和内部程序错误继续明确失败。
2. 探测不创建 execution manager。租约首次登记与续期分开；过期/已关闭 manager 的请求
   必须被拒绝，不能创建一个空 manager 后继续使用旧 execution ID。
3. 连接恢复不重放业务操作。Exec、WriteStdin、写文件或编辑文件的异常仍可能已经产生效果。
4. 文件操作的阻塞阶段离开事件循环；文件锁、manager active operation 必须保持到物理工作结束。
5. 运行期降级由 AppRuntime 监控状态、只读 HTTP 和聊天提示及结构化日志暴露；不修改 Message、Turn 或项目数据。

## 本地实验

使用真实 gRPC UDS、生产 service/client、临时目录和真实子进程；控制传输故障及慢文件边界。
依次验证：探测失联/恢复、租约心跳恢复、过期拒绝旧句柄、boot fencing、响应丢失不重放、
慢文件不阻塞 Probe、取消时物理写入完成前不释放 owner。固定每次源码身份并记录结果。

## 持久状态边界

- 正常业务写入仍由既有 Shell/File 操作执行；实验只写一次性目录。
- manager/health 是内存状态；租约回收只执行原 owner 的进程清理，不减少会话或插件数据。
- RPC 取消不能作为写入回滚证据；实际线程结束才允许排空。
- 本次不改正式数据库 schema、迁移、备份和消息保留协议。

## 可观察状态与故障范围

```text
AppRuntime monitor ──写入── HostBridgeStatus（内存）
  │                           │
  ├─ Probe/identity           └─ GET /api/runtime/host-bridge
  │                               └─ Web Shell → Dashboard UDS → 聊天提示
  └─ UNAVAILABLE/DEADLINE → 降级 → 继续探测 → 恢复

执行调用 ── OpenManager（一次）→ RPC / Heartbeat
  ├─ 响应丢失：报告可能已生效，交给调用者核实
  └─ NOT_FOUND：旧 manager 已失效，不重建或复用旧句柄
```

状态是 disabled/checking/healthy/degraded，包含连续失败数、最近错误分类和检查时间；
不返回 token、路径或命令内容。浏览器每五秒查询，五秒内未取得响应显示“无法确认”；
这与已知 Bridge 降级不同。状态恢复只代表传输与身份正常，不承诺旧 manager 仍存在。

## 本地证据

- 修复前：同一真实服务的 Probe 注入 DEADLINE_EXCEEDED，原 monitor 结束并抛错。
- `.venv/bin/python docker/debug/host_bridge_reliability.py`：八组真实边界实验通过。
  包含实际 UDS 监听停止/重建、Core 主任务监督存活、公开 Web Shell 到 Dashboard UDS 的状态恢复、
  并发首次登记、心跳恢复、过期拒绝、真实命令响应丢失、业务错误后继续执行、四类文件操作、
  慢写入取消后同文件串行与 shutdown 排空，以及认证/boot fencing。
- `node docker/debug/host_bridge_notice.mjs`：Chromium 加载实际 React 组件，验证降级提示、恢复清除、
  HTTP 失败显示未知和 local mode 不显示。这里的 HTTP 状态由实验控制，不冒充完整在线对话验收。
- 概念基线 44 项通过；源码/测试 pyright、前端 typecheck、plugin_boundary、yoyo、
  control schema 与 Host Bridge 生成物检查通过。

## 本次边界与后续风险

- 不重启或修改线上服务，不建立正式 workspace；生产发布与完整聊天/设备验收尚未执行。
- 永久磁盘阻塞仍会延迟物理排空；本方案不会伪造线程已取消。
- 过期 manager 不自动复活。其原 owner 需要结束旧生命周期、创建新客户端；页面整体连接可继续，
  旧执行工具仍会明确报告失效。
- Standard Tools 的同步 skill capture 仍可能在 Core 事件循环执行宿主能力检查。
  本次不修改同步 capture 公共合同或缓存资产生命周期；该路径不属于已验证的文件 I/O 非阻塞保证。
- Memoh 的参考价值是限定故障范围与由连接 owner 恢复；未照搬其容器供应状态或重试业务操作。

# 0008 · 插件运行时只发布已提交快照

- 状态：accepted
- 日期：2026-07-28
- 关联条款：PLG-001～PLG-008、GOV-005、TST-006～TST-008

## 背景

旧插件生命周期用一个 `initialize()` 同时承担候选准备、正式数据访问和后台任务启动。
候选快照在最终校验期间还会短暂成为普通 reader 的 current snapshot。进程内回滚可以恢复
部分内存指针，却无法说明进程在端点切换、快照提交或旧代排空之间崩溃后应该恢复哪一代。

插件位于多个独立仓库。只在核心仓库复制接口测试，或让 CI 跟随插件默认分支，无法证明
某个确定的核心版本与确定的插件组合能够一起发布。

## 决定

1. 插件只支持 API v2。插件类显式声明 `api_version = 2`；旧 `initialize()` 不再兼容。
2. `prepare()` 只使用候选代际的 KV staging、事件 staging 和资源 scope，不取得正式
   `data_dir`、session、memory 或 LLM，也不启动后台任务。
3. 所有门控通过后，`activate()` 在同步提交临界区取得正式资源并启动新代任务。随后只做
   一次 current snapshot 指针切换；普通 reader 永远只能租用 committed snapshot。
4. 指针切换后同步调用旧代 `retire()`，阻止它再接新工作；已有 lease 继续使用原快照。
   lease 排空后再调用 `terminate()` 并逆序清理 scope。
5. 每次热重载把阶段写入 workspace 内独立 SQLite journal。启动时丢弃未提交候选；已经提交
   但未完成 drain 的事务必须按精确 source revision 恢复，否则启动失败并暴露不一致。
6. 核心和外部插件使用一个发布锁固定合同检查器与全部插件的完整 commit SHA。CI 从公开
   HTTPS 仓库检出这组提交，在只读源码挂载和一次性 workspace 中运行静态合同、原子热重载、
   全插件启停和 Fitbit 托管进程场景。

```text
┌──────────────┐    ┌────────────────┐    ┌──────────────┐
│ prepare shadow│───▶│ validate gates │───▶│ commit pointer│
└──────────────┘    └────────────────┘    └──────┬───────┘
                                                  ▼
                                        ┌──────────────────┐
                                        │ retire old       │
                                        │ wait old leases  │
                                        │ terminate + clean│
                                        └──────────────────┘
```

## 理由

这把用户可见的提交点收敛为一个动作：新请求要么完整看到旧代，要么完整看到新代。候选失败
不会产生普通流量可见的脏读，旧请求也不会因为新代发布而丢失自己的 handler、skill、tool、
job、MCP 或 dashboard 绑定。

API v2 把可暂存工作与正式副作用分开，使 Core 能在候选失败时清除资源而不伪装外部效果已经
回滚。journal 不承诺撤销已经发生的远端效果；它负责让重启后的恢复选择明确、可检查且
fail-loud。

## 影响

- 插件作者必须把纯准备放入 `prepare()`，把 task 和正式路径相关动作放入 `activate()`。
- `retire()` 是同步停止接新工作的通知；`terminate()` 是 drain 后的异步最终清理。
- 没有写过 KV 的候选提交不会创建空 `.kv.json`；确实发生的候选 KV 修改只在 commit 落盘。
- watcher 只负责发现 revision；业务请求仍以 snapshot lease 决定所见代际。
- `docker/debug/plugin-api-v2.lock.json` 是本次跨仓库发布组合身份，不能使用 branch 或 tag
  代替完整 SHA。

## 验收

- 校验期间 current snapshot 仍是旧代，且 validating snapshot 无法被普通 reader 租用。
- 无效源码、prepare 失败、activate 失败和提交失败都保留旧代及原 plugin-data。
- 新代提交后，tool、event、job、skill、MCP、service 和请求在同一快照切换；旧代 writer
  被 fence，旧 lease 排空后资源只清理一次。
- 进程在 journal 各阶段重启时得到确定的 discard、restore 或 fail-loud 结果。
- 锁定的 21 个插件全部通过 API v2 静态合同；Docker Gate 能完成原子热重载、19 个可运行
  外部插件全量启停，以及 Fitbit 进程单实例、热重载、禁用和用户数据不变检查。

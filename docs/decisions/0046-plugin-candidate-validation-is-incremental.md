# 0046 · 插件候选只重建依赖闭包

- 状态：accepted / implemented
- 日期：2026-08-27
- 关联条款：PLG-001～PLG-004、PLG-008～PLG-010、PLG-014
- supersedes：`plugin-candidate-root-isolation-task-contract.md` 中“候选重建全部 v3 participants”
- superseded by：无

## 背景

旧实现为每个 candidate 重新 import、挂载并复制全部 v3 插件。修改一个无状态插件也会再次启动 Akasha，并把 `memory/`、`sessions.db` 投影到验证区。普通文件复制不是数据库快照；Akasha 读取到不一致副本时会重建索引。验证因此改变了与候选无关的生命周期和持久状态。

## 决定

candidate 以 stable snapshot 为基线，只创建一个增量 Root：从变更插件的 `inject` 出发，根据 stable Root 真实记录的 Service provider owner，递归计算上游依赖闭包。Core 只 clone、挂载并投影这个闭包；未选中的 Fiber、Effect、module 和 workspace 状态继续属于 stable，候选不得执行其 listener。

```text
stable snapshot
│
├─ 未受影响插件 ───────────────┐
│                              │ immutable catalog selection
└─ provider graph ── closure ──┼─ candidate RuntimeSnapshot
                               │
candidate plugin ─ incremental Root
```

RuntimeSnapshot 把 stable 中未替换 owner 的不可变 catalog contribution 与增量 Root 的 contribution 合并。重复 Service、catalog 名称冲突、required dependency 缺失、manifest/credential 不一致仍在 latest 发布前 fail-loud。snapshot identity 只由最终逻辑拓扑、generation 与最终 catalog 内容计算，不由它物理来自一棵或两棵 Root 决定。

验证期不派发 `runtime.started`/`runtime.stopping`：这两个事件代表正式 effect 准入，scheduler timer 等能力也只存在于 formal Root。候选只验证 lifecycle listener 的注册合同与拓扑；正式 Root 才在旧 owner 停止后启动。

显式 promotion 是同一份正式数据的 owner 交接，不是数据复制，也不是让 stable 与 latest 同时挂载正式数据：publication owner 先停止旧 snapshot 接收新 lease，排空已有 lease，销毁验证 Root，再停止并销毁旧 formal Root，使它释放全部 reader、writer 和 Effect；随后新 formal Root 打开相同的 production 路径。新 Root 完整启动后才提交 stable pointer 和 snapshot。失败时 latest pointer 保留候选事实，stable pointer 不动，并用 stable generation 在相同路径重建旧 formal Root。本决定不改变父 Turn 授权或恢复日志。

## 理由

- `Service provider → inject consumer` 是运行时已经拥有的依赖事实，不新增静态 `provides` 清单或第二套图。
- 复制范围由依赖关系决定，不由插件名称、数据大小或 Akasha 特判决定。
- `Root/Fiber/Effect` 继续拥有生命周期；overlay 只拥有一次不可变 snapshot 选择，不取得插件状态所有权。
- `stable` 指向当前正式 owner，`latest` 指向已验证候选；双指针表达发布状态，不表示两套正式 owner 可以并存。
- 无关 stateful 插件既不重挂载也不复制；候选真正依赖的 provider 才在隔离区重建。

## 失败与数据边界

- 未知 required Service：candidate Fiber 保持 pending，latest 不发布。
- candidate 删除仍被 stable consumer 要求的 Service：完整选择图缺依赖，latest 不发布。
- candidate 与未替换 owner 重复提供 Service 或 catalog key：拒绝候选。
- candidate closure 的 workspace/data 仍按声明采用 isolated copy 或 shared read；未进入 closure 的正式数据零读取、零复制、零清理。
- Python 插件仍是受信代码；绕过 Context 直接访问任意绝对路径不由该机制伪装成安全沙箱。

## 验收

- 无关 stateful fixture 在 candidate prepare 前后 mount count 恒为 1。
- fixture `memory/` 与 `sessions.db` 不出现在 candidate validation root。
- candidate 的直接、传递上游 provider 进入隔离 Root；无关插件不进入。
- 申请不存在 Service 的 candidate 在 latest 前被拒绝，`apply` 不执行。
- 最终合并后的 channel catalog 必须与所有 active manifest 一致。
- candidate event 派发保留正式 Root 的 key/listener 顺序与并发语义。
- candidate 与 formal snapshot 的逻辑 identity 相同；discard、失败和 cancellation 清理增量 module/data，stable Root 不变。

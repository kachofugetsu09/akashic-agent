# 0004 · 跨仓库证据绑定不可变组合

- 状态：accepted
- 日期：2026-07-18
- 关联条款：GOV-005、MOB-002～MOB-004、TST-006～TST-008

## 背景

移动端 stacked PR 同时依赖核心协议、当前 runtime、插件仓库、Android Room schema 和设备环境。只记录分支、PR 链接或数据库版本号，会把多个会移动或发生分叉的对象压成一个含糊身份：插件更新后旧 Gate 可能被误复用，同一个 Room `user_version` 也可能代表不同分支产生的不同 schema。

本次评审还暴露了共享 worktree 的写入竞态。并行 agent 可以从同一份材料审查，但若多个 writer 在同一 worktree 提交，接手者无法可靠判断 merge 和文件变化由谁产生。

## 决定

1. 跨仓库验收把 consumer、协议 source、实际 runtime、provider 和 scenario catalog/profile 分别固定到不可变 identity，再报告它们的组合。
2. 协议历史源和当前 runtime 可以是不同 commit。核心保留已发布协议归档，使旧客户端的 `source_repository + source_commit + source_path + hash` 始终可重建。
3. Provider 的 GitHub ref 在 Gate 开始时解析成 commit SHA；ref 后续移动不会抹掉旧报告，但新组合不能复用旧通过状态。
4. 数据库迁移按真实表、列、索引和外键识别已知 schema lineage，不只读取 `user_version`。已知 lineage 汇合到唯一目标 schema，未知或部分匹配形状 fail-loud。
5. 每个 worktree 同时只有一个 writer。并行审查默认只读；writer lease 记录 repository、worktree、branch、owner、base HEAD、允许路径和状态。产生修改后只能用提交后的 commit 交接；旧 writer 完成或被明确中断前不能转移 owner。
6. CI、Docker、隔离互操作和 Pixel/ADB 是不同证据层。设备 Gate 从干净 source commit/tree 构建，并在构建后、首次 ADB 前再次拒绝任何 source drift；再从 APK 读取实际 app/test identity，使用 run-specific application ID，并在安装前以 `pm list packages -u` 拒绝任何 collision。安装禁止 replace，清理权只属于本进程成功安装的 package；instrumentation 必须核对执行数量、指定方法与成功终态，cleanup 后才写唯一 Gate 终态。设备结果不冒充 CI required check，客户端隔离也不冒充 Mobile Lab workspace 隔离。

## 理由

这让每次通过都回答一个精确问题：“这一组确定的源码、协议、provider、场景和环境是否满足同一组语义不变量？”它不会把浮动链接、安装缓存、版本号或共享目录当成身份，也允许维护者在任何一个输入变化后准确决定需要重跑哪些边界。

## 影响

- 跨仓库 PR 报告增加 source/runtime/provider/scenario identity。
- Stacked PR 的栈顶运行累计迁移矩阵和最终 schema identity 检查。
- Observe 等插件由 canonical GitHub revision 安装到空 plugin home 后验证，不直接使用正式 cache。
- Subagent 可以并行给出 findings；被主 agent 接受的修复在独立 writer worktree 完成，再以 commit 传播。
- Pixel 7 结果只补充 Android OS、Room、通知、文件和 Compose 真实行为；run-specific app/test package 在测试后清理，正式 package 前后身份必须相同。

## 验收

- 任一跨仓库报告能唯一定位 consumer、协议、runtime、provider 和 scenario 输入。
- Provider ref 或任一 source digest 改变后，旧报告不会被复用为新组合通过。
- 同一 `user_version` 的多个已知 schema 都有显式迁移测试；未知形状不会 destructive fallback。
- Worktree handoff 能指出唯一 writer、允许路径、交接 HEAD 和 dirty state，且旧 writer 已停止写入。
- PR 将 CI 与设备结果分开列出；source commit/tree、APK identity、`pm list packages -u` inventory、collision、安装所有权、instrumentation oracle、正式 package 前后状态、cleanup 和唯一 Gate 终态均有证据。

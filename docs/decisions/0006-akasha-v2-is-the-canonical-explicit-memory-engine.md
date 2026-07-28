# 0006 · Akasha V2 是显式记忆的唯一算法实现

- 状态：accepted
- 日期：2026-07-27
- 关联条款：MEM-009、SES-003、GOV-005、TST-002、TST-005

## 背景

旧 Akasha 把在线插件、重放脚本、图快照、Dashboard 检查器和实验性 fast/slow
路径放在同一个宿主目录。在线增长与重放分别维护相似但不相同的状态转换，难以证明
同一组 turn 会得到同一张图。显式 recall 还可能与自动上下文查询共享可变状态，使一次
只读工具调用影响下一次学习。

独立仓库 `akasha-v2-engine` 已经把稀疏索引、模式补全、突触可塑性、遗忘和确定性
持久化整理为一条 `MemoryCycle`。宿主需要采用这条实现，同时保留当前 memory engine
协议和 Agent 的输出约定。

## 决定

1. `akasha-v2-engine/src/akasha` 是 Akasha 算法的唯一源码。宿主只保存字节一致的镜像，
   并在 `plugins/akasha/UPSTREAM.json` 固定 upstream commit、Git tree 和内容摘要。
2. 在线提交与离线重放都调用同一个 `MemoryCycle.retrieve → MemoryCycle.commit`，
   不在宿主重写图学习或遗忘规则。
3. 自动 `intent=context` 查询可以产生一次临时 retrieval ticket。只有对应
   `TurnCommitted` 可以消费它并学习；`recall_memory` 使用 `effect=read_only`，
   不替换 ticket，也不更新图。
4. `sessions.db/messages` 和匹配的 `message_embeddings` 是固定重建输入。
   `akasha.db` 与稀疏索引都是派生 sidecar。完整重建在首次备份或目标写入前审计全部
   合法对话向量，缺失、错模型、错维度、非有限或零向量都 fail-loud。
5. scheduler、带 `skip_post_memory` 的消息以及没有完成 assistant 回复的中断 turn
   既不在线学习，也不进入重放。新中断占位使用结构化 `skip_post_memory`；历史
   assistant 占位只兼容精确正文 `[interrupted]`，避免两条路径对学习样本的定义不同。
6. 用户可见结果保留“左脑记忆：精确回忆”和“右脑记忆：联想补全”两条 lane。
   左脑最多五条 dense 命中；右脑按稳定 turn ID 去重后排除左脑项。每条 lane 内按时间
   从近到远显示，日期为 `MM-DD`，assistant 正文最多 50 个字符。
7. 旧 `/akashalast` 和 Akasha 图 Dashboard 不再属于运行时接口。`AkashaPlugin`
   提供桌面端与移动端只读 Inspector：它只从 V2 sidecar 重建线索、可选扩散路径、
   左右脑召回和实际 Prompt 记忆块，不提供任意 SQL、图遍历或记忆修改；记忆能力仍由
   `MemoryPlugin` 提供。

## 理由

单一 `MemoryCycle` 把“同算法”从代码相似提升为同一个状态机。临时 ticket 与提交事件
形成 read-before-write 边界：模型看到的自动记忆可以参与当前回答，只有真实持久化
完成的 turn 才改变未来检索。显式 recall 保持只读后，Agent 是否调用工具不会偷偷
重排下一轮学习输入。

固定 upstream 身份避免宿主镜像悄悄漂移。严格向量预检保护 MEM-009：重放不能以
“成功退出”掩盖合法学习样本的向量缺口。把 scheduler 和未完成 turn 的排除规则放在
在线与重放共同边界，可以避免重复任务占满连接预算，也避免补算一个从未在线学习的
中断事件。

## 影响

- 宿主 memory engine 仍使用 `engine = "akasha"`，调用者不需要认识 V2 内部类型。
- 更新算法必须先在 upstream 提交，再重新镜像并更新三项身份；禁止直接改宿主镜像。
- 现存旧 Akasha 配置和数据库不能被新 loader 猜测兼容。部署前需要在隔离快照完成
  embedding 审计、全量重建和原子 sidecar 切换；本决策不授权修改正式 workspace。
- Dashboard 与移动端提供只读 Inspector，但不恢复私有图入口。Inspector 没有新的
  持久化 owner；逐节点扩散 capture 缺失时必须明确显示为“未保存路径”，不能推断为
  没有发生扩散。

## 验收

- 镜像校验能发现 commit、tree、文件集合或任一字节差异。
- 真实宿主插件动态加载、fresh runtime 启停和 `recall_memory` 工具调用通过。
- 同一隔离 `sessions.db` 的在线提交与干净重放得到相同 canonical logical state。
- 不同 `PYTHONHASHSEED` 的完整重放得到相同 logical state。
- 缺少合法对话 embedding 时，重建在备份和目标数据库写入前失败并生成缺口报告。
- 中断 turn 保留原始消息，但不产生 embedding 要求、稀疏节点、hub 或图关系。
- Docker runtime 使用独立 workspace 完成 query、持久化、自动上下文、显式 recall 和
  下一轮提交；正式 workspace 的文件摘要与 mtime 不变。
- Inspector 桌面 API、当前回复下方的移动端召回和移动端检索列表都只读取 V2 sidecar；
  左右脑内容与实际 Prompt 重建结果一致，且不暴露 Akasha Graph。
- 移动 recall 卡片使用 `akasha.recall-card.v1` 有界投影，不包含完整正文或未渲染的
  Inspector 字段；最坏 Unicode fixture 的编码结果小于 16KiB。

完整调用链、状态所有权和迁移 Gate 见
[`../design/akasha-v2-runtime-migration.md`](../design/akasha-v2-runtime-migration.md)。

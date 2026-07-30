# 0009 · Akasha 移动卡片完整保留有界召回 lane

- 状态：accepted
- 日期：2026-07-30
- supersedes：[0007](0007-mobile-plugin-control-and-data-planes-are-explicit.md) 第 5 项的每 lane 五项与 16KiB 上限
- 关联条款：MOB-006、PLG-011、TST-006～TST-008

## 背景

Akasha 领域查询已经按用途限制召回集合：自动上下文的精确 lane 最多五条，模式补全
lane 受 `context_recall_limit` 约束且当前最多四十条，显式 recall tool 单次最多请求
四十条。Agent Prompt 和桌面 Inspector 使用这些领域层选出的集合。

移动投影在此之后又把每条 lane 固定裁成五条。真实只读数据库中，92.49% 的已记录
轮次包含超过五条原始 recall item，因此右脑卡片经常隐藏 Agent 实际看到的关联。
这不是传输边界要求：当前最坏的 Agent lane 是左五条、右四十条，使用 card-v1
预览字段编码约 9.4KiB，gzip 后约 3.7KiB。真实库中聚合后的 tool lane 最大为左
三十八条、右三十七条；四条 lane 各四十条的最坏 Unicode fixture 为 110,373B，
仍低于 Mobile 已有的 192KiB 响应上限。

## 决定

1. Akasha `recall.current` 完整投影每条领域 lane 已经选出的 N 条，不在移动 DTO
   再应用固定 top-k。条目顺序和计数与上游 lane 一致。
2. 召回选择职责留在 lane 生产者。当前 dense、completion 和单次 tool recall 的领域
   上限不变；card-v1 仍只包含 100 字用户预览、50 字助手预览、时间和可选分数。
3. Mobile 为 DTO 中每条 lane 创建全部 N 个列表项。长列表使用
   `content-visibility` 和 intrinsic block size 跳过离屏布局与绘制；该优化不得改变
   DOM 条目、顺序或计数。
4. 完整最坏 Unicode fixture 必须低于既有 192KiB 响应边界。超过边界时查询
   fail-loud，不允许尾部丢弃、分页或客户端静默裁切。

```text
┌─────────────────────────┐
│ Akasha lane producer    │  dense ≤ 5；completion ≤ 40；单次 tool ≤ 40
└────────────┬────────────┘
             │ 已选出的完整 N 条
             ▼
┌─────────────────────────┐
│ card-v1 semantic DTO    │  只裁字段与单条预览，不裁条目
└────────────┬────────────┘
             │ compact JSON + gzip，整体 < 192KiB
             ▼
┌─────────────────────────┐
│ Mobile list             │  N 个列表项；离屏项延后 layout/paint
└─────────────────────────┘
```

## 理由

条目是否属于召回应由理解 query、图传播和去重规则的 Akasha 领域层决定。移动层缺少
这些语义，固定 top-k 会让展示结果与 Agent 实际上下文分叉。字段投影、预览长度、HTTP
压缩和整体响应上限能够约束传输成本；`content-visibility` 能约束长列表初次布局和绘制
成本。它们都不需要删除已经选出的记忆。

保留 card-v1 schema 是兼容变化：字段形状没有改变，数组只恢复到领域层原有的有界
长度。已有 Mobile 会按数组长度渲染，不需要数据库或本地缓存迁移。

## 影响

- 右脑或 tool lane 超过五条时，移动卡片会展示更多列表项；左脑常见数量不变。
- HTTPS payload 和 WebView DOM 数量随领域 lane 的 N 增长，但仍受 192KiB 协议上限和
  四十条领域上限约束。
- `akasha.recall-card.v1` 的字段、缓存键和 revision 协议不变；插件 revision 随资源
  内容正常变化。
- 不修改 sessions、Akasha 图、召回排序、Agent Prompt 或桌面 Inspector。

## 验收

- 四条 lane 各输入四十条最坏 Unicode 内容时，投影后每条仍为四十条，完整 card
  小于 192KiB，且不包含完整正文或 Inspector 调试字段。
- JavaScript 不使用 `slice` 或固定 top-k，CSS 对记忆列表项启用离屏布局与绘制优化。
- Akasha 上游单测、Agent 插件单测、Node UI 合同和隔离移动 Gateway 场景通过。
- 镜像 Gate 证明 Agent 的 `plugins/akasha` 与 `UPSTREAM.json` 固定的上游提交逐字节一致。

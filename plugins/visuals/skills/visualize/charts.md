# 数据图写法

## 先定形式，最后定颜色

1. 选形式（有时答案不是图）；
2. 按用途分配颜色；
3. 加图例或直接标注；
4. 看预检截图：标签碰撞、溢出、深色下是否可读。

| 数据是… | 用 | 不用 |
|---|---|---|
| 一个当前值（可带趋势） | hero 数字 + `ak.spark` | 只有一根柱的柱图 |
| 几个关键数 | 一行 hero 或指标卡 | 分组柱图 |
| 七类以上都重要 | `table.dtable`（可配一张图） | 更多颜色 |

| 要表达 | 形式 | 颜色 |
|---|---|---|
| 比大小 | `ak.bars`、`ak.barlist`；二维规律用 `ak.heatmap` | 单色 |
| 随时间变化 | `ak.line` | 单色或少量系列 |
| 区分几个系列 | 堆叠柱、多条线 | 系列色 |
| 一个系列是重点，其余是背景 | 突出一条，其余用 `'gray'` | 一色加灰 |
| 部分与整体 | 堆叠柱（名字长就横向）、`ak.share`、`ak.treemap` | 系列色 |
| 每项的前后对比 | `ak.dumbbell` | 同色深浅 |
| 两个量的关系 | `ak.scatter` | 单色，分组时系列色 |

- 默认单色。故事是"这一项涨了"，就是突出一项，不是给每项一种颜色。
- 颜色跟着实体走，不跟着排名走；不用彩虹色；最多四个系列，再多并入"其他"或拆成几张小图。
- 只有一个坐标轴：两种量纲就画两张图，不画双 Y 轴。
- 数值、轴标签和图例文字用文字色，身份靠旁边的小色块，不把文字染成系列色。
- 两个以上系列加图例，四个以内优先在线尾或柱旁直接标注；只有一个系列不加图例，标题就是它的名字。

## 编辑式版式

- `h3.headline` 标题直接写结论（结论词用 `em`），可选 `p.dek` 一句补充；
- 1–4 个关键数字用 `div.hero > div > b.v`（单位放 `small`）+ `span.k` 短标签 + `span.d` 说明；
  汇总数字也可以用 `div.cards > div.metric > span + b`；
- 每张图上方 `p.chart-title`（可加 `p.chart-sub`），峰值、拐点和阈值直接注释在图上；
- 明细用 `table.dtable`（数字列的 th 和 td 都加 `n`，合计放 tfoot），`div.split` 左图右表，`p.foot` 写口径与来源。

## 内置 ak

不要自己画坐标轴：放空 div，在最后的 script 里调用。

- `ak.bars('#id', {x, series:[{name, values}], stack, annotate:[{at, text, sub}], refLine:{value, label}})`
  画分类或按天计数，默认堆叠；加 `horizontal:true`、`sub`、`range:[{lo, hi}]` 画估计区间与阈值。
- `ak.line('#id', {x, series, milestones:[{at, text, sub}], spans:[{from, to, label}], annotate:[{at, text, series}], projectFrom, yMin})`
  画趋势，多系列在线尾自动标名，`projectFrom` 之后画虚线外推。
- `ak.share('#id', {x, series, annotate})` 画构成随时间的 100% 份额。
- `ak.scatter('#id', {points:[{x, y, label, size, group}], xTitle, yTitle, xFmt, yFmt, highlight, refX, refY, trend})`
  画两个量的关系，`size` 变气泡。
- `ak.dumbbell('#id', {x, from:{name, values}, to:{name, values}})` 画每行前后或两种口径的对比。
- `ak.heatmap('#id', {rows, cols, values})` 画星期 × 小时这类两个维度的规律。
- `ak.treemap('#id', {nodes:[{name, value, group}]})` 画层级与大小。
- `ak.barlist('#id', {items:[{label, value, note}], limit})` 画排名。
- `ak.spark('#id', values)` 是放进 hero 或表格的小趋势线。

通用参数：`fmt` 取 `'raw'`、`'%'`、`'wan'`、`'si'` 或函数，`unit` 是单位，`label` 是可访问名称；
series 可写 `color: 1–6`、`'gray'` 或 `'hot'`。x 写 YYYY-MM-DD 会自动缩写；`ak.fmt(value, fmt, unit)`
让 hero 和表格用同一格式。`ak` 对不一致的数据直接报错，按报错修正。

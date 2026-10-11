# Harness 性能回归检测

在 CI 上用开源的 [harness-perf-benchmark](https://github.com/KonghaYao/harness-perf-benchmark) 对比改动前后 akashic 的 harness 性能，
回答一个问题：**这次改动有没有让 harness 变贵**。结果只报告，不阻塞合并。

## 什么是 harness 性能

> **harness 性能 = 在模型不花时间的前提下，akashic 驱动一段固定的 agent 工作所消耗的本机资源。**

模型由 mock 扮演：首包零延迟，响应内容由剧本逐条给定，因此测到的全部开销都来自 harness 自身，
包括进程启动、插件加载、上下文构建、请求序列化、流式解析、工具执行、账本持久化与关闭。
网络、模型推理与 token 数不在其中。

### 被测工作

| 要素 | 取值 | 理由 |
|---|---|---|
| 剧本 | 上游 `gen-long-run.ts --turns 100 --tool shell --args command+description` | 与公开榜单同一生成器；每轮约 3KB 回复正文 + 1 次 shell 工具调用，上下文逐轮累积到约 550KB |
| 工具命令 | 第 n 轮为 `printf 'bench-tick-000n\n'` | 上游原命令的输出依赖目录内容；换成确定输出才能逐轮校验 |
| 生命周期 | `gateway` 冷启动 → `exec --new --final-only` 跑完剧本 → SIGTERM | 覆盖用户一次完整使用：启动、对话、停服 |
| 组合 | 正式发行版 + `base` bundle，本地执行模式 | 与发布镜像同一条安装链（`build_plugin_distribution.py` → `prepare-wheels` → `ensure_bundle`） |
| 缓存状态 | 每次从模板恢复 workspace，插件 home 保留；每侧先空跑一次并丢弃 | 测“重启后的稳态”，不把首次安装的一次性成本混进每次运行 |

### 指标

主指标与上游 `scripts/perf/score.ts` 的 CU 同口径（CPU 与内存 1:1）：

```
CU = CPU 核·秒 + 内存 GiB·秒
CPU 核·秒   = wait4 回收的整棵进程树 user+sys（gateway、exec 客户端、工具子进程、测量外壳）
内存 GiB·秒 = ∫ 进程树 RSS dt（/proc 每 100ms 采样一次，每拍刷新进程树）
```

CPU 取 wait4 的精确值，不用采样积分：procfs 的 CPU 计数以 10ms 为一个 tick，而工具子进程只活几毫秒。
内存只能靠采样，所以由上游采样器按拍积分。

分项指标用于定位衰退出在哪一段：

| 指标 | 含义 | 主要反映 |
|---|---|---|
| 启动段 | harness 进程启动 → 第一个模型请求 | 导入、插件发现与加载、迁移、就绪 |
| gateway 就绪 CPU | gateway 在控制 socket 出现时自身已用的 CPU | 启动段里与调度无关的那部分，噪声最小 |
| 每轮工具循环 | (最后一个请求 − 第一个请求) / 轮数 | 每轮的上下文构建、持久化、工具执行、流式解析 |
| 收尾段 | 最后一个请求 → 进程退出 | 关闭顺序、checkpoint、Web 服务器退出 |
| exec 客户端 CPU | 插件命令客户端自身的 CPU | CLI 冷启动 |
| 进程树 RSS 峰值 | 采样期间最大的进程树 RSS | 机器规格需要按峰值配 |
| 模型请求总量 | 101 个请求体的字节数之和 | 上下文体积；几乎确定，0.1% 以上的变化都会报告 |

### 有效性

一次运行同时满足以下条件才计入统计，否则在报告中单独列出：

- runner、gateway、exec 都以 0 退出；
- mock 恰好收到 101 个请求；
- 第 n 个请求的最后一条消息恰好是第 n 轮工具的输出 `bench-tick-000n\n`。

这样可以排除“更快是因为少做了事”：漏掉工具调用、提前退出、截断上下文都会让运行失效。

## 如何判定衰退

两侧的发行版都从各自的提交构建，测量代码统一使用候选侧这一份。
A（基线）与 B（候选）在同一台机器上按 AB、BA、AB… 交替运行，抵消机器状态随时间的漂移。

对每个指标：

1. 每组配对计算 log(B/A)，取 Hodges-Lehmann 估计（所有 Walsh 平均的中位数）作为变化量。
   它在小样本下比中位数有效，又不像均值那样被单次停顿拖偏；
2. 对配对重抽样 4000 次，取估计值的 2.5%～97.5% 分位作为置信区间；
3. 置信区间整体高于 1 + 阈值，判为**衰退**；整体低于 1 − 阈值，判为**改善**；否则为无可分辨变化。

阈值：CU、CPU、内存为 3%，分段与峰值为 5%，收尾段为 10%（绝对值小、相对波动大）。
请求总量阈值为 0.1%：它几乎是确定的，只有工具结果里的 `wall_time_ms` 位数变化会带来约 0.03% 的抖动。
系统提示里含 workspace 绝对路径，所以两侧的 work 目录名长度必须一致（compare.py 用 side-a / side-b）。

“无可分辨变化”不等于“没有变化”，只说明这次的配对数分辨不出来。分项指标通常比 CU 更早显出问题：
例如每轮多出几毫秒，在 CU 里会被启动段的波动淹没，但在“每轮工具循环”里很清楚。

GitHub runner 是共享虚拟机，绝对值比开发机慢且有波动，所以**只看同一次 job 内的 A/B 比值**，
不要拿不同 job 的绝对 CU 互比（上游 README 的跨批次规则也是如此）。

## 本地运行

```bash
git clone https://github.com/KonghaYao/harness-perf-benchmark /tmp/hpb
git -C /tmp/hpb checkout e89a895a27a6634bded08f93c886b1a70bb15d63
(cd /tmp/hpb && bun install --frozen-lockfile)
git worktree add --detach /tmp/ak-base origin/main
.venv/bin/python scripts/harness-performance/compare.py \
  --benchmark /tmp/hpb --a-root /tmp/ak-base --b-root . \
  --work /tmp/hp-work --pairs 6 --report /tmp/hp-report.md
```

需要 Linux（采样器读 `/proc`）与 bun。被测 HOME 是空目录，但 shell 工具的 `bash -lic` 快照仍会读
`/etc/profile.d/`；CI 去掉了 runner 的 `bash_completion.sh`，本地机器若装有它也应同样处理。插件从各侧 `HEAD` 提交构建，核心代码运行工作树，
所以对比未提交的插件改动前要先提交。端口 18457（mock）与 18458（Web）需要空闲。

## 文件

| 文件 | 作用 |
|---|---|
| `compare.py` | 生成剧本、准备两侧、交替测量、校验、统计与报告 |
| `prepare.py` | 为一侧构建发行版、安装 base bundle、注册 mock 模型，产出 workspace 模板 |
| `measured.py` | 被测命令：gateway → exec → SIGTERM，记录 wait4 CPU 与就绪时刻 |
| `playground/perf-demo.ts` | 拷入上游 `playground/akashic/` 的接入层：Linux 采样后端与运行编排 |
| `playground/mock.ts` | 上游 mock 外加逐请求记录，用于校验与请求总量 |

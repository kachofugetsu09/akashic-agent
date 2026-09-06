# Computer driver 基线

直接复用 `trycua/cua@aabb2082c170289256f0c8d9db4cce094c778578` 的
`libs/cua-bench/datasets/cua-bench-basic`：题目、HTML、任务变体、参考步骤和判分器均不修改。
`adapter.py` 只将上游 `DesktopSession` 接到现有 Computer Gateway。
`run.py` 管理一次性容器和证据，调用上游 `Environment.reset/solve/evaluate` 与 `Tracing`。

```text
┌───────────────────────────────────────┐
│ Cua Basic 原题、参考步骤、判分器         │
└───────────────────┬───────────────────┘
                    ▼
          DesktopSession 薄适配
             │             │
      动作与截图           准备与判分
             ▼             ▼
      原始 Gateway     Chromium 页面
             └─────┬───────┘
                   ▼
       独立 Computer 容器与空 profile
```

此阶段不启动模型或 Akashic Core，不加载 SessionDB、记忆、调度、插件管理器或正式 workspace。
这份报告衡量的是上游参考步骤经过当前 driver 的表现，不是 Agent 成功率，也不是上游原生环境排行榜成绩。

## 安装

在本 Git worktree 根执行；依赖和上游 checkout 放在忽略的 `benchmark/data/`，不修改项目 `.venv`。

```sh
git clone --filter=blob:none --no-checkout https://github.com/trycua/cua.git benchmark/data/cua
git -C benchmark/data/cua sparse-checkout set libs/cua-bench
git -C benchmark/data/cua checkout aabb2082c170289256f0c8d9db4cce094c778578
uv venv --python 3.13 benchmark/data/venv
uv pip install --no-sources --python benchmark/data/venv/bin/python \
  -r benchmark/computer_driver/requirements.lock ./benchmark/data/cua/libs/cua-bench
curl --fail --location \
  https://cdn.jsdelivr.net/npm/@tailwindcss/browser@4.1.18/dist/index.global.js \
  --output benchmark/data/tailwind-4.1.18.js
```

上游 Python 包声明了 `cua-agent` 等依赖，安装依赖不会启动这些组件或调用模型。
`--no-sources` 仅跳过上游指向缺失相邻目录的开发路径，使用锁定的发行依赖。
运行前检查上游 commit、clean 状态和样式文件 SHA-256，关闭 Cua 遥测。

## 运行

需要本地已有的 Computer 镜像。镜像立即解析为不可变 image ID。
测试使用项目既有 user namespace seccomp 文件、零 capability 和 no-new-privileges；不关闭浏览器 sandbox。

```sh
benchmark/data/venv/bin/python benchmark/computer_driver/run.py \
  --image <existing-computer-image> \
  --output benchmark/data/driver-baseline
```

默认运行所有原始任务变体。快速验证可加 `--task click-button --max-variants 1`。
相同命令加 `--suppress-actions` 并指定新输出目录，可运行不发送输入的负对照。
未来候选用 `--source /absolute/candidate/worktree` 指定 driver 源码，其余条件保持一致。

每次输出保留源码副本与摘要、镜像 ID、浏览器版本、上游版本、筛选条件、逐题动作、
原始 reward、前后截图和上游 trajectory。目录已存在时拒绝覆盖；不会自动清理旧报告。
命令退出 0 表示评测和容器清理完成，题目是否成功应读取 `results.json`。
失败或不支持的题目保留原始状态，不计为成功。

## 解释结果

- 固定 1280×800、DPR=1、随机种子 42 和 Tailwind 4.1.18。题目在真实 Chromium 全屏页面中展示，
  不复现上游 native webview 的窗口尺寸；这些是已声明的评测环境差异。
- 第一阶段只测桌面 `/input` 与 `/screenshot`。结构化 Browser 工具、后台多窗口和 Wayland 不在本次覆盖内。
- `initial_reward` 区分本来就满足目标的题目。负对照应与初始 reward 比较，不能把原本已完成的任务算作动作生效。
- 上游参考步骤不是完美解法。例如 native `<option>` 的坐标点击依赖窗口后端；视频音量参考步骤会直接显示滑块。
  这类失败或辅助操作需要结合动作轨迹判断，不能全部归因于 driver。
- 上游 `click_element` 经 `Environment.step`，部分原始步骤直接调用 `execute_action`；比较动作数量用本地动作记录，
  不把上游 step count 当作完整调用数。
- 判分代码运行于可信测试端。本阶段没有模型；未来加入模型时，只能向其暴露 driver，不能提供判分 JS 或答案。

比较前后版本时固定上游、依赖锁、适配器、镜像 ID、分辨率、种子、题目与变体；只改变 driver 源码。
先比较逐题 reward、错误与实际动作，再比较耗时。可靠性结论需要相同条件重复运行。
模型评测另接上游已有的 Agent 入口，不把完整 Akashic runtime 带进 driver 基线。

## 状态与恢复

唯一新增持久事实是此次评测代码和结果目录。测试 profile 在本次容器私有 tmpfs 中；
runner 在结束或失败时删除自己创建的容器，由 Docker 释放 tmpfs。正式 profile、宿主桌面和 Akashic workspace 均不挂载。
driver 源码通过只读挂载进入测试容器。基线源码与恢复点在输出的 `source/`，旧结果不被候选覆盖。
本地交付不修改、提交、推送或部署 driver。

## 源码版对照

`--driver source` 使用已构建镜像中的正式 `/driver/run`；不挂入另一个 driver host，也不加载
Akashic Agent。`source_adapter.py` 只绑定上游动作：option 使用 selectOption，带类型的 input
使用 fill，其他元素点击使用 locator，坐标动作使用 native desktop。与原版 Browser + Sky
对比使用同一绑定；不改变题目、原解法或判分器。

```sh
python benchmark/computer_driver/run.py --driver source \
  --image akashic-computer:driver-source-20260905 \
  --cua /path/to/pinned/cua --css /path/to/tailwind-4.1.18.js \
  --output /tmp/computer-source-results
```

`--suppress-actions` 是负对照，保留观察但不发送 solver 动作。每次创建独立容器、tmpfs profile
和随机 loopback 端口。manifest 保存镜像 ID、源码快照、依赖、成绩和 cleanup 状态。

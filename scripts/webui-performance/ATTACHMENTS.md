# 文本附件预览基准

`serve-attachment-runtime.py` 新建隔离 Runtime，以真实 ArtifactStore 和 MessageLog 追加附件与消息；`compare-attachment-preview.mjs` 通过真实 API 和浏览器测量。没有模型调用、mock API 或 DOM 造数。只在独占性能窗口串行运行，不与构建或其他基准并发。

固定复现对照是基线 `8b0f0afa0167ffd58b90d9337014bfa9c9c21a1c` 与候选 `7866645abda66469553de8e6bd51d99536757478`（PR #1790）：首开附件读取由2次减为1次。下列计数及收益门槛只验收这项变化，不是任意两版比较都必须满足的合同。

## 准备与运行

需要 Linux `/proc`、`ss`、`rsync`、Python 3、仓库 Python/Node 依赖和 Chromium。脚本不安装依赖、不构建、不启动浏览器以外的服务。服务脚本所在 worktree 需已有 `static/dashboard` 与 `static/chat`：运行 `npm run build:dashboard`、`npm run build:chat`。使用仓库虚拟环境启动服务。

分别在干净基线与候选源码上构建，将各自 `static/chat` 保存到不同制品目录；记录 HEAD、源码差异、`chat-app.build` 和文件 hash。不能把同一份或混入其他优化的制品充当基线。以下路径变量由操作者指定；state、output 必须不存在。

```sh
BASELINE_CHAT=/path/to/baseline/chat
CANDIDATE_CHAT=/path/to/candidate/chat
BROWSER=/path/to/chromium
.venv/bin/python scripts/webui-performance/serve-attachment-runtime.py --state /tmp/attachment-run-1 --port 24719
```

服务报告 origin 后，在同一 worktree 的另一个终端设置上述三个变量，依次执行：

```sh
node scripts/webui-performance/compare-attachment-preview.mjs --state /tmp/attachment-run-1 --baseline "$BASELINE_CHAT" --candidate "$CANDIDATE_CHAT" --browser "$BROWSER" --output /tmp/attachment-performance-1 --runs 7 --mode performance
node scripts/webui-performance/compare-attachment-preview.mjs --state /tmp/attachment-run-1 --baseline "$BASELINE_CHAT" --candidate "$CANDIDATE_CHAT" --browser "$BROWSER" --output /tmp/attachment-correctness-1 --mode correctness
.venv/bin/python scripts/webui-performance/serve-attachment-runtime.py --state /tmp/attachment-run-1 --verify
```

最后在服务终端 Ctrl-C 正常停机，保留 state、日志、所有成功和失败报告，不删除失败后重跑的证据。新实验使用新 output；不得按结果调门槛。服务不写原 checkout 或正式 workspace，初始化后只读比较完整消息、附件引用、导入账及附件字节；认证状态另属其真实 owner。

## 首次运行前固定的门槛

性能：27,000字节文本与8,494字节CSV；1440×900、默认CPU；LAN 20ms/100Mbps、弱网800ms/1Mbps。每组7对AB/BA；每次冷首开使用新context，热重开在同页关闭后再点。计时从真实pointerdown到逐行正文和精确元数据正确、字体就绪、两帧显示；settled另等待有限动画及几何连续两帧稳定。不是“弹窗出现”。

- 每种文件的弱网冷首开，content与settled中位数都须同时改善≥20%且≥600ms。
- LAN冷首开及所有热重开，每种文件、每个指标退步≤max(25ms,baseline中位数的10%)。
- baseline冷首开2次请求、candidate冷首开1次；两侧热重开均1次；所有原文oracle必须通过。
- 300ms绝对目标独立报告，不用相对收益代替达标。每组7样本的经验p95接近最大值，不提供尾延迟保证。

正确性只跑LAN：空文、末尾换行、CSV/TSV引号与字段内换行、4999/5000/5001行、512KiB/2MiB/8MiB上下边界、UTF-8截断。真实复制全文和下载均与独立原文字节比较；超过8MiB复制禁用。两份报告都完成才通过，不把9MiB弱网传输当优化收益。

首个样本之前同时冻结baseline与candidate的完整文件名和SHA256清单，后续轮次和恢复不得接受新的源内容。发布临时制品前核对marker、PID、监听端口及realpath；不允许软链接制品或源/目标互相包含。每次安装先核对源清单，再核对安装后的完整清单和真实HTTP JS/CSS字节，全部对照首次冻结值。结束独立尝试关闭浏览器、恢复baseline、核对完整性，分别保存错误；任何失败都会退出非零。报告保存冻结清单、harness/fixture/oracle hash、原始请求状态/时序和全部样本。

默认关闭web-auth，性能和404不证明登录权限。fixture支持`--auth`新实例；不得把本基准结果当作鉴权验收。插件启动日志也须保留，附件链通过不证明无关插件均健康。

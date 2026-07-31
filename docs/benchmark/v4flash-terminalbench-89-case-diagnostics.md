# V4 Flash Terminal-Bench 2.1 逐题诊断

日期：2026-07-31

状态：89/89 个任务均已取得唯一有效结果；两个被 infra/provider 故障污染的
首轮结果已在修复后替代重跑。

关联文档：

- [V4 Flash Harness 实验 Ledger](v4flash-harness-experiment-ledger.md)
- [Benchmark 诊断循环设计](../spark/2026-07-30-agent-benchmark-diagnostic-loop-design.md)
- [0010：Provider 默认输出与 Benchmark 诊断](../decisions/0010-provider-default-output-and-benchmark-diagnostics.md)

## 1. 冻结运行

| 项目 | 值 |
|---|---|
| Frozen candidate | `94823b730520c3bb61411f75b8fdf64dd20054ed` |
| Replacement candidate | `2db5b1ec59685f377a611e8209dfa30743594fbd` |
| Dataset | Terminal-Bench 2.1，89 tasks |
| Model | DeepSeek V4 Flash，`high` |
| Agent 限制 | `max_output_tokens=0`、`max_iterations=0` |
| Compaction | 按生产配置在模型上下文 74% 触发 |
| 并发 | 最多 6 个独立 Docker runtime/workspace |
| Frozen run root | `/mnt/data/coding/akasic-agent-worktrees/benchmark-runs/final-eval-89-94823b73` |
| Replacement run root | `/mnt/data/coding/akasic-agent-worktrees/benchmark-runs/replacement-eval-89-2db5b1ec` |
| Runtime cache | `akasic-bench-runtime-v1-09168f46e2097f6c44d7daf2` |
| Git cache | `akasic-bench-git-v1-e4706df27562c97921ac2a6c` |

普通 `Trial` 相对于 frozen run root；以 `replacement/` 开头的 `Trial` 相对于
replacement run root。标准证据位置如下：

- Agent trace：`<Trial>/agent/trace.jsonl`
- Agent 终态：`<Trial>/agent/turn-result.json`
- Runtime 日志：`<Trial>/agent/runtime.stderr.log`
- Verifier 输出：`<Trial>/verifier/test-stdout.txt`
- 完整环境、资源和结果清单：`<Trial>/campaign-manifest.json`

旧的两轮 verifier 启动失败结果位于其他 run root，因在线下载 `uv` 失败且
`uvx` 不存在或代理错误，官方测试没有执行，已判为无效，本文不把它们计入任何
case 的有效结果。

## 2. 当前汇总

| 分类 | 数量 | 说明 |
|---|---:|---|
| Verifier 通过 | 63 | reward `1.0` |
| Verifier 真实断言失败 | 13 | reward `0.0`，官方测试确实执行 |
| Agent/模型未收口 | 12 | 达到题目 turn timeout，online/resource 无异常 |
| Agent 未收口并触发资源上限 | 1 | `make-mips-interpreter`，2 GiB OOM |
| 无效结果 | 0 | 两个首轮无效结果均已由修复后的有效结果替代 |

最终唯一结果通过率为 `63 / 89 = 70.8%`。这是当前 Akashic harness 的诊断结果，
不直接等同于官方 leaderboard 分数。

另外两题虽有 OOM 证据，但都进入了 verifier：

- `rstan-to-pystan`：8 GiB 上限记录 2 次 OOM kill，最终 6/6 通过；
- `video-processing`：记录 1 次 OOM kill，最终 4/5，属于有效任务失败。

## 3. 逐题结果

状态含义：

- `PASS`：官方 verifier 通过；
- `ASSERT`：官方 verifier 执行后断言失败；
- `TIMEOUT`：Agent 未在题目 turn budget 内收口，未进入 verifier；
- `RESOURCE`：运行时触发容器资源上限；
- `INFRA-INVALID`：题目没有获得有效 Agent/verifier 机会，修复后必须替代重跑；
- `RUNNING`：尚未形成终态。

| # | Case | 状态 | 当前结论或失败问题 | Trial |
|---:|---|---|---|---|
| 1 | `adaptive-rejection-sampler` | PASS | 官方 verifier 9/9。 | `akasic-bench-v4flash-diagnostic-adaptive-rejection-sampler-20260731-082914-137381` |
| 2 | `bn-fit-modify` | PASS | 官方 verifier 9/9。 | `akasic-bench-v4flash-diagnostic-bn-fit-modify-20260731-082916-853988` |
| 3 | `break-filter-js-from-html` | PASS | 官方 verifier 1/1。 | `akasic-bench-v4flash-diagnostic-break-filter-js-from-html-20260731-082917-908227` |
| 4 | `build-cython-ext` | PASS | 官方 verifier 11/11。 | `akasic-bench-v4flash-diagnostic-build-cython-ext-20260731-082919-061276` |
| 5 | `build-pmars` | PASS | 官方 verifier 4/4。 | `akasic-bench-v4flash-diagnostic-build-pmars-20260731-082921-792261` |
| 6 | `build-pov-ray` | PASS | 官方 verifier 3/3。 | `akasic-bench-v4flash-diagnostic-build-pov-ray-20260731-082922-327715` |
| 7 | `caffe-cifar-10` | PASS | 官方 verifier 6/6；长任务完整收口。 | `akasic-bench-v4flash-diagnostic-caffe-cifar-10-20260731-083359-931477` |
| 8 | `cancel-async-tasks` | ASSERT | 5/6；`test_tasks_cancel_above_max_concurrent` 期望取消 2 个任务，实际为 0。 | `akasic-bench-v4flash-diagnostic-cancel-async-tasks-20260731-083540-438309` |
| 9 | `chess-best-move` | TIMEOUT | 900 秒未收口；trace 终态为 `TimeoutError`。 | `akasic-bench-v4flash-diagnostic-chess-best-move-20260731-083630-388622` |
| 10 | `circuit-fibsqrt` | PASS | 官方 verifier 3/3。 | `akasic-bench-v4flash-diagnostic-circuit-fibsqrt-20260731-083734-370817` |
| 11 | `cobol-modernization` | PASS | 官方 verifier 3/3。 | `akasic-bench-v4flash-diagnostic-cobol-modernization-20260731-084000-193659` |
| 12 | `code-from-image` | TIMEOUT | 1200 秒持续 OCR/像素分析，未形成答案。 | `akasic-bench-v4flash-diagnostic-code-from-image-20260731-084327-047952` |
| 13 | `compile-compcert` | PASS | 官方 verifier 3/3。 | `akasic-bench-v4flash-diagnostic-compile-compcert-20260731-084356-768103` |
| 14 | `configure-git-webserver` | PASS | 官方 verifier 1/1。 | `akasic-bench-v4flash-diagnostic-configure-git-webserver-20260731-084855-000376` |
| 15 | `constraints-scheduling` | PASS | 官方 verifier 3/3。 | `akasic-bench-v4flash-diagnostic-constraints-scheduling-20260731-085152-604753` |
| 16 | `count-dataset-tokens` | PASS | 官方 verifier 1/1。 | `akasic-bench-v4flash-diagnostic-count-dataset-tokens-20260731-085321-853943` |
| 17 | `crack-7z-hash` | PASS | 官方 verifier 2/2。 | `akasic-bench-v4flash-diagnostic-crack-7z-hash-20260731-085352-596737` |
| 18 | `custom-memory-heap-crash` | PASS | 官方 verifier 6/6。 | `akasic-bench-v4flash-diagnostic-custom-memory-heap-crash-20260731-085840-388776` |
| 19 | `db-wal-recovery` | ASSERT | 5/7；只恢复 base 数据，WAL 更新未解密或应用。 | `akasic-bench-v4flash-diagnostic-db-wal-recovery-20260731-090223-465032` |
| 20 | `distribution-search` | PASS | 官方 verifier 4/4。 | `akasic-bench-v4flash-diagnostic-distribution-search-20260731-090309-163004` |
| 21 | `dna-assembly` | ASSERT | 0/1；正反向 primer 的熔解温度差超过 5°C。 | `akasic-bench-v4flash-diagnostic-dna-assembly-20260731-090350-203759` |
| 22 | `dna-insert` | PASS | 官方 verifier 1/1。 | `akasic-bench-v4flash-diagnostic-dna-insert-20260731-090448-210107` |
| 23 | `extract-elf` | ASSERT | 1/2；只匹配到预期值的 0%，要求至少 75%。 | `akasic-bench-v4flash-diagnostic-extract-elf-20260731-090713-402491` |
| 24 | `extract-moves-from-video` | TIMEOUT | 1800 秒未收口。 | `akasic-bench-v4flash-diagnostic-extract-moves-from-video-20260731-091015-290027` |
| 25 | `feal-differential-cryptanalysis` | PASS | 官方 verifier 1/1。 | `akasic-bench-v4flash-diagnostic-feal-differential-cryptanalysis-20260731-091124-575402` |
| 26 | `feal-linear-cryptanalysis` | ASSERT | 0/1；缺少 `/app/plaintexts.txt`，攻击程序未形成所需输入。 | `akasic-bench-v4flash-diagnostic-feal-linear-cryptanalysis-20260731-091240-599774` |
| 27 | `filter-js-from-html` | ASSERT | 0/2；未完整拦截 XSS，且修改了 5 个干净 HTML。 | `akasic-bench-v4flash-diagnostic-filter-js-from-html-20260731-091516-682608` |
| 28 | `financial-document-processor` | PASS | 官方 verifier 7/7。 | `akasic-bench-v4flash-diagnostic-financial-document-processor-20260731-091639-623485` |
| 29 | `fix-code-vulnerability` | PASS | 项目回归 367/367，官方 verifier 6/6。 | `akasic-bench-v4flash-diagnostic-fix-code-vulnerability-20260731-092126-571992` |
| 30 | `fix-git` | PASS | 官方 verifier 2/2。 | `akasic-bench-v4flash-diagnostic-fix-git-20260731-092320-729821` |
| 31 | `fix-ocaml-gc` | PASS | 项目回归 40/40，官方 verifier 1/1。 | `akasic-bench-v4flash-diagnostic-fix-ocaml-gc-20260731-092337-321270` |
| 32 | `gcode-to-text` | TIMEOUT | 900 秒未收口。 | `akasic-bench-v4flash-diagnostic-gcode-to-text-20260731-092449-769434` |
| 33 | `git-leak-recovery` | PASS | 官方 verifier 5/5。 | `akasic-bench-v4flash-diagnostic-git-leak-recovery-20260731-093146-014387` |
| 34 | `git-multibranch` | PASS | 官方 verifier 1/1。 | `akasic-bench-v4flash-diagnostic-git-multibranch-20260731-093327-356022` |
| 35 | `gpt2-codegolf` | TIMEOUT | 900 秒未收口。 | `akasic-bench-v4flash-diagnostic-gpt2-codegolf-20260731-093340-544507` |
| 36 | `headless-terminal` | PASS | 官方 verifier 7/7。 | `akasic-bench-v4flash-diagnostic-headless-terminal-20260731-093539-459730` |
| 37 | `hf-model-inference` | PASS | 官方 verifier 4/4。 | `akasic-bench-v4flash-diagnostic-hf-model-inference-20260731-093811-138691` |
| 38 | `install-windows-3.11` | PASS | 官方 verifier 4/4；约 49 分钟后完整收口。 | `akasic-bench-v4flash-diagnostic-install-windows-3-11-20260731-093919-539503` |
| 39 | `kv-store-grpc` | PASS | 官方 verifier 7/7。 | `akasic-bench-v4flash-diagnostic-kv-store-grpc-20260731-094017-800770` |
| 40 | `large-scale-text-editing` | PASS | 官方 verifier 5/5。 | `akasic-bench-v4flash-diagnostic-large-scale-text-editing-20260731-094039-933799` |
| 41 | `largest-eigenval` | TIMEOUT | 900 秒未收口。 | `akasic-bench-v4flash-diagnostic-largest-eigenval-20260731-094117-533962` |
| 42 | `llm-inference-batching-scheduler` | PASS | 官方 verifier 6/6。 | `akasic-bench-v4flash-diagnostic-llm-inference-batching-scheduler-20260731-094144-710251` |
| 43 | `log-summary-date-ranges` | PASS | 官方 verifier 2/2。 | `akasic-bench-v4flash-diagnostic-log-summary-date-ranges-20260731-094721-175024` |
| 44 | `mailman` | PASS | 官方 verifier 3/3。 | `akasic-bench-v4flash-diagnostic-mailman-20260731-094738-116634` |
| 45 | `make-doom-for-mips` | TIMEOUT | 900 秒未收口。 | `akasic-bench-v4flash-diagnostic-make-doom-for-mips-20260731-094838-396331` |
| 46 | `make-mips-interpreter` | TIMEOUT + RESOURCE | 1800 秒、第 154 个 ReAct step 仍未收口；可见输入约 335k token，2 GiB 容器 OOM kill。 | `akasic-bench-v4flash-diagnostic-make-mips-interpreter-20260731-094909-320676` |
| 47 | `mcmc-sampling-stan` | PASS | 官方 verifier 6/6。 | `akasic-bench-v4flash-diagnostic-mcmc-sampling-stan-20260731-095458-077527` |
| 48 | `merge-diff-arc-agi-task` | PASS | 官方 verifier 5/5。 | `akasic-bench-v4flash-diagnostic-merge-diff-arc-agi-task-20260731-095648-461476` |
| 49 | `model-extraction-relu-logits` | ASSERT | 0/1；提取矩阵第 11 行不匹配。 | `akasic-bench-v4flash-diagnostic-model-extraction-relu-logits-20260731-095723-607733` |
| 50 | `modernize-scientific-stack` | PASS | 官方 verifier 2/2。 | `akasic-bench-v4flash-diagnostic-modernize-scientific-stack-20260731-095941-471880` |
| 51 | `mteb-leaderboard` | PASS | 官方 verifier 2/2。 | `akasic-bench-v4flash-diagnostic-mteb-leaderboard-20260731-100112-872993` |
| 52 | `mteb-retrieve` | PASS | 官方 verifier 2/2。 | `akasic-bench-v4flash-diagnostic-mteb-retrieve-20260731-100404-409631` |
| 53 | `multi-source-data-merger` | PASS | 官方 verifier 3/3。 | `akasic-bench-v4flash-diagnostic-multi-source-data-merger-20260731-100444-271758` |
| 54 | `nginx-request-logging` | PASS | 官方 verifier 8/8。 | `akasic-bench-v4flash-diagnostic-nginx-request-logging-20260731-100644-428010` |
| 55 | `openssl-selfsigned-cert` | PASS | 官方 verifier 6/6。 | `akasic-bench-v4flash-diagnostic-openssl-selfsigned-cert-20260731-100837-334517` |
| 56 | `overfull-hbox` | TIMEOUT | 750 秒未收口。 | `akasic-bench-v4flash-diagnostic-overfull-hbox-20260731-101010-599000` |
| 57 | `password-recovery` | ASSERT | 1/2；输出存在，但没有恢复出正确密码。 | `akasic-bench-v4flash-diagnostic-password-recovery-20260731-101244-006751` |
| 58 | `path-tracing` | PASS | 官方 verifier 5/5。 | `akasic-bench-v4flash-diagnostic-path-tracing-20260731-101355-099995` |
| 59 | `path-tracing-reverse` | TIMEOUT | 1800 秒未收口。 | `akasic-bench-v4flash-diagnostic-path-tracing-reverse-20260731-101616-368246` |
| 60 | `polyglot-c-py` | PASS | 官方 verifier 1/1。 | `akasic-bench-v4flash-diagnostic-polyglot-c-py-20260731-101715-644726` |
| 61 | `polyglot-rust-c` | PASS | 官方 verifier 1/1。 | `akasic-bench-v4flash-diagnostic-polyglot-rust-c-20260731-101931-514360` |
| 62 | `portfolio-optimization` | PASS | 官方 verifier 6/6。 | `akasic-bench-v4flash-diagnostic-portfolio-optimization-20260731-102303-592745` |
| 63 | `protein-assembly` | ASSERT | 0/1；融合蛋白顺序不符合 `flag-donor-dhfr-acceptor-snap`。 | `akasic-bench-v4flash-diagnostic-protein-assembly-20260731-102519-392363` |
| 64 | `prove-plus-comm` | PASS | 修复后继承镜像 `/workspace` WORKDIR；官方 verifier 4/4，无资源异常。首轮硬编码 `/app` 的结果无效。 | `replacement/akasic-bench-v4flash-smoke-prove-plus-comm-20260731-121432-473790` |
| 65 | `pypi-server` | PASS | 官方 verifier 1/1。 | `akasic-bench-v4flash-diagnostic-pypi-server-20260731-102734-653302` |
| 66 | `pytorch-model-cli` | PASS | 官方 verifier 6/6。 | `akasic-bench-v4flash-diagnostic-pytorch-model-cli-20260731-102829-360920` |
| 67 | `pytorch-model-recovery` | PASS | 替代运行 14 个 ReAct step、13 次工具调用后收口；官方 verifier 5/5，无资源异常。首轮第 7 次 stream 的 `incomplete chunked read` 结果无效。 | `replacement/akasic-bench-v4flash-smoke-pytorch-model-recovery-20260731-121617-459331` |
| 68 | `qemu-alpine-ssh` | PASS | 官方 verifier 1/1。 | `akasic-bench-v4flash-diagnostic-qemu-alpine-ssh-20260731-102936-871296` |
| 69 | `qemu-startup` | PASS | 官方 verifier 1/1。 | `akasic-bench-v4flash-diagnostic-qemu-startup-20260731-103026-338644` |
| 70 | `query-optimize` | PASS | 官方 verifier 6/6；verifier 约 9 分 40 秒。 | `akasic-bench-v4flash-diagnostic-query-optimize-20260731-103241-583806` |
| 71 | `raman-fitting` | TIMEOUT | 900 秒未收口。 | `akasic-bench-v4flash-diagnostic-raman-fitting-20260731-103403-871980` |
| 72 | `regex-chess` | PASS | 官方 verifier 4/4；verifier 约 5 分 23 秒。 | `akasic-bench-v4flash-diagnostic-regex-chess-20260731-103507-872963` |
| 73 | `regex-log` | PASS | 官方 verifier 1/1。 | `akasic-bench-v4flash-smoke-regex-log-20260731-082415-955449` |
| 74 | `reshard-c4-data` | PASS | 官方 verifier 1/1。 | `akasic-bench-v4flash-diagnostic-reshard-c4-data-20260731-104014-411187` |
| 75 | `rstan-to-pystan` | PASS + RESOURCE | 官方 verifier 6/6；运行期间 8 GiB 上限记录 2 次 OOM kill。 | `akasic-bench-v4flash-diagnostic-rstan-to-pystan-20260731-104123-745692` |
| 76 | `sam-cell-seg` | ASSERT | 8/9；mask IoU `0.4988`，低于要求。 | `akasic-bench-v4flash-diagnostic-sam-cell-seg-20260731-104627-386438` |
| 77 | `sanitize-git-repo` | PASS | 官方 verifier 3/3。 | `akasic-bench-v4flash-diagnostic-sanitize-git-repo-20260731-104641-163085` |
| 78 | `schemelike-metacircular-eval` | PASS | 内部 63/63，官方 verifier 1/1。 | `akasic-bench-v4flash-diagnostic-schemelike-metacircular-eval-20260731-104934-708725` |
| 79 | `sparql-university` | PASS | 官方 verifier 3/3。 | `akasic-bench-v4flash-diagnostic-sparql-university-20260731-105205-160439` |
| 80 | `sqlite-db-truncate` | PASS | 官方 verifier 1/1。 | `akasic-bench-v4flash-diagnostic-sqlite-db-truncate-20260731-105502-840582` |
| 81 | `sqlite-with-gcov` | PASS | 官方 verifier 3/3。 | `akasic-bench-v4flash-diagnostic-sqlite-with-gcov-20260731-105505-789181` |
| 82 | `torch-pipeline-parallelism` | ASSERT | 2/4；两个 pipeline case 抛 `ProcessRaisedException`。 | `akasic-bench-v4flash-diagnostic-torch-pipeline-parallelism-20260731-105736-036618` |
| 83 | `torch-tensor-parallelism` | ASSERT | 1/13；12 个 column/row parallel case 失败。 | `akasic-bench-v4flash-diagnostic-torch-tensor-parallelism-20260731-105745-116108` |
| 84 | `train-fasttext` | TIMEOUT | 后台训练持续占用 CPU，但 3600 秒内未完成并收口。 | `akasic-bench-v4flash-diagnostic-train-fasttext-20260731-110351-840468` |
| 85 | `tune-mjcf` | TIMEOUT | 900 秒未收口。 | `akasic-bench-v4flash-diagnostic-tune-mjcf-20260731-110452-550757` |
| 86 | `video-processing` | ASSERT + RESOURCE | 4/5；landing frame 为 61，要求 62–64；运行中记录 1 次 OOM kill。 | `akasic-bench-v4flash-diagnostic-video-processing-20260731-110620-368026` |
| 87 | `vulnerable-secret` | PASS | 官方 verifier 3/3。 | `akasic-bench-v4flash-diagnostic-vulnerable-secret-20260731-110759-547245` |
| 88 | `winning-avg-corewars` | PASS | 官方 verifier 3/3。 | `akasic-bench-v4flash-diagnostic-winning-avg-corewars-20260731-110934-129932` |
| 89 | `write-compressor` | PASS | 官方 verifier 3/3。 | `akasic-bench-v4flash-diagnostic-write-compressor-20260731-111521-153498` |

## 4. 当前值得继续验证的 Harness 问题

### H5：任务工作目录不能硬编码为 `/app`

`prove-plus-comm` 的官方 Dockerfile 使用 `WORKDIR /workspace`。Harness 的 gateway 和
driver 启动命令无条件执行 `cd /app`，导致模型调用前失败。这是环境兼容性缺陷，
修复应让进程继承镜像声明的 WORKDIR，不检测 task name，也不改变任务文件或 verifier。

状态：已由 `e166be50` 修复；Harbor 定向测试 18/18，通过替代重跑 4/4。

### H6：流在首个有效 delta 前断开时应有限重试

`pytorch-model-recovery` 的第 7 次 LLM 调用发生
`httpx.RemoteProtocolError: incomplete chunked read`。当前重试只覆盖 stream 创建，
不覆盖 stream 消费。修复必须满足：

1. 只对明确可重试的 transport error 生效；
2. 不能在已经发出用户可见 content 或工具参数后盲目重放，避免重复副作用；
3. 重试次数和 1/2/4 秒退避沿用 provider 现有策略；
4. 非重试错误继续 fail-fast、fail-loud。

状态：已由 `e577dfda` 修复。新增测试分别证明首个有效 delta 前的
`RemoteProtocolError` 会重试，以及收到 content delta 后不会重放；provider 流定向
测试 5/5。替代实网运行未再次发生断线，因此它证明任务恢复为有效结果并通过 5/5，
不作为“真实断线重试被命中”的证据。

### H7：只按上下文百分比触发 compaction 可能晚于内存压力

`make-mips-interpreter` 在第 154 个 ReAct step 时可见输入约 335k token，尚未达到
V4 Flash 1M context 的 74%，但 2 GiB 容器已经 OOM。这个观察不直接授权改变生产
compaction 语义；后续应先区分模型 payload、session history、LLM snapshot、工具子进程
和 task 自身内存，再设计资源维度的消融。

## 5. 当前结论与后续

- 89 题逐题唯一有效结果和失败证据已经冻结；
- H5/H6 属于可泛化的 harness/provider 功能修复，不修改题目语义或 verifier；
- H7 仍只是资源观察，不授权提前 compaction 或针对 benchmark 做特殊处理；
- timeout case 的 step/token/compaction 聚合与 Agent/模型细分归因可在下一轮诊断中
  继续补充，不影响本轮 89 题有效性。

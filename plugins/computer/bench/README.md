# MiniWoB 完整宿主评测

通过正式 Gateway、默认回复循环、模型 Provider 和 Computer 工具执行网页任务。
宿主装配代码由仓库级 `benchmark/computer_runtime/` 拥有；此目录拥有任务、协议与判分。
每个任务与 seed 启动独立 AppRuntime、Workload Controller、workspace 和 Computer 容器。
Web 聊天入口和 subagent 关闭；其他默认插件照常装配，任何启动失败都使该用例失败。

```text
固定任务 / seed ──→ 页面初始化 ──→ Gateway 输入
                                      │
                           Akashic → 模型 → Computer
                                      │
页面 reward ←── 可信判分端 ←──────────┘
```

## 运行

需要 Linux、Docker、已安装项目 Python / Node 依赖。Core 容器只读挂载当前源码、
Python 基础安装及虚拟环境；这验证开发源码的完整后端，不代表发布镜像验收。
Docker socket 只交给宿主 Controller。使用私有目录保存模型凭据，权限设为 0600。

```bash
npm run build:web-plugins
# upstream 是独立 checkout；不要覆盖已有个人修改。
git clone https://github.com/Farama-Foundation/miniwob-plusplus.git /tmp/miniwob
git -C /tmp/miniwob checkout a7658beb68e1d3572329e06734c5822bd284ddc8
.venv/bin/python plugins/computer/bench/run.py \
  --upstream /tmp/miniwob --output /tmp/miniwob-smoke \
  --key-file /private/miniwob-api-key \
  --tasks click-test click-button enter-text --seeds 0
```

省略 `--tasks` 枚举上游 130 个 HTML 任务。默认每个 episode 180 秒、回复最多 30 步。
`--mode noop` 验证不操作时不会误报成功。输出目录必须不存在，避免覆盖旧证据。
Computer 固定占用宿主 OpenCLI 端口 19825，同一 Docker daemon 上按顺序运行。
运行时禁止编辑挂载源码；源码变化会使整批无效并停止。
端点和模型可用 `--endpoint`、`--model` 覆盖；固定 seed 不保证模型输出确定性。

## 证据与判分

每例保存启动状态、容器身份、起止截图、页面分数、消息、事件与宿主日志；
顶层保存任务表、源码提交、工作区 diff、脚本哈希和逐例结果。
成功条件是上游 `done && raw > 0`，不使用模型自述；同时保留原始和时间折扣分数。
超时、回复结束、环境失败分别报告。模型必须只用公开 UI；隐藏状态、奖励变量、
源代码、网络或脚本改写解题均无效。`audit: pending` 的成功在人工轨迹复核前只是候选。
这些约束是评测协议，不是对恶意 Agent 的安全隔离保证。

输出目录包含模型配置、凭据和完整会话，**不能整体提交或上传**。
分享时只导出脱敏汇总、必要轨迹片段和截图。保留失败用例，不用重跑覆盖原结果。
先扫固定 seed，再按共性失败聚类；修复前后用相同任务与 seed 配对复测，
新增 seed 验证泛化。问题优先级分别标记已实测收益和仅估计的潜在收益。

## 配对优化

首次 episode 的结算被冻结，后续重开不会覆盖分数。仍允许回复走到完成或原预算截止，
分别记录首次正分、正常回复、工具错误、结算后耗时与重开。`completed` 要求首次正分、
正常回复且没有已识别违规；`smooth` 再要求没有显式工具错误。两者仍须人工审计，
特别是 Computer 代码中的隐藏状态读取；自动检查不是安全沙箱。

`hover-shape` 没有实现奖励；`moving-items`、`chase-circle`、`stock-market` 有不受
统一预算控制的内部时限；`simon-says` 的序列在冷启动首次观察前已经结束。
这五项标记 `unsupported` 并保留记录，不纳入本目标集；
不能静默减少全量 130 项的分母，也不声称已评估这些动态场景。
先冻结任务与优化顺序：合规首次成功、正常收尾、错误和成本。基线与候选使用相同
任务、seed、模型、预算和全新状态；额外 seed/任务只在候选冻结后用于泛化验证。

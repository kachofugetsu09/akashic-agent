# 声明式设置的热重开

普通 `registerForm` 表单已有模块内配置资源；声明式 `DeclaredForms` 之前只缓存表单清单，每次重建 `Configuration` 都等 GET 返回。按 [ADR 0127](../../docs/decisions/0127-web-modules-own-their-data.md)，每个表单 ID 现在也拥有一份模块内资源，使用已有 `Configuration` 的读取、草稿保护、提交与回执处理。

```text
┌──────────────┐       ┌──────────────────────┐
│ 模块内已读配置 │ ────▶ │ 重开：立即显示字段和原值 │
└──────▲───────┘       └──────────┬───────────┘
       │                          ▼
       └─────────────── 后台 GET 核对最新值
                          ├─ 成功：按原草稿守卫更新
                          └─ 失败：保留字段并显示错误
```

只缓存已读取的配置，不缓存未保存草稿，也不写浏览器持久存储。首次打开仍等真实响应；后台最新值可能晚于已读内容就绪。旧配置的提交继续由服务端 `expected_input` 拒绝，不绕过并发校验。

## 重现

在独立 worktree 中固定 npm 锁文件、浏览器与其余源码，保留候选 `frontend/plugins/plugin_settings/src/index.tsx` 的恢复副本。将该文件放到基线版本 `8b0f0afa` 后构建，启动下面的 baseline；服务会复制制品到隔离目录。随后恢复候选源码、重新构建，再启动 candidate。两份运行时的 Web 模块摘要应只有 `plugin_settings` 不同；不要在采样期间构建或更换制品。

```bash
npm ci
npm run build
# 基线与候选分别在前台终端启动，state 必须尚不存在，端口须空闲：
.venv/bin/python scripts/webui-performance/serve-startup-runtime.py \
  --state /absolute/baseline-state --port 24683 --declared-settings
.venv/bin/python scripts/webui-performance/serve-startup-runtime.py \
  --state /absolute/candidate-state --port 24684 --declared-settings

node scripts/webui-performance/measure-declared-settings.mjs \
  --baseline http://127.0.0.1:24683 --candidate http://127.0.0.1:24684 \
  --browser /usr/bin/chromium --output /absolute/new/results --runs 7
```

两个测试 owner 使用真实插件声明、`ModelSettings` 和配置提交协议，每个含文本、行列表、数字和开关四种字段；没有外部账号或伪造的成功响应。基准使用新的浏览器 context、1440×900、CPU 1/4 倍、LAN 20ms/100Mbps 与弱网 800ms/1Mbps，交错 AB/BA。只在初次读取后计热重开：两张完整表单及其原值出现、完成两次绘制为 `readyMs`；两个后台 GET 全部 200 且完成为 `freshMs`。之后再核对字段与错误状态，不提前关页掩盖失败。未完成和失败样本保留，命令返回非零。

行为验收会向隔离测试 owner 保存一个 note，必须在性能采样之后运行：

```bash
node scripts/webui-performance/verify-declared-settings.mjs \
  --fixture /absolute/candidate-state --browser /usr/bin/chromium \
  --output /absolute/new/behavior-results
```

命令核对隔离标记、活跃进程和监听端口，先备份 bundle patch。覆盖：挂起真实 GET 后编辑，迟到响应不能覆盖草稿；取消离开保留草稿；放弃后重开不把草稿当配置；一次 503 显示错误且保留字段，重试真实端点恢复；真实保存取得 active 回执，新页面读回持久值。性能采样不使用网络拦截；故障注入只在独立行为验收中使用。结束时向两个测试进程发送 SIGINT，核对各自 `integrity.json` 的完整消息摘要不变。

## 2026-10-31 结果

基线 `8b0f0afa`，两份完整本地 Runtime（75 个插件 generation），只有 `plugin_settings` Web 模块摘要不同。7 对、4 个 CPU/网络组合，共 56 个样本全部通过；每个样本还等后台核对成功并比较所有字段。以下为中位数，最大值只描述这批样本，不外推 p95。

| CPU / 网络 | 基线已读内容 ms | 候选已读内容 ms | 候选最大 ms | 候选后台 GET 完成 ms |
|---|---:|---:|---:|---:|
| 1 / LAN | 161.0 | 49.3 | 72.2 | 114.2 |
| 1 / 弱网 | 899.7 | 52.5 | 63.0 | 866.9 |
| 4 / LAN | 112.0 | 81.8 | 99.7 | 75.4 |
| 4 / 弱网 | 912.7 | 83.2 | 91.9 | 879.3 |

本轮候选的已读内容热重开均小于 300ms；最新值仍需网络往返，不计为 300ms 达标。首次读取没有被替换成假数据。草稿、失败重试、真实保存并刷新读回四项验收通过，截图字段与布局一致。

原始 56 样本、请求起止、截图及模块摘要在 `/tmp/akashic-declared-paired-final-20261031/`；行为结果与 bundle 恢复副本在 `/tmp/akashic-declared-behavior-20261031/`。正式服务只用于读取基线现象：实际声明式表单弱网重开约 1.1–3.3 秒。候选未部署，隔离测试 owner 不等同于生产 Computer/GitHub Watch 的完整验收，也不代表所有前端交互已满足 300ms。

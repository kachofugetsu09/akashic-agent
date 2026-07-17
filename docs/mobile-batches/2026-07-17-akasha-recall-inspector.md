# Akasha 回忆检查移动看板

## 任务与边界

手机端只回答一个问题：**最近一轮回答为什么想起这些内容**。桌面全局图、重建、节点管理和分析筛选没有迁入手机；宿主也没有新增 Akasha 专用协议、数据库字段或 Android 页面。

```text
抽屉「插件」
    │
    ▼
运行中插件平面列表
    │ 点 Akasha
    ▼
┌──────────────────────────────┐
│ ←  Akasha                   │  固定宿主返回
│                              │
│ 最近一次检索                 │  唯一主状态区
│ “这轮问题摘要”               │
│ 7/17 10:18 · 注入 5,396 字   │
│ ● 精确 10       ◆ 联想 8     │
│                              │
│ 更早的检索              33 轮 │  平面列表
│ 问题摘要 · 注入字符  ●10 ◆8  │
│ └─ 点击后原位展开左右脑明细  │
└──────────────────────────────┘
```

## 数据所有权

- `AkashaStore.list_query_logs()` 继续拥有最近检索摘要；移动看板固定读取最近 30 轮。
- `AkashaStore.get_query_log()` 继续拥有单轮完整诊断；只有用户展开对应行时才读取 JSON 明细。
- 手机读取路径使用 SQLite `mode=ro` 并启用 `PRAGMA query_only=ON`；只读打开不会创建父目录、数据库或执行 schema migration。写入路径保持既有 owner 和行为。
- `recall.current` 保持消息内 slot 的 session/turn 绑定；新增的 `inspector.recent` 与 `inspector.detail` 是插件自有只读 RPC。
- 明细复用既有 `_mobile_recall_items()` 排序、摘要裁剪和旧消息时间补全；没有另造第二套“左右脑”解释。
- 看板没有绕回 HTTP Dashboard，也没有调用全局 Graph reader 或触发 snapshot rebuild。

## 交互与视觉决策

- ExtraGram 的偏好页与长详情交互只作为结构参考：抽屉入口进入全屏任务面；明细插在父行下方；同一时间只展开一轮；父行、列表顺序与返回栈保持不变。
- Material 3 使用单一 20dp 主状态容器、12dp 语义分区和平面 divider 列表。没有卡片墙、玻璃拟态、阴影、装饰渐变或 badge 堆叠。
- 蓝色只表示左脑精确召回，圆点是其形状语义；亮紫只表示右脑联想召回，菱形保持与 Agent 过程一致的联想语义。空态和普通列表回到中性色。
- 标题、问题、时间、注入字符和分数形成四级排版；计数与分数使用 tabular figures。中文采用宿主 `Noto Sans SC` fallback，按钮和动态 DOM 均继承宿主字体。
- 详情使用 260ms Material easing 原位展开，`prefers-reduced-motion` 时降为 1ms；折叠内容设置 `inert + aria-hidden`。
- Pixel 7 首轮真机发现无空格长记忆压住右侧分数，最终在文本 owner 上使用 `overflow-wrap:anywhere`，未通过隐藏分数或扩大卡片掩盖问题。

## 自动化验证

```bash
/mnt/data/coding/akasic-agent/.venv/bin/pytest -q \
  tests/test_akasha_plugin.py tests/test_plugin_mobile_ui.py
node --test plugins/akasha/tests/test_mobile_ui.mjs
/mnt/data/coding/akasic-agent/.venv/bin/pyright \
  plugins/akasha/plugin.py tests/test_akasha_plugin.py
npm run build:mobile-web
/mnt/data/coding/akasic-agent/.venv/bin/pytest -q tests/
git diff --check
```

结果：定向 Python `73 passed`，插件资产 `3 passed`，Pyright `0 errors / 0 warnings`，移动 Web 生产构建成功，完整测试 `2329 passed`，diff check 通过。除自动化测试外，真机 RPC 还逐项比对数据库哈希、mtime、大小与日志计数。

## Pixel 7 端到端

- 设备：`28151FDH200478`，隔离 Mobile Lab，正式 workspace 未读写。
- 插件 watcher 把 Akasha 资源热更新到手机；`store.py` 的 Python 模块依赖变化需要重启隔离 Mobile Lab Agent，随后 Pixel 7 在 generation 22、connection epoch 278 自动完成 proof、resume、history、command 与插件资源同步。正式 runtime 未重启、正式 workspace 未访问。
- 真机读取隔离库真实 34 轮日志；最近一轮显示 `reply only OK / 精确 10 / 联想 8 / 注入 5,396 字`。
- 点击最近一轮后，WebSocket 发送 `inspector.detail`，原位展开 10 条精确与 8 条联想；点击旧轮只保留一个展开区，系统返回键回到运行中插件目录。
- 首轮截图发现无空格长文本溢出后修复并复测；分数列没有被覆盖，页面可继续滚动到右脑区。
- `inspector.recent` 与 `inspector.detail` 前后，隔离 `akasha.db` 始终为 SHA256 `dae13acb…e97f2f`、mtime `1784254721`、大小 `962560` 字节、`akasha_query_log=34`，证明真机看板没有写库或触发 migration。
- 最终 logcat 只有 generation 22 / epoch 278 的两次 `plugin.ui.call`，没有 FATAL、`RenderProcessGone`、event sequence gap 或协议校验错误。

证据：

- `/tmp/pixel7-akasha-readonly-overview-final.png`
- `/tmp/pixel7-akasha-readonly-detail-final.png`
- `/tmp/pixel7-akasha-readonly-right-brain-final.png`
- `/tmp/pixel7-akasha-single-expand-final.png`
- `/tmp/pixel7-plugin-directory-back-final.png`

# Computer 驱动源码迁移

- 状态：implementation，用户已确认开始。
- 初始基线：`6a15444009c807994d33691e0b756167880fad5d`；交付分支已同步 `origin/main@dcc8c8a23c526133e3883628945bfd9f8632a814`。
- Worktree：`/mnt/data/coding/akasic-agent-worktrees/computer-driver-source`。
- Writer：Codex `/root`。
- 恢复点：`/mnt/data/coding/akasic-agent-backups/computer-driver-source-20260905`。

## 1. 目标与范围

以 `codex-desktop-rev` 中的实现、类型、说明和反编译结果为参考，迁移可维护的 Browser 与 Linux
Desktop 驱动源码，尽可能保持原接口和可观察行为。原二进制只作为开发对照；无法证明等价的
部分必须写明差异。原有容器、同一 Chromium/profile、OpenCLI 与人工接管继续由现有 Computer
插件和 Workload 管理，不增加 Core 的 Computer 专属接口。

`change_type: feature`，`semantic_delta: expanded driver API`。能力 owner 为 Computer 插件；
consumer 为该插件工具与容器内调用者；`runtime_patch: false`。插件数据和容器生命周期遵循
[现有合同](computer-plugin-workload-task-contract.md)。只允许改 `docker/computer`、
`plugins/computer`、相关 benchmark、验证和本文档入口。

不修改正式 workspace、消息、profile、其他插件数据或用户 checkout。不发布、部署或迁移正式
数据。测试仅创建本次拥有的隔离容器和临时 profile；结束由测试 owner 清理。持久 profile
仍由现有单写者使用，不因替换驱动删除或复制登录数据。

## 2. 目标结构

```text
Akashic Agent
      │
Computer 插件：工具与调用生命周期
      │
┌─────▼──────── Computer 容器 ─────────┐
│ JS 调用入口 → Browser 驱动 → Chromium│
│             → Desktop 驱动 → X11    │
│ 观察、错误、取消与连接均有明确 owner   │
└────────────────┬────────────────────┘
                 ▼
          现有持久 profile
```

JS 执行对象、标签连接、拖拽句柄只属于本代进程；重启后重建，不当作持久事实。调用退出、取消和
连接丢失必须释放由该调用按下的输入，不能把超时伪装为操作已回滚。人工输入继续使用原 RFB。

## 3. 原版证据与验收边界

- 参考包：`@oai/cua 0.2.4`、`@oai/sky 0.6.26`、`@oai/browser-desktop 0.1.1`。
- Cua Basic 固定提交：`aabb2082c170289256f0c8d9db4cce094c778578`，原题 68 个变体。
- 当前 desktop 基线 47/68；原版 desktop 53/68；原版 browser + desktop 66/68。
- 两个音量原解法偏离目标；单独使用 browser fill 均通过，不并入 66/68 主成绩。
- 基线和原版证据位于 `computer-driver-benchmark/benchmark/data`。题目、原参考解法和判分器
  不改动；动作绑定变化和补充测试单独记录。它们不是 LLM 成功率或完整 API 等价证明。

原包的 JS 包含第三方依赖和业务实现；先保留原逻辑再整理，不能用格式化冒充源码重建。Rust
反编译缺少可靠类型与完整源码，须与符号、系统调用和原版实验交叉核对。WASM 同样有独立的
重建义务，不以“JS 已迁入”宣称全部可维护。

## 4. 实施与验证

1. 固定原版内容哈希、恢复点和公开 API 清单，区分 Linux、其他平台及未验证能力。
2. 迁移 JS 与容器接入，保留控件语义、观察和组合动作。错误与缺失能力必须显式返回。
3. 重建 Linux 输入、剪贴板、截图和无障碍处理；以原版作逐项行为对照，记录实现差异。
4. 运行原有 68 题、负对照和输入释放、标签生命周期、连接恢复等真实边界验证。
5. 运行相关静态检查、构建、change-impact Gate 和独立概念评审，再整理交付证据。

先复用已有夹具与库级覆盖，缺少现实行为覆盖时再添加有针对性的测试。权限、配置、进程和
浏览器连接接入不得沿用实验 host 的固定窗口 ID、内存配置或绕过权限的测试模式。

## 5. 进度

- [x] 固定源码与原版恢复点，建立独立 worktree。
- [x] 固定能力清单与迁移来源。
- [x] JS 与容器驱动接入。
- [x] Linux 原生源码重建。
- [x] 无障碍源码及其他二进制依赖处理。
- 行为对照、集成验证与独立评审以第 8 节的验证报告为准。
- [ ] 镜像发布、双处 digest 更新与正式插件安装（待发布授权）。

该清单是本次任务执行状态，不改变 `projectneed` 的权威语义，也不代表未完成项已交付。

## 6. 调用、标签和临时状态合同

公开面逐项固定在 [`api-matrix.json`](../../docker/computer/driver/api-matrix.json)，其中
`preserve` 是实现计划，只有 `verified: true` 和对应证据才是已验证能力。上游 CDP 本来不提供的
可选能力继续拒绝；本容器未启用 raw CDP capability，公开调用使用 Browser 的 CUA、DOM 和 AX
接口。macOS 原生 AX 和可选音频不伪装为 Linux 支持。

调用由 Computer 插件附上本代 `generation_id`、真实 `session_id/turn_id` 和 `call_id`，用户
代码不能覆盖这些值。容器 supervisor 串行接收调用，拥有 Worker、CDP 请求表和 Native 连接。
取消顺序为：停止接收本调用的后续 RPC → 停止该 JS Worker → 取消 Native 并 drain 已发送的
CDP 请求 → 释放登记的输入 → 返回结果。释放无法确认时停止接收新调用，要求重启容器；已发生
的页面变化不回滚。`drag_handle` 可跨动作和截图，但必须在同一次 JS 调用内结束。
输入表覆盖键盘、鼠标、直接触摸和鼠标模拟触摸；取消使用原 target/session 释放。
用户可见的 `nodeRepl` 只提供输出和目录信息，内部连接留在参考模块宿主中；Node 执行环境并非
用于运行敌对代码的安全沙箱。

插件在真实 Turn 结束时通知 Browser backend；正常新建标签默认关闭，deliverable/handoff
保留；人工标签必须经当前列表 claim 才允许显式关闭，自动收尾只释放 claim。标签 ID 使用本代
随机起点，重启后旧 ID 拒绝；元素仍由原 JS 校验 frame/loader/backend node。不能拿固定假的
session/turn/window ID 作为正式集成。MCP 缺少的调用上下文和取消通过插件已有工具上下文、事件和其自有
控制连接接入，不新增 Core 接口。

| 状态 | owner / 路径 | 增加与更新 | 失效与物理清理 |
|---|---|---|---|
| Chromium profile | 原 Workload，`/data/profile` | Chromium 正常写入 | 本次不减少或迁移 |
| activity | 原 gateway，`/data/state/activity.json` | 沿用 revision/notice 更新 | 沿用现行合同 |
| JS 绑定、claim、输入表 | 容器 driver 进程 | 调用创建与更新 | 取消、reset 或本代退出失效；不持久化 |
| Driver 临时文件/配置 | driver，`/tmp/akashic-computer-driver-*` | 创建本代专用目录 | 只清理自己创建的目录 |
| MCP 控制 socket | Computer MCP，Root 数据目录对应的 Linux abstract socket | 进程启动建立 | drain 后关闭，内核回收；不产生持久文件 |
| 对外截图文件 | 原插件 `screenshots/` | 沿用原保存接口 | 原 owner 保留最近 32 张 |

原 Browser JS 保留为可读参考源码，接入层、输入源码与 AX renderer 使用普通名称维护。AX
renderer 保留节点身份、完整树和动作所需信息，首版输出完整文本；不声称复刻原 WASM 的所有
排版/剪枝/差分压缩。Native screenshot 使用 XFixes 的真实指针，返回图片字节而非额外生成
临时文件；旧 `/screenshot` 的图片格式也从 PNG 变为 JPEG。这些是明确的输出差异。
ZXing `3.1.2` 的上游 npm WASM 与 rev 中 SHA-256 完全
相同，作为有公开 C++ 源码和构建方法的第三方依赖固定；不从 rev 再分发该二进制。

新旧截图工具统一返回文件路径并提示 `read_file`。读文件工具拥有当前 Agent 的模型能力判断：
多模态模型直接得到图片内容，文字模型才转向 `read_image_vision`。Computer 不重复判断模型能力。

## 7. 已验证的重建细节

- 原 `move_path` ELF 地址 `0x56fb0`，长度 `0x6e0`。根据反汇编与 rodata，步数为
  `ceil(hypot(dx,dy) / 160 * 10).clamp(1,10)`；包含两端，四舍五入到 i16，去掉连续重复
  坐标，采样间隔 8ms。相同起终点仍发送一次 motion，不等待不存在的位移。
- 原 Ghidra 项目的函数边界无法得到可靠反编译，不能把其警告输出当成源码。此次以上述
  精确符号范围的汇编、常量读取和原 Cua 拖放 5 题验证交叉确认。
- 本次 Dockerfile 使用现行 Computer 镜像 digest 作为运行层，构建上下文仅包含
  `docker/computer`；Core、会话、记忆与其他插件不进入镜像。Rust 1.88 Bookworm 编译通过。
- 兼容 `/input` 仍是单请求桌面操作，只有短期输入 owner，不伪造 Akashic Session/Turn；
  新 `computer` 工具才提供带真实 Session/Turn 的持久 JS 上下文。

控制地址由 Core 注入的 `data_root` 路径哈希得到，只负责路由。candidate 使用隔离的数据根；
formal reload 在旧 Root、MCP 和 Workload 完成 drain 后顺序复用正式地址。`generation_id` 是
追踪字段；MCP 不把它冒充可独立认证的凭据，调用身份由 Core exact Tool binding 保证。
容器内标签的 close/mark 只接受同 Session 的 create/claim；跨 Session 须等旧 owner 收尾后再 claim。
标签枚举使用浏览器进程的 metadata，不等待每个页面执行脚本；CDP 不提供可靠活动标签时不编造
`active`。旧 `/browser/action` 也经过同一 supervisor，显式新建标签视为交付，旧调用没有跨 Turn JS 状态。

控制连接的取消回执最多等待 45 秒，覆盖 Native 取消、CDP drain、输入释放与 Native 重建。
普通调用 HTTP/控制总时限 170 秒；Turn 收尾可能等待另一 Session 的当前调用，预算为 330 秒。
取消先于 admission 到达时，driver 保存最多 4096 个、5 分钟有效的 call_id 取消记录；相应 run
在任何输入前消费记录并拒绝。它是临时取消状态，不是副作用回滚或通用幂等存储。

## 8. 交付与发布

源码版已通过一次真实 gateway 的 Cua Basic 全量：66/68，只有 video-player 的第 2/3 变体
reward 为 0，与原版 Browser + Desktop 相同。最终冻结镜像复验记录保存在
`benchmark/data/computer-driver-source-20260905/verification.json`（本机证据，不提交生成数据）。
真实 Root、静态 manifest、MCP 控制连接、Python 取消、Turn 事件和 legacy HTTP 路径在
`tests/test_computer_driver_plugin.py` 验证；需要一次性容器时显式设置 `COMPUTER_TEST_GATEWAY`。
将测试复制到一次性容器后，`node --test /opt/computer/test/*.test.mjs` 验证源码 AX、标签和输入生命周期。

当前未发布或部署。插件 `plugin.py` 与 `akashic.plugin.toml` 的镜像 pin 暂保留已发布基线；
**安装发布前必须一起更新为新镜像的 registry manifest digest**。本地 image ID 不是 registry
manifest digest，不能冒充该 pin。MCP 启动检查 driver v2/source/ready，旧镜像会明确拒绝加载。
发布动作不包含在本次只改驱动的授权内；不能把源码验收当作正式 generation 已升级。

构建命令：`docker build -f docker/computer/Dockerfile -t akashic-computer:driver-source-20260905 .`。
现有镜像工作流仍可发布这个 Dockerfile。发布后更新两处 pin，再走正式插件安装/candidate 链；
不手改 cache。恢复使用开工备份与旧镜像 digest，profile 不迁移也不清空。

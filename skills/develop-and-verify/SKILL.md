---
name: develop-and-verify
description: 改任何带界面或服务的项目（网页、前端、后端、Akashic 自身）并亲自验收时使用：在宿主上隔离地起真实服务，假掉外部依赖，用 computer 浏览器真实操作和截图，验完再提 PR。用户让你修界面、做功能、提 PR、"跑起来看看"时使用。
---

# 在宿主上开发并验收

验收要落在一个**真的在跑**的服务上：真实构建产物、真实后端、真实登录、真实操作。
只跑 typecheck / build、手写一段 DOM 拼成截图、假后端吐一份手编 JSON，都只能当辅助，不能当验收。

```text
隔离工作区 → 起服务(localhost:<端口>，后台运行) → 假掉外部依赖 → computer 打开 http://localhost:<端口>
      ↑                                                                        ↓
      └─────────────── 改代码 → 重启服务 ←──────────── 真实操作、截图、对照
```

## 0. 规矩

- **只删自己建的东西。** 开工时在 `$DEV/created.txt` 记下建的目录、进程 PID、容器、分支；收尾只清这张单。
  不碰 `/tmp` 里别人的目录，不对 `..`、`~`、通配符 `rm -rf`；删之前先 `ls`。当前目录不要删，先 `cd` 出去。
- **停进程用精确 PID**（`ss -ltnp | grep :<端口>` 取 pid）。`pkill -f <端口号>` 会匹配到执行它的 shell 自己。
- **不另起容器当浏览器或 curl 跳板。** 页面用 computer 看；接口在宿主 shell 里直接 `curl`。
- **不碰生产**：不改正在服务的数据目录，不重启生产服务；合并、部署要用户明确授权。
- 临时文件放 `~/.cache/<项目>-dev/<任务>/`（下文记作 `$DEV`），并 `export TMPDIR=$DEV/tmp`。`/tmp` 是很小的 tmpfs，满了构建会报莫名错误。

## 1. 隔离工作区

```sh
DEV=~/.cache/<项目>-dev/<任务>; mkdir -p "$DEV/tmp"; export TMPDIR="$DEV/tmp"
git -C <主仓库> fetch origin && git -C <主仓库> worktree add "$DEV/src" -b <分支> origin/main
cd "$DEV/src" && npm ci        # 或该项目自己的依赖安装；不要把 node_modules 软链到主仓库
```

依赖装不全时先修依赖（看清缺哪个包），不要带着基线错误继续，否则分不清哪些错是你造成的。

## 2. 起服务，让 computer 看得到

computer 的浏览器经宿主出口上网：它打开的 `localhost` 就是宿主本机，和你在宿主 shell 里 `curl` 的是同一个地址。
服务照常监听 `127.0.0.1`，不要绑 `0.0.0.0`，没必要把开发服务开到所有网卡。

| 条件 | 做法 |
|---|---|
| 端口没被占 | 选一个空闲端口，`ss -ltn` 看一眼 |
| 进程一直活着 | 用 shell 工具的后台运行（`run_in_background`），它不设截止，活到 `task_stop` 或会话结束；命令末尾的 `&` 会在命令结束时被回收 |

先用宿主 shell 确认 `curl -sI http://localhost:<端口>/` 有响应，再去 computer 里开 `http://localhost:<端口>/`。

在 computer 里的现象怎么读：

- **`ERR_CONNECTION_REFUSED`**：服务没在跑（被回收了？）。
- **`ERR_BLOCKED_BY_CLIENT`**：网络是通的，但响应不是页面。浏览器顶层只渲染 HTML/PDF/图片/视频，别的 Content-Type 当下载拦掉。看 JSON 接口用 shell 的 curl，或在同源页面里 `fetch()`。

## 3. 假掉外部依赖

真实上游（LLM、第三方 API、支付）不要在验收里调：花钱、不确定、可能有副作用。写一个确定性的假服务，监听 `127.0.0.1`，回复里带上能证明链路的信息（模型名、收到的话、第几步）。

LLM 类依赖直接用 `scripts/mock_model.py`（只依赖标准库，OpenAI 兼容）：

```sh
python3 <本 skill 目录>/scripts/mock_model.py --port 44099 --steps 6 --tool shell --args '{"command": "echo step {n}", "description": "第 {n} 步"}'
```

`--steps N` 让每条用户消息先跑 N 步工具调用再回复，用来造"多步过程""长对话"之类的真实场景；工具名和参数按被测服务真实下发的 `tools` 填。

需要专门的场景（报错、超时、超长输出）就复制一份改 `decide()`，放在 `$DEV` 里，不进仓库。

## 4. 真实操作并截图

1. 加载 computer（`load_tools {"plugin":"computer"}`），读一遍它的 API 文档再操作。
2. 按用户会走的路径操作：登录 → 触发功能（发消息、点按钮、填表单）→ 等它真的完成 → 看结果。
3. 宽屏和手机宽度各看一次；有主题的亮/暗各看一次。
4. **改前 / 改后对照**：在 `origin/main` 上用另一个端口起一份，同一步骤截图对比。
5. 截图保存后给用户看，或推到截图分支在 PR 里引用。

同时跑和改动相关的检查（单测、类型检查、lint），只汇报真实跑过的结果。

## 5. 提 PR 与收尾

- PR 写清改了什么、为什么、取舍（放弃了什么）、验收步骤和结果，附对照截图。提 PR 不等于授权合并。
- 停掉自己起的服务和假依赖（精确 PID），按 `$DEV/created.txt` 删自己的容器、worktree（`git worktree remove`）和 `$DEV`。

## 项目参考

- Akashic 自身：`references/akashic.md`（启动器、接模型、免密登录）。

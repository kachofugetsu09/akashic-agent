---
name: verify-akashic-plugin
description: 把自己写好的插件安装到当前运行实例并验收：提交、安装、核对状态、实际用一次、如实汇报。写完或改完插件、要确认插件是否生效时使用。
---

# 安装并验收插件

已安装、已激活、能用是三件不同的事，每一层都要单独取证。汇报只写拿到证据的那一层。

## 1. 提交

```sh
cd "$AKASHIC_WORKSPACE/plugin-src/<name>" && git add -A && git commit -m "<说明>" && git rev-parse HEAD
```

记下完整 commit。没提交的改动不会被安装。

## 2. 安装

本次可见工具里有 `plugin_install` 就用它：`source` 填源码绝对路径，`marketplace` 填 `local`，`ref` 填完整 commit。

没有这个工具时用当前实例自带的命令行。先找到实例的 python、config、workspace：

```sh
ps -eo args= | grep -F "$AKASHIC_CORE_ROOT/main.py --config" | grep -v grep | head -1
```

输出形如 `<python> <core>/main.py --config <config> --workspace <workspace>`，用同一个 python 和参数：

```sh
<python> "$AKASHIC_CORE_ROOT/main.py" plugin-install \
  --source "$AKASHIC_WORKSPACE/plugin-src/<name>" --marketplace local --ref <完整commit> \
  --config <config> --workspace <workspace>
```

不要另起一个 Akashic 进程，不要猜 python 路径。记下返回的 `update_id` 和 `plugin_id`（形如 `<name>@local`）。

## 3. 核对状态

```sh
<python> "$AKASHIC_CORE_ROOT/main.py" plugin-status <update_id> --config <config> --workspace <workspace>
```

- `state` 为 `active`，且 `active_input_ref` 等于本次的 `input_ref`：本次输入已激活。
- `accepted`：已接纳还没激活，间隔几秒再查，最多十次。
- `failed`：读 `error`，改源码后回到第 1 步。不要反复重装同一个 commit。

## 4. 实际用一次

激活只说明代码加载了，还要按插件类型取能力证据：

| 插件类型 | 怎么验 |
|---|---|
| 工具 | 先 `load_tools` 传插件 ID 拿到 schema（`always_on` 的工具组跳过这步），再真调用一次，贴出返回结果 |
| Skill | `load_skill` 读出新正文，核对 `source_id` 是本插件 |
| WebUI | 有浏览器或截图能力时打开 WebUI，在亮色和暗色主题下各看一次新内容，并检查控制台没有报错；没有就明确写"界面未亲眼验证"，请用户刷新页面查看 |
| 上下文材料、后台任务 | 触发一次对应场景，用它产生的真实结果作证据 |

新工具和 Skill 只出现在之后新取得的调用里；本轮看不到时，在下一轮再验，不要声称已生效。

## 5. 汇报

写清：插件 ID、commit、`update_id`、状态、第 4 步的实际证据、没验证到的部分。
不要把"安装成功"说成"功能可用"，不要编造没跑过的检查。

## 卸载

```sh
<python> "$AKASHIC_CORE_ROOT/main.py" plugin-uninstall <name>@local --config <config> --workspace <workspace> --json
```

卸载不删 `plugin-data` 里的数据；要删数据必须另有用户明确授权。

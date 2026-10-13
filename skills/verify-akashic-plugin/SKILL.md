---
name: verify-akashic-plugin
description: 把自己写好的插件安装到当前运行实例并验收：提交、安装、核对状态、实际用一次、如实汇报。写完或改完插件、要确认插件是否生效时使用。
---

# 安装并验收插件

已安装、已激活、能用是三件不同的事，每一层都要单独取证。汇报只写拿到证据的那一层。

一轮最多约 40 次模型输出。验收控制在 10 次以内；接近上限时先停下汇报已有证据，再说明还缺什么，不能因为步数用完而没有汇报。

## 1. 提交

```sh
cd "$AKASHIC_WORKSPACE/plugin-src/<name>" && git add -A && git commit -m "<说明>" && git rev-parse HEAD
```

记下完整 commit。没提交的改动不会被安装。

## 2. 安装

本次可见工具里有 `plugin_install` 就用它：`source` 填源码绝对路径，`marketplace` 填 `local`，`ref` 填完整 commit。

没有这个工具时用当前实例自带的命令行；后续查状态和新会话验收也使用同一入口。先根据执行环境确定命令前缀，后文记作 `<当前CLI>`：

- Host Bridge 宿主有 `$AKASHIC_RUNTIME_CLI`：使用 `"$AKASHIC_RUNTIME_CLI"`，配置与 workspace 取 `$AKASHIC_CONFIG`、`$AKASHIC_WORKSPACE`。这个入口已绑定对应 release 的解释器和源码，不需要查找容器内 `main.py`。
- 本地开发实例没有该变量：从下面的真实进程中取得 python、config、workspace，使用 `<python> "$AKASHIC_CORE_ROOT/main.py"`。变量或进程缺席时明确报告入口未定位，不拼出 `/main.py` 或另起 Runtime。

```sh
ps -eo args= | grep -F "$AKASHIC_CORE_ROOT/main.py --config" | grep -v grep | head -1
```

安装命令（`<config>`、`<workspace>` 替换为上面取得的路径；子命令必须在配置参数之前）：

```sh
<当前CLI> plugin-install \
  --source "$AKASHIC_WORKSPACE/plugin-src/<name>" --marketplace local --ref <完整commit> \
  --config <config> --workspace <workspace>
```

不要另起一个 Akashic 进程，不要猜 python 路径。记下返回的 `update_id` 和 `plugin_id`（形如 `<name>@local`）。

## 3. 核对状态

```sh
<当前CLI> plugin-status <update_id> --config <config> --workspace <workspace>
```

- `state` 为 `active`，且 `active_input_ref` 等于本次的 `input_ref`：本次输入已激活。
- `accepted`：已接纳还没激活，间隔几秒再查，最多十次。
- `failed`：读 `error`，改源码后回到第 1 步。不要反复重装同一个 commit。
- `restart_required`：输入已选择但需重启，不能报告已生效；按已有重启授权处理，再核对同一输入。`unknown`：保留回执并报告尚未确认，不当成成功。

## 4. 实际用一次

激活只说明代码加载了，还要按插件类型取能力证据。

当前这一轮的工具和 Skill 目录在安装前就固定了，新能力在本轮看不到（`load_tools` 会报"不属于当前获授工具目录"）。
工具和 Skill 用一个新会话验：

```sh
<当前CLI> exec --new --final-only \
  "只做一件事：调用 <工具名>，参数 <...>，把工具返回的原始结果原样贴出来，不要自己算。" \
  --config <config> --workspace <workspace> 2>&1
```

它新建一个程序化会话（不写记忆），等那一轮结束后只打印最终回复；失败时退出码为 1，`reason` 在输出里。

| 插件类型 | 怎么验 |
|---|---|
| 工具 | 用上面的 `exec`：工具组非 `always_on` 时让新会话先 `load_tools` 传插件 ID 再调用；贴出原始返回 |
| Skill | 用上面的 `exec` 让新会话 `load_skill` 并复述要点，核对来源是本插件 |
| WebUI | 从当前实例入口或已确认的 Web 配置取得浏览器可访问的 URL（不是 Gateway RPC 端口）。读取该 URL 的 `/api/chat/web-ui/bootstrap`，检查 `modules[].pluginId` 包含本插件，只能证明目录已发布。再跑下面的截图脚本；退出码 2 表示缺少 Playwright 或浏览器，写明"界面未亲眼验证"，不要用 curl 扒前端资源代替 |
| 上下文材料、后台任务 | 触发一次对应场景，用它产生的真实结果作证据 |

WebUI 截图脚本，不用自己写浏览器脚本：

```sh
python3 "<本 skill 的 base_directory>/scripts/shoot_page.py" "<导航文字>" "$AKASHIC_WORKSPACE/.shots/<name>-<commit前缀>" \
  --url "<当前实例 Web URL>" \
  --click "<按钮文字>" --click "<按钮文字>"
```

`base_directory` 使用 `load_skill` 返回的实际资源目录，不从 Core 路径拼接。解释器选能 `import playwright` 的那个；可用 `--browser <路径>` 指定浏览器。
脚本打开页面、按顺序点按钮并记录点击后的页面文字，
存 `light.png`、`dark.png`、`narrow.png`，输出 JSON：`clicks`、`horizontal_overflow`、`errors`；路由取消请求与浏览器缺省图标提示单列在 `cancelled_requests`、`browser_notes`，结合页面结果核对。
输出目录必须尚不存在，保留每次验收证据；有请求错误或横向溢出时退出码为 1。读截图看布局；能读图就写看到了什么，读图失败就只写 JSON 里的事实。

## 5. 汇报

写清：插件 ID、commit、`update_id`、状态、第 4 步的实际证据、没验证到的部分。
不要把"安装成功"说成"功能可用"，不要编造没跑过的检查。

## 卸载

```sh
<当前CLI> plugin-uninstall <name>@local --config <config> --workspace <workspace> --json
```

卸载不删 `plugin-data` 里的数据；要删数据必须另有用户明确授权。

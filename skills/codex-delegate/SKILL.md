---
name: codex-delegate
description: 把长代码库任务委托给本机 Codex CLI 后台执行。当用户说用 codex skill、codexskill、codex delegate、委托 codex、后台 codex、阻塞 codex exec、subagent 跑 codex 时使用。
metadata: {"akashic": {"always": false, "requires": {"bins": ["codex"]}}}
---

# Codex Delegate

## 目标

把一个可以独立完成的长任务交给后台 subagent；subagent 内部用 `shell` 同步阻塞等待 `codex exec` 完成，最后由后台 subagent 把结果带回主会话。

## 流程

```
┌─ 主会话
│  ├─ shell(command -v codex && codex --version, auto_promote=false)
│  └─ spawn(run_in_background=true, profile="scripting" 或 "general")
│     └─ 后台 subagent
│        ├─ write_file(<task_dir>/prompt.txt)
│        ├─ shell(auto_promote=false)
│        │  └─ codex exec --cd <repo> --output-last-message <task_dir>/codex-result.md - < <task_dir>/prompt.txt
│        │     └─ 阻塞等待完成
│        ├─ read_file(<task_dir>/codex-result.md)
│        └─ read_file(<task_dir>/codex-session.txt)
└─ subagent 完成后回灌结果
```

## 使用规则

1. 主 agent 必须先检查 `codex` 是否存在：调用 `shell(command="command -v codex && codex --version", auto_promote=false)`。如果失败，直接告诉用户本机没有可用 Codex CLI，不要 spawn。不要把这个检查下放给 subagent。
2. 如果用户已经给出 repo 路径，主 agent 不要为了理解任务先 `list_dir`、`read_file`、`shell find/grep` 探索该仓库；只把 repo 路径和任务目标原样传给 spawn。代码库探索由 Codex delegate 完成。
3. 只有用户没给 repo 路径、路径明显缺失、或用户要求主 agent 先确认路径时，主 agent 才做最小路径检查。
4. 主 agent 不要在 spawn task 里写“重点读取这些文件”“我已经发现这些入口”“候选路径如下”这类预探索结果；除非用户原文明确指定文件。spawn task 只能包含用户给出的 repo 路径、用户目标、输出要求和本技能的执行约束。
5. Codex prompt 必须要求 Codex 自己从整个 repo 发现入口、目录和相关文件，而不是沿着主 agent 预先挑出的文件列表工作。
6. 检查通过后，外层必须用 `spawn(run_in_background=true)`，不要在主会话里直接跑长时间 `codex exec`。
7. subagent 的 `profile` 选 `scripting`；如果任务还需要联网调研，选 `general`。
8. subagent 内部调用 `shell` 时必须设置 `auto_promote=false`，不要设置 `run_in_background=true`。
9. `auto_promote=false` 且不传 `timeout` 时，shell 会默认同步等待 21600 秒；只有需要更短硬截止时才显式传 `timeout`。
10. `codex exec` 要用 `--cd <repo>` 指定工作目录，避免依赖 shell 的 `cd` 状态。
11. 默认把任务说明写入 prompt 文件，再用 `codex exec --cd <repo> - < prompt.txt` 读取，避免 shell 引号、换行和特殊字符破坏 prompt。
12. 必须给 `codex exec` 加 `--output-last-message <task_dir>/codex-result.md`，完成后读这个文件作为主要结果；不要从 `/tmp/akashic-shell-*.log` 里 grep 或 tail 输出。
13. 同时把 stdout+stderr 保存到 `<task_dir>/codex-run.log`，因为 `session id: ...` 在 stderr 里。
14. 如果需要后续复用同一个 Codex 会话，必须从 `codex-run.log` 提取 session id，写入 `<task_dir>/codex-session.txt`，并在回复里告诉主会话这个 session id。
15. prompt、result、run log、session id 都必须放在 subagent 任务目录 `<task_dir>`；不要写 `/tmp`，也不要依赖 `/tmp/akashic-shell-*.log`。

## 推荐调用形态

外层：

```text
shell(
  command="command -v codex && codex --version",
  description="检查 codex",
  auto_promote=false
)

spawn(
  task="用户给出的 repo 路径是 /path/to/repo；用户目标是：<原样概括用户目标>。不要让主 agent 先探索这个仓库，不要使用主 agent 预选的文件列表；Codex CLI 必须把 /path/to/repo 当作完整代码库，自行发现入口、目录和相关文件。在当前 subagent 任务目录写入 prompt.txt，然后用 shell 阻塞执行 codex exec --cd /path/to/repo --output-last-message <当前任务目录>/codex-result.md - < <当前任务目录>/prompt.txt，并用 2>&1 | tee <当前任务目录>/codex-run.log 保存完整日志。shell 必须设置 auto_promote=false，不要 run_in_background，不要写 /tmp，不要读取 /tmp/akashic-shell 日志。完成后读取 codex-result.md，总结结果；从 codex-run.log 提取 session id 写入 codex-session.txt 并带回。",
  label="codex delegate",
  profile="scripting",
  run_in_background=true
)
```

subagent 内层：

```text
shell(
  command="bash -lc 'set -o pipefail; codex exec --cd /path/to/repo --output-last-message /path/to/task_dir/codex-result.md - < /path/to/task_dir/prompt.txt 2>&1 | tee /path/to/task_dir/codex-run.log; sed -n \"s/^session id: //p\" /path/to/task_dir/codex-run.log | tail -1 > /path/to/task_dir/codex-session.txt'",
  description="阻塞 codex",
  auto_promote=false
)
```

完成后：

```text
read_file(path="/path/to/task_dir/codex-result.md")
read_file(path="/path/to/task_dir/codex-session.txt")
```

需要续聊同一个 Codex 会话时：

```text
codex exec resume <session_id> --output-last-message /path/to/task_dir/codex-result-2.md - < /path/to/task_dir/prompt-2.txt
```

## 注意

- 不要把 `codex exec` 命令末尾加 `&`、`nohup`、`disown`，也不要用 shell 后台化包装。
- 用户消息里已经有 repo 路径时，主 agent 不要提前读这个 repo 的文件来“了解一下”；直接委托 Codex CLI。
- 主 agent 不要把自己猜的入口文件、相关目录、搜索结果塞进 Codex prompt；这些会污染 Codex 对完整 repo 的自主分析。
- Codex prompt 可以写“请自行在整个 repo 中定位相关实现”，不要写“请重点读取以下文件”，除非这些文件来自用户原文。
- 不要轮询 `task_output` 等 shell 后台结果；这个技能要求 shell 前台阻塞，等待完整结果后再返回。
- 简短 prompt 可以作为参数传给 `codex exec --cd <repo> '<任务说明>'`，但默认优先用 stdin 文件形态。
- `--last` 只适合人工临时使用；自动化续聊必须传明确的 `<session_id>`，避免串到别的 Codex 会话。
- 如果不用 `bash -lc`，也可以分两步执行：先跑 `codex exec ... 2>&1 | tee codex-run.log`，再用 `sed` 从 `codex-run.log` 提取 session id。
- 如果 `write_file` 因路径限制失败，说明路径没用当前 subagent 任务目录；修正为 `<task_dir>/prompt.txt`，不要改用 shell 写 `/tmp`。

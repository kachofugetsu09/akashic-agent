"""脚本来源：watch/sources/ 里的 .sh / .py 由值班每轮替它跑一遍，标准输出的每一行就是一条事件。"""
from __future__ import annotations

import shlex
from pathlib import Path

from agent.plugin_composition import Context
from plugins.host_execution.contract import DEFAULT_MAX_OUTPUT_TOKENS, MAX_YIELD_TIME_MS, PROCESSES

RUNNERS = {".sh": "sh", ".py": "python3"}
MAX_LINES = 200


def find(root: Path) -> list[Path]:
    directory = root / "sources"
    return sorted(path for path in directory.glob("*") if path.suffix in RUNNERS and path.is_file()) if directory.exists() else []


# 和主 Agent 的命令行在同一个环境里执行，脚本在那里试通了，在这里就能跑。返回输出的行，或者一句失败原因。
async def run(ctx: Context, path: Path) -> tuple[list[str], str]:
    errors = path.with_name(f".{path.stem}.err")
    command = f"{RUNNERS[path.suffix]} {shlex.quote(path.name)} 2>{shlex.quote(errors.name)}"
    owner = f"watch-source:{path.stem}"
    try:
        processes = ctx.require(PROCESSES)
        result = await processes.exec_command(
            ctx, owner, command=command, argv=processes.resolve_shell(None).derive_argv(command, login=False), cwd=path.parent, env={},
            tty=False, yield_time_ms=MAX_YIELD_TIME_MS, max_output_tokens=DEFAULT_MAX_OUTPUT_TOKENS, hard_timeout_s=MAX_YIELD_TIME_MS // 1000 + 10)
        if result.execution_id is not None:
            _ = await processes.terminate_execution(ctx, owner, result.execution_id)
            return [], f"超过 {MAX_YIELD_TIME_MS // 1000} 秒没有结束，已终止"
    except Exception as error:
        # 执行环境是外部进程服务：它出任何问题都只算这个来源这一轮失败，不影响值班
        return [], f"没能运行（{type(error).__name__}: {error}）"
    lines = [line for line in result.output.decode("utf-8", "replace").splitlines() if line.strip()][:MAX_LINES]
    if result.exit_code == 0:
        return lines, ""
    detail = errors.read_text(encoding="utf-8", errors="replace").strip()[-400:] if errors.exists() else ""
    return [], f"退出码 {result.exit_code}" + (f"：{detail}" if detail else "")

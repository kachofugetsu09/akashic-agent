import logging
import os
import shlex
from pathlib import Path

from agent.plugins import Plugin, on_tool_pre
from agent.lifecycle.types import PreToolCtx

logger = logging.getLogger("plugin.shell_restore")


def _restore_dir() -> str:
    return os.environ.get("AKASIC_RESTORE_DIR", str(Path.home() / "restore"))


class ShellRestore(Plugin):
    name = "shell_restore"

    @on_tool_pre(tool_name="shell")
    async def rewrite_rm_to_mv(self, event: PreToolCtx) -> dict[str, object] | None:
        command = str(event.arguments.get("command", "")).strip()
        rewritten = self._rewrite_command(command)
        if rewritten is None:
            return None
        Path(_restore_dir()).mkdir(parents=True, exist_ok=True)
        logger.info("rm → mv: %r", rewritten)
        return dict(event.arguments, command=rewritten)

    def _rewrite_command(self, command: str) -> str | None:
        try:
            tokens = shlex.split(command, posix=True)
        except ValueError:
            return None
        if not tokens:
            return None
        # 读取 rm 前面的前缀（sudo、env、VAR=val 等）
        prefix: list[str] = []
        i = 0
        while i < len(tokens):
            token = tokens[i]
            if Path(token).name == "rm":
                break
            if token == "sudo" or token == "env" or "=" in token:
                prefix.append(token)
                i += 1
                continue
            return None
        if i >= len(tokens) or Path(tokens[i]).name != "rm":
            return None
        # 跳过 rm 名字本身
        i += 1
        # 跳过 rm 选项，收集目标路径
        targets: list[str] = []
        parsing_options = True
        while i < len(tokens):
            token = tokens[i]
            i += 1
            if parsing_options and token == "--":
                parsing_options = False
                continue
            if parsing_options and token.startswith("-") and token != "-":
                continue
            parsing_options = False
            targets.append(token)
        if not targets:
            return None
        # 改写为 mv -- targets... restore_dir
        parts = [*prefix, "mv", "--"]
        parts.extend(targets)
        parts.append(_restore_dir())
        return shlex.join(parts)

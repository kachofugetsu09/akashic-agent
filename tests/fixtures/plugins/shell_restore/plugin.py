import logging
import os
import shlex
from pathlib import Path

from agent.lifecycle.types import PreToolCtx
from agent.plugins import Plugin, on_tool_pre

logger = logging.getLogger("plugin.shell_restore_fixture")


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
        logger.info("[%s] rm -> mv: %r", self.name, rewritten)
        return dict(event.arguments, command=rewritten)

    def _rewrite_command(self, command: str) -> str | None:
        try:
            tokens = shlex.split(command, posix=True)
        except ValueError:
            return None
        if not tokens:
            return None
        prefix: list[str] = []
        index = 0
        while index < len(tokens):
            token = tokens[index]
            if Path(token).name == "rm":
                break
            if token == "sudo" or token == "env" or "=" in token:
                prefix.append(token)
                index += 1
                continue
            return None
        if index >= len(tokens) or Path(tokens[index]).name != "rm":
            return None
        index += 1
        targets: list[str] = []
        parsing_options = True
        while index < len(tokens):
            token = tokens[index]
            index += 1
            if parsing_options and token == "--":
                parsing_options = False
                continue
            if parsing_options and token.startswith("-") and token != "-":
                continue
            parsing_options = False
            targets.append(token)
        if not targets:
            return None
        return shlex.join([*prefix, "mv", "--", *targets, _restore_dir()])


__all__ = ["ShellRestore"]

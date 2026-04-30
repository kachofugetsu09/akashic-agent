from agent.lifecycle.types import PreToolCtx
from agent.plugins import on_tool_pre
from plugins.shell_restore.plugin import ShellRestore as _ShellRestore


class ShellRestore(_ShellRestore):
    @on_tool_pre(tool_name="shell")
    async def rewrite_rm_to_mv(self, event: PreToolCtx) -> dict[str, object] | None:
        return await super().rewrite_rm_to_mv(event)


__all__ = ["ShellRestore"]

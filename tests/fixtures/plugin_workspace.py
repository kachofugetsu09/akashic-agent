"""显式准备测试新 workspace；恢复路径不得再次调用。"""
from pathlib import Path

from agent.plugins.selection import PluginSelection


def initialize_plugin_workspace(workspace: Path) -> None:
    """由测试 setup 声明新运行空间，只创建一次空插件选择。"""
    workspace.mkdir(parents=True, exist_ok=True)
    PluginSelection(workspace).initialize()

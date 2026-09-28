"""保留 setup 命令作为非交互的 Core 初始化入口。"""
from pathlib import Path
from bootstrap.init_workspace import init_workspace


def run_setup_wizard(config_path: Path, workspace: Path) -> None:
    """保留现有配置，插件设置在启动后的 Web 页面完成。"""
    summary = init_workspace(config_path=config_path, workspace=workspace)
    for note in summary.notes:
        print(note)
    print("初始化完成。安装插件组合并启动后，在 Web 页面完成初始配置。")

"""在独占 benchmark workspace 中初始化当前插件配置。"""
from pathlib import Path
import sys

from agent.plugin_composition.config_input import save_config
from agent.plugins.bundles import set_plugin_choice
from bootstrap.init_workspace import init_workspace


def prepare(workspace: Path) -> None:
    """先建立空选择，再通过各 owner 写入实验配置。"""
    # 1. 新 workspace 的选择由初始化入口建立，不依赖旧配置迁移。
    init_workspace(config_path=Path(__file__).with_name("config.toml"), workspace=workspace)
    set_plugin_choice(workspace, "subagent", enabled=False)
    # 题目容器没有 Workload Controller，不启动桌面容器 provider。
    set_plugin_choice(workspace, "computer", enabled=False)
    # 2. 保留无限回复步数、近期原文预算、关闭聊天与默认控制监听。
    data = workspace / "plugin-data"
    save_config(data / "reply-builtin", {"max_steps": 0})
    save_config(data / "compaction-builtin", {"keep_recent_tokens": 20000})
    save_config(data / "akashic_clients-builtin", {"enabled": False, "web": {"enabled": False}})
    save_config(data / "gateway-builtin", {"enabled": True, "listen": ""})
    save_config(data / "context-builtin", {
        "prompt_sources": {"default_prompt": "prompt", "markdown_memory": "markdown_memory",
                           "skills": "skills"},
        "summary_source": ["compaction", "compaction"],
    })


if __name__ == "__main__":
    prepare(Path(sys.argv[1]))

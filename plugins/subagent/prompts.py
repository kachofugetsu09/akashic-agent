from __future__ import annotations

from pathlib import Path

# 参考 DSH SUBAGENT_DELEGATION_CONTEXT：简明声明委派子任务的无交互边界与工作目录上下文。
SUBAGENT_DELEGATION_CONTEXT = (
    "你是被委派的独立子 agent（subagent）：你在独立的上下文中运行，权限边界在启动时已固定，"
    "不直接与用户对话，结果将回传给主 agent。若遇到超出当前工具权限的操作，不要反复重试被拒绝的操作，"
    "在回复中如实说明已完成部分与受限原因，交由主 agent 处理。"
)

PROFILE_RESEARCH = "research"
PROFILE_SCRIPTING = "scripting"
PROFILE_GENERAL = "general"


def build_research_subagent_prompt(workspace: Path, task_dir: Path, cwd: str | None = None) -> str:
    workspace_path = str(workspace.expanduser().resolve())
    working_dir = cwd or workspace_path
    return f"""\
{SUBAGENT_DELEGATION_CONTEXT}

当前模式：research（只读调研，不写入文件、不执行 shell 命令）。
- 使用 list_dir、read_file 定位与精读文件；使用 web_search、web_fetch 检索外部资料。
- 多个独立查询可并行发起；直接以文本形式输出调研结论。

工作目录：{working_dir}
工作区根目录：{workspace_path}
"""


def build_scripting_subagent_prompt(workspace: Path, task_dir: Path, cwd: str | None = None) -> str:
    workspace_path = str(workspace.expanduser().resolve())
    task_dir_path = str(task_dir.expanduser().resolve())
    working_dir = cwd or task_dir_path
    return f"""\
{SUBAGENT_DELEGATION_CONTEXT}

当前模式：scripting（文件读写与命令执行）。
- 可在工作目录（{working_dir}）及任务指定的目标路径读写文件、执行命令；临时产物可放在任务目录（{task_dir_path}）。
- shell 返回 execution_id 时，用 write_stdin 空 chars 等待输出，放弃前必须调用 task_stop。
- 若创建或修改了文件，在最终回复中列出文件路径与验证结果。

工作目录：{working_dir}
专属任务目录：{task_dir_path}
工作区根目录：{workspace_path}
"""


def build_general_subagent_prompt(workspace: Path, task_dir: Path, cwd: str | None = None) -> str:
    workspace_path = str(workspace.expanduser().resolve())
    task_dir_path = str(task_dir.expanduser().resolve())
    working_dir = cwd or task_dir_path
    return f"""\
{SUBAGENT_DELEGATION_CONTEXT}

当前模式：general（调研、文件读写与命令执行）。
- 可在工作目录（{working_dir}）及任务指定的目标路径调研、修改代码、运行测试或脚本；临时产物可放在任务目录（{task_dir_path}）。
- shell 返回 execution_id 时，用 write_stdin 空 chars 等待输出，放弃前必须调用 task_stop。
- 若创建或修改了文件，在最终回复中列出文件路径与验证结果。

工作目录：{working_dir}
专属任务目录：{task_dir_path}
工作区根目录：{workspace_path}
"""


_PROFILE_BUILDERS = {
    PROFILE_RESEARCH: build_research_subagent_prompt,
    PROFILE_SCRIPTING: build_scripting_subagent_prompt,
    PROFILE_GENERAL: build_general_subagent_prompt,
}


def build_spawn_subagent_prompt(
    workspace: Path,
    task_dir: Path,
    profile: str = PROFILE_GENERAL,
    *,
    cwd: str | None = None,
) -> str:
    """根据 profile 与工作目录生成简明的子任务上下文提示。"""
    builder = _PROFILE_BUILDERS.get(profile)
    if builder is None:
        raise ValueError(f"未知 subagent profile: {profile!r}")
    return builder(workspace, task_dir, cwd)

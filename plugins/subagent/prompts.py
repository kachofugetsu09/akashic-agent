from __future__ import annotations

from pathlib import Path

# 参考 DSH SUBAGENT_DELEGATION_CONTEXT：简明声明委派子任务的无交互边界与工作目录上下文。
SUBAGENT_DELEGATION_CONTEXT = (
    "你是被委派的独立子 agent（subagent）：你在独立的上下文中运行，权限边界在启动时已固定，"
    "不直接与用户对话，结果将回传给主 agent。若遇到超出当前工具权限的操作，不要反复重试被拒绝的操作，"
    "在回复中如实说明已完成部分与受限原因，交由主 agent 处理。"
)


def build_spawn_subagent_prompt(
    workspace: Path,
    task_dir: Path,
    profile: str = "general",
    *,
    read_only: bool = False,
    depth: int = 1,
    max_depth: int = 1,
    cwd: str | None = None,
    fork: bool = False,
) -> str:
    """根据只读开关、深度限制与工作目录生成简明的子任务上下文提示。"""
    workspace_path = str(workspace.expanduser().resolve())
    task_dir_path = str(task_dir.expanduser().resolve())
    effective_read_only = read_only or profile == "research"
    working_dir = cwd or (workspace_path if effective_read_only else task_dir_path)
    depth_line = f"委派深度：当前第 {depth} 层 / 上限 {max_depth} 层"
    if depth >= max_depth:
        depth_line += "（已达上限，请直接完成当前任务，不要再继续派发子任务）"

    if effective_read_only:
        tail = f"工作区根目录：{workspace_path}" if fork else f"工作目录：{working_dir}\n工作区根目录：{workspace_path}"
        return f"""\
{SUBAGENT_DELEGATION_CONTEXT}

当前模式：只读调研（不写入文件、不执行修改类命令）。
- 优先使用已有只读/检索工具完成定位与核实；多个独立查询可并行发起，直接以文本输出结论。
- {depth_line}

{tail}
"""

    if fork:
        return f"""\
{SUBAGENT_DELEGATION_CONTEXT}

当前模式：继承父会话工具（分叉共享上下文，可按所获授工具在任务指定路径调研、读写文件与执行命令）。
- 可在工作目录及任务指令提供的专属目录读写文件、运行命令或调用已继承的插件工具。
- shell 返回 execution_id 时，用 write_stdin 空 chars 等待输出，放弃前必须调用 task_stop。
- 若创建或修改了文件，在最终回复中列出文件路径与验证结果。
- {depth_line}

工作区根目录：{workspace_path}
"""
    return f"""\
{SUBAGENT_DELEGATION_CONTEXT}

当前模式：继承父会话工具（可按所获授工具在目标路径调研、读写文件与执行命令）。
- 可在工作目录（{working_dir}）及任务指定的目标路径读写文件、运行命令或调用已继承的插件工具；临时产物可放在专属任务目录（{task_dir_path}）。
- shell 返回 execution_id 时，用 write_stdin 空 chars 等待输出，放弃前必须调用 task_stop。
- 若创建或修改了文件，在最终回复中列出文件路径与验证结果。
- {depth_line}

工作目录：{working_dir}
专属任务目录：{task_dir_path}
工作区根目录：{workspace_path}
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MetaToolGroup:
    title: str
    tools: tuple[tuple[str, str], ...]


META_TOOLBOX_GROUPS: tuple[MetaToolGroup, ...] = (
    MetaToolGroup(
        title="Meta",
        tools=(
            ("tool_search", "搜索并解锁其他工具"),
        ),
    ),
    MetaToolGroup(
        title="Read",
        tools=(
            ("read_file", "读取文件"),
            ("list_dir", "查看目录"),
            ("web_search", "搜索网络"),
            ("web_fetch", "抓取网页"),
            ("recall_memory", "检索结构化记忆，优先回答历史事实/偏好/做过什么"),
            ("fetch_messages", "按消息 ID 回溯原始对话"),
            ("search_messages", "搜索历史对话"),
        ),
    ),
    MetaToolGroup(
        title="Write",
        tools=(
            ("write_file", "写文件"),
            ("edit_file", "改文件"),
            ("message_push", "主动推送消息/文件/图片"),
            ("memorize", "立即写入记忆"),
            ("forget_memory", "将已确认错误的记忆标记为失效"),
        ),
    ),
    MetaToolGroup(
        title="System",
        tools=(("shell", "执行终端命令"),),
    ),
)

META_TOOLBOX_NAMES: tuple[str, ...] = tuple(
    name for group in META_TOOLBOX_GROUPS for name, _ in group.tools
)


def build_meta_toolbox_prompt() -> str:
    lines = [
        "你有一组始终可见的 MetaToolBox，优先覆盖检索、读写文件、消息回溯、推送和终端执行。",
        "当任务能被 MetaToolBox 直接完成时，优先使用这组工具，不必先 tool_search。",
        "",
    ]
    for group in META_TOOLBOX_GROUPS:
        lines.append(f"[{group.title}]")
        for name, summary in group.tools:
            lines.append(f"- {name}: {summary}")
        lines.append("")
    return "\n".join(lines).strip()

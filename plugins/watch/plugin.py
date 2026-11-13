from __future__ import annotations

import asyncio
from collections.abc import Mapping
from datetime import datetime

from agent.plugin_composition import RUNTIME_STARTED, RUNTIME_STOPPING, Context, stop_spawned
from agent.plugin_composition.plugin_config import PLUGIN_CONFIG
from core.common.file_io import run_file_io
from plugins.assets.contract import INSTALLED_ASSETS
from plugins.context.contract import MATERIALS_V4
from plugins.host_execution.contract import PROCESSES
from plugins.ledger.contract import Message
from plugins.ledger.contract import MESSAGE_CATALOG
from plugins.models.contract import CHAT_MODELS, MODEL_CALL_STATS, MODEL_CATALOG
from plugins.plugin_settings.contract import SETTINGS_FORMS, declare_settings
from plugins.programmatic.contract import PROGRAMMATIC
from plugins.tools.contract import TOOL_DISPLAY
from plugins.ui.contract import UI_SLOTS, PluginUiDefinition, PluginUiRpcInvalidRequest
from plugins.userinformation.contract import USER_INFORMATION

from . import panel
from .contract import WATCH_EVENTS
from .runtime import Runtime, count_items, count_notes
from .settings import FORM, Config

POINTER = (
    "你在后台留着 {notes} 行便签（watch/notes.md），待看里有 {inbox} 条（watch/inbox.md）。那是你自己的事，不是这次对话的议程。"
    "先听他这句话在说什么、此刻是什么状态，就回应这一件：他报了个状态或一件事，接住它；他交代了事，就去办，能自己定的自己定，不请示，也不问他已经说过的。"
    "回应比提问重要。想问只问一句，而且是顺着他刚说的话问；他没接的问题不问第二遍。他累、在忙、在玩、要睡了，就短一点，不问。"
    "他的话短，你也短：他说一两句，你回三五句就够，不要把一句话展开成一篇。他分享一件高兴的事，就和他一起高兴，不要转成叮嘱和建议；没人问就不给建议。"
    "你平时看到、记下的关于他的事，和他正在说的有关时就用上，让他知道你看见了，说得出是从哪知道的；无关的不提，更不要一件件罗列。"
    "他提到一件过后才有结果的事，自己记一行，到时候挑他有空的时刻问（办法在技能 watch 里）。"
    "他问一件还在变的事现在怎么样了，就是在等它：你够得着的就自己去看，不问他要你本来看得到的东西。"
    "这件事还没到头，回答就不算做完：结束这一轮之前，先按技能 watch 里「他没让你盯，但他在等的事」把后面怎么接着看安排好，别让他问第二遍；已经结束的事和随口一问的常识不用。"
    "他问的事和记忆里某个专题沾边，先读它再答：照上次走通的路走，上次卡住的地方别再去撞，直接告诉他卡在哪、缺什么；上次看到的和这次看到的放在一起说。"
    "他直接问你知道些什么，就坦白全说。")

api_version = 3
name = "watch"
version = "1.6.0"
desc = "值班：便宜的模型看着事件流，只在该叫的时候叫醒主 Agent；主 Agent 在后台醒来后自己决定做什么、说不说话"
inject = (PLUGIN_CONFIG, CHAT_MODELS, MODEL_CATALOG, MESSAGE_CATALOG, PROGRAMMATIC, PROCESSES)
workspace_roots = ("watch",)


class Current:
    """用户信息入口的提供方：各来源最近一条状态事件。"""

    def __init__(self, runtime: Runtime) -> None:
        self._runtime = runtime

    async def describe(self) -> Mapping[str, object]:
        return {"queries": {"current": {
            "description": "各来源最近一次报告的用户状态（位置、睡眠、在线等）和报告时间；没有某个来源不等于否定。",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False}}}}

    async def query(self, query: str, arguments: Mapping[str, object]) -> Mapping[str, object]:
        if query != "current" or arguments:
            return {"status": "invalid_query", "error": "Use current without arguments"}
        items = [{"source": event.source, "since": event.at, "state": " — ".join(part for part in (event.title, event.text) if part)[:1000]}
                 for event in await self._runtime.states()]
        return {"items": items, "semantics": "Latest state each source reported, with the time it started. Not a live measurement."}


async def apply(ctx: Context) -> None:
    """事件入口始终可用；只有配置了主对话，值班循环才启动。"""
    config = Config.model_validate(ctx.config)
    runtime = Runtime(ctx, config)
    _ = await ctx.provide(WATCH_EVENTS, runtime)
    _ = await ctx.inject((SETTINGS_FORMS, PLUGIN_CONFIG, MODEL_CATALOG),
                         lambda child: declare_settings(child, Config, FORM), name="settings")
    # 机制说明是普通技能，主 Agent 需要时用 load_skill 读取。
    _ = await ctx.inject((INSTALLED_ASSETS,), lambda child: child.require(INSTALLED_ASSETS).register(child, "skills", "skills"),
                         name="skills")

    # 用户来主对话聊天时给主 Agent 的一段提醒：后台攒下的东西不是这次对话的议程，先回应他这句话。
    async def pointer(snapshot: tuple[Message, ...], _source: str) -> dict[str, object]:
        if not snapshot or snapshot[-1].session_id != config.target:
            return {}
        notes, inbox = count_notes(runtime.root / "notes.md"), count_items(runtime.root / "inbox.md")
        return {"reminders": ({"name": "watch", "priority": 300, "replay": False, "text": POINTER.format(notes=notes, inbox=inbox)},)}

    _ = await ctx.inject((MATERIALS_V4,), lambda child: child.require(MATERIALS_V4).register(
        child, name="watch", kind="context", prepare=pointer), name="pointer")
    # 主 Agent 平时聊天想知道"用户现在怎么样"时，经统一的用户信息入口读到同一份状态。
    _ = await ctx.inject((USER_INFORMATION,), lambda child: child.require(USER_INFORMATION).register(child, Current(runtime)),
                         name="information")
    watcher: asyncio.Task[None] | None = None

    async def start(_event: object) -> None:
        nonlocal watcher
        if config.target:
            watcher = await ctx.spawn(runtime.follow(), name="watch")

    async def stop(_event: object) -> None:
        await stop_spawned(watcher)

    _ = await ctx.on(RUNTIME_STARTED, start)
    _ = await ctx.on(RUNTIME_STOPPING, stop)

    # 值班面板：侧栏插件面板里的一项，数据从日志、事件文件和账本现读。
    async def register_panel(child: Context) -> object:
        display = child.require(TOOL_DISPLAY)
        call_stats = child.require(MODEL_CALL_STATS)

        def tool_name(binding_id: str) -> str:
            try:
                return display(binding_id).name
            except Exception:
                return binding_id[:8]

        async def query(method: str, payload: dict[str, object], *, session_id: str | None, turn_id: str | None) -> dict[str, object]:
            now = datetime.now(runtime.zone)
            catalog = ctx.require(MESSAGE_CATALOG)
            if method == "round":
                reader = catalog.reader(panel.parse_session(payload))
                steps, usage = await run_file_io(lambda: panel.read_round_detail(reader, tool_name, call_stats))
                return {"steps": steps, "usage": usage}
            if method != "day":
                raise PluginUiRpcInvalidRequest(f"不支持的查询：{method}")
            day = panel.parse_day(payload, now.date())
            talk = await run_file_io(lambda: panel.read_talk(catalog.reader(config.target), day, now)) if config.target else []
            ticks, usage = await run_file_io(lambda: panel.read_day_usage(runtime.root, day, catalog.reader, call_stats))
            lanes = await run_file_io(lambda: panel.read_lanes(runtime.root, day))
            files = {name: runtime.files.text(f"watch/{name}") or "" for name in ("WATCH.md", "notes.md", "inbox.md")}
            return {"date": day.isoformat(), "today": now.date().isoformat(), "now": f"{now:%H:%M}", "interval": config.interval_minutes,
                    "ticks": ticks, "lanes": lanes, "talk": talk, "files": files, "usage": usage}

        return await child.require(UI_SLOTS).register_plugin_ui(
            child, PluginUiDefinition(module="plugin_ui.js", stylesheet="plugin_ui.css", slots=("sidebar.panels",)), query=query)

    _ = await ctx.inject((UI_SLOTS, TOOL_DISPLAY, MODEL_CALL_STATS), register_panel, name="panel")

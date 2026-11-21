"""值班主循环：收事件、攒批给值班员判断、落记录、该叫的时候叫醒主 Agent。"""
from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from agent.plugin_composition import Context
from core.common.file_io import run_file_io
from infra.persistence.json_store import atomic_save_json, load_json
from plugins.ledger.contract import MESSAGE_CATALOG
from plugins.models.contract import CHAT_MODELS, MODEL_CATALOG, ModelError, ModelRequest
from plugins.programmatic.contract import PROGRAMMATIC

from . import orders as watch_orders
from . import panel, recent, scripts, timeline, trail, wake
from .events import Event, EventLog, clip, load_places, name_places, normalize, parse_line, read_day, read_sources
from .settings import WAKE_MODEL, Config
from .watcher import Call, Files, ModelFailure, ModelStats, Outcome, Reply, Verdict, judge

logger = logging.getLogger(__name__)
CHUNK, MAX_CHUNKS, WAKE_EVENTS = 40, 5, 6


@dataclass(slots=True)
class State:
    next_id: int = 1
    judged: int = 0
    offsets: dict[str, int] = field(default_factory=dict)
    # 来源名 → 它最近一条状态事件的编号
    current: dict[str, str] = field(default_factory=dict)
    # 合并相同状态时已接纳但未另写正文的 (来源 → key → 事件编号)，随事件保留期清理
    aliases: dict[str, dict[str, str]] = field(default_factory=dict)
    wakes: list[str] = field(default_factory=list)
    orders_checked: str | None = None
    review_day: str | None = None
    last_agent: str | None = None
    review_tries: int = 0
    dream_day: str | None = None
    # 时间线记到哪了：他说的话按时间、状态事件按编号
    said_until: str | None = None
    timeline_event: int = 0


class Runtime:
    def __init__(self, ctx: Context, config: Config) -> None:
        self.ctx = ctx
        self.config = config
        self.root = ctx.workspace_root("watch")
        self.zone = ZoneInfo(config.timezone)
        self.log = EventLog(self.root)
        self.files = Files(ctx.runtime.workspace, ("memory", "watch"))
        self._state_path = ctx.data_root / "state.json"
        self._lock = asyncio.Lock()
        saved = load_json(self._state_path, {}, domain="watch")
        self.state = State(**{key: value for key, value in saved.items() if key in State.__slots__})
        self.state.next_id = max(self.state.next_id, self.log.last_number() + 1)
        self._keys: dict[tuple[str, str], str] = {}
        self._sync_keys(self.log.keys(), self.log.live_ids())
        self._skip_noted = ""

    def _sync_keys(self, log_keys: dict[tuple[str, str], str], live_ids: set[str]) -> None:
        cleaned_aliases: dict[str, dict[str, str]] = {}
        keys = dict(log_keys)
        for source, mapping in self.state.aliases.items():
            if not isinstance(mapping, dict):
                continue
            kept = {k: eid for k, eid in mapping.items() if eid in live_ids}
            if kept:
                cleaned_aliases[source] = kept
                for k, eid in kept.items():
                    keys.setdefault((source, k), eid)
        self.state.aliases = cleaned_aliases
        self._keys = keys

    async def follow(self) -> None:
        while True:
            try:
                async with self.ctx.runtime_scope():
                    await self.tick(datetime.now(self.zone))
            except Exception as error:
                # 一次轻唤醒失败不终止值班；事件留在原处，下一次再判。
                logger.exception("值班轻唤醒失败")
                _ = self.ctx.report_incident("watch_tick_failed", f"{type(error).__name__}: {error}"[:500])
            await asyncio.sleep(self.config.interval_minutes * 60)

    # 事件入口：其他插件经 WATCH_EVENTS 交事件，和文件来源落在同一处。
    async def emit(self, source: str, *, title: str = "", text: str = "", url: str = "", at: datetime | None = None,
                   state: bool = False, key: str = "") -> str:
        if not source.strip() or not (title.strip() or text.strip()):
            raise ValueError("事件需要来源名，以及标题或正文之一")
        raw = {"title": title, "text": text, "url": url, "at": at.isoformat() if at else "", "key": key}
        now = datetime.now(self.zone)
        added = await self._append([(source.strip(), raw, state)], now)
        # 交过的 key、和当前状态一模一样的重复上报都不产生新事件，返回原来那条的编号
        name = source.strip()[:80]
        return added[0].id if added else self._keys.get((name, key[:300])) or self.state.current[name]

    # 事件落盘和读取游标在同一次保存里前进：中途崩溃最多重读一批，不会丢。
    async def _append(self, rows: Sequence[tuple[str, Mapping[str, object] | str, bool]], now: datetime, *,
                      offsets: dict[str, int] | None = None) -> list[Event]:
        async with self._lock:
            events: list[Event] = []
            merged_keys: dict[tuple[str, str], str] = {}
            latest = await run_file_io(lambda: self._current(now))
            places = await run_file_io(lambda: load_places(self.root))
            for source, raw, state in rows:
                event = name_places(normalize(self.state.next_id + len(events), source, raw, now, state), places)
                previous = latest.get(event.source)
                if event.key and (
                    (event.source, event.key) in self._keys
                    or (event.source, event.key) in merged_keys
                    or any(other.key == event.key and other.source == event.source for other in events)
                ):
                    continue
                # 状态没变的重复上报不算新事件，但保留期内仍需记住已接纳的 key 以防重试倒流
                if event.state and previous is not None and (previous.title, previous.text) == (event.title, event.text):
                    if event.key:
                        merged_keys[(event.source, event.key)] = previous.id
                    continue
                events.append(event)
                if event.state:
                    latest[event.source] = event
            written = False

            def append_and_mark() -> None:
                nonlocal written
                self.log.append(events, now)
                written = True

            try:
                await run_file_io(append_and_mark)
            finally:
                if written:
                    self._keys.update({(event.source, event.key): event.id for event in events if event.key})
                    for (src, k), eid in merged_keys.items():
                        self._keys[(src, k)] = eid
                        self.state.aliases.setdefault(src, {})[k] = eid
                    self.state.next_id += len(events)
                    self.state.current = {source: event.id for source, event in latest.items()}
                    if offsets is not None:
                        self.state.offsets = offsets
            await self._save()
            return events

    # 每个来源最近一条状态事件；事件过了保留期就不再算数。
    def _current(self, now: datetime) -> dict[str, Event]:
        found = {source: self.log.get(event_id, now) for source, event_id in self.state.current.items()}
        return {source: event for source, event in found.items() if event is not None}

    async def _save(self) -> None:
        data = asdict(self.state)
        await run_file_io(lambda: atomic_save_json(self._state_path, data, domain="watch"))

    async def tick(self, now: datetime) -> None:
        # 1. 收进新事件，整理嘱托文件
        await self._ingest(now)
        try:
            await self._timeline(now)
        except Exception:
            # 时间线是给后台回合查的旁路，记不上不该让这一次值班停下；游标没动，下一次接着补。
            logger.exception("[skipped] 时间线这一次没有记上，下一次轻唤醒重试")
        orders = await run_file_io(lambda: watch_orders.maintain(self.root, now.date()))
        after = datetime.fromisoformat(self.state.orders_checked) if self.state.orders_checked else now
        timed = [order for order in orders if watch_orders.due(order, after, now)]
        pending = await run_file_io(lambda: self.log.since(self.state.judged, now))
        # 2. 有新事件或到点的嘱托才动用模型
        if pending or timed:
            await self._judge(now, orders, timed, pending[: CHUNK * MAX_CHUNKS])
        else:
            self.state.orders_checked = now.isoformat()
        # 3. 待看列表定时过一眼、每日复盘与清理
        await self._digest(now)
        await self._review(now, orders)
        await self._dream(now)
        log_keys, live_ids = await run_file_io(lambda: self._clean(now))
        async with self._lock:
            self._sync_keys(log_keys, live_ids)
        await self._save()

    # 把上次以来他说的话和状态变化追加进时间线。第一次往回补到账本和事件文件够得着的地方。
    async def _timeline(self, now: datetime) -> None:
        if not self.config.target:
            return
        reader = self.ctx.require(MESSAGE_CATALOG).reader(self.config.target)
        since = datetime.fromisoformat(self.state.said_until) if self.state.said_until else now - timedelta(days=60)
        back = 1 if self.state.said_until else 14

        def work() -> tuple[list[recent.Turn], list[Event]]:
            said = [turn for turn in recent.read_turns(reader, now, (now - since).total_seconds() / 3600, 9000) if turn.kind == "user" and turn.at > since]
            known = [event for offset in range(back, -1, -1) for event in read_day(self.root, (now - timedelta(days=offset)).date()) if event.state]
            fresh = [event for event in known if event.number > self.state.timeline_event]
            timeline.append(self.root, timeline.rows(fresh, said, known, self.zone))
            return said, fresh
        said, fresh = await run_file_io(work)
        self.state.said_until = now.isoformat()
        self.state.timeline_event = max([self.state.timeline_event, *(event.number for event in fresh)])
        await self._save()

    # 事件全文与值班记录留十四天；待看列表三天没处理的条目自动清掉。
    def _clean(self, now: datetime) -> tuple[dict[tuple[str, str], str], set[str]]:
        self.log.prune(now, 14)
        _prune_dated(self.root / "inbox.md", now, 3)
        _prune_dated(self.root / "trail.md", now, trail.KEPT_DAYS)
        _prune_logs(self.root, now, 14)
        return self.log.keys(), self.log.live_ids()

    async def _ingest(self, now: datetime) -> None:
        offsets = dict(self.state.offsets)
        rows = [(source, raw, False) for source, raw in await run_file_io(lambda: read_sources(self.root, offsets))]
        for path in await run_file_io(lambda: scripts.find(self.root)):
            lines, failure = await scripts.run(self.ctx, path)
            # 脚本坏了也是一条事件，同一天只记一次，写它的主 Agent 会在待看里看到
            failed: list[Mapping[str, object] | str] = [{"title": "来源脚本运行失败", "text": failure, "key": f"failed:{now:%Y-%m-%d}"}] if failure else []
            rows += [(path.stem, raw, False) for raw in failed + [parse_line(line) for line in lines]]
        _ = await self._append(rows, now, offsets=offsets)

    async def _judge(self, now: datetime, orders: list[watch_orders.Order], timed: list[watch_orders.Order], pending: list[Event]) -> None:
        text, wake_orders = watch_orders.render(orders)
        standing, live = await self._context(now, text)
        checks = [Event(f"t{index}", "到点查看", now.isoformat(timespec="seconds"), order.text, "", "") for index, order in enumerate(timed, 1)]
        by_check = {event.id: order for event, order in zip(checks, timed)}
        batch = checks + pending
        chain: tuple[str | None, ...] = self.config.watch_models or (None,)
        fired: set[str] = set()
        wakes: list[Verdict] = []
        for start in range(0, len(batch), CHUNK):
            chunk = batch[start:start + CHUNK]
            lookup = {event.id: event for event in chunk}
            outcome = await judge(self._complete, chain, standing, live, chunk, self.files,
                                  lambda event_id: lookup.get(event_id) or self.log.get(event_id, now))
            await self._record(now, outcome)
            if outcome.model is None:
                # 整条模型链都失败：这批留到下一次轻唤醒再判
                logger.warning("[paused] 值班模型链全部失败，这批事件留到下一轮再判：失败数=%s 本批=%s 条", len(outcome.failures), len(chunk))
                _ = self.ctx.report_incident("watch_models_failed", "; ".join(outcome.failures)[:500])
                break
            for verdict in outcome.verdicts:
                if verdict.to == "wake":
                    wakes.append(verdict)
                    if verdict.order is not None and 1 <= verdict.order <= len(wake_orders):
                        fired.add(wake_orders[verdict.order - 1].raw)
                if verdict.event.id in by_check:
                    # 到点查看过的嘱托算触发过一次；一次性的随后归档，周期的保留
                    fired.add(by_check[verdict.event.id].raw)
            numbers = [event.number for event in chunk if event.id[0] == "e"]
            self.state.judged = max([self.state.judged, *numbers])
            self.state.orders_checked = now.isoformat()
            await self._save()
        if fired:
            _ = await run_file_io(lambda: watch_orders.maintain(self.root, now.date(), frozenset(fired)))
        if wakes:
            await self._wake(now, wakes)

    # 值班员每次看到的现场，分成两段各占一条消息：前一段是很少变的（长期记忆、嘱托、判例），后一段每次都变（最近对话、用户状态、时间）。
    # 有的模型只在"最后一条消息之前的内容和上次完全一样"时才命中缓存，会变的东西混进前一段，整段就白算了。
    async def _context(self, now: datetime, order_text: str) -> tuple[str, str]:
        turns = await self._turns(now)
        memory = self.files.text(self.config.memory_file) or ""
        cases = _tail(self.root / "cases.md", 20)
        standing = "\n\n".join(part for part in (
            "用户的长期记忆：\n" + memory if memory else "",
            "嘱托（主 Agent 写给你的）。\n" + order_text,
            "主 Agent 对你过去判断的纠正：\n" + cases if cases else "",
        ) if part)
        live = "\n\n".join((
            "主 Agent 和用户最近的对话（从旧到新）：\n" + recent.render(turns, now),
            recent.stats(turns, now),
            "用户当前状态：\n" + await self._user_state(now),
            f"现在：{now:%Y-%m-%d %H:%M}（周{'一二三四五六日'[now.weekday()]}）",
        ))
        return standing, live

    async def _turns(self, now: datetime) -> list[recent.Turn]:
        if not self.config.target:
            return []
        reader = self.ctx.require(MESSAGE_CATALOG).reader(self.config.target)
        return await run_file_io(lambda: recent.read_turns(reader, now))

    # 各来源当前的状态事件，给用户信息入口用。
    async def states(self) -> list[Event]:
        now = datetime.now(self.zone)
        latest = await run_file_io(lambda: self._current(now))
        return [latest[source] for source in sorted(latest)]

    # 当前状态不另外去查：就是各来源交来的最近一条状态事件。
    async def _user_state(self, now: datetime) -> str:
        latest = await run_file_io(lambda: self._current(now))
        lines = [f"- [{source}] {event.at[5:16].replace('T', ' ')} 起：{clip(' — '.join(part for part in (event.title, event.text) if part), 400)}"
                 for source, event in sorted(latest.items())]
        return "\n".join(lines) or "（还没有来源报告过状态）"

    async def _complete(self, model_id: str | None, messages: Sequence[Mapping[str, object]], tools: Sequence[Mapping[str, object]]) -> Reply:
        # 宿主遇到不可用的模型会悄悄换成默认模型；退避链要的是明确失败、换下一个。
        if model_id is not None and not any(
                model.model_id == model_id and model.kind == "chat" and model.availability == "available"
                for model in self.ctx.require(MODEL_CATALOG).snapshot().models):
            raise ModelFailure("模型未注册或当前不可用")
        try:
            async with asyncio.timeout(240), self.ctx.require(CHAT_MODELS).independent_execution(model_id=model_id) as execution:
                chat = execution.chat()
                response = await chat.complete(ModelRequest(messages=tuple(messages), tools=tuple(tools), max_attempts=2, max_retry_wait=20))
        except (RuntimeError, TimeoutError) as error:
            if ModelError.matches(error) or isinstance(error, TimeoutError):
                raise ModelFailure(str(error) or type(error).__name__) from error
            raise
        return Reply(
            response.content or "",
            tuple(Call(call.id, call.name, call.arguments) for call in response.tool_calls),
            model=chat.descriptor.model or model_id or "default",
            usage=response.usage,
        )

    # 值班记录：每条事件的去向与理由，按天存成可读的文件；待看的另记一份给主 Agent。
    async def _record(self, now: datetime, outcome: Outcome) -> None:
        def write() -> None:
            (self.root / "log").mkdir(parents=True, exist_ok=True)
            with (self.root / "log" / f"{now:%Y-%m-%d}.md").open("a", encoding="utf-8") as file:
                for failure in outcome.failures:
                    file.write(f"- {now:%H:%M} 模型失败：{failure}\n")
                for stat in outcome.usage.values():
                    file.write(f"- {now:%H:%M} 值班模型：{_format_usage(stat)}\n")
                for verdict in outcome.verdicts:
                    event = verdict.event
                    file.write(f"- {now:%H:%M} #{event.id} [{event.source}] {clip(event.title or event.text, 90)} → {verdict.to}"
                               + (f"｜{verdict.why}" if verdict.why else "") + f"（{outcome.model}）\n")
            _inbox(self.root, now, [verdict for verdict in outcome.verdicts if verdict.to == "inbox"])
            # 叫醒判定在游标推进前先落入待看列表，避免提交游标后、进入 _wake 前中断导致叫醒丢失（ADR 0149 / #2103）
            _inbox(self.root, now, [verdict for verdict in outcome.verdicts if verdict.to == "wake"], "值班员认为该叫醒，待交接")
        await run_file_io(write)

    async def _scene(self, now: datetime) -> wake.Scene:
        turns = await self._turns(now)
        channel, _, chat_id = self.config.target.partition(":")
        return wake.Scene(now, await self._user_state(now), recent.render(turns, now, 4000), recent.stats(turns, now),
                          count_items(self.root / "inbox.md"), count_notes(self.root / "notes.md"), channel, chat_id,
                          str(self.root.parent), await run_file_io(lambda: trail.render(self.root)))

    async def _wake(self, now: datetime, verdicts: list[Verdict]) -> None:
        recent_wakes = [stamp for stamp in self.state.wakes if datetime.fromisoformat(stamp) > now - timedelta(hours=24)]
        if len(recent_wakes) >= self.config.daily_wakes:
            # 今日额度用完：不丢，放进待看列表
            logger.warning("[fallback] 今日叫醒额度已用完，叫醒改写进待看列表：额度=%s 事件=%s 条", self.config.daily_wakes, len(verdicts))
            await run_file_io(lambda: _inbox(self.root, now, verdicts, "今日叫醒额度已用完；值班员认为该叫醒"))
            return
        shown, rest = verdicts[:WAKE_EVENTS], verdicts[WAKE_EVENTS:]
        # 叫醒的事件也记进待看：主 Agent 处理完自己删掉；它这一轮没跑成或中途断了，事情还留在待看里
        await run_file_io(lambda: (_inbox(self.root, now, shown, "已为这件事叫醒你，处理完请删掉这一行"), _inbox(self.root, now, rest, "同批叫醒")))
        prompt = wake.event_prompt(await self._scene(now), [(verdict.event, verdict.why) for verdict in shown], len(rest), await self._owed(now))
        self.state.wakes = [*recent_wakes, now.isoformat()]
        await self._save()
        _ = await self._run_agent(now, "wake", prompt, "叫醒主 Agent（" + "、".join("#" + verdict.event.id for verdict in shown) + "）")

    # 上次后台回合之后他说过话、现在已经停了半小时：那段对话还没人收尾。
    async def _owed(self, now: datetime) -> bool:
        said = [turn.at for turn in await self._turns(now) if turn.kind == "user"]
        last = datetime.fromisoformat(self.state.last_agent) if self.state.last_agent else None
        return bool(said) and (last is None or said[-1] > last) and now - said[-1] >= timedelta(minutes=30)

    # 待看列表和没收尾的对话都不能只等复盘：攒了新条目或他说过话，主 Agent 又很久没醒，就叫它过一眼。
    # 只因为额度用完而没跑待看过目：一天内同一原因只记一次。
    def _note_digest_skipped(self, now: datetime, fresh: int) -> None:
        reason = "待看有新条目" if fresh else "对话没收尾"
        stamp = f"{now:%Y-%m-%d}:{reason}"
        if self._skip_noted != stamp:
            self._skip_noted = stamp
            logger.info("[skipped] 待看过目没有执行：今日叫醒额度已用完（%s，额度=%s）", reason, self.config.daily_wakes)

    async def _digest(self, now: datetime) -> None:
        last = datetime.fromisoformat(self.state.last_agent) if self.state.last_agent else None
        if not self.config.digest_hours or (last is not None and now - last < timedelta(hours=self.config.digest_hours)):
            return
        fresh = await run_file_io(lambda: _fresh(self.root / "inbox.md", last))
        recent_wakes = [stamp for stamp in self.state.wakes if datetime.fromisoformat(stamp) > now - timedelta(hours=24)]
        owed = bool(fresh) or await self._owed(now)
        if len(recent_wakes) >= self.config.daily_wakes or not owed:
            if owed:
                self._note_digest_skipped(now, fresh)
            return
        self.state.wakes = [*recent_wakes, now.isoformat()]
        _ = await self._run_agent(now, "inbox", wake.digest_prompt(await self._scene(now), fresh), "待看列表过一眼" if fresh else "对话收尾")

    async def _review(self, now: datetime, orders: list[watch_orders.Order]) -> None:
        if self.config.review_at == "off" or self.state.review_day == f"{now:%Y-%m-%d}":
            return
        hour, minute = (int(part) for part in self.config.review_at.split(":"))
        if (now.hour, now.minute) < (hour, minute):
            return
        soon = [order.raw for order in watch_orders.expiring(orders, now.date())]
        status = await self._run_agent(now, "review", wake.review_prompt(await self._scene(now), soon), "每日复盘")
        self.state.review_tries = 0 if status == "complete" else self.state.review_tries + 1
        if status == "complete" or self.state.review_tries >= 3:
            # 成功才算当天复盘过；连续三次没跑成就放到明天，不反复重试
            self.state.review_day, self.state.review_tries = f"{now:%Y-%m-%d}", 0
            if status != "complete":
                logger.warning("[paused] 每日复盘连续 3 次没有成功，今天不再重试：最后状态=%s", status)
        await self._save()

    # 复盘跑完的当天再单独跑一轮整理记忆：复盘管对外的事，这一轮只把这些天看到的并成对他的认识。一天只试一次。
    async def _dream(self, now: datetime) -> None:
        today = f"{now:%Y-%m-%d}"
        if self.state.review_day != today or self.state.dream_day == today:
            return
        self.state.dream_day = today
        _ = await self._run_agent(now, "dream", wake.dream_prompt(await self._scene(now)), "整理记忆")

    # 每一轮用自己的会话名：同一次轻唤醒里可能先后跑事件叫醒、待看过目和复盘。
    async def _run_agent(self, now: datetime, kind: str, prompt: str, label: str) -> str:
        selection = await self.ctx.require(CHAT_MODELS).resolve(WAKE_MODEL, self.config.model)
        session_id = f"programmatic:watch-{datetime.now(self.zone):%Y%m%d-%H%M%S}-{kind}"
        self.state.last_agent = now.isoformat()
        await self._save()
        memory_before = self.files.text(self.config.memory_file) or ""
        try:
            status = await wake.run(self.ctx.require(PROGRAMMATIC), session_id, prompt, model_id=selection.model_id,
                                    reasoning_effort=selection.reasoning_effort, timeout=1800)
        except Exception as error:
            # 主 Agent 这一轮没跑起来：记下来，不影响后面的值班
            status = f"没有跑起来（{type(error).__name__}）"
            logger.warning("[paused] 主 Agent %s 回合没有跑起来，这一轮放弃：%s", kind, type(error).__name__)
            _ = self.ctx.report_incident("watch_agent_failed", f"{type(error).__name__}: {error}"[:500])

        def write() -> None:
            (self.root / "log").mkdir(parents=True, exist_ok=True)
            with (self.root / "log" / f"{now:%Y-%m-%d}.md").open("a", encoding="utf-8") as file:
                file.write(f"- {now:%H:%M} {label}：{status}（会话 {session_id}）\n")
                # 记忆会被之后每一轮值班读到：它这一轮改了哪几行，留一行让人能翻到
                for mark, line in _changed(memory_before, self.files.text(self.config.memory_file) or ""):
                    file.write(f"- {now:%H:%M} 记忆改动：{mark} {clip(line, 300)}\n")
        await run_file_io(write)
        await self._leave_trail(now, label, status, session_id)
        return status

    # 这一轮实际去了哪、留下了什么，记一行给下一轮：它是全新的会话，不记得这一轮。
    async def _leave_trail(self, now: datetime, label: str, status: str, session_id: str) -> None:
        try:
            reader = self.ctx.require(MESSAGE_CATALOG).reader(session_id)
            messages = await run_file_io(lambda: panel.read_tail_messages(reader))
            await run_file_io(lambda: trail.record(self.root, now, label, status, trail.places(messages), panel.read_left(messages)))
        except Exception as error:
            logger.warning("[skipped] 这一轮的足迹没有记下，下一轮看不到它：%s", type(error).__name__)


def _format_usage(stat: ModelStats) -> str:
    model = stat.model.replace("｜", "|").replace("\n", " ").strip() or "default"
    if stat.reported_calls == 0:
        return f"{model}｜{stat.calls} 次｜用量未知"
    if stat.cache_reported_calls == 0:
        return f"{model}｜{stat.calls} 次｜输入 {stat.input_tokens}｜输出 {stat.output_tokens}"
    hit = f"命中 {stat.hit_tokens}" if stat.hit_input_tokens == stat.input_tokens else f"命中 {stat.hit_tokens}/{stat.hit_input_tokens}"
    return f"{model}｜{stat.calls} 次｜输入 {stat.input_tokens}（{hit}）｜输出 {stat.output_tokens}"


# 两版文本之间增删的行；只看非空行，顺序按新增在前。
def _changed(before: str, after: str) -> list[tuple[str, str]]:
    old, new = [line.strip() for line in before.splitlines() if line.strip()], [line.strip() for line in after.splitlines() if line.strip()]
    return [("+", line) for line in new if line not in old][:20] + [("-", line) for line in old if line not in new][:20]


def _inbox(root: Path, now: datetime, verdicts: Sequence[Verdict], note: str = "") -> None:
    if not verdicts:
        return
    root.mkdir(parents=True, exist_ok=True)
    path = root / "inbox.md"
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    for verdict in verdicts:
        event = verdict.event
        reason = "；".join(part for part in (note, verdict.why) if part)
        rendered = (
            f"- [{now:%Y-%m-%d %H:%M}] #{event.id} [{event.source}] {clip(event.title or event.text, 120)}"
            + (f" {event.url}" if event.url else "")
            + (f"｜{reason}" if reason else "")
        )
        marker = f"] #{event.id} "
        replaced = False
        for idx, existing in enumerate(lines):
            if marker in existing:
                lines[idx] = rendered
                replaced = True
                break
        if not replaced:
            lines.append(rendered)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# 上次主 Agent 醒来之后新进待看列表的条数。
def _fresh(path: Path, since: datetime | None) -> int:
    if not path.exists():
        return 0
    floor = f"- [{since:%Y-%m-%d %H:%M}]" if since else ""
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.startswith("- [20") and line[:20] > floor)


def _tail(path: Path, count: int) -> str:
    if not path.exists():
        return ""
    return "\n".join([line for line in path.read_text(encoding="utf-8").splitlines() if line.startswith("- ")][-count:])


# 便签是自由格式：非空且不是标题的行都算。
def count_notes(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith("#"))


# 文件里以 "- " 开头的条目数。
def count_items(path: Path) -> int:
    return len(_tail(path, 100000).splitlines()) if path.exists() else 0


# 待看列表里带日期的行过期就删：主 Agent 几天没看的东西不再占它的注意力。
def _prune_dated(path: Path, now: datetime, days: int) -> None:
    if not path.exists():
        return
    floor = f"- [{now - timedelta(days=days):%Y-%m-%d %H:%M}]"
    lines = path.read_text(encoding="utf-8").splitlines()
    kept = [line for line in lines if not (line.startswith("- [20") and line[:20] < floor)]
    if len(kept) != len(lines):
        path.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")


def _prune_logs(root: Path, now: datetime, days: int) -> None:
    floor = f"{now - timedelta(days=days):%Y-%m-%d}"
    for path in (root / "log").glob("*.md") if (root / "log").exists() else ():
        if path.stem < floor:
            path.unlink()

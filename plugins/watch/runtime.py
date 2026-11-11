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
from plugins.eventmail.contract import EVENTMAIL_STREAM
from plugins.ledger.contract import MESSAGE_CATALOG
from plugins.models.contract import CHAT_MODELS, MODEL_CATALOG, ModelError, ModelRequest
from plugins.programmatic.contract import PROGRAMMATIC

from . import orders as watch_orders
from . import recent, wake
from .events import Event, EventLog, clip, normalize, read_sources
from .settings import WAKE_MODEL, Config
from .watcher import Call, Files, ModelFailure, Outcome, Reply, Verdict, judge

logger = logging.getLogger(__name__)
CHUNK, MAX_CHUNKS, WAKE_EVENTS = 40, 5, 6


@dataclass(slots=True)
class State:
    next_id: int = 1
    judged: int = 0
    mail_seq: int | None = None
    offsets: dict[str, int] = field(default_factory=dict)
    # 来源名 → 它最近一条状态事件的编号
    current: dict[str, str] = field(default_factory=dict)
    wakes: list[str] = field(default_factory=list)
    orders_checked: str | None = None
    review_day: str | None = None
    last_agent: str | None = None
    review_tries: int = 0


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
                   state: bool = False) -> str:
        if not source.strip() or not (title.strip() or text.strip()):
            raise ValueError("事件需要来源名，以及标题或正文之一")
        raw = {"title": title, "text": text, "url": url, "at": at.isoformat() if at else ""}
        now = datetime.now(self.zone)
        added = await self._append([(source.strip(), raw, state)], now)
        # 和当前状态一模一样的重复上报不产生新事件，返回原来那条的编号
        return added[0].id if added else self.state.current[source.strip()[:80]]

    # 事件落盘和读取游标在同一次保存里前进：中途崩溃最多重读一批，不会丢。
    async def _append(self, rows: Sequence[tuple[str, Mapping[str, object] | str, bool]], now: datetime, *,
                      offsets: dict[str, int] | None = None, mail_seq: int | None = None) -> list[Event]:
        async with self._lock:
            events: list[Event] = []
            latest = await run_file_io(lambda: self._current(now))
            for source, raw, state in rows:
                event = normalize(self.state.next_id + len(events), source, raw, now, state)
                previous = latest.get(event.source)
                # 状态没变的重复上报不算新事件
                if event.state and previous is not None and (previous.title, previous.text) == (event.title, event.text):
                    continue
                events.append(event)
                if event.state:
                    latest[event.source] = event
            await run_file_io(lambda: self.log.append(events, now))
            self.state.next_id += len(events)
            self.state.current = {source: event.id for source, event in latest.items()}
            if offsets is not None:
                self.state.offsets = offsets
            if mail_seq is not None:
                self.state.mail_seq = mail_seq
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
        orders = await run_file_io(lambda: watch_orders.maintain(self.root, now.date()))
        after = datetime.fromisoformat(self.state.orders_checked) if self.state.orders_checked else now
        timed = [order for order in orders if order.section == watch_orders.ORDERS and watch_orders.due(order, after, now)]
        pending = await run_file_io(lambda: self.log.since(self.state.judged, now))
        # 2. 有新事件或到点的嘱托才动用模型
        if pending or timed:
            await self._judge(now, orders, timed, pending[: CHUNK * MAX_CHUNKS])
        else:
            self.state.orders_checked = now.isoformat()
        # 3. 待看列表定时过一眼、每日复盘与清理
        await self._digest(now)
        await self._review(now, orders)
        await run_file_io(lambda: self._clean(now))
        await self._save()

    # 事件全文与值班记录留十四天；待看列表三天没处理的条目自动清掉。
    def _clean(self, now: datetime) -> None:
        self.log.prune(now, 14)
        _prune_dated(self.root / "inbox.md", now, 3)
        _prune_logs(self.root, now, 14)

    async def _ingest(self, now: datetime) -> None:
        offsets = dict(self.state.offsets)
        rows = [(source, raw, False) for source, raw in await run_file_io(lambda: read_sources(self.root, offsets))]
        mail_seq = self.state.mail_seq
        with self.ctx.borrow(EVENTMAIL_STREAM) as stream:
            if stream is not None:
                if mail_seq is None:
                    # 第一次接上事件流时从当前位置开始，不回放历史
                    mail_seq = await stream.watermark()
                for _ in range(5):
                    envelopes = await stream.after(mail_seq, 200)
                    if not envelopes:
                        break
                    for envelope in envelopes:
                        expires = envelope.get("expires_at")
                        if isinstance(expires, str) and datetime.fromisoformat(expires) <= now:
                            # 来源声明的有效期已过：不再当成新事交给值班员
                            continue
                        payload = dict(envelope["payload"]) if isinstance(envelope["payload"], Mapping) else {}
                        payload["at"] = envelope["observed_at"]
                        rows.append((str(payload.get("source_name") or envelope["source_id"]), payload, False))
                    mail_seq = int(str(envelopes[-1]["seq"]))
        _ = await self._append(rows, now, offsets=offsets, mail_seq=mail_seq)

    async def _judge(self, now: datetime, orders: list[watch_orders.Order], timed: list[watch_orders.Order], pending: list[Event]) -> None:
        text, wake_orders = watch_orders.render(orders)
        context = await self._context(now, text)
        checks = [Event(f"t{index}", "到点查看", now.isoformat(timespec="seconds"), order.text, "", "") for index, order in enumerate(timed, 1)]
        by_check = {event.id: order for event, order in zip(checks, timed)}
        batch = checks + pending
        chain: tuple[str | None, ...] = self.config.watch_models or (None,)
        fired: set[str] = set()
        wakes: list[Verdict] = []
        for start in range(0, len(batch), CHUNK):
            chunk = batch[start:start + CHUNK]
            lookup = {event.id: event for event in chunk}
            outcome = await judge(self._complete, chain, context, chunk, self.files,
                                  lambda event_id: lookup.get(event_id) or self.log.get(event_id, now))
            await self._record(now, outcome)
            if outcome.model is None:
                # 整条模型链都失败：这批留到下一次轻唤醒再判
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

    # 值班员每次看到的现场：时间、用户状态、嘱托、判例、最近对话和长期记忆。
    async def _context(self, now: datetime, order_text: str) -> str:
        turns = await self._turns(now)
        memory = self.files.text(self.config.memory_file) or ""
        cases = _tail(self.root / "cases.md", 20)
        return "\n\n".join(part for part in (
            f"现在：{now:%Y-%m-%d %H:%M}（周{'一二三四五六日'[now.weekday()]}）",
            "用户当前状态：\n" + await self._user_state(now),
            "嘱托（主 Agent 写给你的）。\n" + order_text,
            "主 Agent 对你过去判断的纠正：\n" + cases if cases else "",
            "主 Agent 和用户最近的对话（从旧到新）：\n" + recent.render(turns, now),
            recent.stats(turns, now),
            "用户的长期记忆：\n" + memory if memory else "",
        ) if part)

    async def _turns(self, now: datetime) -> list[recent.Turn]:
        if not self.config.target:
            return []
        reader = self.ctx.require(MESSAGE_CATALOG).reader(self.config.target)
        return await run_file_io(lambda: recent.read_turns(reader, now))

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
                response = await execution.chat().complete(ModelRequest(messages=tuple(messages), tools=tuple(tools), max_attempts=2, max_retry_wait=20))
        except (RuntimeError, TimeoutError) as error:
            if ModelError.matches(error) or isinstance(error, TimeoutError):
                raise ModelFailure(str(error) or type(error).__name__) from error
            raise
        return Reply(response.content or "", tuple(Call(call.id, call.name, call.arguments) for call in response.tool_calls))

    # 值班记录：每条事件的去向与理由，按天存成可读的文件；待看的另记一份给主 Agent。
    async def _record(self, now: datetime, outcome: Outcome) -> None:
        def write() -> None:
            (self.root / "log").mkdir(parents=True, exist_ok=True)
            with (self.root / "log" / f"{now:%Y-%m-%d}.md").open("a", encoding="utf-8") as file:
                for failure in outcome.failures:
                    file.write(f"- {now:%H:%M} 模型失败：{failure}\n")
                for verdict in outcome.verdicts:
                    event = verdict.event
                    file.write(f"- {now:%H:%M} #{event.id} [{event.source}] {clip(event.title or event.text, 90)} → {verdict.to}"
                               + (f"｜{verdict.why}" if verdict.why else "") + f"（{outcome.model}）\n")
            _inbox(self.root, now, [verdict for verdict in outcome.verdicts if verdict.to == "inbox"])
        await run_file_io(write)

    async def _scene(self, now: datetime) -> wake.Scene:
        turns = await self._turns(now)
        channel, _, chat_id = self.config.target.partition(":")
        return wake.Scene(now, await self._user_state(now), recent.render(turns, now, 4000), recent.stats(turns, now),
                          count_items(self.root / "inbox.md"), count_notes(self.root / "notes.md"), channel, chat_id)

    async def _wake(self, now: datetime, verdicts: list[Verdict]) -> None:
        recent_wakes = [stamp for stamp in self.state.wakes if datetime.fromisoformat(stamp) > now - timedelta(hours=24)]
        if len(recent_wakes) >= self.config.daily_wakes:
            # 今日额度用完：不丢，放进待看列表
            await run_file_io(lambda: _inbox(self.root, now, verdicts, "今日叫醒额度已用完；值班员认为该叫醒"))
            return
        shown, rest = verdicts[:WAKE_EVENTS], verdicts[WAKE_EVENTS:]
        # 叫醒的事件也记进待看：主 Agent 处理完自己删掉；它这一轮没跑成或中途断了，事情还留在待看里
        await run_file_io(lambda: (_inbox(self.root, now, shown, "已为这件事叫醒你，处理完请删掉这一行"), _inbox(self.root, now, rest, "同批叫醒")))
        prompt = wake.event_prompt(await self._scene(now), [(verdict.event, verdict.why) for verdict in shown], len(rest))
        self.state.wakes = [*recent_wakes, now.isoformat()]
        await self._save()
        _ = await self._run_agent(now, "wake", prompt, "叫醒主 Agent（" + "、".join("#" + verdict.event.id for verdict in shown) + "）")

    # 待看列表不能只等复盘：攒了新条目、主 Agent 又很久没醒，就叫它过一眼。
    async def _digest(self, now: datetime) -> None:
        last = datetime.fromisoformat(self.state.last_agent) if self.state.last_agent else None
        if not self.config.digest_hours or (last is not None and now - last < timedelta(hours=self.config.digest_hours)):
            return
        fresh = await run_file_io(lambda: _fresh(self.root / "inbox.md", last))
        recent_wakes = [stamp for stamp in self.state.wakes if datetime.fromisoformat(stamp) > now - timedelta(hours=24)]
        if not fresh or len(recent_wakes) >= self.config.daily_wakes:
            return
        self.state.wakes = [*recent_wakes, now.isoformat()]
        _ = await self._run_agent(now, "inbox", wake.digest_prompt(await self._scene(now), fresh), "待看列表过一眼")

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
        await self._save()

    # 每一轮用自己的会话名：同一次轻唤醒里可能先后跑事件叫醒、待看过目和复盘。
    async def _run_agent(self, now: datetime, kind: str, prompt: str, label: str) -> str:
        selection = await self.ctx.require(CHAT_MODELS).resolve(WAKE_MODEL, self.config.model)
        session_id = f"programmatic:watch-{datetime.now(self.zone):%Y%m%d-%H%M%S}-{kind}"
        self.state.last_agent = now.isoformat()
        await self._save()
        try:
            status = await wake.run(self.ctx.require(PROGRAMMATIC), session_id, prompt, model_id=selection.model_id,
                                    reasoning_effort=selection.reasoning_effort, timeout=1800)
        except Exception as error:
            # 主 Agent 这一轮没跑起来：记下来，不影响后面的值班
            status = f"没有跑起来（{type(error).__name__}）"
            _ = self.ctx.report_incident("watch_agent_failed", f"{type(error).__name__}: {error}"[:500])

        def write() -> None:
            (self.root / "log").mkdir(parents=True, exist_ok=True)
            with (self.root / "log" / f"{now:%Y-%m-%d}.md").open("a", encoding="utf-8") as file:
                file.write(f"- {now:%H:%M} {label}：{status}（会话 {session_id}）\n")
        await run_file_io(write)
        return status


def _inbox(root: Path, now: datetime, verdicts: Sequence[Verdict], note: str = "") -> None:
    if not verdicts:
        return
    root.mkdir(parents=True, exist_ok=True)
    with (root / "inbox.md").open("a", encoding="utf-8") as file:
        for verdict in verdicts:
            event = verdict.event
            reason = "；".join(part for part in (note, verdict.why) if part)
            file.write(f"- [{now:%Y-%m-%d %H:%M}] #{event.id} [{event.source}] {clip(event.title or event.text, 120)}"
                       + (f" {event.url}" if event.url else "") + (f"｜{reason}" if reason else "") + "\n")


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

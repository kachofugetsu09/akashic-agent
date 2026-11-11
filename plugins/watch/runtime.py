"""值班主循环（第一步）：把三种入口的事件收成同一种形状，按天落盘。"""
from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime
from zoneinfo import ZoneInfo

from agent.plugin_composition import Context
from core.common.file_io import run_file_io
from infra.persistence.json_store import atomic_save_json, load_json
from plugins.eventmail.contract import EVENTMAIL_STREAM

from .events import Event, EventLog, normalize, read_sources
from .settings import Config

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class State:
    next_id: int = 1
    mail_seq: int | None = None
    offsets: dict[str, int] = field(default_factory=dict)
    current: dict[str, str] = field(default_factory=dict)


class Runtime:
    def __init__(self, ctx: Context, config: Config) -> None:
        self.ctx = ctx
        self.config = config
        self.root = ctx.workspace_root("watch")
        self.zone = ZoneInfo(config.timezone)
        self.log = EventLog(self.root)
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
                # 一次失败不终止循环；游标没有前进，下一次重读。
                logger.exception("值班收事件失败")
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
        await self._ingest(now)
        # 事件全文留十四天
        await run_file_io(lambda: self.log.prune(now, 14))

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
                        state = envelope["kind"] == "context"
                        expires = envelope.get("expires_at")
                        if not state and isinstance(expires, str) and datetime.fromisoformat(expires) <= now:
                            # 来源声明的有效期已过：不再当成新事交给值班员。状态的变化发生过就是发生过，照收
                            continue
                        payload = dict(envelope["payload"]) if isinstance(envelope["payload"], Mapping) else {}
                        payload["at"] = envelope["observed_at"]
                        rows.append((str(payload.get("source_name") or envelope["source_id"]), payload, state))
                    mail_seq = int(str(envelopes[-1]["seq"]))
        _ = await self._append(rows, now, offsets=offsets, mail_seq=mail_seq)

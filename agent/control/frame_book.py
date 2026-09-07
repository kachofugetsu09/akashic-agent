from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol, cast

from agent.plugin_composition import ServiceKey
from session.message import CallRef


class FrameResolver(Protocol):
    """Resolve the one completed Output belonging to one Input."""

    def __call__(self) -> str | None: ...


class FrameRouteReleased(RuntimeError):
    """The short-lived route no longer accepts a delivery wait."""


@dataclass(frozen=True, slots=True)
class _RouteKey:
    session_id: str
    input_id: str


class FrameReservation:
    """Wait on one active Input route without owning the route itself."""

    def __init__(self, book: FrameBook, route: _Route) -> None:
        self._book = book
        self._route = route

    async def wait_output(self, message_id: str) -> None:
        await self._book.wait_output(self._route, message_id)


class FrameClaim:
    """Keep one exact ending drain alive until its caller consumes or aborts."""

    def __init__(self, book: FrameBook, route: _Route, call_ref: CallRef) -> None:
        self._book = book
        self._route = route
        self.call_ref = call_ref
        self.ending_message_id: str | None = None
        self._closed = False

    async def wait_output(self) -> None:
        if self._closed:
            raise FrameRouteReleased("frame claim 已结束")
        await self._book.wait_claim(self)

    def consume(self) -> None:
        if not self._closed:
            self._closed = True
            self._book._release_claim(self)  # pyright: ignore[reportPrivateUsage]

    def abort(self) -> None:
        if not self._closed:
            self._closed = True
            self._book._release_claim(self)  # pyright: ignore[reportPrivateUsage]


@dataclass(slots=True)
class _Route:
    key: _RouteKey
    connection_id: str
    resolver: FrameResolver
    reservation: FrameReservation = field(init=False)
    expected: str | None = None
    resolved: str | None = None
    written: asyncio.Future[None] | None = None
    drained: bool = False
    error: BaseException | None = None
    waiters: set[asyncio.Future[None]] = field(default_factory=set)
    claims: set[FrameClaim] = field(default_factory=set)
    staged: bool = False


@dataclass(eq=False, slots=True)
class FrameRouteStage:
    """A replacement route that becomes visible only after its operation succeeds."""

    _book: FrameBook
    _route: _Route
    _closed: bool = False

    @property
    def reservation(self) -> FrameReservation:
        return self._route.reservation

    def commit(self) -> FrameReservation:
        if self._closed:
            raise FrameRouteReleased("frame route stage 已结束")
        self._closed = True
        return self._book._commit_stage(self)  # pyright: ignore[reportPrivateUsage]

    def abort(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._book._abort_stage(self)  # pyright: ignore[reportPrivateUsage]


class FrameBook:
    """Own active control routes and one exact writer drain per route."""

    def __init__(self) -> None:
        self._routes: dict[_RouteKey, _Route] = {}
        self._stages: set[FrameRouteStage] = set()
        self._claims: dict[tuple[str, CallRef], FrameClaim] = {}
        self._closed: BaseException | None = None

    def route_input(
        self,
        session_id: str,
        input_id: str,
        connection_id: str,
        resolver: FrameResolver,
    ) -> FrameReservation:
        """Keep the first connection owner; a duplicate send cannot steal it."""
        self._check_open()
        key = _RouteKey(session_id, input_id)
        existing = self._routes.get(key)
        if existing is not None:
            return existing.reservation
        route = _Route(
            key=key,
            connection_id=connection_id,
            resolver=resolver,
        )
        route.reservation = FrameReservation(self, route)
        self._routes[key] = route
        return route.reservation

    def route_input_with_owner(
        self,
        session_id: str,
        input_id: str,
        connection_id: str,
        resolver: FrameResolver,
    ) -> tuple[FrameReservation, bool]:
        """Route one Input and report whether this call created its owner."""
        self._check_open()
        key = _RouteKey(session_id, input_id)
        existing = self._routes.get(key)
        if existing is not None:
            return existing.reservation, False
        return self.route_input(session_id, input_id, connection_id, resolver), True

    def stage_input(
        self,
        session_id: str,
        input_id: str,
        connection_id: str,
        resolver: FrameResolver,
    ) -> FrameRouteStage:
        """Prepare a resume owner without changing the active route."""
        self._check_open()
        key = _RouteKey(session_id, input_id)
        route = _Route(
            key=key,
            connection_id=connection_id,
            resolver=resolver,
            staged=True,
        )
        route.reservation = FrameReservation(self, route)
        stage = FrameRouteStage(self, route)
        self._stages.add(stage)
        return stage

    def resolve_page(
        self, connection_id: str, page: Mapping[str, object]
    ) -> tuple[_Route, ...]:
        """Resolve exact endings synchronously before a writer future is made."""
        rows = page.get("items")
        if not isinstance(rows, list):
            raise TypeError("message page items 必须是列表")
        checked_rows: list[Mapping[str, object]] = []
        for row in rows:
            if not isinstance(row, Mapping):
                raise TypeError("message page item 必须是对象")
            checked_rows.append(cast(Mapping[str, object], row))
        complete: set[tuple[str, str]] = {
            (cast(str, row["session_id"]), cast(str, row["id"]))
            for row in checked_rows
            if _complete_output(row)
        }
        tracked: list[_Route] = []
        routes = tuple(self._routes.values()) + tuple(stage._route for stage in self._stages)
        for route in routes:
            if route.connection_id != connection_id:
                continue
            ending = route.resolver()
            key = route.key
            if ending is None or (key.session_id, ending) not in complete:
                continue
            if route.expected is not None and route.expected != ending:
                raise ValueError("一次 Input 不能绑定多个最终 Output")
            route.expected = ending
            route.resolved = ending
            for claim in route.claims:
                claim.ending_message_id = ending
            tracked.append(route)
        return tuple(tracked)

    def attach_page(
        self, tracked: tuple[_Route, ...], written: asyncio.Future[None]
    ) -> None:
        """Attach one real writer drain to the routes selected by resolve_page."""
        for route in tracked:
            if route.written is not None or route.drained:
                continue
            route.written = written
            written.add_done_callback(
                lambda future, exact_route=route: self._frame_done(exact_route, future)
            )

    async def wait_output(
        self,
        route: _Route,
        message_id: str,
    ) -> None:
        if route.expected is not None and route.expected != message_id:
            raise ValueError("一次 Input 不能绑定多个最终 Output")
        route.expected = message_id
        await self._wait_route(route)

    async def wait_claim(self, claim: FrameClaim) -> None:
        """等待同一 CallRef 解析出的最终 Output drain，保留 claim 直到消费。"""
        route = claim._route
        while claim.ending_message_id is None or not route.drained:
            await self._wait_route(route)
            if route.error is not None:
                raise route.error
            if claim.ending_message_id is None and route.drained:
                raise FrameRouteReleased("frame claim 未解析到最终 Output")
        if route.error is not None:
            raise route.error

    async def _wait_route(self, route: _Route) -> None:
        if route.error is not None:
            raise route.error
        if route.drained:
            return
        loop = asyncio.get_running_loop()
        waiter = loop.create_future()
        route.waiters.add(waiter)
        try:
            await asyncio.shield(waiter)
        finally:
            route.waiters.discard(waiter)
        if route.error is not None:
            raise route.error
        self._maybe_release(route)

    async def wait_input(self, session_id: str, input_id: str, ending: str) -> None:
        """Wait on one route without exposing its internal key type."""
        route = self._routes.get(_RouteKey(session_id, input_id))
        if route is None:
            raise FrameRouteReleased("frame route 已释放")
        await self.wait_output(route, ending)

    def release_input(self, session_id: str, input_id: str) -> None:
        """Release an unclaimed route after a normal non-streaming completion."""
        key = _RouteKey(session_id, input_id)
        route = self._routes.get(key)
        if route is not None and not route.claims:
            self._release_route(key)

    def active_input_ids(self, session_id: str) -> tuple[str, ...]:
        """Return active Input identities for one Session, including staged routes."""
        return tuple(
            route.key.input_id
            for route in (*self._routes.values(), *(stage._route for stage in self._stages))
            if route.key.session_id == session_id
        )

    def settle_input(
        self, session_id: str, input_id: str, error: BaseException | None = None,
    ) -> None:
        """结束来源已确认的普通 route；explicit claim 继续保留或接收失败。"""
        key = _RouteKey(session_id, input_id)
        route = self._routes.get(key)
        if route is None:
            return
        if route.claims:
            if error is not None:
                self._fail_route(key, error, exact_route=route)
            return
        self._release_route(key)

    def arm_claim(self, session_id: str, input_id: str, call_ref: CallRef) -> FrameClaim:
        """登记一个 ToolCall claim；最终 Output 由同一 resolver 绑定。"""
        route = self._routes.get(_RouteKey(session_id, input_id))
        if route is None:
            raise FrameRouteReleased("frame route 已释放")
        key = (session_id, call_ref)
        existing = self._claims.get(key)
        if existing is not None and not existing._closed:  # pyright: ignore[reportPrivateUsage]
            return existing
        claim = FrameClaim(self, route, call_ref)
        claim.ending_message_id = route.resolved
        route.claims.add(claim)
        self._claims[key] = claim
        return claim

    def claim_for(self, session_id: str, call_ref: CallRef) -> FrameClaim | None:
        """Return the live claim for one exact ToolCall, if it was pre-armed."""
        claim = self._claims.get((session_id, call_ref))
        return claim if claim is not None and not claim._closed else None  # pyright: ignore[reportPrivateUsage]

    def fail_connection(self, connection_id: str, error: BaseException) -> None:
        """Fail and release every route owned by a disconnected connection."""
        for key, route in tuple(self._routes.items()):
            if route.connection_id == connection_id:
                self._fail_route(key, error)
        for stage in tuple(self._stages):
            if stage._route.connection_id == connection_id:
                stage.abort()

    def close(self, error: BaseException | None = None) -> None:
        self._closed = error or ConnectionError("control frame book closed")
        for key in tuple(self._routes):
            self._fail_route(key, self._closed)
        for stage in tuple(self._stages):
            stage.abort()

    def _frame_done(self, route: _Route, future: asyncio.Future[None]) -> None:
        try:
            future.result()
        except BaseException as error:
            self._fail_route(route.key, error, exact_route=route)
            return
        route.written = None
        route.drained = True
        for waiter in route.waiters:
            if not waiter.done():
                waiter.set_result(None)
        self._maybe_release(route)

    def _release_claim(self, claim: FrameClaim) -> None:
        route = claim._route
        route.claims.discard(claim)
        key = (route.key.session_id, claim.call_ref)
        if self._claims.get(key) is claim:
            _ = self._claims.pop(key)
        self._maybe_release(route)

    def _commit_stage(self, stage: FrameRouteStage) -> FrameReservation:
        self._stages.discard(stage)
        old = self._routes.get(stage._route.key)  # pyright: ignore[reportPrivateUsage]
        if old is not None:
            self._fail_route(old.key, FrameRouteReleased("resume replaced old route"), exact_route=old)
        route = stage._route  # pyright: ignore[reportPrivateUsage]
        route.staged = False
        self._routes[route.key] = route
        self._maybe_release(route)
        return route.reservation

    def _abort_stage(self, stage: FrameRouteStage) -> None:
        self._stages.discard(stage)
        self._fail_route(stage._route.key, FrameRouteReleased("resume route aborted"),
                        exact_route=stage._route)

    def _release_route(self, key: _RouteKey, *, exact_route: _Route | None = None) -> None:
        route = self._routes.get(key)
        if route is None or (exact_route is not None and route is not exact_route):
            return
        self._routes.pop(key)
        if route.error is None and not route.drained:
            route.error = FrameRouteReleased("frame route 已结束但没有最终 Output")
        for waiter in route.waiters:
            if not waiter.done():
                waiter.set_exception(route.error or FrameRouteReleased("frame route 已释放"))
        route.waiters.clear()

    def _fail_route(
        self, key: _RouteKey, error: BaseException, *, exact_route: _Route | None = None,
    ) -> None:
        route = exact_route or self._routes.get(key)
        if route is None:
            return
        if self._routes.get(key) is route:
            self._routes.pop(key)
        self._stages = {stage for stage in self._stages if stage._route is not route}
        for claim in route.claims:
            claim_key = (route.key.session_id, claim.call_ref)
            if self._claims.get(claim_key) is claim:
                self._claims.pop(claim_key)
        route.error = error
        for waiter in route.waiters:
            if not waiter.done():
                waiter.set_exception(error)
        route.waiters.clear()

    def _maybe_release(self, route: _Route) -> None:
        if route.staged or not route.drained or route.claims:
            return
        self._release_route(route.key, exact_route=route)

    def _check_open(self) -> None:
        if self._closed is not None:
            raise self._closed


def _complete_output(row: Mapping[str, object]) -> bool:
    body = row.get("body")
    return (
        isinstance(row.get("id"), str)
        and isinstance(row.get("session_id"), str)
        and isinstance(body, Mapping)
        and cast(Mapping[str, object], body).get("kind") == "output"
        and cast(Mapping[str, object], body).get("finish") == "complete"
        and isinstance(cast(Mapping[str, object], body).get("parts"), list)
    )


CONTROL_FRAMES = ServiceKey[FrameBook]("core.control_frames.v1")

"""
FitbitSleepProvider — 从本地 fitbit-monitor 服务轮询睡眠状态。

后台线程每 poll_interval 秒拉取一次 /api/data，结果缓存在内存中。
缓存超过 30 分钟或服务不可达时降级返回 unavailable，对 proactive 行为无影响。
"""

from __future__ import annotations

import asyncio
from contextlib import suppress
import logging
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class SleepContext:
    state: str  # sleeping | awake | uncertain | unknown
    prob: float | None  # 0-1，None=无数据
    prob_source: str  # ml | heuristic | unavailable
    data_lag_min: int | None
    fetched_at: float  # time.time()
    available: bool  # False=服务不可达或数据过期
    sleeping_modifier: float = 0.15
    health_events: list[dict[str, Any]] = field(default_factory=list)

    @property
    def sleep_modifier(self) -> float:
        """
        用于乘以 interrupt_factor 的修正系数。
        不设为 0，保留 chat 的概率可能性。
        """
        if not self.available:
            return 1.0  # 降级：不影响现有行为
        if self.state == "sleeping":
            return self.sleeping_modifier
        if self.state == "uncertain":
            # 睡眠高概率的 uncertain 也按睡眠保护处理，减少夜间/午睡打扰。
            if (
                self.prob is not None
                and self.prob >= 0.60
                and (self.data_lag_min is None or self.data_lag_min <= 15)
            ):
                return 0.20
            return 0.50  # 普通 uncertain：chat 概率降约 50%
        if self.state == "awake":
            return 1.0
        return 0.88  # unknown：轻微保守

    @property
    def has_urgent_health_event(self) -> bool:
        """是否存在高优先级健康事件（需要主动触达）。"""
        return any(
            str((e or {}).get("severity", "")).lower() == "high"
            for e in self.health_events
        )


_FALLBACK = SleepContext(
    state="unknown",
    prob=None,
    prob_source="unavailable",
    data_lag_min=None,
    sleeping_modifier=0.15,
    health_events=[],
    fetched_at=0.0,
    available=False,
)


class FitbitSleepProvider:
    """
    后台线程轮询 fitbit-monitor 本地服务，缓存最新睡眠状态。

    用法：
        provider = FitbitSleepProvider()
        ctx = provider.get()   # 随时取最新缓存，非阻塞
    """

    STALE_SECONDS = 1800  # 缓存超过 30 分钟视为过期
    FAIL_RETRY_SECONDS = 10  # 首次启动/短暂不可达时快速重试，避免 300s 空窗
    STARTUP_GRACE_SECONDS = (
        4  # 与 run_fitbit_monitor 并发启动时，给 monitor 预留启动时间
    )

    def __init__(
        self,
        url: str = "http://127.0.0.1:18765",
        poll_interval: int = 300,
        timeout: int = 10,
        sleeping_modifier: float = 0.15,
    ) -> None:
        self._url = url.rstrip("/")
        self._poll_interval = poll_interval
        self._timeout = timeout
        self._sleeping_modifier = max(0.0, min(float(sleeping_modifier), 1.0))
        self._cached: SleepContext = _FALLBACK
        self._lock = threading.Lock()
        self._consecutive_failures = 0
        t = threading.Thread(target=self._loop, daemon=True, name="fitbit-sleep-poller")
        t.start()

    def get(self) -> SleepContext:
        """返回最新缓存的睡眠上下文，非阻塞。缓存过期时降级。"""
        with self._lock:
            ctx = self._cached
        if not ctx.available:
            return ctx
        if time.time() - ctx.fetched_at > self.STALE_SECONDS:
            logger.debug("[fitbit_sleep] 缓存过期（>30min），降级为 unavailable")
            return _FALLBACK
        return ctx

    def _loop(self) -> None:
        retry_sec = max(1, min(self.FAIL_RETRY_SECONDS, int(self._poll_interval)))
        if self.STARTUP_GRACE_SECONDS > 0:
            logger.info(
                "[fitbit_sleep] 启动缓冲 %ds，等待 fitbit-monitor 就绪",
                self.STARTUP_GRACE_SECONDS,
            )
            time.sleep(self.STARTUP_GRACE_SECONDS)
        while True:
            sleep_sec = self._poll_interval
            try:
                self._fetch_once()
                if self._consecutive_failures > 0:
                    logger.info(
                        "[fitbit_sleep] 轮询恢复（连续失败 %d 次后成功）",
                        self._consecutive_failures,
                    )
                self._consecutive_failures = 0
            except Exception as e:
                self._consecutive_failures += 1
                if (
                    self._consecutive_failures == 1
                    or self._consecutive_failures % 12 == 0
                ):
                    logger.warning(
                        "[fitbit_sleep] 轮询失败（连续 %d 次，%ds 后重试）: %s",
                        self._consecutive_failures,
                        retry_sec,
                        e,
                    )
                sleep_sec = retry_sec
            time.sleep(sleep_sec)

    def refresh_now(self, timeout: float = 2.5) -> bool:
        """主动刷新一次本地 /api/agent 缓存，供 proactive 决策前调用。"""
        try:
            self._fetch_once(timeout=timeout)
            return True
        except Exception as e:
            logger.debug("[fitbit_sleep] 主动刷新失败: %s", e)
            return False

    def acknowledge_events(self, event_ids: list[str]) -> None:
        """通知 fitbit-monitor 服务，事件已被 LLM 处理（fire-and-forget）。"""
        import requests

        for eid in event_ids:
            try:
                r = requests.post(f"{self._url}/api/agent/acknowledge/{eid}", timeout=3)
                r.raise_for_status()
                logger.debug("[fitbit_sleep] acknowledged event_id=%s", eid)
            except Exception as e:
                logger.warning(
                    "[fitbit_sleep] acknowledge 失败 event_id=%s: %s", eid, e
                )

    def _fetch_once(self, timeout: float | None = None) -> None:
        import requests

        req_timeout = float(timeout) if timeout is not None else float(self._timeout)
        r = requests.get(f"{self._url}/api/agent", timeout=req_timeout)
        r.raise_for_status()
        d = r.json()

        sleep = d.get("sleep", {}) or {}
        health_events = d.get("health_events") or []

        ctx = SleepContext(
            state=sleep.get("state", "unknown"),
            prob=sleep.get("prob"),
            prob_source=sleep.get("prob_source", "unavailable"),
            data_lag_min=sleep.get("data_lag_min"),
            sleeping_modifier=self._sleeping_modifier,
            health_events=health_events if isinstance(health_events, list) else [],
            fetched_at=time.time(),
            available=True,
        )
        with self._lock:
            self._cached = ctx
        logger.debug(
            "[fitbit_sleep] 已更新 state=%s prob=%s source=%s lag=%s health_events=%d",
            ctx.state,
            ctx.prob,
            ctx.prob_source,
            ctx.data_lag_min,
            len(ctx.health_events),
        )


def _monitor_wait_ready_and_refresh(base_url: str, timeout_sec: int = 3) -> None:
    import requests

    url = base_url.rstrip("/")
    requests.get(f"{url}/api/data", timeout=timeout_sec).raise_for_status()
    try:
        requests.get(f"{url}/api/refresh", timeout=timeout_sec).raise_for_status()
    except Exception:
        # refresh 失败不影响服务可用性
        pass


async def _bootstrap_monitor_data(base_url: str, max_wait_sec: int = 30) -> None:
    deadline = time.monotonic() + max_wait_sec
    last_err: Exception | None = None
    while time.monotonic() < deadline:
        try:
            await asyncio.to_thread(_monitor_wait_ready_and_refresh, base_url, 3)
            logger.info("[fitbit_monitor] 服务就绪并已触发首次 refresh")
            return
        except Exception as e:
            last_err = e
            await asyncio.sleep(1)
    if last_err is not None:
        logger.warning(
            "[fitbit_monitor] 启动后 %ds 内未就绪，继续等待下轮轮询: %s",
            max_wait_sec,
            last_err,
        )


async def run_fitbit_monitor(
    monitor_dir: str | Path,
    base_url: str = "http://127.0.0.1:18765",
) -> None:
    """
    将 fitbit-monitor/server.py 作为子进程启动，并在崩溃时自动重启。
    挂入 asyncio.gather，随 agent 生命周期一起结束。
    """
    server_py = Path(monitor_dir).expanduser().resolve() / "server.py"
    if not server_py.exists():
        logger.error("[fitbit_monitor] server.py 不存在: %s", server_py)
        return

    runtime_log = server_py.parent / "monitor.runtime.log"
    logger.info("[fitbit_monitor] 启动 %s（日志: %s）", server_py, runtime_log)
    while True:
        with runtime_log.open("ab") as log_f:
            env = dict(os.environ)
            env["PYTHONUNBUFFERED"] = "1"
            proc = await asyncio.create_subprocess_exec(
                sys.executable,
                str(server_py),
                cwd=str(server_py.parent),
                stdout=log_f,
                stderr=log_f,
                env=env,
            )
            bootstrap_task = asyncio.create_task(
                _bootstrap_monitor_data(base_url=base_url, max_wait_sec=30)
            )
            rc = await proc.wait()
            if not bootstrap_task.done():
                bootstrap_task.cancel()
                with suppress(asyncio.CancelledError):
                    await bootstrap_task
        if rc == 0:
            logger.info("[fitbit_monitor] 进程正常退出，不再重启")
            return
        logger.warning("[fitbit_monitor] 进程退出 rc=%d，5秒后重启...", rc)
        await asyncio.sleep(5)

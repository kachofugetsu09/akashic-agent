from __future__ import annotations

import asyncio
import logging
import socket
import urllib.request
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

from utils.process_group import (
    OwnedProcessGroup,
    owned_process_env,
    process_group_spawn_kwargs,
)

logger = logging.getLogger(__name__)
_STOP_TIMEOUT_SECONDS = 5.0
_RECOVERY_BACKOFF_SECONDS = (0.25, 1.0, 3.0)
_RECOVERY_STABLE_SECONDS = 60.0


@dataclass
class _ServiceEpoch:
    spec: dict[str, Any]
    epoch: int
    candidate_generation_id: str | None
    restart_attempts: int = 0
    ready_at: float | None = None
    stopping: bool = False


@dataclass
class _RunningService:
    epoch: _ServiceEpoch
    process: asyncio.subprocess.Process
    process_group: OwnedProcessGroup
    watch_task: asyncio.Task[None] | None = None
    stopping: bool = False


class PluginServiceHost:
    def __init__(self) -> None:
        self._bindings: dict[str, dict[str, dict[str, Any]]] = {}
        self._running: dict[tuple[str, str], _RunningService] = {}
        self._epochs: dict[tuple[str, str], _ServiceEpoch] = {}
        self._validation_services: dict[str, tuple[str, ...]] = {}
        self._candidate_failures: dict[str, RuntimeError] = {}
        self._next_epoch = 0
        self._fatal_failure: RuntimeError | None = None
        self._fatal_future: asyncio.Future[RuntimeError] | None = None

    def bind_plugin_services(
        self,
        services: dict[str, dict[str, dict[str, Any]]],
    ) -> None:
        self._bindings = {
            plugin_id: dict(plugin_services)
            for plugin_id, plugin_services in services.items()
        }

    async def start_all(self) -> None:
        started: list[tuple[str, str]] = []
        try:
            for plugin_id, services in sorted(self._bindings.items()):
                for service_id, spec in sorted(services.items()):
                    await self._start(plugin_id, service_id, spec)
                    started.append((plugin_id, service_id))
        except BaseException as start_error:
            rollback_errors: list[str] = []
            for key in reversed(started):
                try:
                    await self._stop(*key)
                except (asyncio.CancelledError, Exception) as error:
                    rollback_errors.append(f"{key[0]}:{key[1]}: {error}")
            if rollback_errors:
                rollback_error = RuntimeError(
                    "managed service 启动回滚失败: " + "; ".join(rollback_errors)
                )
                raise start_error from rollback_error
            raise

    async def stop_all(self) -> None:
        errors: list[str] = []
        cancellation: asyncio.CancelledError | None = None
        for plugin_id, service_id in reversed(tuple(self._epochs)):
            try:
                await self._stop(plugin_id, service_id)
            except asyncio.CancelledError as error:
                if cancellation is None:
                    cancellation = error
            except Exception as error:
                errors.append(f"{plugin_id}:{service_id}: {error}")
        if cancellation is not None:
            raise cancellation
        if errors:
            raise RuntimeError("managed service 停止失败: " + "; ".join(errors))

    async def start_candidate(
        self,
        generation_id: str,
        services: dict[str, dict[str, Any]],
    ) -> None:
        """Start candidate services under generation-scoped isolated ownership."""

        # 1. Each validation generation owns a disjoint process namespace key.
        if generation_id in self._validation_services:
            raise RuntimeError(f"候选 managed service 已启动: {generation_id}")
        self._candidate_failures.pop(generation_id, None)
        owner = f"validation:{generation_id}"
        started: list[str] = []
        try:
            for service_id, spec in sorted(services.items()):
                await self._start(owner, service_id, spec)
                started.append(service_id)
        except BaseException:
            for service_id in reversed(started):
                await self._stop(owner, service_id)
            raise
        self._validation_services[generation_id] = tuple(started)

    async def stop_candidate(self, generation_id: str) -> None:
        """Stop every isolated service owned by one candidate generation."""

        service_ids = self._validation_services.pop(generation_id, ())
        owner = f"validation:{generation_id}"
        errors: list[str] = []
        for service_id in reversed(service_ids):
            try:
                await self._stop(owner, service_id)
            except Exception as error:
                errors.append(f"{service_id}: {error}")
        self._candidate_failures.pop(generation_id, None)
        if errors:
            raise RuntimeError("候选 managed service 停止失败: " + "; ".join(errors))

    async def wait_fatal_failure(self) -> None:
        """Wait for the first exhausted active service and raise its stable failure."""

        if self._fatal_failure is not None:
            raise self._fatal_failure
        if self._fatal_future is None:
            self._fatal_future = asyncio.get_running_loop().create_future()
        failure = await asyncio.shield(self._fatal_future)
        raise failure

    async def assert_candidate_healthy(self, generation_id: str) -> None:
        """Reprobe every live candidate service before allowing promotion."""

        # 1. Resolve the exact validation ownership before crossing an async probe.
        service_ids = self._validation_services.get(generation_id)
        if service_ids is None:
            raise RuntimeError(f"候选 managed service generation 不存在: {generation_id}")
        failure = self._candidate_failures.get(generation_id)
        if failure is not None:
            raise failure
        owner = f"validation:{generation_id}"
        observed: list[tuple[tuple[str, str], _ServiceEpoch, _RunningService]] = []
        for service_id in service_ids:
            key = (owner, service_id)
            epoch = self._epochs.get(key)
            running = self._running.get(key)
            if (
                epoch is None
                or epoch.candidate_generation_id != generation_id
                or epoch.stopping
                or running is None
                or running.epoch is not epoch
                or running.stopping
                or running.process.returncode is not None
            ):
                raise RuntimeError(
                    "候选 managed service 当前不可晋升: "
                    f"{generation_id}:{service_id} 没有健康的当前 process epoch"
                )
            observed.append((key, epoch, running))

        # 2. Reprobe readiness, then reject any ownership or process change during I/O.
        for key, epoch, running in observed:
            readiness_url = str(epoch.spec.get("readiness_url") or "")
            if readiness_url and not await asyncio.to_thread(_url_ready, readiness_url):
                raise RuntimeError(
                    "候选 managed service readiness 失败: "
                    f"{key[0]}:{key[1]} url={readiness_url}"
                )
            await asyncio.sleep(0)
            if (
                self._epochs.get(key) is not epoch
                or self._running.get(key) is not running
                or epoch.stopping
                or running.stopping
                or running.process.returncode is not None
            ):
                raise RuntimeError(
                    "候选 managed service 晋升探测期间代际变化: "
                    f"{key[0]}:{key[1]} epoch={epoch.epoch}"
                )

    async def swap_plugin_services(
        self,
        plugin_id: str,
        old_services: dict[str, dict[str, Any]],
        new_services: dict[str, dict[str, Any]],
    ) -> None:
        if self._bindings.get(plugin_id, {}) != old_services:
            raise RuntimeError(f"插件 managed service 代际不一致: {plugin_id}")
        changed = {
            service_id
            for service_id in old_services.keys() | new_services.keys()
            if old_services.get(service_id) != new_services.get(service_id)
        }
        stopped: list[str] = []
        try:
            for service_id in sorted(changed.intersection(old_services), reverse=True):
                try:
                    await self._stop(plugin_id, service_id)
                finally:
                    if (plugin_id, service_id) not in self._running:
                        stopped.append(service_id)
        except BaseException as stop_error:
            restore_errors = await self._restore(plugin_id, old_services, stopped)
            if restore_errors:
                raise RuntimeError(
                    "旧 managed service 恢复失败: " + "; ".join(restore_errors)
                ) from stop_error
            raise

        started: list[str] = []
        try:
            for service_id in sorted(changed.intersection(new_services)):
                await self._start(plugin_id, service_id, new_services[service_id])
                started.append(service_id)
        except BaseException as start_error:
            for service_id in reversed(started):
                await self._stop(plugin_id, service_id)
            restore_errors = await self._restore(plugin_id, old_services, stopped)
            if restore_errors:
                raise RuntimeError(
                    "旧 managed service 恢复失败: " + "; ".join(restore_errors)
                ) from start_error
            raise
        self._bindings[plugin_id] = dict(new_services)

    async def _restore(
        self,
        plugin_id: str,
        services: dict[str, dict[str, Any]],
        service_ids: list[str],
    ) -> list[str]:
        errors: list[str] = []
        for service_id in reversed(service_ids):
            try:
                await self._start(plugin_id, service_id, services[service_id])
            except Exception as error:
                errors.append(f"{service_id}: {error}")
        return errors

    async def _start(
        self,
        plugin_id: str,
        service_id: str,
        spec: dict[str, Any],
    ) -> None:
        key = (plugin_id, service_id)
        if key in self._epochs:
            raise RuntimeError(f"managed service 已运行: {plugin_id}:{service_id}")
        self._next_epoch += 1
        generation_id = (
            plugin_id.removeprefix("validation:")
            if plugin_id.startswith("validation:")
            else None
        )
        epoch = _ServiceEpoch(
            spec=spec,
            epoch=self._next_epoch,
            candidate_generation_id=generation_id,
        )
        self._epochs[key] = epoch
        try:
            await self._spawn_attempt(key, epoch)
        except BaseException:
            if self._epochs.get(key) is epoch:
                del self._epochs[key]
            raise

    async def _spawn_attempt(
        self,
        key: tuple[str, str],
        epoch: _ServiceEpoch,
    ) -> None:
        """Spawn and publish one attempt only after its readiness contract passes."""

        readiness_url = str(epoch.spec.get("readiness_url") or "")
        if readiness_url and await asyncio.to_thread(
            _endpoint_listening,
            readiness_url,
        ):
            raise RuntimeError(
                f"managed service readiness 监听端口已被占用: {readiness_url}"
            )
        process = await asyncio.create_subprocess_exec(
            *epoch.spec["command"],
            cwd=epoch.spec["cwd"],
            env=owned_process_env(epoch.spec["env"]),
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
            **process_group_spawn_kwargs(),
        )
        running = _RunningService(
            epoch=epoch,
            process=process,
            process_group=OwnedProcessGroup.from_process(process),
        )
        self._running[key] = running
        try:
            await self._wait_ready(running)
        except BaseException:
            running.stopping = True
            await self._terminate_attempt(key, running)
            raise
        epoch.ready_at = asyncio.get_running_loop().time()
        running.watch_task = asyncio.create_task(
            self._watch_process_exit(key, running),
            name=f"managed_service:{key[0]}:{key[1]}:epoch-{epoch.epoch}",
        )

    async def _wait_ready(self, service: _RunningService) -> None:
        timeout = float(service.epoch.spec["startup_timeout_seconds"])
        deadline = asyncio.get_running_loop().time() + timeout
        readiness_url = str(service.epoch.spec.get("readiness_url") or "")
        if not readiness_url:
            try:
                exit_code = await asyncio.wait_for(
                    asyncio.shield(service.process.wait()),
                    timeout=min(0.2, timeout),
                )
            except TimeoutError:
                return
            raise RuntimeError(f"managed service 启动失败: exit={exit_code}")
        while asyncio.get_running_loop().time() < deadline:
            if service.process.returncode is not None:
                raise RuntimeError(
                    f"managed service 启动失败: exit={service.process.returncode}"
                )
            if await asyncio.to_thread(
                _url_ready,
                readiness_url,
            ):
                await asyncio.sleep(0)
                if service.process.returncode is not None:
                    raise RuntimeError(
                        f"managed service 启动失败: exit={service.process.returncode}"
                    )
                return
            await asyncio.sleep(0.1)
        raise RuntimeError("managed service 启动超时")

    async def _stop(self, plugin_id: str, service_id: str) -> None:
        key = (plugin_id, service_id)
        epoch = self._epochs.get(key)
        if epoch is None:
            return
        epoch.stopping = True
        running = self._running.get(key)
        if running is not None:
            running.stopping = True

        async def reap() -> None:
            if running is not None:
                try:
                    await running.process_group.terminate(
                        timeout_s=_STOP_TIMEOUT_SECONDS,
                    )
                    if running.watch_task is not None:
                        _ = await asyncio.gather(
                            running.watch_task,
                            return_exceptions=True,
                        )
                except BaseException:
                    running.stopping = False
                    epoch.stopping = False
                    raise
                if self._running.get(key) is running:
                    del self._running[key]
            if self._epochs.get(key) is epoch:
                del self._epochs[key]

        task = asyncio.create_task(
            reap(), name=f"stop_service:{plugin_id}:{service_id}"
        )
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            _ = await task
            raise

    async def _watch_process_exit(
        self,
        key: tuple[str, str],
        running: _RunningService,
    ) -> None:
        """Clean an exited attempt, then recover only its still-current epoch."""
        exit_code = await running.process.wait()
        if running.stopping or self._running.get(key) is not running:
            return
        logger.error(
            "managed service 意外退出: %s:%s exit=%s，开始清理进程组 %s",
            key[0],
            key[1],
            exit_code,
            running.process_group.group_id,
        )
        try:
            await self._terminate_attempt(key, running)
        except Exception as error:
            self._exhaust_epoch(key, running.epoch, error)
            return
        await self._recover_epoch(
            key,
            running.epoch,
            RuntimeError(
                f"managed service 意外退出: {key[0]}:{key[1]} exit={exit_code}"
            ),
        )

    async def _terminate_attempt(
        self,
        key: tuple[str, str],
        running: _RunningService,
    ) -> None:
        await running.process_group.terminate(timeout_s=_STOP_TIMEOUT_SECONDS)
        if self._running.get(key) is running:
            del self._running[key]

    async def _recover_epoch(
        self,
        key: tuple[str, str],
        epoch: _ServiceEpoch,
        failure: Exception,
    ) -> None:
        """Retry one current epoch with bounded backoff and stable-window reset."""

        while self._epochs.get(key) is epoch and not epoch.stopping:
            now = asyncio.get_running_loop().time()
            if (
                epoch.ready_at is not None
                and now - epoch.ready_at >= _RECOVERY_STABLE_SECONDS
            ):
                epoch.restart_attempts = 0
            if epoch.restart_attempts >= len(_RECOVERY_BACKOFF_SECONDS):
                self._exhaust_epoch(key, epoch, failure)
                return

            delay = _RECOVERY_BACKOFF_SECONDS[epoch.restart_attempts]
            epoch.restart_attempts += 1
            await asyncio.sleep(delay)
            if self._epochs.get(key) is not epoch or epoch.stopping:
                return
            try:
                await self._spawn_attempt(key, epoch)
            except asyncio.CancelledError:
                raise
            except Exception as error:
                failure = error
                continue
            return

    def _exhaust_epoch(
        self,
        key: tuple[str, str],
        epoch: _ServiceEpoch,
        cause: Exception,
    ) -> None:
        """Publish one terminal generation failure without reviving stale ownership."""

        if self._epochs.get(key) is not epoch or epoch.stopping:
            return
        if key not in self._running:
            del self._epochs[key]
        failure = RuntimeError(
            f"managed service recovery 耗尽: {key[0]}:{key[1]} epoch={epoch.epoch}"
        )
        failure.__cause__ = cause
        if epoch.candidate_generation_id is not None:
            self._candidate_failures.setdefault(
                epoch.candidate_generation_id,
                failure,
            )
            return
        if self._fatal_failure is None:
            self._fatal_failure = failure
            if self._fatal_future is not None and not self._fatal_future.done():
                self._fatal_future.set_result(failure)


def _url_ready(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=1):
            return True
    except OSError:
        return False


def _endpoint_listening(url: str) -> bool:
    """验证 HTTP readiness endpoint，并判断其监听端口是否已被占用。"""
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or parsed.hostname is None:
        raise RuntimeError(f"managed service readiness_url 无效: {url}")
    try:
        port = parsed.port
    except ValueError as error:
        raise RuntimeError(f"managed service readiness_url 无效: {url}") from error
    if port is None:
        port = 443 if parsed.scheme == "https" else 80
    try:
        with socket.create_connection((parsed.hostname, port), timeout=1):
            return True
    except OSError:
        return False

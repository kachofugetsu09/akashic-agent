from __future__ import annotations

import asyncio
import os
import urllib.request
from dataclasses import dataclass
from typing import Any


@dataclass
class _RunningService:
    spec: dict[str, Any]
    process: asyncio.subprocess.Process


class PluginServiceHost:
    def __init__(self) -> None:
        self._bindings: dict[str, dict[str, dict[str, Any]]] = {}
        self._running: dict[tuple[str, str], _RunningService] = {}

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
        except BaseException:
            for key in reversed(started):
                await self._stop(*key)
            raise

    async def stop_all(self) -> None:
        errors: list[str] = []
        for plugin_id, service_id in reversed(tuple(self._running)):
            try:
                await self._stop(plugin_id, service_id)
            except Exception as error:
                errors.append(f"{plugin_id}:{service_id}: {error}")
        if errors:
            raise RuntimeError("managed service 停止失败: " + "; ".join(errors))

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
                await self._stop(plugin_id, service_id)
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
        if key in self._running:
            raise RuntimeError(f"managed service 已运行: {plugin_id}:{service_id}")
        process = await asyncio.create_subprocess_exec(
            *spec["command"],
            cwd=spec["cwd"],
            env={**os.environ, **spec["env"]},
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        running = _RunningService(spec=spec, process=process)
        self._running[key] = running
        try:
            await self._wait_ready(running)
        except BaseException:
            await self._stop(plugin_id, service_id)
            raise

    async def _wait_ready(self, service: _RunningService) -> None:
        timeout = float(service.spec["startup_timeout_seconds"])
        deadline = asyncio.get_running_loop().time() + timeout
        readiness_url = str(service.spec.get("readiness_url") or "")
        while asyncio.get_running_loop().time() < deadline:
            if service.process.returncode is not None:
                raise RuntimeError(
                    f"managed service 启动失败: exit={service.process.returncode}"
                )
            if not readiness_url or await asyncio.to_thread(
                _url_ready,
                readiness_url,
            ):
                return
            await asyncio.sleep(0.1)
        raise RuntimeError("managed service 启动超时")

    async def _stop(self, plugin_id: str, service_id: str) -> None:
        running = self._running.get((plugin_id, service_id))
        if running is None:
            return
        process = running.process
        if process.returncode is None:
            try:
                process.terminate()
            except ProcessLookupError:
                pass
            try:
                _ = await asyncio.wait_for(process.wait(), timeout=5)
            except TimeoutError:
                try:
                    process.kill()
                except ProcessLookupError:
                    pass
                _ = await process.wait()
        _ = self._running.pop((plugin_id, service_id), None)


def _url_ready(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=1):
            return True
    except OSError:
        return False

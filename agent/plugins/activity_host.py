from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType

from agent.plugins.jobs import (
    EventTrigger,
    IntervalTrigger,
    RegisteredPluginJob,
    plugin_job_key,
)
from agent.plugins.specs import RegisteredProactiveSource, proactive_source_key


@dataclass(frozen=True)
class PreparedJobCatalog:
    generation_id: str
    jobs: Mapping[str, RegisteredPluginJob]


@dataclass(frozen=True)
class PreparedProactiveCatalog:
    generation_id: str
    sources: Mapping[str, RegisteredProactiveSource]


class PluginJobHost:
    def __init__(self) -> None:
        self._catalogs: dict[str, PreparedJobCatalog] = {}

    def prepare(
        self,
        generation_id: str,
        jobs: tuple[RegisteredPluginJob, ...],
    ) -> PreparedJobCatalog:
        compiled: dict[str, RegisteredPluginJob] = {}
        for job in jobs:
            key = plugin_job_key(job)
            if key in compiled:
                raise RuntimeError(f"插件 Job 稳定键重复: {key}")
            self._validate(job)
            compiled[key] = replace(
                job,
                spec=replace(job.spec, triggers=tuple(job.spec.triggers)),
            )
        catalog = PreparedJobCatalog(
            generation_id=generation_id,
            jobs=MappingProxyType(compiled),
        )
        self._catalogs[generation_id] = catalog
        return catalog

    def close(self, generation_id: str) -> None:
        _ = self._catalogs.pop(generation_id, None)

    def get(self, generation_id: str) -> PreparedJobCatalog | None:
        return self._catalogs.get(generation_id)

    @staticmethod
    def _validate(job: RegisteredPluginJob) -> None:
        spec = job.spec
        if not callable(spec.handler):
            raise RuntimeError(f"插件 Job handler 不可调用: {plugin_job_key(job)}")
        if (
            isinstance(spec.debounce_seconds, bool)
            or not isinstance(spec.debounce_seconds, int)
            or spec.debounce_seconds < 0
            or not isinstance(spec.coalesce, bool)
        ):
            raise RuntimeError(f"插件 Job 策略无效: {plugin_job_key(job)}")
        for trigger in spec.triggers:
            if isinstance(trigger, IntervalTrigger):
                if (
                    isinstance(trigger.seconds, bool)
                    or not isinstance(trigger.seconds, int)
                    or trigger.seconds <= 0
                ):
                    raise RuntimeError(f"插件 Job interval 无效: {plugin_job_key(job)}")
            elif isinstance(trigger, EventTrigger):
                if not isinstance(trigger.event_type, type):
                    raise RuntimeError(f"插件 Job event type 无效: {plugin_job_key(job)}")
            else:
                raise RuntimeError(f"插件 Job trigger 无效: {plugin_job_key(job)}")


class PluginProactiveHost:
    def __init__(self) -> None:
        self._catalogs: dict[str, PreparedProactiveCatalog] = {}

    def prepare(
        self,
        generation_id: str,
        sources: tuple[RegisteredProactiveSource, ...],
    ) -> PreparedProactiveCatalog:
        compiled: dict[str, RegisteredProactiveSource] = {}
        for source in sources:
            key = proactive_source_key(source)
            if key in compiled:
                raise RuntimeError(f"proactive source 稳定键重复: {key}")
            compiled[key] = replace(
                source,
                spec=replace(source.spec, channels=tuple(source.spec.channels)),
            )
        catalog = PreparedProactiveCatalog(
            generation_id=generation_id,
            sources=MappingProxyType(compiled),
        )
        self._catalogs[generation_id] = catalog
        return catalog

    def close(self, generation_id: str) -> None:
        _ = self._catalogs.pop(generation_id, None)

    def get(self, generation_id: str) -> PreparedProactiveCatalog | None:
        return self._catalogs.get(generation_id)

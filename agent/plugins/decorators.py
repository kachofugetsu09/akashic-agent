from __future__ import annotations

from typing import Any, Callable

from agent.plugins.registry import (
    HandlerType,
    MetadataKind,
    PluginEventType,
    PluginHandlerMetadata,
    plugin_registry,
)


def _get_or_create_handler(
    func: Callable[..., Any],
    event_type: PluginEventType,
    handler_type: HandlerType,
    **kwargs: Any,
) -> PluginHandlerMetadata:
    # 1. 幂等：同一函数重复装饰时直接返回已有记录
    existing = plugin_registry._handlers.get_by_name(
        event_type, func.__name__, func.__module__
    )
    if existing:
        return existing
    # 2. 构建元数据并按 priority 插入全局 handler 列表
    md = PluginHandlerMetadata(
        kind=MetadataKind.LIFECYCLE,
        event_type=event_type,
        handler_type=handler_type,
        handler=func,
        handler_name=func.__name__,
        plugin_module_path=func.__module__,
        **kwargs,
    )
    plugin_registry._handlers.append(md)
    return md


def on_before_turn(**options: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def deco(func: Callable[..., Any]) -> Callable[..., Any]:
        _get_or_create_handler(func, PluginEventType.BEFORE_TURN, HandlerType.GATE, **options)
        return func
    return deco


def on_before_reasoning(**options: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def deco(func: Callable[..., Any]) -> Callable[..., Any]:
        _get_or_create_handler(func, PluginEventType.BEFORE_REASONING, HandlerType.GATE, **options)
        return func
    return deco


def on_before_step(**options: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def deco(func: Callable[..., Any]) -> Callable[..., Any]:
        _get_or_create_handler(func, PluginEventType.BEFORE_STEP, HandlerType.GATE, **options)
        return func
    return deco


def on_after_step(**options: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def deco(func: Callable[..., Any]) -> Callable[..., Any]:
        _get_or_create_handler(func, PluginEventType.AFTER_STEP, HandlerType.TAP, **options)
        return func
    return deco


def on_after_reasoning(**options: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def deco(func: Callable[..., Any]) -> Callable[..., Any]:
        _get_or_create_handler(func, PluginEventType.AFTER_REASONING, HandlerType.GATE, **options)
        return func
    return deco


def on_after_turn(**options: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def deco(func: Callable[..., Any]) -> Callable[..., Any]:
        _get_or_create_handler(func, PluginEventType.AFTER_TURN, HandlerType.TAP, **options)
        return func
    return deco

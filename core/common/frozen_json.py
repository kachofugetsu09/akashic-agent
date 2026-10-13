"""复制并深冻结 JSON 值；重复边界复用已经冻结的对象。"""
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Never, cast


class _FrozenJson(dict[str, object]):
    """深冻结的 JSON 使用原生字典读取与编码，只禁止修改操作。"""

    __slots__ = ()

    def __new__(cls, value: Mapping[str, object]) -> _FrozenJson:
        result = dict.__new__(cls)
        dict.update(result, value)
        return result

    def __init__(self, value: Mapping[str, object]) -> None:
        # 仅在新建时填充；对既有对象再次调用 __init__ 不能修改内容。
        pass

    def _immutable(self: object, *args: object, **kwargs: object) -> Never:
        raise TypeError("冻结的 JSON 对象不能修改")

    __setitem__ = _immutable
    __delitem__ = _immutable
    __ior__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable
    setdefault = _immutable
    update = _immutable


class _FrozenJsonArray(tuple[object, ...]):
    """标记已经深冻结的 JSON 数组，边界间直接复用。"""

    __slots__ = ()


def _freeze(item: object, active: set[int]) -> object:
    # 精确类型先命中普通 JSON；ABC 检查只留给 Mapping 等结构子类。
    item_type = type(item)
    if item_type in _TRIVIAL_TYPES:
        return item
    if item_type is float:
        if not math.isfinite(cast(float, item)):
            raise ValueError("JSON 不接受非有限浮点数")
        return item
    if item_type is dict or item_type is list or item_type is tuple:
        identity = id(item)
        if identity in active:
            raise ValueError("JSON value 不允许循环引用")
        active.add(identity)
        try:
            if item_type is dict:
                # 直接向冻结实例填充，省掉临时字典到 _FrozenJson 的二次复制。
                frozen: dict[str, object] = dict.__new__(_FrozenJson)
                setitem = dict.__setitem__
                for key, nested in cast(dict[object, object], item).items():
                    if not isinstance(key, str):
                        raise TypeError("JSON 对象的 key 必须是字符串")
                    setitem(frozen, key, _freeze(nested, active))
                return frozen
            return _FrozenJsonArray(
                [_freeze(nested, active) for nested in cast(list[object] | tuple[object, ...], item)]
            )
        finally:
            active.remove(identity)
    # 标量子类与原 isinstance 语义一致：按标量接受，不因精确类型失败误拒。
    if isinstance(item, (str, bool, int, _FrozenJson, _FrozenJsonArray)):
        return item
    if isinstance(item, float):
        if not math.isfinite(item):
            raise ValueError("JSON 不接受非有限浮点数")
        return item
    if isinstance(item, Mapping):
        identity = id(item)
        if identity in active:
            raise ValueError("JSON value 不允许循环引用")
        active.add(identity)
        try:
            frozen = dict.__new__(_FrozenJson)
            setitem = dict.__setitem__
            for key, nested in cast(Mapping[object, object], item).items():
                if not isinstance(key, str):
                    raise TypeError("JSON 对象的 key 必须是字符串")
                setitem(frozen, key, _freeze(nested, active))
            return frozen
        finally:
            active.remove(identity)
    if isinstance(item, (list, tuple)):
        identity = id(item)
        if identity in active:
            raise ValueError("JSON value 不允许循环引用")
        active.add(identity)
        try:
            return _FrozenJsonArray(
                [_freeze(nested, active) for nested in cast(list[object] | tuple[object, ...], item)]
            )
        finally:
            active.remove(identity)
    raise TypeError(f"值必须是 JSON 值，实际为 {type(item).__name__}")


_TRIVIAL_TYPES = frozenset({str, bool, int, type(None), _FrozenJson, _FrozenJsonArray})


def freeze_json(value: object) -> object:
    """复制外部 JSON；已经冻结的对象不重复校验或复制。"""
    if isinstance(value, (_FrozenJson, _FrozenJsonArray)):
        return value
    return _freeze(value, set())


def json_value(value: object) -> object:
    """把已校验的不可变 JSON 转回可序列化容器。"""
    if isinstance(value, Mapping):
        return {
            key: json_value(item)
            for key, item in cast(Mapping[str, object], value).items()
        }
    if isinstance(value, tuple):
        return [json_value(item) for item in cast(tuple[object, ...], value)]
    return value

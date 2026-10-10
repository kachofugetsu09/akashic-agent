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


def freeze_json(value: object) -> object:
    """复制外部 JSON；已经冻结的对象不重复校验或复制。"""
    if isinstance(value, (_FrozenJson, _FrozenJsonArray)):
        return value
    active: set[int] = set()

    def freeze(item: object) -> object:
        if item is None or isinstance(item, (str, bool, int, _FrozenJson, _FrozenJsonArray)):
            return item
        if isinstance(item, float):
            if not math.isfinite(item):
                raise ValueError("JSON 不接受非有限浮点数")
            return item
        if isinstance(item, (Mapping, list, tuple)):
            identity = id(item)
            if identity in active:
                raise ValueError("JSON value 不允许循环引用")
            active.add(identity)
            try:
                if isinstance(item, Mapping):
                    mapping = cast(Mapping[object, object], item)
                    frozen: dict[str, object] = {}
                    for key, nested in mapping.items():
                        if not isinstance(key, str):
                            raise TypeError("JSON 对象的 key 必须是字符串")
                        frozen[key] = freeze(nested)
                    return _FrozenJson(frozen)
                return _FrozenJsonArray(freeze(nested) for nested in cast(list[object] | tuple[object, ...], item))
            finally:
                active.remove(identity)
        raise TypeError(f"值必须是 JSON 值，实际为 {type(item).__name__}")

    return freeze(value)

"""工具图标声明的校验与规范化；只允许受限语义名与单色安全 SVG 几何。"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from typing import cast
import xml.etree.ElementTree as ET

from core.common.frozen_json import freeze_json

_LUCIDE_NAME_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_SVG_NS = "http://www.w3.org/2000/svg"
_ALLOWED_TAGS = frozenset({"path", "circle", "rect", "line", "polyline", "polygon", "ellipse"})
_ALLOWED_ATTRS = frozenset(
    {
        "d",
        "cx",
        "cy",
        "r",
        "rx",
        "ry",
        "x",
        "y",
        "x1",
        "y1",
        "x2",
        "y2",
        "width",
        "height",
        "points",
        "fill",
        "stroke",
        "fillRule",
        "clipRule",
        "strokeLinecap",
        "strokeLinejoin",
        "strokeWidth",
    }
)
_ATTR_ALIASES = {
    "fill-rule": "fillRule",
    "clip-rule": "clipRule",
    "stroke-linecap": "strokeLinecap",
    "stroke-linejoin": "strokeLinejoin",
    "stroke-width": "strokeWidth",
}
_SVG_MAX_BYTES = 8192
_SVG_MAX_NODES = 32


def _local_tag(tag: str) -> str:
    if tag.startswith("{"):
        ns, _, name = tag[1:].partition("}")
        if ns != _SVG_NS:
            raise ValueError(f"工具图标不支持的 SVG 命名空间: {ns}")
        return name
    return tag


def _validate_view_box(raw: object) -> str:
    if not isinstance(raw, str):
        raise ValueError("工具图标 SVG viewBox 必须是字符串")
    parts = raw.replace(",", " ").split()
    if len(parts) != 4:
        raise ValueError("工具图标 SVG viewBox 必须包含 4 个数值")
    try:
        nums = [float(item) for item in parts]
    except ValueError as error:
        raise ValueError("工具图标 SVG viewBox 数值无效") from error
    if any(not math.isfinite(val) for val in nums) or nums[2] <= 0 or nums[3] <= 0:
        raise ValueError("工具图标 SVG viewBox 宽高必须大于 0")
    return " ".join(parts)


def _normalize_paint(attr: str, value: str) -> str:
    cleaned = value.strip()
    lower = cleaned.lower()
    if lower == "none":
        return "none"
    if lower == "currentcolor":
        return "currentColor"
    raise ValueError(f"工具图标 {attr} 只允许 none 或 currentColor，禁止写死颜色: {value}")


def _normalize_attrs(raw_attrs: Mapping[str, object]) -> dict[str, str]:
    attrs: dict[str, str] = {}
    for key, value in raw_attrs.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise ValueError("工具图标 SVG 属性键值必须是字符串")
        if key.startswith("{"):
            raise ValueError(f"工具图标 SVG 不允许带命名空间的属性: {key}")
        canonical = _ATTR_ALIASES.get(key, key)
        if canonical not in _ALLOWED_ATTRS:
            raise ValueError(f"工具图标 SVG 不允许的属性: {key}")
        cleaned = value.strip()
        if not cleaned or len(cleaned) > 2048:
            raise ValueError(f"工具图标 SVG 属性值无效: {key}")
        if canonical in {"fill", "stroke"}:
            cleaned = _normalize_paint(canonical, cleaned)
        attrs[canonical] = cleaned
    return attrs


def _collect_svg_nodes(element: ET.Element, inherited: Mapping[str, str], nodes: list[dict[str, object]]) -> None:
    for child in element:
        tag = _local_tag(child.tag)
        if tag == "g":
            group_attrs = {**inherited, **_normalize_attrs(child.attrib)}
            _collect_svg_nodes(child, group_attrs, nodes)
            continue
        if tag not in _ALLOWED_TAGS:
            raise ValueError(f"工具图标 SVG 不允许的元素: <{tag}>")
        if len(child):
            raise ValueError(f"工具图标 SVG <{tag}> 不允许嵌套子元素")
        if len(nodes) >= _SVG_MAX_NODES:
            raise ValueError(f"工具图标 SVG 节点数超过上限 ({_SVG_MAX_NODES})")
        merged = {**inherited, **_normalize_attrs(child.attrib)}
        nodes.append({"tag": tag, "attrs": merged})


def _parse_svg_string(raw: str) -> Mapping[str, object]:
    text = raw.strip()
    if len(text.encode("utf-8")) > _SVG_MAX_BYTES:
        raise ValueError("工具图标 SVG 超过大小上限")
    if "<!" in text or "<?" in text:
        raise ValueError("工具图标 SVG 不允许包含声明或处理指令")
    try:
        root = ET.fromstring(text)
    except ET.ParseError as error:
        raise ValueError("工具图标 SVG 解析失败") from error
    if _local_tag(root.tag) != "svg":
        raise ValueError("工具图标 SVG 根元素必须是 <svg>")
    view_box = _validate_view_box(root.attrib.get("viewBox", "0 0 24 24"))
    inherited: dict[str, str] = {}
    for key, value in root.attrib.items():
        canonical = _ATTR_ALIASES.get(key, key)
        if canonical in _ALLOWED_ATTRS:
            if canonical in {"fill", "stroke"}:
                inherited[canonical] = _normalize_paint(canonical, value)
            else:
                inherited[canonical] = value.strip()
    nodes: list[dict[str, object]] = []
    _collect_svg_nodes(root, inherited, nodes)
    if not nodes:
        raise ValueError("工具图标 SVG 至少需要一个几何绘制节点")
    return cast(
        Mapping[str, object],
        freeze_json({"kind": "svg", "viewBox": view_box, "nodes": nodes}),
    )


def normalize_tool_icon(icon: str | Mapping[str, object] | None) -> Mapping[str, object] | None:
    """校验并冻结工具图标声明；接受 Lucide 短名、受限 SVG 字符串或规范化字典。"""
    if icon is None:
        return None
    if isinstance(icon, str):
        trimmed = icon.strip()
        if not trimmed:
            raise ValueError("工具图标不能为空字符串")
        if trimmed.startswith("<"):
            return _parse_svg_string(trimmed)
        if len(trimmed) > 64 or _LUCIDE_NAME_RE.fullmatch(trimmed) is None:
            raise ValueError(f"工具图标名称无效: {icon!r}")
        return cast(Mapping[str, object], freeze_json({"kind": "lucide", "name": trimmed}))
    if isinstance(icon, Mapping):
        kind = icon.get("kind")
        if kind == "lucide":
            if set(icon) != {"kind", "name"}:
                raise ValueError("lucide 工具图标字段无效")
            name = icon.get("name")
            if not isinstance(name, str) or len(name) > 64 or _LUCIDE_NAME_RE.fullmatch(name) is None:
                raise ValueError(f"工具图标名称无效: {name!r}")
            return cast(Mapping[str, object], freeze_json({"kind": "lucide", "name": name}))
        if kind == "svg":
            if set(icon) != {"kind", "viewBox", "nodes"}:
                raise ValueError("svg 工具图标字段无效")
            view_box = _validate_view_box(icon.get("viewBox"))
            raw_nodes = icon.get("nodes")
            if not isinstance(raw_nodes, (list, tuple)) or not (1 <= len(raw_nodes) <= _SVG_MAX_NODES):
                raise ValueError("工具图标 SVG nodes 数量无效")
            nodes: list[dict[str, object]] = []
            for item in raw_nodes:
                if not isinstance(item, Mapping) or set(item) != {"tag", "attrs"}:
                    raise ValueError("工具图标 SVG node 结构无效")
                tag = item.get("tag")
                attrs = item.get("attrs")
                if not isinstance(tag, str) or tag not in _ALLOWED_TAGS:
                    raise ValueError(f"工具图标 SVG 不允许的节点标签: {tag!r}")
                if not isinstance(attrs, Mapping):
                    raise ValueError("工具图标 SVG node.attrs 必须是映射")
                nodes.append({"tag": tag, "attrs": _normalize_attrs(attrs)})
            return cast(
                Mapping[str, object],
                freeze_json({"kind": "svg", "viewBox": view_box, "nodes": nodes}),
            )
        raise ValueError(f"未知工具图标类型: {kind!r}")
    raise TypeError("工具图标必须是字符串、映射或 None")

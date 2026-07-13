from __future__ import annotations

import re
from collections.abc import Mapping
from typing import TypedDict, cast

_ASCII_ALIAS_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9_]*")
_NEGATIVE_TOOL_PREFIXES = (
    "不能直接使用",
    "不能直接用",
    "不要直接使用",
    "不要直接用",
    "别直接使用",
    "别直接用",
    "不能先使用",
    "不能先用",
    "不要先使用",
    "不要先用",
    "别先使用",
    "别先用",
    "不能使用",
    "不能用",
    "不要使用",
    "不要用",
    "别使用",
    "别用",
    "禁止使用",
    "禁止用",
)
_POSITIVE_TOOL_PREFIXES = (
    "必须先使用",
    "必须先用",
    "必须使用",
    "必须用",
    "先使用",
    "先用",
    "优先使用",
    "优先用",
    "应先使用",
    "应先用",
    "应该使用",
    "应该用",
    "直接使用",
    "直接用",
)


class ProcedureRuleSchema(TypedDict):
    required_tools: list[str]
    forbidden_tools: list[str]
    mentioned_tools: list[str]


def _extract_ascii_aliases(text: str) -> set[str]:
    aliases: set[str] = set()
    matches = list(_ASCII_ALIAS_PATTERN.finditer(text))
    for match in matches:
        token = match.group(0).lower()
        if len(token) >= 2:
            aliases.add(token)
    for index in range(len(matches) - 1):
        left = matches[index]
        right = matches[index + 1]
        if text[left.end() : right.start()].strip() != "":
            continue
        phrase = f"{left.group(0).lower()}_{right.group(0).lower()}"
        if len(phrase) >= 2:
            aliases.add(phrase)
    return aliases


def build_procedure_rule_schema(
    summary: str,
    tool_requirement: str | None = None,
    steps: list[str] | None = None,
    rule_schema: ProcedureRuleSchema | dict[str, list[str]] | None = None,
) -> ProcedureRuleSchema:
    """汇总显式元数据和文本约束，生成 procedure 规则结构。"""

    # 1. 合并已有规则和文本中提到的工具。
    actual_steps = [] if steps is None else steps
    required = set(rule_schema.get("required_tools", []) if rule_schema else [])
    forbidden = set(rule_schema.get("forbidden_tools", []) if rule_schema else [])
    mentioned = set(rule_schema.get("mentioned_tools", []) if rule_schema else [])
    mentioned.update(_extract_ascii_aliases(summary))
    for step in actual_steps:
        mentioned.update(_extract_ascii_aliases(step))

    # 2. 仅为缺失的约束补充文本推断结果。
    if not required or not forbidden:
        inferred_required, inferred_forbidden = _infer_rule_constraints(summary, actual_steps)
        if not required:
            required.update(inferred_required)
        if not forbidden:
            forbidden.update(inferred_forbidden)

    # 3. 显式工具要求优先，并消除自相矛盾的禁用项。
    if tool_requirement:
        normalized = tool_requirement.strip().lower()
        if normalized:
            required.add(normalized)
            mentioned.add(normalized)
    forbidden.difference_update(required)
    return {
        "required_tools": sorted(required),
        "forbidden_tools": sorted(forbidden),
        "mentioned_tools": sorted(mentioned),
    }


def resolve_procedure_rule_schema(
    summary: str,
    extra: Mapping[str, object] | None,
) -> ProcedureRuleSchema:
    """校验持久化 procedure 元数据并生成统一规则结构。"""

    # 1. 在元数据消费边界解析三个 procedure 字段。
    payload: Mapping[str, object] = {} if extra is None else extra
    tool_requirement = parse_procedure_tool_requirement(payload.get("tool_requirement"))
    steps = parse_procedure_steps(
        payload.get("steps", []),
        context="procedure metadata steps",
    )
    rule_schema = _parse_rule_schema(payload.get("rule_schema"))

    # 2. 边界后只把已验证类型交给规则构造器。
    return build_procedure_rule_schema(
        summary=summary,
        tool_requirement=tool_requirement,
        steps=steps,
        rule_schema=rule_schema,
    )


def procedure_rules_conflict(
    new_schema: ProcedureRuleSchema,
    old_schema: ProcedureRuleSchema,
) -> bool:
    new_terms = _schema_terms(new_schema)
    old_terms = _schema_terms(old_schema)
    if not new_terms or not old_terms or not (new_terms & old_terms):
        return False
    new_required = set(new_schema["required_tools"])
    new_forbidden = set(new_schema["forbidden_tools"])
    old_required = set(old_schema["required_tools"])
    old_forbidden = set(old_schema["forbidden_tools"])
    return bool((new_required & old_forbidden) or (new_forbidden & old_required))


def parse_procedure_steps(
    value: object,
    *,
    context: str = "procedure steps",
) -> list[str]:
    """校验 procedure 步骤为非空字符串列表。"""
    if not isinstance(value, list):
        raise TypeError(f"{context} 必须是字符串数组")
    steps: list[str] = []
    for index, step in enumerate(cast(list[object], value)):
        if not isinstance(step, str):
            raise TypeError(f"{context}[{index}] 必须是字符串")
        if not step.strip():
            raise ValueError(f"{context}[{index}] 不能为空")
        steps.append(step)
    return steps


def parse_procedure_tool_requirement(value: object) -> str | None:
    """校验 procedure 的工具要求，只允许字符串或明确缺省。"""
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError("procedure metadata tool_requirement 必须是字符串或 null")
    return value


def _parse_rule_schema(value: object) -> ProcedureRuleSchema | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TypeError("procedure metadata rule_schema 必须是 object")
    schema = cast(Mapping[object, object], value)
    return {
        "required_tools": _parse_schema_list(schema, "required_tools"),
        "forbidden_tools": _parse_schema_list(schema, "forbidden_tools"),
        "mentioned_tools": _parse_schema_list(schema, "mentioned_tools"),
    }


def _parse_schema_list(schema: Mapping[object, object], field: str) -> list[str]:
    if field not in schema:
        return []
    value = schema[field]
    if not isinstance(value, list):
        raise TypeError(f"rule_schema.{field} 必须是字符串数组")
    normalized: set[str] = set()
    for index, item in enumerate(cast(list[object], value)):
        if not isinstance(item, str):
            raise TypeError(f"rule_schema.{field}[{index}] 必须是字符串")
        token = item.strip().lower()
        if not token:
            raise ValueError(f"rule_schema.{field}[{index}] 不能为空")
        normalized.add(token)
    return sorted(normalized)


def _schema_terms(schema: ProcedureRuleSchema) -> set[str]:
    return (
        set(schema["mentioned_tools"])
        | set(schema["required_tools"])
        | set(schema["forbidden_tools"])
    )


def _infer_rule_constraints(
    summary: str,
    steps: list[str],
) -> tuple[set[str], set[str]]:
    required: set[str] = set()
    forbidden: set[str] = set()
    for text in [summary, *steps]:
        for clause in re.split(r"[，。！？；;\n]", text):
            for alias, prefix in _iter_alias_prefixes(clause):
                if any(prefix.endswith(cue) for cue in _NEGATIVE_TOOL_PREFIXES):
                    forbidden.add(alias)
                    continue
                if any(prefix.endswith(cue) for cue in _POSITIVE_TOOL_PREFIXES):
                    required.add(alias)
    return required, forbidden


def _iter_alias_prefixes(clause: str) -> list[tuple[str, str]]:
    matches = list(_ASCII_ALIAS_PATTERN.finditer(clause))
    pairs: list[tuple[str, str]] = []
    index = 0
    while index < len(matches):
        match = matches[index]
        prefix = _normalize_prefix(clause[max(0, match.start() - 12) : match.start()])
        if index < len(matches) - 1:
            next_match = matches[index + 1]
            if clause[match.end() : next_match.start()].strip() == "":
                alias = f"{match.group(0).lower()}_{next_match.group(0).lower()}"
                pairs.append((alias, prefix))
                index += 2
                continue
        pairs.append((match.group(0).lower(), prefix))
        index += 1
    return pairs


def _normalize_prefix(text: str) -> str:
    return re.sub(r"\s+", "", text)

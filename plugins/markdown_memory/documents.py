"""Markdown 记忆的文件模型：SELF、MEMORY、topics/ 与按 scope 归属的文件夹，以及提炼操作的校验和叠加。"""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
import math
import re
from typing import Literal, cast, get_args

MEMORY = "MEMORY.md"
SELF = "SELF.md"
TOPICS = "topics"
SCOPE_FILE = "_scope.md"
MEMORY_HEADINGS = ("# 用户长期记忆", "## 用户事实", "## 用户偏好", "## 用户明确要求长期记住的关键内容")
SELF_HEADINGS = ("# Akashic 的自我认知", "## 人格与形象", "## 我对当前用户的理解", "## 我们关系的定义")
# 这些章节记录用户本人或关系判断，只能以用户原话为证据。
USER_SECTIONS: dict[str, frozenset[str]] = {MEMORY: frozenset(MEMORY_HEADINGS[1:]), SELF: frozenset(SELF_HEADINGS[2:])}
USER_HEADINGS: dict[str, tuple[str, ...]] = {MEMORY: MEMORY_HEADINGS[1:], SELF: SELF_HEADINGS[1:]}
MAX_LINE_CHARS = 120
MAX_DESCRIPTION_CHARS = 80
_FOLDER = re.compile(r"[^\s/\\._][^/\\]{0,39}")
_TOPIC_FILE = re.compile(r"[a-z0-9][a-z0-9-]{0,63}\.md")

OpName = Literal["add", "supersede", "retract", "create", "create_folder", "merge", "prune"]
_FIELDS: dict[str, frozenset[str]] = {
    "add": frozenset({"op", "doc", "line", "message_ids", "section"}),
    "supersede": frozenset({"op", "doc", "old", "line", "message_ids"}),
    "retract": frozenset({"op", "doc", "old", "message_ids"}),
    "create": frozenset({"op", "doc", "description", "line", "message_ids"}),
    "create_folder": frozenset({"op", "folder", "description"}),
    "merge": frozenset({"op", "doc", "olds", "line"}),
    "prune": frozenset({"op", "doc", "old", "line"}),
}


class InvalidDraft(ValueError):
    """模型草稿违反档案合同；原文或持久状态错误不属于模型重试。"""


@dataclass(frozen=True, slots=True)
class Op:
    op: OpName
    doc: str
    line: str = ""
    old: str = ""
    section: str = ""
    description: str = ""
    message_ids: tuple[str, ...] = ()
    olds: tuple[str, ...] = ()

    # 只写出该操作类型的字段，receipt 重放时与模型输出走同一解析。
    def to_json(self) -> dict[str, object]:
        values: dict[str, object] = {key: value for key, value in asdict(self).items() if key in _FIELDS[self.op]}
        if self.op == "create_folder":
            values["folder"] = self.doc.split("/", 1)[1]
        elif self.op == "merge":
            values["olds"] = list(self.olds)
        elif self.op == "prune":
            if not self.line:
                del values["line"]
        else:
            values["message_ids"] = list(self.message_ids)
        if self.op == "add" and not self.section:
            del values["section"]
        return values


# 解析模型或 receipt 中的操作；字段集合按操作类型严格匹配。
def parse_ops(raw: object) -> tuple[Op, ...]:
    if not isinstance(raw, dict) or set(cast(dict[str, object], raw)) != {"ops"}:
        raise InvalidDraft("模型必须只返回 {\"ops\": [...]}")
    items = cast(dict[str, object], raw)["ops"]
    if not isinstance(items, list):
        raise InvalidDraft("ops 必须是数组")
    return tuple(_parse_op(item) for item in cast(list[object], items))


def _parse_op(item: object) -> Op:
    if not isinstance(item, dict):
        raise InvalidDraft("每个操作必须是 JSON object")
    data = cast(dict[str, object], item)
    name = data.get("op")
    if name not in get_args(OpName):
        raise InvalidDraft(f"未知操作: {name!r}")
    allowed = _FIELDS[cast(str, name)]
    required = allowed - ({"section"} if name == "add" else {"line"} if name == "prune" else set())
    if not required <= set(data) <= allowed:
        raise InvalidDraft(f"{name} 操作字段必须是 {sorted(allowed)}")
    values = {key: value for key, value in data.items() if key not in {"op", "message_ids", "folder", "olds"}}
    if any(not isinstance(value, str) for value in values.values()):
        raise InvalidDraft(f"{name} 操作的文本字段必须是字符串")
    # 条目在操作边界校验，不能靠文档差异判断：空行或已有标题也可能是非法替换结果。
    if "line" in values:
        line = cast(str, values["line"])
        if (not line.startswith("- ") or not line[2:].strip() or line.splitlines() != [line]
                or len(line) > MAX_LINE_CHARS or "```" in line):
            raise InvalidDraft(f"新条目必须是不超过 {MAX_LINE_CHARS} 字的非空单行 '- ' 条目: {line[:40]}")
    if "description" in values:
        description = cast(str, values["description"])
        if (not description.strip() or len(description) > MAX_DESCRIPTION_CHARS
                or description.splitlines() != [description]):
            raise InvalidDraft(f"description 必须是不超过 {MAX_DESCRIPTION_CHARS} 字的非空单行文本")
    olds = data.get("olds", [])
    if name == "merge" and (not isinstance(olds, list) or len(cast(list[object], olds)) < 2
                            or any(not isinstance(old, str) for old in cast(list[object], olds))):
        raise InvalidDraft("merge 的 olds 必须是至少两条旧条目原文")
    ids = data.get("message_ids", [])
    if name not in {"create_folder", "merge", "prune"} and (not isinstance(ids, list) or not ids
                                     or any(not isinstance(key, str) for key in cast(list[object], ids))):
        raise InvalidDraft("message_ids 必须是非空消息 ID 数组")
    if name == "create_folder":
        folder = data["folder"]
        if not isinstance(folder, str):
            raise InvalidDraft("folder 必须是字符串")
        values["doc"] = f"{TOPICS}/{folder}"
    return Op(cast(OpName, name), message_ids=tuple(cast(list[str], ids)), olds=tuple(cast(list[str], olds)),
              **cast(dict[str, str], values))


def split_frontmatter(text: str) -> tuple[dict[str, str], list[str]]:
    lines = text.splitlines()
    if lines[:1] != ["---"] or "---" not in lines[1:]:
        return {}, lines
    end = lines.index("---", 1)
    fields = {key.strip(): value.strip() for key, sep, value in
              (row.partition(":") for row in lines[1:end]) if sep}
    return fields, lines[end + 1:]


def scope_text(scope: str, description: str) -> str:
    return f"---\nscope: {scope}\ndescription: {description}\n---\n"


MEMORY_TEMPLATE = "\n\n".join(MEMORY_HEADINGS) + "\n"


def topic_text(description: str, line: str) -> str:
    return f"---\ndescription: {description}\n---\n{line}\n"


# 当前会话可写的路径：两份档案、全局主题和本 scope 文件夹内的主题。
def writable(doc: str, folder: str | None) -> bool:
    parts = doc.split("/")
    if doc in {MEMORY, SELF}:
        return True
    if parts[0] != TOPICS or not _TOPIC_FILE.fullmatch(parts[-1]):
        return False
    return len(parts) == 2 or (len(parts) == 3 and folder is not None and parts[1] == folder)


@dataclass(slots=True)
class Workspace:
    """一次提炼可见的文本副本；folders 为文件夹名 → scope 宽键。"""

    texts: dict[str, str]
    folders: dict[str, str]
    scope: str | None
    # 重放时前提失效而跳过的操作；与"已应用"区分，写入 applied receipt。
    conflicts: list[str] = field(default_factory=lambda: [])

    def folder(self) -> str | None:
        return next((name for name, key in self.folders.items() if key == self.scope), None)

    # 叠加操作；strict 校验模型给出的原始目标与前提，否则按已验证的 scope 重放并收敛（供 rebase 与崩溃恢复）。
    def apply(self, ops: tuple[Op, ...], *, strict: bool) -> None:
        for index, op in enumerate(ops):
            if op.op == "create_folder":
                self._create_folder(op, strict)
                continue
            doc = op.doc if strict else self._replay_target(op.doc)
            if doc is None:
                self.conflicts.append(f"#{index} {op.op} {op.doc}: 当前 scope 没有文件夹")
                continue
            if strict and not writable(doc, self.folder()):
                raise InvalidDraft(f"不可写入的路径: {op.doc}")
            if op.op == "create" and doc not in self.texts:
                self.texts[doc] = topic_text(op.description, op.line)
            elif op.op in {"add", "create"}:
                self._add(doc, op, strict, index)
            elif op.op == "merge":
                self._merge(doc, op, strict, index)
            else:
                self._replace(doc, op, strict, index)

    def _replay_target(self, doc: str) -> str | None:
        # 校验时已确认三段路径只指向本 scope；重放按 scope 找实际文件夹，文件夹名只负责定位。
        parts = doc.split("/")
        if len(parts) != 3:
            return doc
        folder = self.folder()
        return None if folder is None else "/".join((parts[0], folder, parts[2]))

    def _create_folder(self, op: Op, strict: bool) -> None:
        name = op.doc.split("/", 1)[1]
        if strict and (self.scope is None or self.folder() is not None
                       or name in self.folders or not _FOLDER.fullmatch(name)):
            raise InvalidDraft("只能为尚无文件夹的当前 scope 新建一个未占用的合法文件夹")
        if self.scope is None or self.folder() is not None:
            return
        while name in self.folders:
            name += "-2"
        self.folders[name] = self.scope
        self.texts[f"{TOPICS}/{name}/{SCOPE_FILE}"] = scope_text(self.scope, op.description)

    def _add(self, doc: str, op: Op, strict: bool, index: int) -> None:
        text = self.texts.get(doc, "")
        if op.line in text.splitlines():
            if strict:
                raise InvalidDraft(f"重复已有条目: {op.line}")
            return
        if doc == MEMORY and not text.strip():
            text = MEMORY_TEMPLATE
        if strict and doc in USER_SECTIONS and op.section not in USER_HEADINGS[doc]:
            raise InvalidDraft(f"{doc} 的 section 必须是 {list(USER_HEADINGS[doc])} 之一")
        if doc in USER_SECTIONS:
            self.texts[doc] = append_section_lines(text, op.section, [op.line])
        elif doc in self.texts:
            self.texts[doc] = text.rstrip("\n") + "\n" + op.line + "\n"
        elif strict:
            raise InvalidDraft(f"主题不存在，应使用 create: {doc}")
        else:
            self.conflicts.append(f"#{index} add {doc}: 主题文件已不存在")

    def _replace(self, doc: str, op: Op, strict: bool, index: int) -> None:
        lines = self.texts.get(doc, "").splitlines()
        if op.old not in lines:
            if strict:
                raise InvalidDraft(f"旧条目不存在: {op.old}")
            # retract 的结果就是旧条目不在；supersede 只有新条目已在才算已应用。
            if (op.op == "supersede" or (op.op == "prune" and op.line)) and op.line not in lines:
                self.conflicts.append(f"#{index} {op.op} {doc}: 旧条目与新条目都不在")
            return
        index = lines.index(op.old)
        remove = op.op == "retract" or (op.op == "prune" and not op.line) or op.line in lines
        lines[index:index + 1] = [] if remove else [op.line]
        self.texts[doc] = "\n".join(lines) + "\n"


    def _merge(self, doc: str, op: Op, strict: bool, index: int) -> None:
        # 只有全部旧条目仍在才合并；旧条目全不在且结果已在算已应用，其余视为前提失效。
        lines = self.texts.get(doc, "").splitlines()
        present = {line for line in lines if line in op.olds}
        if len(set(op.olds)) == len(op.olds) and len(present) == len(op.olds):
            first = next(position for position, line in enumerate(lines) if line in present)
            keep = op.line not in [line for line in lines if line not in present]
            lines = [row for position, line in enumerate(lines)
                     for row in (([op.line] if keep else []) if position == first
                                 else [] if line in present else [line])]
            self.texts[doc] = "\n".join(lines) + "\n"
        elif strict:
            raise InvalidDraft(f"merge 的旧条目必须各不相同且都在 {doc} 中")
        elif present or op.line not in lines:
            self.conflicts.append(f"#{index} merge {doc}: 旧条目已部分或全部变化（现存 {len(present)}/{len(op.olds)}）")


def append_section_lines(content: str, heading: str, lines: list[str]) -> str:
    values = content.rstrip().splitlines()
    if heading not in values:
        values.extend(["", heading])
    start = values.index(heading) + 1
    end = next((index for index in range(start, len(values)) if values[index].startswith("#")), len(values))
    while end > start and not values[end - 1].strip():
        end -= 1
    values[end:end] = [line for line in lines if line not in values]
    return "\n".join(values) + "\n"


def section_of(text: str, line: str) -> str:
    heading = ""
    for row in text.splitlines():
        if row.startswith("#"):
            heading = row.strip()
        elif row == line:
            return heading
    return ""


def _section_sizes(text: str) -> dict[str, int]:
    sizes: dict[str, int] = {}
    heading = ""
    for row in text.splitlines():
        if row.startswith("#"):
            heading = row.strip()
        elif row.startswith("- "):
            sizes[heading] = sizes.get(heading, 0) + len(row)
    return sizes


# 只限制本次增加的用量；既有手写内容超额时仍允许减少。
def check_budgets(before: Mapping[str, str], after: Mapping[str, str],
                  *, memory_budget: int, self_budgets: Mapping[str, int]) -> None:
    for doc, text in after.items():
        old = before.get(doc, "")
        if text == old:
            continue
        if doc in USER_SECTIONS:
            _check_budget(doc, old, text, memory_budget, self_budgets)


def budget_usage(doc: str, text: str, memory_budget: int,
                 self_budgets: Mapping[str, int]) -> dict[str, tuple[int, int]]:
    """返回受管理章节的实际字数与配额；手写的其他章节不占配额。"""
    sizes = _section_sizes(text)
    if doc == MEMORY:
        return {"": (sum(sizes.get(heading, 0) for heading in MEMORY_HEADINGS[1:]), memory_budget)}
    return {heading: (sizes.get(heading, 0), budget) for heading, budget in self_budgets.items()}


def _check_budget(doc: str, old: str, text: str, memory_budget: int, self_budgets: Mapping[str, int]) -> None:
    before = budget_usage(doc, old, memory_budget, self_budgets)
    for heading, (size, budget) in budget_usage(doc, text, memory_budget, self_budgets).items():
        if size > budget and size > before[heading][0]:
            raise InvalidDraft(f"{doc} {heading} 超出 {budget} 字配额（当前 {size} 字）")


PINNED_CHARS = 4000


# 常驻记忆：使用规则、两份档案、当前 scope 与全局主题的 pinned 全文和自动索引、其他文件夹清单。
def render_memory(root: str, workspace: Workspace, others: Mapping[str, str]) -> str:
    folder = workspace.folder()
    parts = [_usage(root, workspace.scope, folder)]
    for title, doc in (("Akashic 自我认知", SELF), ("用户长期记忆", MEMORY)):
        if text := workspace.texts.get(doc, "").strip():
            parts.append(f"## {title}（{doc}）\n\n{text}")
    if folder is not None:
        parts.append(f"## 当前项目记忆（{TOPICS}/{folder}/）\n\n"
                     + (_topics(workspace.texts, f"{TOPICS}/{folder}/") or "（暂无主题）"))
    if global_topics := _topics(workspace.texts, f"{TOPICS}/"):
        parts.append(f"## 全局主题（{TOPICS}/）\n\n{global_topics}")
    if others:
        names = "；".join(f"{name}：{description}" for name, description in sorted(others.items())[:20])
        parts.append(f"## 其他项目记忆（未加载，需要时用 rg 搜索）\n{len(others)} 个文件夹：{names}")
    return "\n\n".join(parts)


def _topics(texts: Mapping[str, str], prefix: str) -> str:
    pinned: list[str] = []
    index: list[str] = []
    budget = PINNED_CHARS
    for doc in sorted(texts):
        rest = doc.removeprefix(prefix)
        if not doc.startswith(prefix) or "/" in rest or rest == SCOPE_FILE:
            continue
        fields, body = split_frontmatter(texts[doc])
        content = "\n".join(body).strip()
        description = fields.get("description") or "（无 description）"
        if fields.get("pinned", "").lower() == "true" and content and budget > 0:
            # 配额只计正文；截断提示另附，剩余额度不会为负。
            shown = content[:budget]
            budget -= len(shown)
            if len(shown) < len(content):
                shown += f"\n（已截断：pinned 正文超过 {PINNED_CHARS} 字配额，完整内容用 read_file 读取，或精简、取消 pinned）"
            pinned.append(f"### {doc}\n{shown}")
        elif fields.get("pinned", "").lower() == "true" and content:
            index.append(f"- {doc} — {description}（pinned，但配额已用尽，未展开）")
        else:
            index.append(f"- {doc} — {description}")
    return "\n\n".join([*pinned, *(["索引：\n" + "\n".join(index)] if index else [])])


def _usage(root: str, scope: str | None, folder: str | None) -> str:
    if scope is None:
        where = "当前会话不属于任何项目：项目相关内容写进全局主题。"
    elif folder is None:
        where = (f"当前项目（scope {scope}）还没有记忆文件夹：需要时自取一个名字，先写 "
                 f"{TOPICS}/<名字>/{SCOPE_FILE}，内容为 `---` / `scope: {scope}` / `description: 一句话` / `---`。")
    else:
        where = f"当前项目文件夹是 {TOPICS}/{folder}/。"
    return f"""## 长期记忆

记忆目录：{root}。下面是会话开始（或上次上下文压缩）时的快照；你之后写入的内容已经生效，只是不会出现在这里。

读：这里只有档案、pinned 主题和索引。需要某个主题时用 read_file 读完整文件；找不到或要看其他项目时，用 shell 在记忆目录执行 rg。记忆可能过时：涉及路径、版本、配置和当前状态的条目，先对照实际文件或命令核实再依赖。

写：直接用 write_file / edit_file 编辑文件。用户要求记住时立即写；要求忘记时删除对应条目。后台也会在上下文压缩后自动提炼，不必为了保险重复写。
- 用户本人的身份、偏好、明确要求 → MEMORY.md 对应标题下。
- 用户对你工作方式的纠正或确认 → 跨项目适用的写全局主题，只对本项目适用的写项目文件夹；写明原因。
- 项目的约定、决策、背景与外部资源位置 → 项目文件夹的主题文件。{where}
- 已有的记忆文件只用 edit_file 修改或追加，write_file 只用来新建文件，避免整文件覆盖掉后台刚写入的条目。
- 每条是一行 `- ` 文本；新主题文件以 `---` / `description: 一句话` / `---` 开头；先更新已有主题和条目，不追加相互矛盾的条目。
- 在 frontmatter 加 `pinned: true` 会让该主题常驻，只用于每次都需要的少量内容。
- 不要保存：能从仓库、配置或命令直接读到的内容，临时任务状态，调试步骤，git 历史。"""


_WORD = re.compile(r"[a-z0-9_]+|[\u4e00-\u9fff]+")


# ASCII 词与中文二元组；不依赖分词器，足以对短主题文件排序。
def terms(text: str) -> list[str]:
    result: list[str] = []
    for token in _WORD.findall(text.lower()):
        if token[0].isascii():
            result.append(token)
        else:
            result.extend(token[index:index + 2] for index in range(max(len(token) - 1, 1)))
    return result


def rank(query: str, docs: Mapping[str, str], *, k1: float = 1.2, b: float = 0.75) -> list[tuple[str, float]]:
    bags = {doc: Counter(terms(text)) for doc, text in docs.items()}
    if not bags:
        return []
    average = sum(sum(bag.values()) for bag in bags.values()) / len(bags) or 1.0
    scores: dict[str, float] = {}
    for term in set(terms(query)):
        hits = [doc for doc, bag in bags.items() if term in bag]
        idf = math.log(1 + (len(bags) - len(hits) + 0.5) / (len(hits) + 0.5))
        for doc in hits:
            bag = bags[doc]
            frequency = bag[term]
            size = sum(bag.values())
            scores[doc] = scores.get(doc, 0.0) + idf * frequency * (k1 + 1) / (
                frequency + k1 * (1 - b + b * size / average))
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


def surfaceable(texts: Mapping[str, str], folder: str | None) -> dict[str, str]:
    prefixes = (f"{TOPICS}/", *((f"{TOPICS}/{folder}/",) if folder is not None else ()))
    result: dict[str, str] = {}
    for doc, text in texts.items():
        if not any(doc.startswith(prefix) and "/" not in doc.removeprefix(prefix) for prefix in prefixes):
            continue
        fields, body = split_frontmatter(text)
        if doc.endswith(SCOPE_FILE) or fields.get("pinned", "").lower() == "true":
            continue
        result[doc] = fields.get("description", "") + "\n" + "\n".join(body)
    return result

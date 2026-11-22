"""首次安装的扩展包选择：全部 / 最小 / 自选组；有人在操作终端才提问，参数可覆盖。只依赖标准库与 Core 的 index 读取，宿主脚本共用。

选择在提问后先落成"首装意图"记录（状态目录里，原子写），之后写 patch、预取、ensure 都从它推导；失败重跑沿用它，不再提问。
"""
from __future__ import annotations

import json
import os
import sys
import re
import tempfile
import tomllib
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import TextIO, cast

from agent.plugins.package_index import PackageIndex, closure, load_index
from scripts.akashic_release.extra import MARKETPLACE, provided_ids, _read_rows  # pyright: ignore[reportPrivateUsage]

PROFILES = ("full", "minimal", "custom")
INTENT_FILE = "first-install.json"


@dataclass(frozen=True)
class Choice:
    profile: str
    # 要从 extra 装上的全部包（manifest 名）。
    install: tuple[str, ...]
    # 要关掉的预设行（行 id）与要启用的非预设包（插件身份）。
    disable_rows: tuple[str, ...]
    enable_ids: tuple[str, ...]
    # 自选时因依赖自动带上的包：(包, 需要它的包)；只用于提示。
    added: tuple[tuple[str, str], ...] = ()
    groups: tuple[str, ...] = ()
    # 首装自己写进 bundle.patch.toml 的行：(行键, 插件身份, disabled)；重跑只替换这些行，其余（用户手写的、用户改过的）原样保留。
    written: tuple[tuple[str, str, bool], ...] = ()


def _megabytes(size: int) -> str:
    return f"{size / 1048576:.1f} MB"


# 有人在操作 = stdin 与 stdout 都是终端；容器、systemd、CI、管道里都不是。
def interactive() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty()


# 预设里来自 extra 的行（行 id 到插件身份）：发行版自己不提供的那些。
def preset_extra_rows(distribution: Path, mode: str) -> dict[str, str]:
    rows = _read_rows(distribution / "bundles/base.toml")
    if mode != "base":
        rows.update(_read_rows(distribution / "bundles" / f"{mode}.toml"))
    provided = provided_ids(distribution)
    return {row_id: str(row["plugin"]) for row_id, row in rows.items() if str(row["plugin"]) not in provided}


# 反复询问直到得到合法答案；EOF 当作取消。
def _ask_until(ask: Callable[[str], str], prompt: str, parse: Callable[[str], object], out: TextIO) -> object:
    while True:
        try:
            answer = ask(prompt)
        except EOFError as error:
            raise RuntimeError("已取消，未做任何改动") from error
        try:
            return parse(answer.strip())
        except ValueError as error:
            print(f"  {error}", file=out)


def _parse_profile(answer: str) -> str:
    choices = {"": "full", "1": "full", "2": "minimal", "3": "custom"}
    if answer not in choices:
        raise ValueError("请输入 1、2 或 3（回车=1）")
    return choices[answer]


def _parse_numbers(answer: str, count: int) -> list[int]:
    if not answer:
        return list(range(1, count + 1))
    numbers: list[int] = []
    for item in answer.replace("，", ",").split(","):
        if not item.strip():
            continue
        if not item.strip().isdigit() or not 1 <= int(item) <= count:
            raise ValueError(f"编号 {item.strip()!r} 无效，请输入 1 到 {count} 之间的编号（逗号分隔，回车=全部）")
        numbers.append(int(item))
    if not numbers:
        raise ValueError("没有有效的编号")
    return numbers


# 全部/最小/自选三选一；回车 = 全部。非法输入重新询问。
def ask_profile(index: PackageIndex | None, ask: Callable[[str], str], out: TextIO) -> tuple[str, list[str]]:
    total = 0 if index is None else sum(item.size_bytes for item in index.packages.values())
    count = 0 if index is None else len(index.packages)
    print("Akashic 首次安装。选择要安装的功能：", file=out)
    print(f"  [1] 全部（推荐）  {count} 个扩展包，约 {_megabytes(total)}", file=out)
    print("  [2] 最小          只装核心，扩展包以后用 pkg install 按需安装", file=out)
    print("  [3] 自选          按组选择", file=out)
    profile = cast(str, _ask_until(ask, "选择 [1]: ", _parse_profile, out))
    if profile != "custom":
        return profile, []
    assert index is not None
    groups = list(index.groups)
    for number, group_id in enumerate(groups, 1):
        group = index.groups[group_id]
        size = sum(index.packages[name].size_bytes for name in group.members)
        print(f"  {number:>2}  {group.title:<14} {len(group.members)} 个包  {_megabytes(size):>8}  {group.description}", file=out)
    numbers = cast(list[int], _ask_until(ask, "输入编号（逗号分隔，回车=全部）: ", lambda text: _parse_numbers(text, len(groups)), out))
    return "custom", [groups[number - 1] for number in numbers]


# 按档位算出结果。preset_rows 是预设里来自 extra 的行（行 id 到插件身份）。最小档不需要 index。
def decide(
    index: PackageIndex | None, preset_rows: dict[str, str], profile: str, groups: Iterable[str],
    absorbed: frozenset[str] = frozenset(),
) -> Choice:
    preset_names = {identity.partition("@")[0]: row for row, identity in preset_rows.items()}
    if profile == "minimal":
        return Choice("minimal", (), tuple(sorted(preset_rows)), ())
    if index is None:
        raise RuntimeError("需要读取扩展包组信息才能确定要安装的包，但现在取不到；联网后重试，或选择最小安装（--profile minimal）")
    groups = tuple(groups)
    added: list[tuple[str, str]] = []
    if profile == "full":
        names = list(index.packages)
    else:
        unknown = [group for group in groups if group not in index.groups]
        if unknown:
            raise RuntimeError("没有这些包组: " + ", ".join(unknown))
        requested = [name for group in groups for name in index.groups[group].members if name not in absorbed]
        names = closure(index, requested)
        added = [(name, next(owner for owner in names if name in index.packages[owner].requires))
                 for name in names if name not in requested]
    # 已被 Core 吸收的包（index 里可能仍列着）不装，也不出现在清单里。
    names = [name for name in names if name not in absorbed]
    selected = set(names)
    return Choice(profile, tuple(names), tuple(sorted(row for name, row in preset_names.items() if name not in selected)),
                  tuple(f"{name}@{MARKETPLACE}" for name in names if name not in preset_names), tuple(added), groups)


# 这个选择要写进用户 patch 的行：{行键: (插件身份, disabled)}；未选的预设行关掉，非预设的已选包启用。
def desired_rows(choice: Choice, preset_rows: dict[str, str]) -> dict[str, tuple[str, bool]]:
    rows = {row: (preset_rows[row], True) for row in choice.disable_rows}
    rows.update({identity: (identity, False) for identity in choice.enable_ids})
    return rows


def _row_text(key: str, plugin: str, disabled: bool) -> str:
    return f'[rows."{key}"]\nplugin = "{plugin}"\ndisabled = {"true" if disabled else "false"}\n'


def _current_rows(text: str, path: Path) -> dict[str, dict[str, object]]:
    try:
        rows = tomllib.loads(text).get("rows", {})
    except tomllib.TOMLDecodeError as error:
        raise RuntimeError(f"{path} 不是有效的 TOML（{error}）；修正后重试，首装没有改动它") from error
    return cast(dict[str, dict[str, object]], rows) if isinstance(rows, dict) else {}


# 删掉 [rows."键"] 表（到下一个表头为止）；只认首装自己写的这种写法。表头前紧挨着的注释属于下一张表，保留。
def _drop_tables(text: str, keys: set[str]) -> str:
    kept: list[str] = []
    held: list[str] = []
    dropping = False
    for line in text.splitlines(keepends=True):
        header = re.match(r'^\s*\[(.*)\]\s*$', line)
        if header is not None:
            match = re.fullmatch(r'rows\."(.*)"', header.group(1))
            dropping = match is not None and match.group(1) in keys
            if not dropping:
                kept.extend(held)
            held = []
        elif dropping and (line.strip() == "" or line.lstrip().startswith("#")):
            held.append(line)
            continue
        else:
            held = []
        if not dropping:
            kept.append(line)
    return "".join(kept)


# 合并写：只替换"首装上次写的、且内容没被改过"的行；用户手写的行、被用户改过的首装行都保留并提示；
# 返回新文本与这次真正由首装拥有的行。
def merge_patch(existing: str | None, choice: Choice, preset_rows: dict[str, str], path: Path, out: TextIO) -> tuple[str, tuple[tuple[str, str, bool], ...]]:
    wanted = desired_rows(choice, preset_rows)
    previous = {key: (plugin, disabled) for key, plugin, disabled in choice.written}
    current = _current_rows(existing, path) if existing else {}
    replace_keys: set[str] = set()
    for key, value in previous.items():
        row = current.get(key)
        if row is None:
            print(f"  {path.name} 里的 {key} 已被你删除，保留你的修改", file=out)
        elif (row.get("plugin"), row.get("disabled")) == value and set(row) <= {"plugin", "disabled"}:
            replace_keys.add(key)
        else:
            print(f"  {path.name} 里的 {key} 已被你修改，保留你的版本", file=out)
    mine = {key: value for key, value in wanted.items() if key in replace_keys or (key not in previous and key not in current)}
    for key, value in wanted.items():
        if key not in previous and key in current and (current[key].get("plugin"), current[key].get("disabled")) != value:
            print(f"  {path.name} 里已有你手写的 {key}，保留你的版本", file=out)
    text = (existing or "schema_version = 1\n")
    text = _drop_tables(text, replace_keys)
    text += ("\n" if text and not text.endswith("\n\n") else "") + "\n".join(_row_text(key, *value) for key, value in mine.items())
    left = _current_rows(text, path)
    if any(key in left for key in replace_keys - set(mine)):
        raise RuntimeError(f"{path} 里首装写的行格式被改动，无法安全替换；手工整理 {', '.join(sorted(replace_keys))} 后重试")
    return text, tuple((key, *value) for key, value in mine.items())


# 按意图写 patch（start 与 release 共用）：合并写入，并把"首装拥有的行"记回意图记录。返回新的意图。
def write_patch(state: Path, choice: Choice, preset_rows: dict[str, str], patch: Path, out: TextIO = sys.stdout) -> Choice:
    existing = patch.read_text(encoding="utf-8") if patch.exists() else None
    text, written = merge_patch(existing, choice, preset_rows, patch, out)
    temporary = patch.with_name(patch.name + ".tmp")
    _ = temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, patch)
    updated = replace(choice, written=written)
    save_intent(state, updated)
    return updated


# 清单：先列包、数量与总大小；每包一行；自选时自动带上的依赖单独提示。
def print_listing(index: PackageIndex | None, choice: Choice, out: TextIO) -> None:
    if not choice.install:
        print("最小安装：只装核心，不安装扩展包。", file=out)
        return
    assert index is not None
    size = sum(index.packages[name].size_bytes for name in choice.install)
    print(f"将安装 {len(choice.install)} 个扩展包，总计 {_megabytes(size)}：", file=out)
    for name in choice.install:
        package = index.packages[name]
        notes = ([f"需要 {', '.join(package.needs)}（不满足则装上但保持关闭）"] if package.needs else [])
        print(f"  {name:<22} {_megabytes(package.size_bytes):>8}  {index.groups[package.group].title}" + (f"  [{'; '.join(notes)}]" if notes else ""), file=out)
    for name, owner in choice.added:
        print(f"将同时安装 {name}（{owner} 需要）", file=out)


# ---- 首装意图记录：提问后先落盘，之后所有步骤都从它推导；失败重跑沿用，成功后删除 ----

def _intent_path(state: Path) -> Path:
    return state / INTENT_FILE


def save_intent(state: Path, choice: Choice) -> None:
    path = _intent_path(state)
    descriptor, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=state)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            _ = stream.write(json.dumps(asdict(choice), ensure_ascii=False, indent=1))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def load_intent(state: Path) -> Choice | None:
    path = _intent_path(state)
    if not path.exists():
        return None
    try:
        raw = cast(dict[str, object], json.loads(path.read_text(encoding="utf-8")))
        return Choice(str(raw["profile"]), tuple(cast(list[str], raw["install"])), tuple(cast(list[str], raw["disable_rows"])),
                      tuple(cast(list[str], raw["enable_ids"])), tuple((a, b) for a, b in cast(list[list[str]], raw.get("added", []))),
                      tuple(cast(list[str], raw.get("groups", []))),
                      tuple((str(a), str(b), bool(c)) for a, b, c in cast(list[list[object]], raw.get("written", []))))
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise RuntimeError(f"首装意图记录 {path} 无法读取（{error}）；确认没有安装在进行后删除它，重新选择") from error


def clear_intent(state: Path) -> None:
    _intent_path(state).unlink(missing_ok=True)


# 已提交的选择（有 root_ref）：判断"是否首次安装"的唯一依据之一，与 ensure 的拒绝条件同一个判定。
def selection_committed(workspace: Path) -> bool:
    path = workspace / "runtime/plugin-stable.json"
    try:
        return bool(json.loads(path.read_text(encoding="utf-8")).get("root_ref"))
    except (OSError, ValueError, AttributeError):
        return False


# 入口：解析参数、必要时提问与确认。index 取不到时：最小档照常（不需要它），其余明确报错退出。
# 有未完成的首装意图且没有显式参数、没有 --reselect：沿用它，不提问；显式参数以参数为准。
def choose(
    *, repo: str, home: Path, preset_rows: dict[str, str], profile: str | None, groups: list[str],
    assume_yes: bool, state: Path, reselect: bool = False, absorbed: frozenset[str] = frozenset(),
    ask: Callable[[str], str] = input, out: TextIO = sys.stdout,
) -> Choice:
    pending = load_intent(state)
    explicit = profile is not None or bool(groups)
    if pending is not None and not reselect and not explicit:
        print(f"沿用上次的选择：{pending.profile}（{len(pending.install)} 个扩展包）；要重新选择请加 --reselect 或 --profile", file=out)
        return pending
    if groups and profile in (None, "custom"):
        profile = "custom"
    index: PackageIndex | None = None
    if profile != "minimal":
        try:
            index = load_index(repo, home)
        except RuntimeError as error:
            raise RuntimeError(f"{error}；取不到扩展包组信息，未做任何改动，联网后重试，或选择最小安装（--profile minimal）") from error
        # 首装只接受这一次取到的 index：用本机缓存的旧包组信息会把过期的包集合当成"全部"。
        if index.note:
            raise RuntimeError(f"取不到最新的包组信息（{index.note}）；未做任何改动，联网后重试，或选择最小安装（--profile minimal）")
    asked = False
    if profile is None:
        if interactive() and not assume_yes:
            profile, groups = ask_profile(index, ask, out)
            asked = True
        else:
            profile = "full"
    choice = decide(index, preset_rows, profile, groups, absorbed)
    if pending is not None:
        choice = replace(choice, written=pending.written)
    print_listing(index, choice, out)
    if asked and choice.install and ask("继续？[Y/n] ").strip().lower() in ("n", "no"):
        raise RuntimeError("已取消，未做任何改动")
    save_intent(state, choice)
    return choice

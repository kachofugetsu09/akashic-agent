"""首次安装的扩展包选择：全部 / 最小 / 自选组；有人在操作终端才提问，参数可覆盖。只依赖标准库与 Core 的 index 读取，宿主脚本共用。

选择在提问后先落成"首装意图"记录（状态目录里，原子写），之后写 patch、预取、ensure 都从它推导；失败重跑沿用它，不再提问。
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
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


# 用户 patch（与 set_plugin_choice 同形）：未选的预设行关掉，非预设的已选包启用。
def render_patch(choice: Choice, preset_rows: dict[str, str]) -> str:
    lines = ["schema_version = 1", "[rows]"]
    for row in choice.disable_rows:
        lines += [f'[rows."{row}"]', f'plugin = "{preset_rows[row]}"', "disabled = true", ""]
    for identity in choice.enable_ids:
        lines += [f'[rows."{identity}"]', f'plugin = "{identity}"', "disabled = false", ""]
    return "\n".join(lines)


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
                      tuple(cast(list[str], raw.get("groups", []))))
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
    asked = False
    if profile is None:
        if interactive() and not assume_yes:
            profile, groups = ask_profile(index, ask, out)
            asked = True
        else:
            profile = "full"
    choice = decide(index, preset_rows, profile, groups, absorbed)
    print_listing(index, choice, out)
    if asked and choice.install and ask("继续？[Y/n] ").strip().lower() in ("n", "no"):
        raise RuntimeError("已取消，未做任何改动")
    save_intent(state, choice)
    return choice

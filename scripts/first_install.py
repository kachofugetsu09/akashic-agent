"""首次安装的扩展包选择：全部 / 最小 / 自选组；有人在操作终端才提问，参数可覆盖。只依赖标准库与 Core 的 index 读取，宿主脚本共用。"""
from __future__ import annotations

import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

from agent.plugins.package_index import PackageIndex, closure, load_index
from scripts.akashic_release.extra import MARKETPLACE, provided_ids, _read_rows  # pyright: ignore[reportPrivateUsage]

PROFILES = ("full", "minimal", "custom")


@dataclass(frozen=True)
class Choice:
    profile: str
    # 要从 extra 装上的全部包（manifest 名）；None 表示没有 index，沿用预设（旧行为）。
    install: tuple[str, ...] | None
    # 要关掉的预设行（行 id）与要启用的非预设包（插件身份）。
    disable_rows: tuple[str, ...]
    enable_ids: tuple[str, ...]


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


# 全部/最小/自选三选一；回车 = 全部。
def ask_profile(index: PackageIndex | None, ask: Callable[[str], str], out: TextIO) -> tuple[str, list[str]]:
    total = 0 if index is None else sum(item.size_bytes for item in index.packages.values())
    count = 0 if index is None else len(index.packages)
    print("Akashic 首次安装。选择要安装的功能：", file=out)
    print(f"  [1] 全部（推荐）  {count} 个扩展包，约 {_megabytes(total)}", file=out)
    print("  [2] 最小          只装核心，扩展包以后用 pkg install 按需安装", file=out)
    print("  [3] 自选          按组选择", file=out)
    answer = ask("选择 [1]: ").strip() or "1"
    if answer == "2":
        return "minimal", []
    if answer != "3":
        return "full", []
    if index is None:
        raise RuntimeError("自选需要读取扩展包组信息，但现在取不到；联网后重试，或选择全部/最小")
    groups = list(index.groups)
    for number, group_id in enumerate(groups, 1):
        group = index.groups[group_id]
        size = sum(index.packages[name].size_bytes for name in group.members)
        print(f"  {number:>2}  {group.title:<14} {len(group.members)} 个包  {_megabytes(size):>8}  {group.description}", file=out)
    picked = ask("输入编号（逗号分隔，回车=全部）: ").strip()
    chosen = groups if not picked else [groups[int(item) - 1] for item in picked.replace("，", ",").split(",") if item.strip()]
    return "custom", chosen


# 按档位算出结果。preset_rows 是预设里来自 extra 的行（行 id 到插件身份）；index 为 None 表示取不到。
def decide(
    index: PackageIndex | None, preset_rows: dict[str, str], profile: str, groups: Iterable[str],
) -> Choice:
    preset_names = {identity.partition("@")[0]: row for row, identity in preset_rows.items()}
    if profile == "minimal":
        return Choice("minimal", (), tuple(sorted(preset_rows)), ())
    if profile == "full":
        if index is None:
            return Choice("full", None, (), ())
        names = list(index.packages)
    else:
        if index is None:
            raise RuntimeError("自选需要读取扩展包组信息，但现在取不到；联网后重试，或选择全部/最小")
        unknown = [group for group in groups if group not in index.groups]
        if unknown:
            raise RuntimeError("没有这些包组: " + ", ".join(unknown))
        names = closure(index, [name for group in groups for name in index.groups[group].members])
    selected = set(names)
    return Choice(profile, tuple(names), tuple(sorted(row for name, row in preset_names.items() if name not in selected)),
                  tuple(f"{name}@{MARKETPLACE}" for name in names if name not in preset_names))


# 用户 patch（与 set_plugin_choice 同形）：未选的预设行关掉，非预设的已选包启用。
def render_patch(choice: Choice, preset_rows: dict[str, str]) -> str:
    lines = ["schema_version = 1", "[rows]"]
    for row in choice.disable_rows:
        lines += [f'[rows."{row}"]', f'plugin = "{preset_rows[row]}"', "disabled = true", ""]
    for identity in choice.enable_ids:
        lines += [f'[rows."{identity}"]', f'plugin = "{identity}"', "disabled = false", ""]
    return "\n".join(lines)


# 清单：先列包、数量与总大小；每包一行（组、大小、依赖自动带上的标注）。
def print_listing(index: PackageIndex | None, choice: Choice, out: TextIO) -> None:
    if choice.install is None:
        print("将安装默认组合（取不到扩展包组信息，沿用预设）。", file=out)
        return
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


# 入口：解析参数、必要时提问与确认；取不到 index 时全部沿用预设，自选直接失败。
def choose(
    *, repo: str, home: Path, preset_rows: dict[str, str], profile: str | None, groups: list[str],
    assume_yes: bool, ask: Callable[[str], str] = input, out: TextIO = sys.stdout,
) -> Choice:
    try:
        index: PackageIndex | None = load_index(repo, home)
    except RuntimeError as error:
        index = None
        if profile == "custom" or groups:
            raise RuntimeError(f"{error}；自选需要组信息，联网后重试") from error
    if groups and profile in (None, "custom"):
        profile = "custom"
    asked = False
    if profile is None:
        if interactive() and not assume_yes:
            profile, groups = ask_profile(index, ask, out)
            asked = True
        else:
            profile = "full"
    choice = decide(index, preset_rows, profile, groups)
    print_listing(index, choice, out)
    if asked and choice.install and ask("继续？[Y/n] ").strip().lower() in ("n", "no"):
        raise RuntimeError("已取消，未做任何改动")
    return choice

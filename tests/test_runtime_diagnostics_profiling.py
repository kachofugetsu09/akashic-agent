"""导出定位路径的脱敏回归：已知根相对化、其余一律退化为不含身份目录的基名。"""
from pathlib import Path

from plugins.runtime_diagnostics.profiling import _export_locator

ROOTS = ((Path("/home/alice"), "~"), (Path("/srv/app"), "<install>"))


def test_home_root_is_relativized() -> None:
    assert _export_locator("/home/alice/app.py", roots=ROOTS) == "~/app.py"
    assert _export_locator("/home/alice/project/app.py", roots=ROOTS) == "~/project/app.py"


def test_install_root_is_relativized() -> None:
    assert _export_locator("/srv/app/plugins/x/y.py", roots=ROOTS) == "<install>/plugins/x/y.py"


def test_unknown_absolute_root_drops_parent_identity() -> None:
    assert _export_locator("/opt/acme/project/app.py", roots=ROOTS) == "app.py"
    assert _export_locator("/tmp/q.py", roots=ROOTS) == "q.py"


def test_relative_paths_drop_parent_identity() -> None:
    assert _export_locator("home/alice/private/app.py", roots=ROOTS) == "app.py"
    assert _export_locator("../alice/private/app.py", roots=ROOTS) == "app.py"
    assert _export_locator("app.py", roots=ROOTS) == "app.py"


def test_parent_segments_cannot_bypass_root_match() -> None:
    # 词法匹配 home 前缀后仍会带出 /home/bob；折叠 `..` 后才正确退化。
    assert _export_locator("/home/alice/../bob/private/app.py", roots=ROOTS) == "app.py"
    assert _export_locator("/srv/app/../other/x.py", roots=ROOTS) == "x.py"


def test_virtual_labels_are_preserved() -> None:
    assert _export_locator("<frozen importlib._bootstrap>", roots=ROOTS) == "<frozen importlib._bootstrap>"
    assert _export_locator("<string>", roots=ROOTS) == "<string>"
    # 含路径分隔符的名字不是虚拟标签，按路径规则处理。
    assert _export_locator("<other/dir>", roots=ROOTS) == "dir>"


def test_no_identity_leaks_for_representative_inputs() -> None:
    for filename in (
        "/home/alice/app.py",
        "/home/alice/deep/x/y.py",
        "/home/alice/../bob/private/app.py",
        "home/alice/private/app.py",
        "../alice/private/app.py",
        "/opt/other/z.py",
        "/tmp/q.py",
    ):
        rendered = _export_locator(filename, roots=ROOTS)
        assert "alice" not in rendered, f"{filename} -> {rendered}"
        assert "bob" not in rendered, f"{filename} -> {rendered}"

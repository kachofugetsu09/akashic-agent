"""导出定位路径的脱敏回归：已知根相对化、未知根不带父目录身份。"""
from pathlib import Path

from plugins.runtime_diagnostics.profiling import _export_locator

ROOTS = ((Path("/home/alice"), "~"), (Path("/srv/app"), "<install>"))


def test_home_root_is_relativized() -> None:
    assert _export_locator("/home/alice/app.py", roots=ROOTS) == "~/app.py"
    assert _export_locator("/home/alice/project/app.py", roots=ROOTS) == "~/project/app.py"
    assert _export_locator("/home/alice/a/b/c/d.py", roots=ROOTS) == "~/a/b/c/d.py"


def test_install_root_is_relativized() -> None:
    assert _export_locator("/srv/app/plugins/x/y.py", roots=ROOTS) == "<install>/plugins/x/y.py"


def test_unknown_root_drops_parent_identity() -> None:
    assert _export_locator("/opt/acme/project/app.py", roots=ROOTS) == "app.py"
    assert _export_locator("/tmp/q.py", roots=ROOTS) == "q.py"


def test_non_regular_names_keep_basename_only_for_unknown_root() -> None:
    assert _export_locator("<frozen importlib._bootstrap>", roots=ROOTS) == "<frozen importlib._bootstrap>"
    assert _export_locator("/other/dir/<string>", roots=ROOTS) == "<string>"


def test_no_identity_leaks_for_any_absolute_path() -> None:
    for filename in (
        "/home/alice/app.py",
        "/home/alice/deep/x/y.py",
        "/home/aliceuser/other.py",
        "/opt/other/z.py",
        "/tmp/q.py",
    ):
        rendered = _export_locator(filename, roots=ROOTS)
        assert "alice" not in rendered, rendered

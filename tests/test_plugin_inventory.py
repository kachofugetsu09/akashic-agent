"""清单不能通过独立的一套宽泛公开目录规则隐藏实现依赖。"""
import subprocess

from scripts.plugin_inventory import build_inventory


def test_inventory_includes_support_packages_and_uses_frozen_boundary_rules(tmp_path):
    files = {
        "plugins/one/plugin.py": "from agent.plugin_composition.future_private import Hidden\nfrom plugins.two.impl import run\n",
        "plugins/two/impl.py": "def run(): pass\n",
        "bootstrap/app.py": "from plugins.two.impl import run\n",
    }
    for name, content in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    subprocess.run(["git", "init", str(tmp_path)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], check=True, capture_output=True)
    report = build_inventory(tmp_path)
    assert {item["package"] for item in report["packages"]} == {"one", "two"}
    assert report["summary"]["support_package_count"] == 2
    assert {item["kind"] for item in report["static_boundary_violations"]} == {
        "R1", "R2", "R3", "not_installable_artifact"}
    assert report["core_consumer_imports"] == [{"file": "bootstrap/app.py", "target": "plugins.two.impl"}]

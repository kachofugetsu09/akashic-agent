"""用真实 PluginManager 验证首次守卫准备失败不发布无门选择，修复后仍可启动。"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.plugin_composition.endpoints import load_endpoint_plan_requirements
from agent.plugins.manager import PluginManager
from agent.plugins.selection import PluginSelection


async def boot(output: Path, *, broken: bool) -> dict[str, object]:
    """失败不得改变选择；成功必须把守卫要求与后端一起发布。"""
    workspace = output / "workspace"
    selection = PluginSelection(workspace)
    before = selection.read()
    manager = PluginManager([output / "sources"], workspace=workspace)
    try:
        if broken:
            try:
                await manager.load_all()
            except RuntimeError as error:
                assert "入口守卫准备失败" in str(error), str(error)
            else:
                raise AssertionError("坏守卫被跳过，启动意外成功")
            assert selection.read() == before
            assert not (workspace / "runtime/endpoints.json").exists()
            return {"startup": "rejected", "selection_unchanged": True, "endpoints_published": False}
        await manager.load_all()
        endpoints, required = load_endpoint_plan_requirements(workspace / "runtime/endpoints.json")
        assert "guard+unix" in required
        assert any(endpoint.name == "web" for endpoint in endpoints)
        assert selection.read() is not None
        return {"startup": "ready", "requires": list(required)}
    finally:
        await manager.terminate_all()


def main() -> None:
    """创建一次性来源和 workspace，逐进程验证失败、修复和重启。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--boot", choices=("broken", "repaired"))
    args = parser.parse_args()
    output = args.output.resolve()
    os.environ.update(HOME=str(output / "home"), AKASHIC_PLUGIN_HOME=str(output / "plugin-home"))
    if args.boot:
        print(json.dumps(asyncio.run(boot(output, broken=args.boot == "broken"))))
        return

    # 1. 初始化真实状态；只有辅助模块损坏，静态守卫声明仍合法。
    output.mkdir()
    (output / "home").mkdir()
    subprocess.run([sys.executable, str(ROOT / "main.py"), "init", "--workspace", str(output / "workspace"),
                    "--config", str(output / "config.toml")], check=True, capture_output=True)
    gate = output / "sources/gate"
    gate.mkdir(parents=True)
    (gate / "plugin.py").write_text('api_version=3\nname="gate"\nversion="1"\ningress_guard=True\nasync def apply(ctx):\n pass\n')
    broken = gate / "broken.py"
    broken.write_text("def broken(:\n")
    backend = output / "sources/backend"
    backend.mkdir()
    (backend / "plugin.py").write_text(
        'api_version=3\nname="backend"\nversion="1"\nasync def apply(ctx):\n'
        ' await ctx.endpoint("web", protocol="http+unix", address="/tmp/unused-fixture.sock", routes=("/api",))\n'
    )

    # 2. 两次失败启动都不能把守卫从选择中永久丢掉；保留坏源码后再修复。
    report: dict[str, object] = {}
    for name, state in (("first_failure", "broken"), ("second_failure", "broken"),
                        ("repaired", "repaired"), ("restart", "repaired")):
        if name == "repaired":
            broken.rename(gate / "broken.py.before-fix")
            broken.write_text("value = 1\n")
        result = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--output", str(output), "--boot", state],
                                check=True, capture_output=True, text=True)
        report[name] = json.loads(result.stdout)
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()

"""安装诊断只读取制品，不执行第二套插件加载流程。"""

from pathlib import Path

import pytest

from agent.plugins.artifacts import ArtifactPointer, write_pointers
from agent.plugins.doctor import run_plugin_doctor
from agent.plugins.manifest import write_plugin_manifest


def installed_artifact(tmp_path: Path, source: str, *, candidate: bool) -> Path:
    """创建仅供诊断读取的安装指针和制品。"""
    home = tmp_path / "home"
    base = home / "cache/lab/demo"
    artifact = base / ".artifacts/one"
    artifact.mkdir(parents=True)
    (artifact / "plugin.py").write_text(source)
    pointer = ArtifactPointer(".artifacts/one")
    write_pointers(base, stable=ArtifactPointer(None) if candidate else pointer, latest=pointer)
    write_plugin_manifest({"demo@lab": True}, plugins_home=home)
    return home


@pytest.mark.parametrize("candidate", [False, True])
def test_doctor_does_not_import_plugin_and_defers_runtime_checks(tmp_path: Path, candidate: bool):
    marker = tmp_path / "imported"
    source = (
        'name = "demo"\nversion = "1.0.0"\napi_version = 3\n'
        'from pathlib import Path\n'
        f'Path({str(marker)!r}).write_text("import side effect")\n'
        'raise RuntimeError("runtime import must belong to actual Root")\n'
    )
    home = installed_artifact(tmp_path, source, candidate=candidate)
    report = run_plugin_doctor(workspace=tmp_path / "workspace", plugins_home=home)
    assert not marker.exists()
    assert report["status"] == ("broken" if candidate else "degraded")
    checks = report["plugins"][0]["checks"]
    if candidate:
        assert report["status"] == "broken"
        assert any(check["name"] == "install" and check["status"] == "error" for check in checks)
    else:
        assert any(check["name"] == "runtime" and check["status"] == "deferred" for check in checks)
    assert not (tmp_path / "workspace").exists()


@pytest.mark.parametrize("source", [
    "not valid python !!!",
    'name = "demo"\nversion = "1.0.0"\napi_version = 2\n',
])
def test_doctor_reports_invalid_artifact_without_running_it(tmp_path: Path, source: str):
    home = installed_artifact(
        tmp_path, 'name = "demo"\nversion = "1.0.0"\napi_version = 3\n',
        candidate=False,
    )
    # Corrupt the artifact after pointer publication to exercise doctor input.
    (home / "cache/lab/demo/.artifacts/one/plugin.py").write_text(source)
    report = run_plugin_doctor(workspace=tmp_path / "workspace", plugins_home=home)
    assert report["status"] == "broken"
    assert any(check["name"] == "install" and check["status"] == "error"
               for check in report["plugins"][0]["checks"])

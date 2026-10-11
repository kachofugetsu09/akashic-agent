"""插件宿主只在 numpy 首次加载时给出 BLAS 单线程默认值，不改变进程环境。"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

_PROBE = """
import importlib.resources, json, os, sys
from pathlib import Path
from agent.plugins.importer import FreshPluginImporter
importer = FreshPluginImporter()
if sys.argv[1] == "host":
    importer.register("_probe_plugin", Path(sys.argv[2]))
import numpy
installed = len(sys.meta_path)
importer.unregister("_probe_plugin")
print(json.dumps({
    "env": {name: os.environ.get(name) for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")},
    "threads": len(os.listdir("/proc/self/task")),
    "resources": importlib.resources.files("numpy").joinpath("__init__.py").is_file(),
    "finders_released": len(sys.meta_path) == installed - 2 if sys.argv[1] == "host" else True,
    "sum": float(numpy.ones(3).sum()),
}))
"""


def _probe(tmp_path: Path, mode: str, **env: str) -> dict[str, object]:
    root = Path(__file__).resolve().parents[1]
    environment = {key: value for key, value in os.environ.items()
                   if key not in {"OPENBLAS_NUM_THREADS", "GOTO_NUM_THREADS", "OMP_NUM_THREADS"}}
    environment.update(PYTHONPATH=os.pathsep.join((str(root), str(root / "sdk/python/src"))), **env)
    output = subprocess.check_output([sys.executable, "-c", _PROBE, mode, str(tmp_path)], env=environment, text=True)
    return json.loads(output)


@pytest.mark.skipif(not Path("/proc/self/task").is_dir(), reason="需要 /proc 统计线程")
def test_plugin_host_loads_numpy_with_single_blas_thread_and_restores_environment(tmp_path: Path):
    result = _probe(tmp_path, "host")
    assert result["env"] == {"OPENBLAS_NUM_THREADS": None, "OMP_NUM_THREADS": None}
    assert result["threads"] == 1
    assert result["resources"] is True and result["finders_released"] is True and result["sum"] == 3.0


@pytest.mark.skipif(not Path("/proc/self/task").is_dir(), reason="需要 /proc 统计线程")
def test_explicit_thread_setting_is_left_to_the_user(tmp_path: Path):
    result = _probe(tmp_path, "host", OMP_NUM_THREADS="2")
    assert result["env"] == {"OPENBLAS_NUM_THREADS": None, "OMP_NUM_THREADS": "2"}
    assert result["threads"] == 2


@pytest.mark.skipif((os.cpu_count() or 1) < 2 or not Path("/proc/self/task").is_dir(), reason="需要多核")
def test_numpy_outside_plugin_host_keeps_library_default(tmp_path: Path):
    threads = _probe(tmp_path, "plain")["threads"]
    assert isinstance(threads, int) and threads > 1

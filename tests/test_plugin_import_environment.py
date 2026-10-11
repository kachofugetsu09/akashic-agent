"""插件加载依赖时，并发子进程仍继承用户原来的环境。"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


_THREAD_ENV = ("OPENBLAS_NUM_THREADS", "GOTO_NUM_THREADS", "OMP_NUM_THREADS")
_PROBE = """
import importlib.machinery, importlib.resources, json, os, subprocess, sys, threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from agent.plugins.importer import FreshPluginImporter

keys = ("OPENBLAS_NUM_THREADS", "GOTO_NUM_THREADS", "OMP_NUM_THREADS")
entered, release = threading.Event(), threading.Event()
original = importlib.machinery.SourceFileLoader.exec_module

def paused(self, module):
    if module.__name__ == "numpy":
        entered.set()
        if not release.wait(10):
            raise TimeoutError("numpy import was not released")
    return original(self, module)

def load():
    import numpy
    return float(numpy.ones(3).sum())

host = FreshPluginImporter()
host.register("_probe_plugin", Path.cwd())
importlib.machinery.SourceFileLoader.exec_module = paused
try:
    with ThreadPoolExecutor(max_workers=1) as worker:
        loading = worker.submit(load)
        try:
            if not entered.wait(10):
                raise TimeoutError("numpy import did not start")
            during = {key: os.environ.get(key) for key in keys}
            child = json.loads(subprocess.check_output([
                sys.executable, "-c",
                "import json, os, sys; print(json.dumps({key: os.environ.get(key) for key in sys.argv[1:]}))",
                *keys,
            ], text=True, timeout=10))
        finally:
            release.set()
        total = loading.result(timeout=10)
finally:
    importlib.machinery.SourceFileLoader.exec_module = original
    host.unregister("_probe_plugin")

print(json.dumps({
    "during": during,
    "child": child,
    "after": {key: os.environ.get(key) for key in keys},
    "sum": total,
    "resources": importlib.resources.files("numpy").joinpath("__init__.py").is_file(),
}))
"""


@pytest.mark.parametrize("setting", [None, *_THREAD_ENV])
def test_numpy_import_preserves_concurrent_child_environment(setting: str | None):
    """用真实 numpy 和子进程，在受控导入窗口核对环境及加载结果。"""
    root = Path(__file__).resolve().parents[1]
    environment = {key: value for key, value in os.environ.items() if key not in _THREAD_ENV}
    if setting is not None:
        environment[setting] = "2"
    environment["PYTHONPATH"] = os.pathsep.join((str(root), str(root / "sdk/python/src")))
    output = subprocess.check_output(
        [sys.executable, "-c", _PROBE], env=environment, text=True, timeout=30,
    )
    result = json.loads(output)
    expected = {key: environment.get(key) for key in _THREAD_ENV}
    assert result["during"] == result["child"] == result["after"] == expected
    assert result["sum"] == 3.0 and result["resources"] is True

"""Akasha 的 numpy 入口只在加载时限制 OpenBLAS 线程，不改变进程环境。"""
import json
import os
import subprocess
import sys
from pathlib import Path

_PROBE = """
import json, os
from plugins.akasha.numeric import np
print(json.dumps({"env": os.environ.get("OPENBLAS_NUM_THREADS"),
                  "threads": len(os.listdir("/proc/self/task")), "sum": float(np.ones(3).sum())}))
"""


def _probe(**env: str) -> dict[str, object]:
    root = Path(__file__).resolve().parents[1]
    environment = {key: value for key, value in os.environ.items() if key != "OPENBLAS_NUM_THREADS"}
    environment.update(PYTHONPATH=os.pathsep.join((str(root), str(root / "sdk/python/src"))), **env)
    output = subprocess.check_output([sys.executable, "-c", _PROBE], env=environment, text=True)
    return json.loads(output)


def test_numeric_entry_loads_single_threaded_blas_and_restores_environment():
    result = _probe()
    assert result == {"env": None, "threads": 1, "sum": 3.0}


def test_numeric_entry_keeps_explicit_user_setting():
    result = _probe(OPENBLAS_NUM_THREADS="2")
    assert result["env"] == "2" and result["threads"] == 2

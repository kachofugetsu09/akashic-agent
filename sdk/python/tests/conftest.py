from __future__ import annotations

import sys
from pathlib import Path

SDK_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(SDK_ROOT / "src"))
sys.path.insert(0, str(SDK_ROOT.parents[1]))

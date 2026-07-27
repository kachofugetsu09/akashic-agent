#!/usr/bin/env python3
"""Rebuild Akasha V2 through the engine-owned deterministic CLI."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plugins.akasha.cli import main


if __name__ == "__main__":
    main()

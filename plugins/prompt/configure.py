"""首次安装时只创建缺失的人格文件。"""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path

from persona import initialize_veda_if_missing


def main() -> None:
    workspace_value = os.environ.get("AKASHIC_SETUP_WORKSPACE", "").strip()
    if not workspace_value:
        raise RuntimeError("缺少 AKASHIC_SETUP_WORKSPACE")
    result = initialize_veda_if_missing(Path(workspace_value))
    print(json.dumps(asdict(result), ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()

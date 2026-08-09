from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import tomllib
from pathlib import Path
from typing import Any

_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")
_VERSION_COMMANDS = {
    "node": ("node", "--version"),
    "npm": ("npm", "--version"),
    "python": ("python", "--version"),
    "uv": ("uv", "--version"),
    "npm:@jackwener/opencli": ("opencli", "--version"),
    "opencode": ("opencode", "--version"),
}


def _run(*arguments: str, cwd: Path) -> str:
    result = subprocess.run(
        list(arguments), cwd=cwd, check=True, capture_output=True, text=True
    )
    return (result.stdout or result.stderr).strip()


def resolve_toolchain_identity(repository: Path, mise: Path) -> dict[str, Any]:
    """Verify the release checkout and exact mise tools, then return its identity."""

    # 1. Bind the host process to one clean Git generation.
    repository = repository.resolve(strict=True)
    commit = _run("git", "rev-parse", "HEAD", cwd=repository)
    if _COMMIT_PATTERN.fullmatch(commit) is None:
        raise RuntimeError("Host Bridge checkout HEAD 不是完整 commit")
    if _run("git", "status", "--porcelain", "--untracked-files=all", cwd=repository):
        raise RuntimeError("Host Bridge release checkout 必须保持 clean")

    # 2. Require an exact, complete tool profile owned by the same release.
    config_path = repository / "mise.toml"
    config_bytes = config_path.read_bytes()
    document = tomllib.loads(config_bytes.decode("utf-8"))
    tools = document.get("tools")
    if not isinstance(tools, dict) or set(tools) != set(_VERSION_COMMANDS):
        raise RuntimeError("mise.toml tools 与 Host Bridge capability contract 不一致")
    declared = {name: str(value) for name, value in tools.items()}
    if any(not re.fullmatch(r"\d+(?:\.\d+)+", value) for value in declared.values()):
        raise RuntimeError("Host Bridge tool version 必须是精确数字版本")

    # 3. Resolve every command through mise and prove the observed version.
    observed: dict[str, str] = {}
    for tool, command in _VERSION_COMMANDS.items():
        output = _run(str(mise), "exec", "--", *command, cwd=repository)
        if declared[tool] not in output:
            raise RuntimeError(
                f"Host Bridge tool version 不一致: {tool}={output!r}, "
                f"expected={declared[tool]}"
            )
        observed[tool] = declared[tool]

    identity = {
        "schemaVersion": 1,
        "releaseCommit": commit,
        "miseConfigSha256": hashlib.sha256(config_bytes).hexdigest(),
        "tools": observed,
    }
    encoded = json.dumps(
        identity, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    identity["toolchainDigest"] = hashlib.sha256(encoded).hexdigest()
    return identity


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify Host Bridge toolchain identity"
    )
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--mise", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    identity = resolve_toolchain_identity(args.repository, args.mise)
    rendered = json.dumps(identity, ensure_ascii=False, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()

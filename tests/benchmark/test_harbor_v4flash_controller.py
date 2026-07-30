import shutil
import subprocess
import tomllib
from pathlib import Path

from benchmark.harbor_v4flash.isolation import create_source_bundle


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.strip()


def test_v4flash_high_uses_provider_output_limit() -> None:
    config_path = (
        Path(__file__).parents[2]
        / "benchmark"
        / "harbor_v4flash"
        / "config.toml"
    )
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))

    assert config["llm"]["runtimes"]["main"]["max_output_tokens"] == 0
    assert config["agent"]["max_tokens"] == 0


def test_source_bundle_restores_history_and_keeps_worktree_overlay(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    _git(source, "init")
    _git(source, "config", "user.name", "Benchmark Test")
    _git(source, "config", "user.email", "benchmark@localhost")
    tracked = source / "tracked.txt"
    tracked.write_text("baseline\n", encoding="utf-8")
    _git(source, "add", "tracked.txt")
    _git(source, "commit", "-m", "baseline")
    baseline = _git(source, "rev-parse", "HEAD")
    tracked.write_text("head\n", encoding="utf-8")
    _git(source, "commit", "-am", "head")
    head = _git(source, "rev-parse", "HEAD")
    tracked.write_text("dirty overlay\n", encoding="utf-8")

    bundle = tmp_path / "inputs" / "source.bundle"
    info = create_source_bundle(
        source,
        bundle,
        migration_baseline=baseline,
    )

    restored = tmp_path / "restored"
    restored.mkdir()
    _git(restored, "init")
    shutil.copyfile(tracked, restored / "tracked.txt")
    _git(
        restored,
        "fetch",
        str(bundle),
        "+refs/heads/*:refs/remotes/benchmark/*",
    )
    _git(restored, "reset", "--mixed", head)
    assert _git(restored, "cat-file", "-t", baseline) == "commit"
    assert _git(restored, "rev-parse", "HEAD") == head
    assert (restored / "tracked.txt").read_text(encoding="utf-8") == "dirty overlay\n"
    assert _git(restored, "status", "--short") == "M tracked.txt"
    assert info["head"] == head
    assert info["migration_baseline"] == baseline

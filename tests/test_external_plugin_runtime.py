from __future__ import annotations

import json
from pathlib import Path

from docker.debug.plugin_external_acceptance import _source_checkout, main
from scripts.plugin_inventory import build_inventory


def test_external_runtime_rejects_repository_checkout() -> None:
    repo_root = Path(__file__).parents[1].resolve()

    try:
        _source_checkout(str(repo_root / "plugins" / "content"), repo_root)
    except ValueError as error:
        assert "本仓库内" in str(error)
    else:
        raise AssertionError("repository plugin source was accepted as external")


def test_all_inventory_requires_a_real_source_mapping(tmp_path: Path, capsys) -> None:
    repo_root = Path(__file__).parents[1]
    inventory_path = tmp_path / "inventory.json"
    inventory_path.write_text(
        json.dumps(build_inventory(repo_root), ensure_ascii=False),
        encoding="utf-8",
    )
    sources_path = tmp_path / "sources.json"
    sources_path.write_text("{}\n", encoding="utf-8")
    report_path = tmp_path / "external-report.json"

    result = main(
        [
            "--all",
            "--inventory",
            str(inventory_path),
            "--sources-json",
            str(sources_path),
            "--repo-root",
            str(repo_root),
            "--output",
            str(report_path),
        ]
    )

    assert result == 1
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "failed"
    assert report["reports"]
    assert all(item["status"] == "failed" for item in report["reports"])
    assert "one example" in " ".join(
        str(item.get("error", "")) for item in report["reports"]
    )
    assert not capsys.readouterr().out

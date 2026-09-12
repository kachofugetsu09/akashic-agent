from __future__ import annotations

from pathlib import Path

from scripts.plugin_inventory import build_inventory


def test_inventory_covers_manifest_plugins_and_support_packages() -> None:
    report = build_inventory(Path(__file__).parents[1])
    packages = {entry["package"]: entry for entry in report["packages"]}

    assert report["schema_version"] == 1
    assert report["summary"]["package_count"] == len(packages)
    assert report["summary"]["manifest_plugin_count"] >= 1
    assert report["summary"]["support_package_count"] >= 1
    assert packages["akasha"]["classification"] == "manifest-plugin"
    assert packages["content"]["classification"] == "support-package"
    assert all(entry["source_digest"] for entry in packages.values())
    assert all("static_imports" in entry for entry in packages.values())


def test_inventory_keeps_current_boundary_debt_as_failure_evidence() -> None:
    report = build_inventory(Path(__file__).parents[1])
    violations = report["static_boundary_violations"]
    kinds = {item["kind"] for item in violations}

    # This is intentionally a red baseline on the pre-migration tree.  The
    # inventory must expose debt instead of granting a blanket historical
    # allowlist or claiming one external example proves the fleet.
    assert violations
    assert "not_installable_artifact" in kinds
    assert "sibling_plugin_import" in kinds
    assert "private_core_import" in kinds


def test_inventory_records_core_side_plugin_consumers() -> None:
    report = build_inventory(Path(__file__).parents[1])
    consumers = report["core_consumer_imports"]

    assert consumers
    assert all({"file", "line", "target"} <= set(item) for item in consumers)
    assert any(item["target"].startswith("plugins.") for item in consumers)

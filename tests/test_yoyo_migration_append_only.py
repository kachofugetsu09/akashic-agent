from __future__ import annotations

import hashlib
import shutil
import subprocess
from pathlib import Path
from typing import Iterable

import pytest

import scripts.check_yoyo_migrations as checker


def _git(repo: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *arguments],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def _commit(repo: Path, message: str) -> str:
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", message)
    return _git(repo, "rev-parse", "HEAD")


def _repository(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "repo"
    migration = repo / "migrations/yoyo/20260802_01_origin.py"
    migration.parent.mkdir(parents=True)
    migration.write_text("steps = []\n", encoding="utf-8")
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    return repo, _commit(repo, "baseline")


def _write_bundle(
    repo: Path,
    entries: Iterable[tuple[str, str]],
    *,
    plugin_name: str = "future_owner",
) -> Path:
    artifact = repo / "plugins" / plugin_name
    migration_root = artifact / "migration_steps"
    migration_root.mkdir(parents=True, exist_ok=True)
    init = migration_root / "__init__.py"
    init.write_text("\n", encoding="utf-8")
    entry_list = list(entries)
    for migration_id, source in entry_list:
        (migration_root / f"{migration_id}.py").write_text(source, encoding="utf-8")

    files: list[tuple[str, str]] = [
        ("__init__.py", hashlib.sha256(init.read_bytes()).hexdigest())
    ]
    files.extend(
        (
            f"{migration_id}.py",
            hashlib.sha256(
                (migration_root / f"{migration_id}.py").read_bytes()
            ).hexdigest(),
        )
        for migration_id, _source in entry_list
    )
    catalog_lines = [
        "schema_version = 1",
        f"bundle_id = {plugin_name!r}",
        "version = '1.0.0'",
        "migration_root = 'migration_steps'",
        "package_name = 'migration_steps'",
        "",
    ]
    for relative, digest in files:
        catalog_lines.extend(
            ["[[files]]", f"path = {relative!r}", f"sha256 = {digest!r}", ""]
        )
    for migration_id, _source in entry_list:
        digest = dict(files)[f"{migration_id}.py"]
        catalog_lines.extend(
            [
                "[[migrations]]",
                f"id = {migration_id!r}",
                f"path = {migration_id + '.py'!r}",
                "depends = []",
                "transactional = false",
                f"sha256 = {digest!r}",
                "",
            ]
        )
    catalog = artifact / "migration.catalog.toml"
    catalog.write_text("\n".join(catalog_lines), encoding="utf-8")
    catalog_digest = hashlib.sha256(catalog.read_bytes()).hexdigest()
    (artifact / "akashic.plugin.toml").write_text(
        "schema_version = 1\n"
        f"name = {plugin_name!r}\n"
        "version = '1.0.0'\n"
        "api_version = 3\n"
        "entrypoint = 'plugin.py'\n\n"
        "[migration]\n"
        "catalog = 'migration.catalog.toml'\n"
        f"catalog_sha256 = {catalog_digest!r}\n",
        encoding="utf-8",
    )
    return artifact


def _write_retirement_metadata(
    repo: Path,
    commits: Iterable[str],
) -> None:
    previous_root = checker.ROOT
    checker.ROOT = repo
    try:
        lines = ["schema_version = 1", ""]
        for commit in sorted(commits):
            lines.extend(["[[sources]]", f"recovery_commit = {commit!r}", ""])
            for path, digest in checker._retirement_inventory(commit):
                lines.extend(
                    [
                        "[[sources.files]]",
                        f"path = {path!r}",
                        f"sha256 = {digest!r}",
                        "",
                    ]
                )
        (repo / checker.RETIREMENT_METADATA).parent.mkdir(parents=True, exist_ok=True)
        (repo / checker.RETIREMENT_METADATA).write_text(
            "\n".join(lines), encoding="utf-8"
        )
    finally:
        checker.ROOT = previous_root


def test_existing_yoyo_migration_cannot_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, base = _repository(tmp_path)
    migration = repo / "migrations/yoyo/20260802_01_origin.py"
    migration.write_text("steps = ['changed']\n", encoding="utf-8")
    monkeypatch.setattr(checker, "ROOT", repo)

    assert checker.violations(base) == [
        "registered Yoyo migration changed: migrations/yoyo/20260802_01_origin.py"
    ]


def test_new_plugin_bundle_migration_is_allowed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, base = _repository(tmp_path)
    _write_bundle(repo, [("20260803_01_next", "steps = []\n")])
    _git(repo, "add", ".")
    monkeypatch.setattr(checker, "ROOT", repo)

    assert checker.violations(base) == []


def test_retired_yoyo_path_cannot_be_reintroduced(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, base = _repository(tmp_path)
    path = repo / "migrations/yoyo/20260803_01_next.py"
    path.write_text("steps = []\n", encoding="utf-8")
    _git(repo, "add", str(path.relative_to(repo)))
    monkeypatch.setattr(checker, "ROOT", repo)

    assert checker.violations(base) == [
        "retired migration path reintroduced: migrations/yoyo/20260803_01_next.py"
    ]


def test_future_core_migration_is_allowed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, base = _repository(tmp_path)
    path = repo / "migrations/core/20260913_01_future_owner.py"
    path.parent.mkdir(parents=True)
    path.write_text("steps = []\n", encoding="utf-8")
    _git(repo, "add", str(path.relative_to(repo)))
    monkeypatch.setattr(checker, "ROOT", repo)

    assert checker.violations(base) == []


def test_retired_core_origin_cannot_be_reintroduced(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, base = _repository(tmp_path)
    path = repo / "migrations/core/20260802_01_yoyo_origin.py"
    path.parent.mkdir(parents=True)
    path.write_text("steps = []\n", encoding="utf-8")
    _git(repo, "add", str(path.relative_to(repo)))
    monkeypatch.setattr(checker, "ROOT", repo)

    assert checker.violations(base) == [
        "retired migration path reintroduced: "
        "migrations/core/20260802_01_yoyo_origin.py"
    ]


def test_registered_yoyo_migration_cannot_move_to_legacy_plugin(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, base = _repository(tmp_path)
    old = repo / "migrations/yoyo/20260802_01_origin.py"
    replacement = (
        repo / "plugins/legacy_upgrade/legacy_upgrade_migrations/20260802_01_origin.py"
    )
    replacement.parent.mkdir(parents=True)
    replacement.write_text(old.read_text(encoding="utf-8"), encoding="utf-8")
    old.unlink()
    _git(repo, "add", "-A")
    monkeypatch.setattr(checker, "ROOT", repo)

    assert checker.violations(base) == [
        "registered Yoyo migration changed: migrations/yoyo/20260802_01_origin.py",
        "retired migration path reintroduced: "
        "plugins/legacy_upgrade/legacy_upgrade_migrations/20260802_01_origin.py",
    ]


def test_plugin_bundle_keeps_old_sources_and_catalog_entries_append_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    _write_bundle(repo, [("20260803_01_old", "steps = []\n")])
    base = _commit(repo, "bundle baseline")
    monkeypatch.setattr(checker, "ROOT", repo)

    old = repo / "plugins/future_owner/migration_steps/20260803_01_old.py"
    old.write_text("steps = ['changed']\n", encoding="utf-8")
    assert any(item.endswith("20260803_01_old.py") for item in checker.violations(base))

    old.write_text("steps = []\n", encoding="utf-8")
    _write_bundle(
        repo,
        [
            ("20260803_01_old", "steps = []\n"),
            ("20260804_01_new", "steps = ['new']\n"),
        ],
    )
    _git(repo, "add", ".")
    assert checker.violations(base) == []

    # Updating the catalog digest does not authorize rewriting an old ID.
    old.write_text("steps = ['rewritten']\n", encoding="utf-8")
    _write_bundle(
        repo,
        [
            ("20260803_01_old", "steps = ['rewritten']\n"),
            ("20260804_01_new", "steps = ['new']\n"),
        ],
    )
    _git(repo, "add", ".")
    problems = checker.violations(base)
    assert any(
        "plugins/future_owner/migration_steps/20260803_01_old.py" in item
        for item in problems
    )


def test_one_time_retirement_requires_both_real_base_inventories(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, old_base = _repository(tmp_path)
    unrelated = repo / "README.md"
    unrelated.write_text("child\n", encoding="utf-8")
    child_base = _commit(repo, "unrelated child")
    old = repo / "migrations/yoyo/20260802_01_origin.py"
    old.unlink()
    _write_bundle(repo, [("20260803_01_origin", "steps = []\n")])
    new_base = _commit(repo, "move migrations to plugin")

    shutil_root = repo / "plugins/future_owner"
    shutil.rmtree(shutil_root)
    monkeypatch.setattr(checker, "ROOT", repo)
    monkeypatch.setattr(
        checker,
        "RETIREMENT_RECOVERY_COMMITS",
        frozenset({old_base, new_base}),
    )
    _write_retirement_metadata(repo, (old_base, new_base))

    assert checker.violations(old_base) == []
    assert checker.violations(new_base) == []
    assert any(
        "registered Yoyo migration changed" in item
        for item in checker.violations(child_base)
    )


def test_retirement_metadata_digest_must_match_fixed_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, base = _repository(tmp_path)
    (repo / "migrations/yoyo/20260802_01_origin.py").unlink()
    monkeypatch.setattr(checker, "ROOT", repo)
    monkeypatch.setattr(checker, "RETIREMENT_RECOVERY_COMMITS", frozenset({base}))
    _write_retirement_metadata(repo, (base,))
    metadata = repo / checker.RETIREMENT_METADATA
    text = metadata.read_text(encoding="utf-8")
    metadata.write_text(
        text.replace("sha256 = '", "sha256 = '" + "0" * 64, 1), encoding="utf-8"
    )

    assert any(
        "invalid migration retirement metadata" in item
        for item in checker.violations(base)
    )


def test_archived_migration_is_outside_the_catalog(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, base = _repository(tmp_path)
    archived = repo / "migrations/provider_runtimes_and_akasha/migration.py"
    archived.parent.mkdir(parents=True)
    archived.write_text("historical = True\n", encoding="utf-8")
    _git(repo, "add", "migrations/provider_runtimes_and_akasha/migration.py")
    monkeypatch.setattr(checker, "ROOT", repo)

    assert checker.violations(base) == []


def test_existing_yoyo_migration_cannot_be_deleted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, base = _repository(tmp_path)
    (repo / "migrations/yoyo/20260802_01_origin.py").unlink()
    monkeypatch.setattr(checker, "ROOT", repo)

    assert checker.violations(base) == [
        "registered Yoyo migration changed: migrations/yoyo/20260802_01_origin.py"
    ]

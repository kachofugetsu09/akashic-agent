from pathlib import Path

import pytest

import agent.plugins.selection as selection_module
from agent.plugin_composition.archive import PluginArchive
from agent.plugins.selection import (
    PluginSelection,
    SelectionConflictError,
    SelectionFormatError,
    SelectionWriteError,
)
from bootstrap.init_workspace import init_workspace


def component(workspace: Path, value: str) -> str:
    """存储层只引用输入；不解释组件的身份、配置或能力。"""
    archive = PluginArchive(workspace / "runtime/plugin-archives")
    return archive.save_descriptor({"config": {"value": value}})


def test_missing_selection_is_not_an_implicit_first_boot(tmp_path: Path) -> None:
    selection = PluginSelection(tmp_path)
    with pytest.raises(SelectionFormatError, match="显式"):
        selection.read()
    with pytest.raises(SelectionFormatError, match="显式"):
        selection.commit((), expected_ref=None)
    assert not (tmp_path / "runtime").exists()


def test_commits_preserve_history_and_reject_stale_and_aba(tmp_path: Path) -> None:
    selection = PluginSelection(tmp_path)
    selection.initialize()
    assert PluginSelection(tmp_path).read() is None
    a, b = component(tmp_path, "a"), component(tmp_path, "b")
    first = selection.commit((a,), expected_ref=None)
    first_bytes = (selection.archive.path / f"{first}.json").read_bytes()
    second = selection.commit((a, b), expected_ref=first)
    third = selection.commit((a,), expected_ref=second)
    assert first != third
    assert PluginSelection(tmp_path).read() == third
    assert selection.archive.read_descriptor(third) == {
        "version": 1, "components": (a,), "previous": second,
    }
    assert selection.archive.read_descriptor(second)["previous"] == first
    assert (selection.archive.path / f"{first}.json").read_bytes() == first_bytes
    with pytest.raises(SelectionConflictError):
        selection.commit((b,), expected_ref=first)
    with pytest.raises(SelectionConflictError):
        selection.commit((b,), expected_ref=None)
    assert selection.read() == third


def test_empty_whole_selection_is_distinct_from_uninitialized(tmp_path: Path) -> None:
    selection = PluginSelection(tmp_path)
    selection.initialize()
    ref = selection.commit((), expected_ref=None)
    assert ref is not None and selection.read() == ref


@pytest.mark.parametrize("value", ['{}', 'broken', '{"version":true,"root_ref":null}'])
def test_unknown_pointer_is_never_overwritten(tmp_path: Path, value: str) -> None:
    selection = PluginSelection(tmp_path)
    selection.path.parent.mkdir()
    selection.path.write_text(value)
    with pytest.raises(SelectionFormatError):
        selection.initialize()
    with pytest.raises(SelectionFormatError):
        selection.read()
    assert selection.path.read_text() == value


def test_initialization_cannot_reset_committed_selection(tmp_path: Path) -> None:
    selection = PluginSelection(tmp_path)
    selection.initialize()
    ref = selection.commit((), expected_ref=None)
    with pytest.raises(SelectionFormatError):
        selection.initialize()
    assert selection.read() == ref


def test_missing_or_corrupt_record_does_not_fall_back(tmp_path: Path) -> None:
    selection = PluginSelection(tmp_path)
    selection.initialize()
    ref = selection.commit((), expected_ref=None)
    record = selection.archive.path / f"{ref}.json"
    record.write_text('{}')
    with pytest.raises(SelectionFormatError, match="损坏"):
        selection.read()
    record.unlink()
    with pytest.raises(SelectionFormatError, match="缺失"):
        selection.read()


def test_pointer_symlink_is_rejected_without_touching_target(tmp_path: Path) -> None:
    selection = PluginSelection(tmp_path)
    selection.path.parent.mkdir()
    target = tmp_path / "unrelated"
    target.write_text("keep")
    selection.path.symlink_to(target)
    with pytest.raises(SelectionFormatError):
        selection.initialize()
    with pytest.raises(SelectionFormatError):
        selection.read()
    assert target.read_text() == "keep"


def test_missing_component_cannot_change_selection(tmp_path: Path) -> None:
    selection = PluginSelection(tmp_path)
    selection.initialize()
    previous = selection.commit((), expected_ref=None)
    with pytest.raises(FileNotFoundError):
        selection.commit(("a" * 64,), expected_ref=previous)
    assert selection.read() == previous


def test_binding_root_cannot_be_read_as_selection_record(tmp_path: Path) -> None:
    selection = PluginSelection(tmp_path)
    selection.initialize()
    archive = PluginArchive(selection.archive.path)
    binding_ref = archive.save_descriptor({"components": []})
    selection.path.write_text('{"version":1,"root_ref":"' + binding_ref + '"}')
    with pytest.raises(SelectionFormatError, match="完整选择"):
        selection.read()


def test_record_write_failure_reports_unchanged(tmp_path: Path, monkeypatch) -> None:
    selection = PluginSelection(tmp_path)
    selection.initialize()

    def fail(*args, **kwargs):
        raise OSError("archive write failed")

    monkeypatch.setattr(PluginArchive, "save_descriptor", fail)
    with pytest.raises(SelectionWriteError) as failure:
        selection.commit((), expected_ref=None)
    assert failure.value.outcome == "unchanged"
    assert failure.value.observed_ref is None
    assert failure.value.observation_error is None


@pytest.mark.parametrize("replace_first", [False, True])
def test_replace_failure_reports_observed_pointer_without_rollback(
    tmp_path: Path, monkeypatch, replace_first: bool,
) -> None:
    selection = PluginSelection(tmp_path)
    selection.initialize()
    previous = selection.commit((), expected_ref=None)
    replace = selection_module.os.replace

    def fail(source, target):
        if replace_first:
            replace(source, target)
        raise OSError("replace failed")

    monkeypatch.setattr(selection_module.os, "replace", fail)
    with pytest.raises(SelectionWriteError) as failure:
        selection.commit((), expected_ref=previous)
    error = failure.value
    assert error.outcome == "uncertain"
    assert error.observation_error is None
    assert error.observed_ref == (error.target_ref if replace_first else previous)
    assert selection.read() == error.observed_ref
    assert error.target_ref is not None
    assert selection.archive.read_descriptor(error.target_ref)["previous"] == previous


def test_directory_sync_failure_does_not_claim_commit_or_restore_old(tmp_path: Path, monkeypatch) -> None:
    selection = PluginSelection(tmp_path)
    selection.initialize()
    previous = selection.commit((), expected_ref=None)

    def fail(path):
        raise OSError("directory fsync failed")

    monkeypatch.setattr(selection_module, "sync_directory", fail)
    with pytest.raises(SelectionWriteError) as failure:
        selection.commit((), expected_ref=previous)
    assert failure.value.outcome == "uncertain"
    assert failure.value.observed_ref == failure.value.target_ref
    assert selection.read() != previous


def test_interruption_after_replace_still_reports_uncertain(tmp_path: Path, monkeypatch) -> None:
    selection = PluginSelection(tmp_path)
    selection.initialize()
    replace = selection_module.os.replace

    def interrupt(source, target):
        replace(source, target)
        raise KeyboardInterrupt

    monkeypatch.setattr(selection_module.os, "replace", interrupt)
    with pytest.raises(SelectionWriteError) as failure:
        selection.commit((), expected_ref=None)
    assert isinstance(failure.value.__cause__, KeyboardInterrupt)
    assert failure.value.outcome == "uncertain"
    assert selection.read() == failure.value.target_ref


def test_failed_observation_is_not_reported_as_null(tmp_path: Path, monkeypatch) -> None:
    selection = PluginSelection(tmp_path)
    selection.initialize()

    def fail(source, target):
        selection.path.write_text("unreadable pointer")
        raise OSError("write failed")

    monkeypatch.setattr(selection_module.os, "replace", fail)
    with pytest.raises(SelectionWriteError) as failure:
        selection.commit((), expected_ref=None)
    assert failure.value.outcome == "uncertain"
    assert isinstance(failure.value.observation_error, SelectionFormatError)


def test_init_only_initializes_selection_for_directory_it_created(tmp_path: Path) -> None:
    workspace = tmp_path / "new"
    config = tmp_path / "config.toml"
    init_workspace(config_path=config, workspace=workspace)
    assert (workspace / "migrations.sqlite3").is_file()
    selection = PluginSelection(workspace)
    ref = selection.commit((), expected_ref=None)
    init_workspace(config_path=config, workspace=workspace)
    assert selection.read() == ref

    existing = tmp_path / "existing"
    existing.mkdir()
    summary = init_workspace(config_path=config, workspace=existing)
    assert not PluginSelection(existing).path.exists()
    assert any("显式升级" in note for note in summary.notes)

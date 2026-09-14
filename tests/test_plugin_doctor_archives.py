from pathlib import Path


from agent.plugins.archive import PluginArchive


def test_read_only_archive_does_not_create_directory(tmp_path):
    root = tmp_path / "missing"
    PluginArchive(root, create=False)
    assert not root.exists()

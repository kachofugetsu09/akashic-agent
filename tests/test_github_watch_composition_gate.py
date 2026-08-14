from __future__ import annotations

# pyright: reportPrivateUsage=false

import json
from copy import deepcopy
from pathlib import Path

import pytest

from docker.debug.github_watch_composition_gate import (
    DEFAULT_LOCK,
    ProviderSource,
    _expected_observations,
    _load_lock,
    _validate_provider_report,
)
from docker.debug import github_watch_composition_gate as gate_module


def test_checked_in_lock_binds_exact_github_watch_candidate() -> None:
    lock = _load_lock(DEFAULT_LOCK)

    assert lock.profile == "github-watch-v3-composition-v1"
    assert lock.provider.resolved_sha == (
        "3613956a6c6b95f31abd7d6d58464b878600ac05"
    )
    assert lock.provider.tree == "310449a8d7977cf5d5c71a44b3038e20b48bc955"
    assert lock.provider.source_digest == (
        "5aad531c1741f9c970928a29c687ede4087cf7ed05fa33378a2e3b3236eb6976"
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ({"schema_version": 2}, "不支持的 GitHub Watch 组合锁版本"),
        ({"extra": True}, "GitHub Watch 组合锁根结构无效"),
        ({"protocol_sources": []}, "protocol_sources 集合无效"),
    ),
)
def test_github_watch_lock_rejects_schema_drift(
    tmp_path: Path,
    mutation: dict[str, object],
    message: str,
) -> None:
    raw = json.loads(DEFAULT_LOCK.read_text(encoding="utf-8"))
    raw.update(mutation)
    path = tmp_path / "lock.json"
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        _ = _load_lock(path)


def test_requested_ref_fetch_uses_isolated_git_checkout(
    tmp_path: Path,
    monkeypatch,
) -> None:
    commands: list[tuple[tuple[str, ...], Path]] = []
    checkout = tmp_path / "provider"

    def fake_run(command: tuple[str, ...], *, cwd: Path) -> None:
        commands.append((command, cwd))

    def fake_git(root: Path, *args: str) -> str:
        assert root == checkout
        assert args == ("rev-parse", "FETCH_HEAD")
        return "d" * 40

    monkeypatch.setattr(gate_module, "_run", fake_run)
    monkeypatch.setattr(gate_module, "_git", fake_git)

    resolved = gate_module._fetch_requested_ref(_provider_source(), checkout)

    assert resolved == "d" * 40
    assert commands == [
        (("git", "init", "--quiet", str(checkout)), gate_module.ROOT),
        (
            (
                "git",
                "remote",
                "add",
                "origin",
                "https://github.com/kachofugetsu09/github-watch.git",
            ),
            checkout,
        ),
        (
            (
                "git",
                "fetch",
                "--quiet",
                "--depth=1",
                "origin",
                "refs/pull/1/head",
            ),
            checkout,
        ),
    ]


@pytest.mark.parametrize(
    ("mutate", "label"),
    (
        (lambda value: value.update(fake_watch_count=2), "candidate-tick"),
        (
            lambda value: value.update(
                listeners=["emit:turn.after_turn.committed:github-watch"]
            ),
            "event-mode",
        ),
        (
            lambda value: value.update(old_root_effects_after_drain=["timer:poll"]),
            "effect-drain",
        ),
    ),
)
def test_core_oracle_kills_known_github_watch_mutants(mutate, label: str) -> None:
    del label
    source = _provider_source()
    report = _valid_provider_report(source)
    observations = deepcopy(_expected_observations())
    mutate(observations)
    report["observations"] = observations

    with pytest.raises(RuntimeError, match="行为漂移"):
        _validate_provider_report(
            report,
            source,
            core_head="b" * 40,
            core_tree="c" * 40,
        )


def _provider_source() -> ProviderSource:
    return _load_lock(DEFAULT_LOCK).provider


def _valid_provider_report(source: ProviderSource) -> dict[str, object]:
    return {
        "status": "passed",
        "core_head": "b" * 40,
        "core_tree": "c" * 40,
        "plugin_head": source.resolved_sha,
        "plugin_tree": source.tree,
        "plugin_dirty": [],
        "plugin_source_digest": source.source_digest,
        "observations": _expected_observations(),
    }

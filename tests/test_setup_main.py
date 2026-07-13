from __future__ import annotations

import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.config import load_config
from agent.model_runtime.auth import Credential, CredentialStore
from bootstrap.setup_main import patch_main_model_config, run_main_model_setup
from bootstrap.setup_wizard import WizardAnswers, _phase_codex_llm


_CONFIG = """\
# 顶部说明必须保留
[llm]
main = "old_main" # 当前主模型
fast = "fast"
agent = "agent"
vl = "vl"

[llm.runtimes.old_main]
provider = "openai"
model = "old-model"
api_key = "old-secret"
base_url = "https://old.example/v1"
context_window = 32000

# fast 注释必须原样保留
[llm.runtimes.fast]
provider = "openai"
model = "fast-model"
api_key = "fast-secret"
base_url = "https://fast.example/v1"
context_window = 16000

[llm.runtimes.agent]
provider = "openai"
model = "agent-model"
api_key = "agent-secret"
base_url = "https://agent.example/v1"
context_window = 64000

[llm.runtimes.vl]
provider = "openai"
model = "vl-model"
api_key = "vl-secret"
base_url = "https://vl.example/v1"
context_window = 64000
input_modalities = ["text", "image"]

[agent]
max_tokens = 777 # 不由 setup-main 修改

[agent.context]
memory_window = 20 # 根据主上下文更新

[channels.telegram]
enabled = false
token = "telegram-stays"

# plugin/default-memory 自定义配置文本
[plugins.custom]
enabled = true
"""


def _api_answers() -> WizardAnswers:
    return WizardAnswers(
        provider="openai",
        model="new-main",
        api_key="new-secret",
        auth_id="main_default",
        base_url="https://new.example/v1",
        context_window=64_000,
        max_output_tokens=8192,
        memory_window=40,
    )


def test_patch_main_preserves_non_main_config_and_comments() -> None:
    updated = patch_main_model_config(_CONFIG, _api_answers())
    parsed = tomllib.loads(updated)

    assert parsed["llm"]["main"] == "api_main"
    assert parsed["llm"]["fast"] == "fast"
    assert parsed["llm"]["agent"] == "agent"
    assert parsed["llm"]["vl"] == "vl"
    assert parsed["llm"]["runtimes"]["api_main"]["model"] == "new-main"
    assert parsed["agent"]["context"]["memory_window"] == 40
    assert parsed["agent"]["max_tokens"] == 777
    assert "# fast 注释必须原样保留" in updated
    assert 'token = "telegram-stays"' in updated
    assert "# plugin/default-memory 自定义配置文本" in updated
    assert "new-secret" not in updated


def test_patch_main_is_idempotent_without_duplicate_runtime() -> None:
    once = patch_main_model_config(_CONFIG, _api_answers())
    twice = patch_main_model_config(once, _api_answers())

    assert twice == once
    assert twice.count("[llm.runtimes.api_main]") == 1
    assert tomllib.loads(twice)["llm"]["main"] == "api_main"


def test_run_main_setup_backs_up_and_only_rewrites_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    config_path = tmp_path / "config.toml"
    config_path.write_text(_CONFIG, encoding="utf-8")
    original = config_path.read_text(encoding="utf-8")

    def fill_answers(answers: WizardAnswers, **_: object) -> None:
        selected = _api_answers()
        answers.__dict__.update(selected.__dict__)

    monkeypatch.setattr("bootstrap.setup_main._phase_main_llm", fill_answers)

    run_main_model_setup(config_path)

    backup = tmp_path / "config.toml.before-setup-main.bak"
    assert backup.read_text(encoding="utf-8") == original
    config = load_config(config_path)
    assert config.runtime_id == "api_main"
    assert config.model == "new-main"
    assert config.fast_runtime_id == "fast"
    assert CredentialStore().get("main_default").access_token == "new-secret"


def test_codex_phase_reuses_existing_auth_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    CredentialStore().put(
        "codex_default",
        Credential(driver="codex", access_token="existing-token"),
    )
    login_started = False

    class FakeAuth:
        def __init__(self, store: CredentialStore, credential_id: str) -> None:
            assert credential_id == "codex_default"

        def begin_device_login(self):
            nonlocal login_started
            login_started = True
            raise AssertionError("不应重新登录")

    capabilities = SimpleNamespace(
        supported_reasoning_efforts=(),
        default_reasoning_effort="",
        context_window=128_000,
        input_modalities=("text",),
    )
    model = SimpleNamespace(
        slug="gpt-test",
        capabilities=capabilities,
        input_modalities_known=True,
    )

    class FakeCatalog:
        def __init__(self, auth: FakeAuth) -> None:
            pass

        async def list_models(self):
            return [model]

    monkeypatch.setattr("agent.model_runtime.auth.CodexAuthDriver", FakeAuth)
    monkeypatch.setattr("agent.model_runtime.catalog.CodexModelCatalog", FakeCatalog)
    monkeypatch.setattr(
        "bootstrap.setup_wizard.click.prompt",
        lambda _text, **kwargs: kwargs.get("default", ""),
    )
    monkeypatch.setattr(
        "bootstrap.setup_wizard.click.confirm",
        lambda _text, *, default=False: default,
    )
    answers = WizardAnswers()

    _phase_codex_llm(answers, reuse_existing_auth=True)

    assert login_started is False
    assert answers.model == "gpt-test"
    assert answers.context_window == 128_000

from __future__ import annotations

import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.config import load_config
from agent.model_runtime.auth.store import Credential, CredentialStore
from bootstrap.setup_main import patch_main_model_config, run_main_model_setup
from bootstrap.setup_wizard import WizardAnswers, _phase_codex_llm


_CONFIG = """\
# 顶部说明必须保留
[llm]
main = "api_main" # 当前主模型
fast = "fast"

[llm.runtimes.api_main]
provider = "openai"
model = "old"
api_key = "old-secret"
base_url = "https://old.example/v1"
context_window = 32000

# fast 注释必须保留
[llm.runtimes.fast]
provider = "deepseek"
model = "fast"
api_key = "fast-secret"
base_url = "https://api.deepseek.com/v1"
context_window = 16000

[agent.context]
memory_window = 20 # 自动更新

[plugins.custom]
enabled = true
"""


def _answers() -> WizardAnswers:
    return WizardAnswers(
        provider="deepseek",
        model="new-main",
        api_key="new-secret",
        auth_id="main_default",
        base_url="https://api.deepseek.com/v1",
        context_window=64_000,
        max_output_tokens=8192,
        memory_window=40,
    )


def test_patch_main_is_scoped_secret_free_and_idempotent() -> None:
    once = patch_main_model_config(_CONFIG, _answers())
    parsed = tomllib.loads(once)

    assert parsed["llm"]["main"] == "api_main"
    assert parsed["llm"]["fast"] == "fast"
    assert parsed["llm"]["runtimes"]["api_main"]["model"] == "new-main"
    assert "api_key" not in parsed["llm"]["runtimes"]["api_main"]
    assert parsed["agent"]["context"]["memory_window"] == 40
    assert "# fast 注释必须保留" in once
    assert "[plugins.custom]" in once
    assert "new-secret" not in once
    assert patch_main_model_config(once, _answers()) == once


def test_setup_main_backs_up_config_and_persists_credential(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    path = tmp_path / "config.toml"
    path.write_text(_CONFIG, encoding="utf-8")

    def fill(target: WizardAnswers, **_: object) -> None:
        target.__dict__.update(_answers().__dict__)

    monkeypatch.setattr("bootstrap.setup_main._phase_main_llm", fill)
    workspace = tmp_path / "workspace"
    run_main_model_setup(path, workspace)

    assert path.with_name("config.toml.before-setup-main.bak").read_text() == _CONFIG
    config = load_config(path, workspace=workspace)
    assert (config.model, config.fast_runtime_id) == ("new-main", "fast")
    assert CredentialStore().get("main_default").access_token == "new-secret"


def test_codex_setup_reuses_existing_login_and_catalog(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    CredentialStore().put(
        "codex_default",
        Credential(driver="codex", access_token="existing-token"),
    )

    class Auth:
        def __init__(self, store: CredentialStore, credential_id: str) -> None:
            assert credential_id == "codex_default"

        def begin_device_login(self) -> None:
            raise AssertionError("不应重新登录")

    capabilities = SimpleNamespace(
        supported_reasoning_efforts=(),
        default_reasoning_effort="",
        context_window=128_000,
        max_context_window=1_000_000,
        max_output_tokens=32_768,
        effective_context_percent=0.95,
        input_modalities=("text",),
        use_responses_lite=False,
        supports_parallel_tool_calls=True,
        supports_reasoning_summaries=True,
    )

    class Catalog:
        def __init__(self, auth: Auth) -> None:
            pass

        async def list_models(self) -> list[SimpleNamespace]:
            return [SimpleNamespace(
                slug="gpt-test",
                capabilities=capabilities,
                input_modalities_known=True,
            )]

    monkeypatch.setattr("agent.model_runtime.auth.codex.CodexAuthDriver", Auth)
    monkeypatch.setattr("agent.model_runtime.catalog.codex.CodexModelCatalog", Catalog)
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

    assert (answers.model, answers.context_window) == ("gpt-test", 128_000)
    assert answers.reasoning_summary == "auto"

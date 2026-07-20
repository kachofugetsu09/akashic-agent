from __future__ import annotations

import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.config import load_config
from agent.model_runtime.auth.store import Credential, CredentialStore
from bootstrap.setup_main import patch_main_model_config, run_main_model_setup
from bootstrap.setup_wizard import WizardAnswers, _phase_api_key_llm, _phase_codex_llm


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


def test_patch_main_is_scoped_inline_key_and_idempotent() -> None:
    once = patch_main_model_config(_CONFIG, _answers())
    parsed = tomllib.loads(once)

    assert parsed["llm"]["main"] == "deepseek_main"
    assert parsed["llm"]["fast"] == "fast"
    assert parsed["llm"]["runtimes"]["deepseek_main"]["model"] == "new-main"
    assert parsed["llm"]["runtimes"]["deepseek_main"]["api_key"] == "new-secret"
    assert parsed["agent"]["context"]["memory_window"] == 40
    assert "# fast 注释必须保留" in once
    assert "[plugins.custom]" in once
    assert "new-secret" in once
    assert patch_main_model_config(once, _answers()) == once


def test_setup_main_backs_up_config_and_persists_inline_key(
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
    parsed = tomllib.loads(path.read_text(encoding="utf-8"))
    assert parsed["llm"]["runtimes"]["deepseek_main"]["api_key"] == "new-secret"


def test_patch_main_reuses_saved_inline_key() -> None:
    first = patch_main_model_config(_CONFIG, _answers())
    answers = _answers()
    answers.api_key = ""

    second = patch_main_model_config(first, answers)

    assert tomllib.loads(second)["llm"]["runtimes"]["deepseek_main"]["api_key"] == "new-secret"


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


def test_opencode_go_setup_uses_dynamic_catalog_and_forces_text_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Catalog:
        def __init__(self, api_key: str, *, base_url: str) -> None:
            assert api_key == "secret"
            assert base_url == "https://opencode.ai/zen/go/v1"

        async def list_models(self) -> list[SimpleNamespace]:
            return [SimpleNamespace(slug="glm-5.99")]

    confirms: list[str] = []

    def prompt(text: str, **kwargs: object) -> object:
        if text == "服务商":
            return "opencode-go"
        return kwargs.get("default", "")

    def confirm(text: str, *, default: bool = False) -> bool:
        confirms.append(text)
        return default

    monkeypatch.setattr(
        "agent.model_runtime.catalog.opencode_go.OpenCodeGoModelCatalog",
        Catalog,
    )
    monkeypatch.setattr("bootstrap.setup_wizard.click.prompt", prompt)
    monkeypatch.setattr("bootstrap.setup_wizard.click.confirm", confirm)
    monkeypatch.setattr("bootstrap.setup_wizard._secret_prompt", lambda _text: "secret")
    answers = WizardAnswers()

    _phase_api_key_llm(answers)

    assert answers.provider == "opencode-go"
    assert answers.model == "glm-5.99"
    assert answers.base_url == "https://opencode.ai/zen/go/v1"
    assert answers.multimodal is False
    assert "主模型原生支持图片输入？" not in confirms

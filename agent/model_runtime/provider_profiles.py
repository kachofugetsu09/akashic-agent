from __future__ import annotations

from dataclasses import dataclass

OPENCODE_GO_BASE_URL = "https://opencode.ai/zen/go/v1"


@dataclass(frozen=True)
class ProviderProfile:
    provider_id: str
    default_base_url: str
    chat_model_prefixes: tuple[str, ...]
    messages_model_prefixes: tuple[str, ...]
    input_modalities: tuple[str, ...] = ("text",)

    def classify_model(self, model: str) -> str:
        """按稳定模型家族判断 OpenAI Chat Completions 兼容性。"""
        normalized = model.strip().lower()
        if normalized.startswith(self.chat_model_prefixes):
            return "chat_completions"
        if normalized.startswith(self.messages_model_prefixes):
            return "messages"
        return "unknown"


OPENCODE_GO_PROFILE = ProviderProfile(
    provider_id="opencode-go",
    default_base_url=OPENCODE_GO_BASE_URL,
    chat_model_prefixes=("grok-", "glm-", "kimi-", "deepseek-", "mimo-"),
    messages_model_prefixes=("minimax-", "qwen"),
)

_PROFILES = {
    OPENCODE_GO_PROFILE.provider_id: OPENCODE_GO_PROFILE,
}


def get_provider_profile(provider: str) -> ProviderProfile | None:
    return _PROFILES.get(provider.strip().lower())


def validate_profile_runtime(
    *,
    provider: str,
    model: str,
    input_modalities: tuple[str, ...],
) -> None:
    """在配置边界拒绝 profile 不支持的协议和输入模态。"""
    profile = get_provider_profile(provider)
    if profile is None:
        return

    # 1. 模型家族决定 wire protocol；Messages 和未知家族都不得误发到 Chat。
    protocol = profile.classify_model(model)
    if protocol == "messages":
        raise ValueError(
            f"provider {profile.provider_id} 的模型 {model} 使用 Messages API，"
            "当前仅支持 Chat Completions 模型"
        )
    if protocol == "unknown":
        raise ValueError(
            f"provider {profile.provider_id} 的模型 {model} 不属于已支持的 "
            "Chat Completions 家族"
        )

    # 2. OpenCode Go profile 只声明文本输入能力。
    if input_modalities != profile.input_modalities:
        raise ValueError(
            f"provider {profile.provider_id} 仅支持 input_modalities = ['text']"
        )

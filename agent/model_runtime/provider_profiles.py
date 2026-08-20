from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlsplit, urlunsplit

OPENCODE_GO_BASE_URL = "https://opencode.ai/zen/go/v1"
OPENCODE_GO_MAX_TOOL_SCHEMAS = 16


@dataclass(frozen=True)
class ProviderProfile:
    provider_id: str
    default_base_url: str
    messages_model_prefixes: tuple[str, ...]
    input_modalities: tuple[str, ...] = ("text",)
    max_tool_schemas: int = 0

    def classify_model(self, model: str) -> str:
        """排除已知 Messages 家族，其余模型默认走 Chat Completions。"""
        normalized = model.strip().lower()
        if not normalized:
            return "unknown"
        if normalized.startswith(self.messages_model_prefixes):
            return "messages"
        return "chat_completions"


OPENCODE_GO_PROFILE = ProviderProfile(
    provider_id="opencode-go",
    default_base_url=OPENCODE_GO_BASE_URL,
    messages_model_prefixes=("minimax-", "qwen"),
    # Verified compatibility ceiling for the OpenCode Go endpoint. Larger local
    # catalogs remain available through tool_search instead of being discarded.
    max_tool_schemas=OPENCODE_GO_MAX_TOOL_SCHEMAS,
)

_PROFILES = {
    OPENCODE_GO_PROFILE.provider_id: OPENCODE_GO_PROFILE,
}


def get_provider_profile(provider: str) -> ProviderProfile | None:
    return _PROFILES.get(provider.strip().lower())


def is_opencode_go_base_url(base_url: str) -> bool:
    """Match the persisted wire endpoint, independent of the runtime display name."""

    value = base_url.strip()
    if not value:
        return False
    parsed = urlsplit(value)
    normalized = urlunsplit(
        (
            parsed.scheme.lower(),
            parsed.netloc.lower(),
            parsed.path.rstrip("/"),
            "",
            "",
        )
    )
    return normalized == OPENCODE_GO_BASE_URL


def get_runtime_provider_profile(
    *,
    provider: str,
    base_url: str,
) -> ProviderProfile | None:
    """Resolve compatibility from the actual endpoint before the logical provider id."""

    if is_opencode_go_base_url(base_url):
        return OPENCODE_GO_PROFILE
    return get_provider_profile(provider)


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

    # 1. 已知 Messages 家族不得误发到 Chat；新家族由真实请求继续验证。
    protocol = profile.classify_model(model)
    if protocol == "messages":
        raise ValueError(
            f"provider {profile.provider_id} 的模型 {model} 使用 Messages API，"
            "当前仅支持 Chat Completions 模型"
        )
    if protocol == "unknown":
        raise ValueError(f"provider {profile.provider_id} 的模型 ID 不能为空")

    # 2. OpenCode Go profile 只声明文本输入能力。
    if input_modalities != profile.input_modalities:
        raise ValueError(
            f"provider {profile.provider_id} 仅支持 input_modalities = ['text']"
        )

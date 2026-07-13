from .codex import CodexAuthDriver, DeviceCode
from .store import Credential, CredentialStore


class ApiKeyAuthDriver:
    """从统一凭据存储取得 API key。"""

    def __init__(self, store: CredentialStore, credential_id: str) -> None:
        self.store = store
        self.credential_id = credential_id

    def api_key(self) -> str:
        credential = self.store.get(self.credential_id)
        if credential.driver != "api_key" or not credential.access_token:
            raise ValueError(f"凭据 {self.credential_id} 不是有效 API key")
        return credential.access_token

__all__ = [
    "ApiKeyAuthDriver",
    "CodexAuthDriver",
    "Credential",
    "CredentialStore",
    "DeviceCode",
]

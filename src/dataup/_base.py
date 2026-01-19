"""Shared base code for DataUp clients."""

from __future__ import annotations

from abc import ABC

DEFAULT_BASE_URL = "https://api.data-up.io"
DEFAULT_TIMEOUT = 30.0
API_VERSION = "v1"


class BaseClient(ABC):
    """Abstract base class for DataUp API clients."""

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._validate_api_key()

    def _validate_api_key(self) -> None:
        """Validate API key format (key_id.key_secret)."""
        if not self.api_key or "." not in self.api_key:
            raise ValueError("Invalid API key format. Expected: 'key_id.key_secret'")

    @property
    def _headers(self) -> dict[str, str]:
        """Get default headers for requests."""
        return {
            "X-API-Key": self.api_key,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def _build_url(self, path: str) -> str:
        """Build full URL from path."""
        return f"{self.base_url}/api/{API_VERSION}{path}"

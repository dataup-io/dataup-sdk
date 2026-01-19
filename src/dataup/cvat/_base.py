"""Shared base code for CVAT clients."""

from __future__ import annotations

from abc import ABC

DEFAULT_CVAT_URL = "https://app.cvat.ai"
DEFAULT_TIMEOUT = 60.0
API_VERSION = "api"


class BaseCVATClient(ABC):
    """Abstract base class for CVAT API clients."""

    def __init__(
        self, api_token: str, *, base_url: str = DEFAULT_CVAT_URL, timeout: float = DEFAULT_TIMEOUT
    ) -> None:
        self.api_token = api_token
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._validate_token()

    def _validate_token(self) -> None:
        """Validate API token is provided."""
        if not self.api_token:
            raise ValueError("API token is required")

    @property
    def _headers(self) -> dict[str, str]:
        """Get default headers for requests."""
        return {
            "Authorization": f"Bearer {self.api_token}",
            # "Accept": "application/json",
        }

    @property
    def _json_headers(self) -> dict[str, str]:
        """Get headers for JSON requests."""
        return {
            **self._headers,
            "Content-Type": "application/json",
        }

    def _build_url(self, path: str) -> str:
        """Build full URL from path."""
        path = path.lstrip("/")
        return f"{self.base_url}/{API_VERSION}/{path}"

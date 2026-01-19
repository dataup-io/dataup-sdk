"""Custom exceptions for the CVAT client."""

from __future__ import annotations


class CVATError(Exception):
    """Base exception for all CVAT client errors."""

    pass


class CVATAPIError(CVATError):
    """API error with status code and message."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)

    def __str__(self) -> str:
        if self.status_code:
            return f"[{self.status_code}] {self.message}"
        return self.message


class CVATAuthenticationError(CVATAPIError):
    """401 Unauthorized - Invalid or missing API token."""

    pass


class CVATPermissionDeniedError(CVATAPIError):
    """403 Forbidden - Insufficient permissions."""

    pass


class CVATNotFoundError(CVATAPIError):
    """404 Not Found - Resource does not exist."""

    pass


class CVATValidationError(CVATAPIError):
    """400 Bad Request - Invalid request data."""

    pass


class CVATRateLimitError(CVATAPIError):
    """429 Too Many Requests - Rate limit exceeded."""

    pass


class CVATConnectionError(CVATError):
    """Network connectivity error."""

    pass


class CVATTimeoutError(CVATError):
    """Request timeout error."""

    pass

"""Custom exceptions for the DataUp SDK."""

from __future__ import annotations


class DataUpError(Exception):
    """Base exception for all DataUp SDK errors."""

    pass


class DataUpAPIError(DataUpError):
    """API error with status code and message."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)

    def __str__(self) -> str:
        if self.status_code:
            return f"[{self.status_code}] {self.message}"
        return self.message


class AuthenticationError(DataUpAPIError):
    """401 Unauthorized - Invalid or missing API key."""

    pass


class PermissionDeniedError(DataUpAPIError):
    """403 Forbidden - Insufficient permissions."""

    pass


class NotFoundError(DataUpAPIError):
    """404 Not Found - Resource does not exist."""

    pass


class ConflictError(DataUpAPIError):
    """409 Conflict - Resource already exists or state conflict."""

    pass


class ValidationError(DataUpAPIError):
    """400 Bad Request - Invalid request data."""

    pass


class RateLimitError(DataUpAPIError):
    """429 Too Many Requests - Rate limit exceeded."""

    pass


class ConnectionError(DataUpError):
    """Network connectivity error."""

    pass


class TimeoutError(DataUpError):
    """Request timeout error."""

    pass

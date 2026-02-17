"""Custom exceptions for mistral-ratelimit."""


class MistralRatelimitError(Exception):
    """Base exception for all mistral-ratelimit errors."""

    pass


class RateLimitExceeded(MistralRatelimitError):
    """Raised when rate limit is exceeded and retries are exhausted.

    Attributes:
        retry_after: Seconds to wait before retrying (if known)
        limit_type: Type of limit that was exceeded ("requests" or "tokens")
    """

    def __init__(
        self,
        message: str,
        retry_after: float | None = None,
        limit_type: str | None = None,
    ):
        super().__init__(message)
        self.retry_after = retry_after
        self.limit_type = limit_type


class TokenCountingError(MistralRatelimitError):
    """Raised when token counting fails."""

    pass


class ConfigurationError(MistralRatelimitError):
    """Raised when configuration is invalid."""

    pass

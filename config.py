"""Configuration for mistral-ratelimit."""

from dataclasses import dataclass


@dataclass
class RatelimitConfig:
    """Configuration for the rate-limited Mistral client.

    Attributes:
        api_key: Mistral API key (required). Can also be set via MISTRAL_API_KEY env var.
        requests_per_second: Maximum requests per second (default: 10.0)
        tokens_per_minute: Maximum tokens per minute (default: 100,000)
        max_retries: Maximum number of retries on rate limit errors (default: 3)
        base_delay: Base delay for exponential backoff in seconds (default: 1.0)
        max_delay: Maximum delay between retries in seconds (default: 32.0)
        timeout: Request timeout in seconds (default: 60.0)
    """

    api_key: str | None = None
    requests_per_second: float = 1.0  # Conservative default (matches free tier)
    tokens_per_minute: int = 500_000  # Free tier default
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 32.0
    timeout: float = 60.0

    def __post_init__(self):
        """Validate configuration after initialization."""
        import os

        # Auto-load API key from environment if not provided
        if self.api_key is None:
            self.api_key = os.environ.get("MISTRAL_API_KEY")

        if self.api_key is None:
            raise ValueError(
                "api_key must be provided either directly or via MISTRAL_API_KEY environment variable"
            )

        if self.requests_per_second <= 0:
            raise ValueError("requests_per_second must be positive")

        if self.tokens_per_minute <= 0:
            raise ValueError("tokens_per_minute must be positive")

        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")

        if self.base_delay <= 0:
            raise ValueError("base_delay must be positive")

        if self.max_delay <= 0:
            raise ValueError("max_delay must be positive")

        if self.timeout <= 0:
            raise ValueError("timeout must be positive")

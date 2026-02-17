"""mistral-ratelimit - A rate-limited wrapper for Mistral AI client.

Simple SDK to easily add rate limits per second and tokens per minute.
"""

from mistral_ratelimit.client import MistralRatelimitClient
from mistral_ratelimit.config import RatelimitConfig
from mistral_ratelimit.exceptions import (
    ConfigurationError,
    MistralRatelimitError,
    RateLimitExceeded,
    TokenCountingError,
)

# For local development
try:
    from mistral_ratelimit.client import MistralRatelimitClient
except ImportError:
    from client import MistralRatelimitClient
try:
    from mistral_ratelimit.config import RatelimitConfig
except ImportError:
    from config import RatelimitConfig
try:
    from mistral_ratelimit.exceptions import (
        ConfigurationError,
        MistralRatelimitError,
        RateLimitExceeded,
        TokenCountingError,
    )
except ImportError:
    from exceptions import (
        ConfigurationError,
        MistralRatelimitError,
        RateLimitExceeded,
        TokenCountingError,
    )

__version__ = "0.1.0"

__all__ = [
    "MistralRatelimitClient",
    "RatelimitConfig",
    "MistralRatelimitError",
    "RateLimitExceeded",
    "TokenCountingError",
    "ConfigurationError",
]

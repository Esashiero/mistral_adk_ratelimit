"""Rate limiter using token bucket algorithm."""

import asyncio
import time
import threading
from dataclasses import dataclass


@dataclass
class RateLimitState:
    """State for a single rate limit bucket.

    Attributes:
        available: Current available tokens
        max_tokens: Maximum token capacity
        refill_rate: Tokens added per second
        last_update: Timestamp of last update
    """

    available: float
    max_tokens: float
    refill_rate: float
    last_update: float


class RateLimiter:
    """Thread-safe rate limiter using token bucket algorithm.

    Supports two independent limits:
    - Requests per second (for controlling request rate)
    - Tokens per minute (for controlling token usage)

    Uses token bucket for both, which allows burst handling while
    maintaining average rate.
    """

    def __init__(
        self,
        requests_per_second: float = 10.0,
        tokens_per_minute: int = 100_000,
    ):
        """Initialize rate limiter.

        Args:
            requests_per_second: Maximum requests per second
            tokens_per_minute: Maximum tokens per minute
        """
        # Request limiter: capacity = refill_rate (allows 1 burst)
        self._request_bucket = RateLimitState(
            available=requests_per_second,
            max_tokens=requests_per_second,
            refill_rate=requests_per_second,
            last_update=time.monotonic(),
        )

        # Token limiter: convert to per-second rate
        tokens_per_second = tokens_per_minute / 60.0
        self._token_bucket = RateLimitState(
            available=tokens_per_minute,
            max_tokens=tokens_per_minute,
            refill_rate=tokens_per_second,
            last_update=time.monotonic(),
        )

        self._lock = threading.Lock()

    def _refill(self, bucket: RateLimitState) -> None:
        """Refill tokens in a bucket based on elapsed time.

        Args:
            bucket: The bucket to refill
        """
        now = time.monotonic()
        elapsed = now - bucket.last_update

        # Add tokens based on elapsed time
        new_tokens = elapsed * bucket.refill_rate
        bucket.available = min(bucket.max_tokens, bucket.available + new_tokens)
        bucket.last_update = now

    def _acquire_request(self, tokens: int = 1) -> float:
        """Acquire tokens from request bucket.

        Args:
            tokens: Number of tokens to acquire

        Returns:
            Time waited (0 if acquired immediately)
        """
        self._refill(self._request_bucket)

        if self._request_bucket.available >= tokens:
            self._request_bucket.available -= tokens
            return 0.0

        # Calculate wait time for tokens to become available
        needed = tokens - self._request_bucket.available
        wait_time = needed / self._request_bucket.refill_rate

        return wait_time

    def _acquire_token(self, tokens: int = 1) -> float:
        """Acquire tokens from token bucket.

        Args:
            tokens: Number of tokens to acquire

        Returns:
            Time waited (0 if acquired immediately)
        """
        self._refill(self._token_bucket)

        if self._token_bucket.available >= tokens:
            self._token_bucket.available -= tokens
            return 0.0

        # Calculate wait time for tokens to become available
        needed = tokens - self._token_bucket.available
        wait_time = needed / self._token_bucket.refill_rate

        return wait_time

    def acquire(self, request_tokens: int = 1, token_count: int = 0) -> float:
        """Acquire from both rate limits.

        Blocks until both limits allow the request.

        Args:
            request_tokens: Tokens to acquire from request bucket (default: 1)
            token_count: Tokens to acquire from token bucket (default: 0)

        Returns:
            Total time waited in seconds
        """
        total_wait = 0.0

        with self._lock:
            # Acquire from request bucket
            wait = self._acquire_request(request_tokens)
            total_wait += wait

            if wait > 0:
                # Need to wait - update time and retry
                time.sleep(wait)
                # Re-acquire after waiting
                self._refill(self._request_bucket)
                self._request_bucket.available -= request_tokens

            # Acquire from token bucket
            wait = self._acquire_token(token_count)
            total_wait += wait

            if wait > 0:
                time.sleep(wait)
                # Re-acquire after waiting
                self._refill(self._token_bucket)
                self._token_bucket.available -= token_count

        return total_wait

    def get_wait_time(
        self,
        request_tokens: int = 1,
        token_count: int = 0,
    ) -> float:
        """Calculate wait time without actually acquiring.

        Useful for pre-checking before making a request.

        Args:
            request_tokens: Tokens to check in request bucket
            token_count: Tokens to check in token bucket

        Returns:
            Estimated wait time in seconds
        """
        with self._lock:
            self._refill(self._request_bucket)
            self._refill(self._token_bucket)

            wait_request = 0.0
            if self._request_bucket.available < request_tokens:
                needed = request_tokens - self._request_bucket.available
                wait_request = needed / self._request_bucket.refill_rate

            wait_token = 0.0
            if self._token_bucket.available < token_count:
                needed = token_count - self._token_bucket.available
                wait_token = needed / self._token_bucket.refill_rate

            return max(wait_request, wait_token)


class AsyncRateLimiter:
    """Async version of RateLimiter.

    Same functionality but uses asyncio.Lock for thread safety
    in async contexts.
    """

    def __init__(
        self,
        requests_per_second: float = 10.0,
        tokens_per_minute: int = 100_000,
    ):
        """Initialize async rate limiter.

        Args:
            requests_per_second: Maximum requests per second
            tokens_per_minute: Maximum tokens per minute
        """
        # Request limiter
        self._request_bucket = RateLimitState(
            available=requests_per_second,
            max_tokens=requests_per_second,
            refill_rate=requests_per_second,
            last_update=time.monotonic(),
        )

        # Token limiter
        tokens_per_second = tokens_per_minute / 60.0
        self._token_bucket = RateLimitState(
            available=tokens_per_minute,
            max_tokens=tokens_per_minute,
            refill_rate=tokens_per_second,
            last_update=time.monotonic(),
        )

        self._lock = asyncio.Lock()

    def _refill(self, bucket: RateLimitState) -> None:
        """Refill tokens in a bucket based on elapsed time."""
        now = time.monotonic()
        elapsed = now - bucket.last_update
        new_tokens = elapsed * bucket.refill_rate
        bucket.available = min(bucket.max_tokens, bucket.available + new_tokens)
        bucket.last_update = now

    async def acquire(
        self,
        request_tokens: int = 1,
        token_count: int = 0,
    ) -> float:
        """Acquire from both rate limits asynchronously.

        Args:
            request_tokens: Tokens to acquire from request bucket
            token_count: Tokens to acquire from token bucket

        Returns:
            Total time waited in seconds
        """
        total_wait = 0.0

        async with self._lock:
            # Request bucket
            self._refill(self._request_bucket)
            if self._request_bucket.available >= request_tokens:
                self._request_bucket.available -= request_tokens
            else:
                needed = request_tokens - self._request_bucket.available
                wait_time = needed / self._request_bucket.refill_rate
                total_wait += wait_time
                await asyncio.sleep(wait_time)
                self._refill(self._request_bucket)
                self._request_bucket.available -= request_tokens

            # Token bucket
            self._refill(self._token_bucket)
            if self._token_bucket.available >= token_count:
                self._token_bucket.available -= token_count
            else:
                needed = token_count - self._token_bucket.available
                wait_time = needed / self._token_bucket.refill_rate
                total_wait += wait_time
                await asyncio.sleep(wait_time)
                self._refill(self._token_bucket)
                self._token_bucket.available -= token_count

        return total_wait

    async def get_wait_time(
        self,
        request_tokens: int = 1,
        token_count: int = 0,
    ) -> float:
        """Calculate wait time without acquiring."""
        async with self._lock:
            self._refill(self._request_bucket)
            self._refill(self._token_bucket)

            wait_request = 0.0
            if self._request_bucket.available < request_tokens:
                needed = request_tokens - self._request_bucket.available
                wait_request = needed / self._request_bucket.refill_rate

            wait_token = 0.0
            if self._token_bucket.available < token_count:
                needed = token_count - self._token_bucket.available
                wait_token = needed / self._token_bucket.refill_rate

            return max(wait_request, wait_token)

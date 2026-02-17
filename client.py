"""Rate-limited Mistral AI client using official SDK with beta.conversations API."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterator
from typing import Any, Mapping

from mistralai import Mistral
from mistralai.models import (
    ConversationResponse,
    ConversationStreamRequest,
    ConversationEvents,
)

# We import these for type checking, but use Any in signatures to avoid circular imports if strictly typed
try:
    from mistralai.utils import RetryConfig
except ImportError:
    RetryConfig = Any

from config import RatelimitConfig
from exceptions import MistralRatelimitError, RateLimitExceeded
from rate_limiter import AsyncRateLimiter, RateLimiter
from token_counter import TokenCounter


class RateLimitedConversations:
    """Rate-limited wrapper for Mistral beta.conversations API.

    Wraps all conversation methods with rate limiting using token bucket algorithm.
    Supports sync, async, and streaming variants with full parameter support.
    """

    def __init__(
        self,
        client: Mistral,
        rate_limiter: RateLimiter,
        async_rate_limiter: AsyncRateLimiter,
        token_counter: TokenCounter,
        config: RatelimitConfig,
    ):
        """Initialize rate-limited conversations wrapper."""
        self._client = client
        self._rate_limiter = rate_limiter
        self._async_rate_limiter = async_rate_limiter
        self._token_counter = token_counter
        self._config = config

    def _estimate_tokens(self, inputs: Any) -> int:
        """Estimate tokens from conversation inputs."""
        if isinstance(inputs, str):
            return self._token_counter.count_text(inputs)
        elif isinstance(inputs, list):
            # List of messages or entries
            if inputs and isinstance(inputs[0], dict):
                # Check if it's a message dict or entry dict
                if "content" in inputs[0]:
                    return self._token_counter.count_messages(inputs)
            # List of strings or mixed
            tokens = 0
            for item in inputs:
                if isinstance(item, str):
                    tokens += self._token_counter.count_text(item)
                elif isinstance(item, dict):
                    content = item.get("content", "")
                    if isinstance(content, str):
                        tokens += self._token_counter.count_text(content)
            return tokens
        elif isinstance(inputs, dict):
            content = inputs.get("content", "")
            if isinstance(content, str):
                return self._token_counter.count_text(content)
        return 100  # Conservative default

    def _refund_tokens(self, estimated: int, actual_tokens: int) -> None:
        """Refund unused tokens to the sync rate limiter bucket."""
        if actual_tokens < estimated:
            # Direct access to bucket is necessary here as RateLimiter doesn't expose refund
            self._rate_limiter._token_bucket.available += estimated - actual_tokens

    async def _refund_tokens_async(self, estimated: int, actual_tokens: int) -> None:
        """Refund unused tokens to the async rate limiter bucket."""
        if actual_tokens < estimated:
            async with self._async_rate_limiter._lock:
                self._async_rate_limiter._token_bucket.available += estimated - actual_tokens

    def _extract_usage(self, response: Any) -> int:
        """Extract total tokens used from a response object."""
        try:
            if hasattr(response, "usage"):
                usage = response.usage
                return getattr(usage, "total_tokens", 0) or 0
            elif isinstance(response, dict):
                usage = response.get("usage") or {}
                return usage.get("total_tokens", 0) or 0
        except Exception:
            pass
        return 0

    def _filter_none(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Filter out None values from kwargs to avoid sending nulls to API.

        The Mistral API rejects certain None values (e.g., handoff_execution with model).
        This ensures we only send parameters that are explicitly set.
        """
        return {k: v for k, v in kwargs.items() if v is not None}

    def _process_stream_response(self, stream: Iterator[Any], estimated: int) -> Iterator[Any]:
        """Intercept stream events to track usage and refund tokens."""
        for event in stream:
            # Check for usage in 'conversation.response.done' event
            # event.data might be an object or dict depending on SDK version
            if hasattr(event, "data"):
                data = event.data
                event_type = getattr(data, "type", "")
                if event_type == "conversation.response.done":
                    actual = self._extract_usage(data)
                    self._refund_tokens(estimated, actual)
            yield event

    async def _process_stream_response_async(
        self, stream: AsyncIterator[Any], estimated: int
    ) -> AsyncIterator[Any]:
        """Intercept async stream events to track usage and refund tokens."""
        async for event in stream:
            if hasattr(event, "data"):
                data = event.data
                event_type = getattr(data, "type", "")
                if event_type == "conversation.response.done":
                    actual = self._extract_usage(data)
                    await self._refund_tokens_async(estimated, actual)
            yield event

    # ==================== START ====================

    def start(
        self,
        inputs: Any,
        model: str | None = None,
        agent_id: str | None = None,
        instructions: str | None = None,
        tools: list[Any] | None = None,
        completion_args: dict[str, Any] | None = None,
        store: bool | None = None,
        handoff_execution: str | None = None,
        metadata: dict[str, Any] | None = None,
        name: str | None = None,
        description: str | None = None,
        agent_version: Any | None = None,
        retries: Any | None = None,
        server_url: str | None = None,
        timeout_ms: int | None = None,
        http_headers: Mapping[str, str] | None = None,
    ) -> Any:
        """Start a new conversation (synchronous)."""
        estimated = self._estimate_tokens(inputs) + 100
        self._rate_limiter.acquire(request_tokens=1, token_count=estimated)

        try:
            kwargs = self._filter_none(
                {
                    "model": model,
                    "agent_id": agent_id,
                    "instructions": instructions,
                    "tools": tools,
                    "completion_args": completion_args,
                    "store": store,
                    "handoff_execution": handoff_execution,
                    "metadata": metadata,
                    "name": name,
                    "description": description,
                    "agent_version": agent_version,
                    "retries": retries,
                    "server_url": server_url,
                    "timeout_ms": timeout_ms,
                    "http_headers": http_headers,
                }
            )
            kwargs["inputs"] = inputs
            kwargs["stream"] = False
            response = self._client.beta.conversations.start(**kwargs)
            actual = self._extract_usage(response)
            self._refund_tokens(estimated, actual)
            return response
        except Exception as e:
            raise self._handle_error(e) from e

    async def start_async(
        self,
        inputs: Any,
        model: str | None = None,
        agent_id: str | None = None,
        instructions: str | None = None,
        tools: list[Any] | None = None,
        completion_args: dict[str, Any] | None = None,
        store: bool | None = None,
        handoff_execution: str | None = None,
        metadata: dict[str, Any] | None = None,
        name: str | None = None,
        description: str | None = None,
        agent_version: Any | None = None,
        retries: Any | None = None,
        server_url: str | None = None,
        timeout_ms: int | None = None,
        http_headers: Mapping[str, str] | None = None,
    ) -> Any:
        """Start a new conversation (asynchronous)."""
        estimated = self._estimate_tokens(inputs) + 100
        await self._async_rate_limiter.acquire(request_tokens=1, token_count=estimated)

        try:
            kwargs = self._filter_none(
                {
                    "model": model,
                    "agent_id": agent_id,
                    "instructions": instructions,
                    "tools": tools,
                    "completion_args": completion_args,
                    "store": store,
                    "handoff_execution": handoff_execution,
                    "metadata": metadata,
                    "name": name,
                    "description": description,
                    "agent_version": agent_version,
                    "retries": retries,
                    "server_url": server_url,
                    "timeout_ms": timeout_ms,
                    "http_headers": http_headers,
                }
            )
            kwargs["inputs"] = inputs
            kwargs["stream"] = False
            response = await self._client.beta.conversations.start_async(**kwargs)
            actual = self._extract_usage(response)
            await self._refund_tokens_async(estimated, actual)
            return response
        except Exception as e:
            raise self._handle_error(e) from e

    # ==================== START STREAM ====================

    def start_stream(
        self,
        inputs: Any,
        model: str | None = None,
        agent_id: str | None = None,
        instructions: str | None = None,
        tools: list[Any] | None = None,
        completion_args: dict[str, Any] | None = None,
        store: bool | None = None,
        handoff_execution: str | None = None,
        metadata: dict[str, Any] | None = None,
        name: str | None = None,
        description: str | None = None,
        agent_version: Any | None = None,
        retries: Any | None = None,
        server_url: str | None = None,
        timeout_ms: int | None = None,
        http_headers: Mapping[str, str] | None = None,
    ) -> Iterator[Any]:
        """Start a conversation with streaming (synchronous)."""
        estimated = self._estimate_tokens(inputs) + 100
        self._rate_limiter.acquire(request_tokens=1, token_count=estimated)

        try:
            kwargs = self._filter_none(
                {
                    "model": model,
                    "agent_id": agent_id,
                    "instructions": instructions,
                    "tools": tools,
                    "completion_args": completion_args,
                    "store": store,
                    "handoff_execution": handoff_execution,
                    "metadata": metadata,
                    "name": name,
                    "description": description,
                    "agent_version": agent_version,
                    "retries": retries,
                    "server_url": server_url,
                    "timeout_ms": timeout_ms,
                    "http_headers": http_headers,
                }
            )
            kwargs["inputs"] = inputs
            kwargs["stream"] = True
            stream = self._client.beta.conversations.start_stream(**kwargs)
            yield from self._process_stream_response(stream, estimated)
        except Exception as e:
            raise self._handle_error(e) from e

    async def start_stream_async(
        self,
        inputs: Any,
        model: str | None = None,
        agent_id: str | None = None,
        instructions: str | None = None,
        tools: list[Any] | None = None,
        completion_args: dict[str, Any] | None = None,
        store: bool | None = None,
        handoff_execution: str | None = None,
        metadata: dict[str, Any] | None = None,
        name: str | None = None,
        description: str | None = None,
        agent_version: Any | None = None,
        retries: Any | None = None,
        server_url: str | None = None,
        timeout_ms: int | None = None,
        http_headers: Mapping[str, str] | None = None,
    ) -> AsyncIterator[Any]:
        """Start a conversation with streaming (asynchronous)."""
        estimated = self._estimate_tokens(inputs) + 100
        await self._async_rate_limiter.acquire(request_tokens=1, token_count=estimated)

        try:
            kwargs = self._filter_none(
                {
                    "model": model,
                    "agent_id": agent_id,
                    "instructions": instructions,
                    "tools": tools,
                    "completion_args": completion_args,
                    "store": store,
                    "handoff_execution": handoff_execution,
                    "metadata": metadata,
                    "name": name,
                    "description": description,
                    "agent_version": agent_version,
                    "retries": retries,
                    "server_url": server_url,
                    "timeout_ms": timeout_ms,
                    "http_headers": http_headers,
                }
            )
            kwargs["inputs"] = inputs
            kwargs["stream"] = True
            stream = await self._client.beta.conversations.start_stream_async(**kwargs)
            async for event in self._process_stream_response_async(stream, estimated):
                yield event
        except Exception as e:
            raise self._handle_error(e) from e

    # ==================== APPEND ====================

    def append(
        self,
        conversation_id: str,
        inputs: Any,
        completion_args: dict[str, Any] | None = None,
        store: bool | None = None,
        handoff_execution: str | None = None,
        retries: Any | None = None,
        server_url: str | None = None,
        timeout_ms: int | None = None,
        http_headers: Mapping[str, str] | None = None,
    ) -> Any:
        """Append new entries to an existing conversation (synchronous)."""
        estimated = self._estimate_tokens(inputs) + 100
        self._rate_limiter.acquire(request_tokens=1, token_count=estimated)

        try:
            kwargs = self._filter_none(
                {
                    "completion_args": completion_args,
                    "store": store,
                    "handoff_execution": handoff_execution,
                    "retries": retries,
                    "server_url": server_url,
                    "timeout_ms": timeout_ms,
                    "http_headers": http_headers,
                }
            )
            kwargs["conversation_id"] = conversation_id
            kwargs["inputs"] = inputs
            kwargs["stream"] = False
            response = self._client.beta.conversations.append(**kwargs)
            actual = self._extract_usage(response)
            self._refund_tokens(estimated, actual)
            return response
        except Exception as e:
            raise self._handle_error(e) from e

    async def append_async(
        self,
        conversation_id: str,
        inputs: Any,
        completion_args: dict[str, Any] | None = None,
        store: bool | None = None,
        handoff_execution: str | None = None,
        retries: Any | None = None,
        server_url: str | None = None,
        timeout_ms: int | None = None,
        http_headers: Mapping[str, str] | None = None,
    ) -> Any:
        """Append new entries to an existing conversation (asynchronous)."""
        estimated = self._estimate_tokens(inputs) + 100
        await self._async_rate_limiter.acquire(request_tokens=1, token_count=estimated)

        try:
            kwargs = self._filter_none(
                {
                    "completion_args": completion_args,
                    "store": store,
                    "handoff_execution": handoff_execution,
                    "retries": retries,
                    "server_url": server_url,
                    "timeout_ms": timeout_ms,
                    "http_headers": http_headers,
                }
            )
            kwargs["conversation_id"] = conversation_id
            kwargs["inputs"] = inputs
            kwargs["stream"] = False
            response = await self._client.beta.conversations.append_async(**kwargs)
            actual = self._extract_usage(response)
            await self._refund_tokens_async(estimated, actual)
            return response
        except Exception as e:
            raise self._handle_error(e) from e

    # ==================== APPEND STREAM ====================

    def append_stream(
        self,
        conversation_id: str,
        inputs: Any,
        completion_args: dict[str, Any] | None = None,
        store: bool | None = None,
        handoff_execution: str | None = None,
        retries: Any | None = None,
        server_url: str | None = None,
        timeout_ms: int | None = None,
        http_headers: Mapping[str, str] | None = None,
    ) -> Iterator[Any]:
        """Append entries with streaming (synchronous)."""
        estimated = self._estimate_tokens(inputs) + 100
        self._rate_limiter.acquire(request_tokens=1, token_count=estimated)

        try:
            kwargs = self._filter_none(
                {
                    "completion_args": completion_args,
                    "store": store,
                    "handoff_execution": handoff_execution,
                    "retries": retries,
                    "server_url": server_url,
                    "timeout_ms": timeout_ms,
                    "http_headers": http_headers,
                }
            )
            kwargs["conversation_id"] = conversation_id
            kwargs["inputs"] = inputs
            kwargs["stream"] = True
            stream = self._client.beta.conversations.append_stream(**kwargs)
            yield from self._process_stream_response(stream, estimated)
        except Exception as e:
            raise self._handle_error(e) from e

    async def append_stream_async(
        self,
        conversation_id: str,
        inputs: Any,
        completion_args: dict[str, Any] | None = None,
        store: bool | None = None,
        handoff_execution: str | None = None,
        retries: Any | None = None,
        server_url: str | None = None,
        timeout_ms: int | None = None,
        http_headers: Mapping[str, str] | None = None,
    ) -> AsyncIterator[Any]:
        """Append entries with streaming (asynchronous)."""
        estimated = self._estimate_tokens(inputs) + 100
        await self._async_rate_limiter.acquire(request_tokens=1, token_count=estimated)

        try:
            kwargs = self._filter_none(
                {
                    "completion_args": completion_args,
                    "store": store,
                    "handoff_execution": handoff_execution,
                    "retries": retries,
                    "server_url": server_url,
                    "timeout_ms": timeout_ms,
                    "http_headers": http_headers,
                }
            )
            kwargs["conversation_id"] = conversation_id
            kwargs["inputs"] = inputs
            kwargs["stream"] = True
            stream = await self._client.beta.conversations.append_stream_async(**kwargs)
            async for event in self._process_stream_response_async(stream, estimated):
                yield event
        except Exception as e:
            raise self._handle_error(e) from e

    # ==================== RESTART ====================

    def restart(
        self,
        conversation_id: str,
        from_entry_id: str,
        inputs: Any,
        completion_args: dict[str, Any] | None = None,
        store: bool | None = None,
        handoff_execution: str | None = None,
        metadata: dict[str, Any] | None = None,
        agent_version: Any | None = None,
        retries: Any | None = None,
        server_url: str | None = None,
        timeout_ms: int | None = None,
        http_headers: Mapping[str, str] | None = None,
    ) -> Any:
        """Restart a conversation from a given entry (synchronous)."""
        estimated = self._estimate_tokens(inputs) + 100
        self._rate_limiter.acquire(request_tokens=1, token_count=estimated)

        try:
            kwargs = self._filter_none(
                {
                    "completion_args": completion_args,
                    "store": store,
                    "handoff_execution": handoff_execution,
                    "metadata": metadata,
                    "agent_version": agent_version,
                    "retries": retries,
                    "server_url": server_url,
                    "timeout_ms": timeout_ms,
                    "http_headers": http_headers,
                }
            )
            kwargs["conversation_id"] = conversation_id
            kwargs["from_entry_id"] = from_entry_id
            kwargs["inputs"] = inputs
            kwargs["stream"] = False
            response = self._client.beta.conversations.restart(**kwargs)
            actual = self._extract_usage(response)
            self._refund_tokens(estimated, actual)
            return response
        except Exception as e:
            raise self._handle_error(e) from e

    async def restart_async(
        self,
        conversation_id: str,
        from_entry_id: str,
        inputs: Any,
        completion_args: dict[str, Any] | None = None,
        store: bool | None = None,
        handoff_execution: str | None = None,
        metadata: dict[str, Any] | None = None,
        agent_version: Any | None = None,
        retries: Any | None = None,
        server_url: str | None = None,
        timeout_ms: int | None = None,
        http_headers: Mapping[str, str] | None = None,
    ) -> Any:
        """Restart a conversation from a given entry (asynchronous)."""
        estimated = self._estimate_tokens(inputs) + 100
        await self._async_rate_limiter.acquire(request_tokens=1, token_count=estimated)

        try:
            kwargs = self._filter_none(
                {
                    "completion_args": completion_args,
                    "store": store,
                    "handoff_execution": handoff_execution,
                    "metadata": metadata,
                    "agent_version": agent_version,
                    "retries": retries,
                    "server_url": server_url,
                    "timeout_ms": timeout_ms,
                    "http_headers": http_headers,
                }
            )
            kwargs["conversation_id"] = conversation_id
            kwargs["from_entry_id"] = from_entry_id
            kwargs["inputs"] = inputs
            kwargs["stream"] = False
            response = await self._client.beta.conversations.restart_async(**kwargs)
            actual = self._extract_usage(response)
            await self._refund_tokens_async(estimated, actual)
            return response
        except Exception as e:
            raise self._handle_error(e) from e

    # ==================== RESTART STREAM ====================

    def restart_stream(
        self,
        conversation_id: str,
        from_entry_id: str,
        inputs: Any,
        completion_args: dict[str, Any] | None = None,
        store: bool | None = None,
        handoff_execution: str | None = None,
        metadata: dict[str, Any] | None = None,
        agent_version: Any | None = None,
        retries: Any | None = None,
        server_url: str | None = None,
        timeout_ms: int | None = None,
        http_headers: Mapping[str, str] | None = None,
    ) -> Iterator[Any]:
        """Restart a conversation with streaming (synchronous)."""
        estimated = self._estimate_tokens(inputs) + 100
        self._rate_limiter.acquire(request_tokens=1, token_count=estimated)

        try:
            kwargs = self._filter_none(
                {
                    "completion_args": completion_args,
                    "store": store,
                    "handoff_execution": handoff_execution,
                    "metadata": metadata,
                    "agent_version": agent_version,
                    "retries": retries,
                    "server_url": server_url,
                    "timeout_ms": timeout_ms,
                    "http_headers": http_headers,
                }
            )
            kwargs["conversation_id"] = conversation_id
            kwargs["from_entry_id"] = from_entry_id
            kwargs["inputs"] = inputs
            kwargs["stream"] = True
            stream = self._client.beta.conversations.restart_stream(**kwargs)
            yield from self._process_stream_response(stream, estimated)
        except Exception as e:
            raise self._handle_error(e) from e

    async def restart_stream_async(
        self,
        conversation_id: str,
        from_entry_id: str,
        inputs: Any,
        completion_args: dict[str, Any] | None = None,
        store: bool | None = None,
        handoff_execution: str | None = None,
        metadata: dict[str, Any] | None = None,
        agent_version: Any | None = None,
        retries: Any | None = None,
        server_url: str | None = None,
        timeout_ms: int | None = None,
        http_headers: Mapping[str, str] | None = None,
    ) -> AsyncIterator[Any]:
        """Restart a conversation with streaming (asynchronous)."""
        estimated = self._estimate_tokens(inputs) + 100
        await self._async_rate_limiter.acquire(request_tokens=1, token_count=estimated)

        try:
            kwargs = self._filter_none(
                {
                    "completion_args": completion_args,
                    "store": store,
                    "handoff_execution": handoff_execution,
                    "metadata": metadata,
                    "agent_version": agent_version,
                    "retries": retries,
                    "server_url": server_url,
                    "timeout_ms": timeout_ms,
                    "http_headers": http_headers,
                }
            )
            kwargs["conversation_id"] = conversation_id
            kwargs["from_entry_id"] = from_entry_id
            kwargs["inputs"] = inputs
            kwargs["stream"] = True
            stream = await self._client.beta.conversations.restart_stream_async(**kwargs)
            async for event in self._process_stream_response_async(stream, estimated):
                yield event
        except Exception as e:
            raise self._handle_error(e) from e

    # ==================== GETTERS & HELPERS ====================

    def get(self, conversation_id: str, **kwargs: Any) -> Any:
        """Get conversation information."""
        self._rate_limiter.acquire(request_tokens=1, token_count=50)
        try:
            return self._client.beta.conversations.get(conversation_id=conversation_id, **kwargs)
        except Exception as e:
            raise self._handle_error(e) from e

    async def get_async(self, conversation_id: str, **kwargs: Any) -> Any:
        """Get conversation information (async)."""
        await self._async_rate_limiter.acquire(request_tokens=1, token_count=50)
        try:
            return await self._client.beta.conversations.get_async(
                conversation_id=conversation_id, **kwargs
            )
        except Exception as e:
            raise self._handle_error(e) from e

    def get_history(self, conversation_id: str, **kwargs: Any) -> Any:
        """Get all entries in a conversation."""
        self._rate_limiter.acquire(request_tokens=1, token_count=50)
        try:
            return self._client.beta.conversations.get_history(
                conversation_id=conversation_id, **kwargs
            )
        except Exception as e:
            raise self._handle_error(e) from e

    async def get_history_async(self, conversation_id: str, **kwargs: Any) -> Any:
        """Get all entries in a conversation (async)."""
        await self._async_rate_limiter.acquire(request_tokens=1, token_count=50)
        try:
            return await self._client.beta.conversations.get_history_async(
                conversation_id=conversation_id, **kwargs
            )
        except Exception as e:
            raise self._handle_error(e) from e

    def get_messages(self, conversation_id: str, **kwargs: Any) -> Any:
        """Get all messages in a conversation."""
        self._rate_limiter.acquire(request_tokens=1, token_count=50)
        try:
            return self._client.beta.conversations.get_messages(
                conversation_id=conversation_id, **kwargs
            )
        except Exception as e:
            raise self._handle_error(e) from e

    async def get_messages_async(self, conversation_id: str, **kwargs: Any) -> Any:
        """Get all messages in a conversation (async)."""
        await self._async_rate_limiter.acquire(request_tokens=1, token_count=50)
        try:
            return await self._client.beta.conversations.get_messages_async(
                conversation_id=conversation_id, **kwargs
            )
        except Exception as e:
            raise self._handle_error(e) from e

    def list(self, page: int = 0, page_size: int = 100, **kwargs: Any) -> Any:
        """List all conversations."""
        self._rate_limiter.acquire(request_tokens=1, token_count=50)
        try:
            return self._client.beta.conversations.list(page=page, page_size=page_size, **kwargs)
        except Exception as e:
            raise self._handle_error(e) from e

    async def list_async(self, page: int = 0, page_size: int = 100, **kwargs: Any) -> Any:
        """List all conversations (async)."""
        await self._async_rate_limiter.acquire(request_tokens=1, token_count=50)
        try:
            return await self._client.beta.conversations.list_async(
                page=page, page_size=page_size, **kwargs
            )
        except Exception as e:
            raise self._handle_error(e) from e

    def delete(self, conversation_id: str, **kwargs: Any) -> Any:
        """Delete a conversation."""
        self._rate_limiter.acquire(request_tokens=1, token_count=10)
        try:
            return self._client.beta.conversations.delete(conversation_id=conversation_id, **kwargs)
        except Exception as e:
            raise self._handle_error(e) from e

    async def delete_async(self, conversation_id: str, **kwargs: Any) -> Any:
        """Delete a conversation (async)."""
        await self._async_rate_limiter.acquire(request_tokens=1, token_count=10)
        try:
            return await self._client.beta.conversations.delete_async(
                conversation_id=conversation_id, **kwargs
            )
        except Exception as e:
            raise self._handle_error(e) from e

    def _handle_error(self, error: Exception) -> Exception:
        """Handle and transform errors."""
        error_str = str(error).lower()
        if "429" in error_str or "rate limit" in error_str:
            return RateLimitExceeded(
                f"Rate limit exceeded: {error}",
                limit_type="requests",
            )
        return MistralRatelimitError(f"API error: {error}")


class RateLimitedAgents:
    """Rate-limited wrapper for Mistral beta.agents API."""

    def __init__(
        self,
        client: Mistral,
        rate_limiter: RateLimiter,
        async_rate_limiter: AsyncRateLimiter,
        token_counter: TokenCounter,
        config: RatelimitConfig,
    ):
        self._client = client
        self._rate_limiter = rate_limiter
        self._async_rate_limiter = async_rate_limiter
        self._token_counter = token_counter
        self._config = config

    def list(self, page: int = 0, page_size: int = 100, **kwargs: Any) -> Any:
        self._rate_limiter.acquire(request_tokens=1, token_count=50)
        return self._client.beta.agents.list(page=page, page_size=page_size, **kwargs)

    async def list_async(self, page: int = 0, page_size: int = 100, **kwargs: Any) -> Any:
        await self._async_rate_limiter.acquire(request_tokens=1, token_count=50)
        return await self._client.beta.agents.list_async(page=page, page_size=page_size, **kwargs)

    def get(self, agent_id: str, **kwargs: Any) -> Any:
        self._rate_limiter.acquire(request_tokens=1, token_count=50)
        return self._client.beta.agents.get(agent_id=agent_id, **kwargs)

    async def get_async(self, agent_id: str, **kwargs: Any) -> Any:
        await self._async_rate_limiter.acquire(request_tokens=1, token_count=50)
        return await self._client.beta.agents.get_async(agent_id=agent_id, **kwargs)

    def create(self, **kwargs: Any) -> Any:
        self._rate_limiter.acquire(request_tokens=1, token_count=100)
        return self._client.beta.agents.create(**kwargs)

    async def create_async(self, **kwargs: Any) -> Any:
        await self._async_rate_limiter.acquire(request_tokens=1, token_count=100)
        return await self._client.beta.agents.create_async(**kwargs)

    def update(self, agent_id: str, **kwargs: Any) -> Any:
        self._rate_limiter.acquire(request_tokens=1, token_count=100)
        return self._client.beta.agents.update(agent_id=agent_id, **kwargs)

    async def update_async(self, agent_id: str, **kwargs: Any) -> Any:
        await self._async_rate_limiter.acquire(request_tokens=1, token_count=100)
        return await self._client.beta.agents.update_async(agent_id=agent_id, **kwargs)

    def delete(self, agent_id: str, **kwargs: Any) -> Any:
        self._rate_limiter.acquire(request_tokens=1, token_count=10)
        return self._client.beta.agents.delete(agent_id=agent_id, **kwargs)

    async def delete_async(self, agent_id: str, **kwargs: Any) -> Any:
        await self._async_rate_limiter.acquire(request_tokens=1, token_count=10)
        return await self._client.beta.agents.delete_async(agent_id=agent_id, **kwargs)


class MistralRatelimitClient:
    """Rate-limited Mistral AI client using official SDK."""

    def __init__(self, config: RatelimitConfig):
        """Initialize the rate-limited client.

        Args:
            config: Rate limit configuration
        """
        self._config = config
        self._token_counter = TokenCounter()

        # Initialize rate limiters
        self._rate_limiter = RateLimiter(
            requests_per_second=config.requests_per_second,
            tokens_per_minute=config.tokens_per_minute,
        )
        self._async_rate_limiter = AsyncRateLimiter(
            requests_per_second=config.requests_per_second,
            tokens_per_minute=config.tokens_per_minute,
        )

        # Initialize the official Mistral SDK client
        self._client = Mistral(api_key=config.api_key)

        # Wrap beta APIs
        self.beta = _BetaNamespace(
            client=self._client,
            rate_limiter=self._rate_limiter,
            async_rate_limiter=self._async_rate_limiter,
            token_counter=self._token_counter,
            config=config,
        )

    @property
    def config(self) -> RatelimitConfig:
        """Get the current configuration."""
        return self._config


class _BetaNamespace:
    """Namespace for beta API wrappers."""

    def __init__(
        self,
        client: Mistral,
        rate_limiter: RateLimiter,
        async_rate_limiter: AsyncRateLimiter,
        token_counter: TokenCounter,
        config: RatelimitConfig,
    ):
        self.conversations = RateLimitedConversations(
            client=client,
            rate_limiter=rate_limiter,
            async_rate_limiter=async_rate_limiter,
            token_counter=token_counter,
            config=config,
        )
        self.agents = RateLimitedAgents(
            client=client,
            rate_limiter=rate_limiter,
            async_rate_limiter=async_rate_limiter,
            token_counter=token_counter,
            config=config,
        )

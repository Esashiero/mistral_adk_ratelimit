"""Token counting utilities using tiktoken."""

from typing import Any

import tiktoken

try:
    from mistral_ratelimit.exceptions import TokenCountingError
except ImportError:
    from exceptions import TokenCountingError

# Default encoding for Mistral models (compatible with GPT models)
DEFAULT_ENCODING = "cl100k_base"


class TokenCounter:
    """Token counter using tiktoken.

    Provides accurate token counting for messages using tiktoken's
    cl100k_base encoding, which is compatible with Mistral models.
    """

    def __init__(self, encoding_name: str = DEFAULT_ENCODING):
        """Initialize token counter.

        Args:
            encoding_name: Tiktoken encoding to use (default: cl100k_base)
        """
        try:
            self._encoding = tiktoken.get_encoding(encoding_name)
        except Exception as e:
            raise TokenCountingError(f"Failed to load encoding '{encoding_name}': {e}")

    def count_messages(self, messages: list[dict[str, Any]]) -> int:
        """Count tokens in a list of messages.

        Uses the same tokenization scheme as OpenAI/Mistral chat models.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys

        Returns:
            Total token count
        """
        if not messages:
            return 0

        tokens_per_message = 4  # Every message has overhead
        tokens_per_name = 1  # Names add 1 token

        total_tokens = tokens_per_message * len(messages)

        for message in messages:
            content = message.get("content", "")
            if isinstance(content, str):
                total_tokens += len(self._encoding.encode_ordinary(content))
                total_tokens += tokens_per_name if message.get("name") else 0

            # Handle content as list (for multimodal messages)
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict):
                        if item.get("type") == "text":
                            total_tokens += self._encoding.encodeordinary(item.get("text", ""))
                        elif item.get("type") == "image_url":
                            # Approximate tokens for image URLs
                            # This is a rough estimate - actual varies by model
                            total_tokens += 85

        # Add 2 tokens for the assistant message prefix
        total_tokens += 2

        return total_tokens

    def count_text(self, text: str) -> int:
        """Count tokens in plain text.

        Args:
            text: Text to count tokens for

        Returns:
            Token count
        """
        return len(self._encoding.encode(text))

    def estimate_response_tokens(
        self,
        prompt_tokens: int,
        model: str | None = None,
    ) -> int:
        """Estimate tokens for model response.

        Uses a simple heuristic based on common response patterns.

        Args:
            prompt_tokens: Number of tokens in the prompt
            model: Model name (currently unused, reserved for future)

        Returns:
            Estimated response token count
        """
        # Common heuristic: response is ~25% of input, min 50 tokens
        # This can be adjusted based on actual usage patterns
        estimated = max(50, int(prompt_tokens * 0.25))
        return estimated

    def count_completion(self, text: str) -> int:
        """Count tokens in a completion response.

        Args:
            text: Completion text

        Returns:
            Token count
        """
        return self.count_text(text)

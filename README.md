# Mistral Rate-Limited SDK

A Python SDK for Mistral AI with configurable rate limiting. Wraps the official `mistralai` SDK with token bucket rate limiting for RPS + TPM control.

## ⚠️ CRITICAL: Beta Conversations API Only

**This SDK uses the Beta Conversations API (`/v1/conversations`) ONLY.**

The old `/v1/chat/completions` API is NOT supported and should NEVER be used.

**📖 Official Beta Conversations API Docs:** https://docs.mistral.ai/api/endpoint/beta/conversations

## Features

- **Rate Limiting**: Configurable requests/second and tokens/minute
- **Token Bucket Algorithm**: Allows bursts while maintaining average rate
- **Sync + Async**: Full support for both modes with native async methods
- **Streaming**: Both sync and async streaming with typed SSE events
- **Auto Retry**: Exponential backoff on 429 errors
- **Token Counting**: Tiktoken-based token estimation
- **Token Refunds**: Optimizes capacity by refunding unused tokens
- **Beta Conversations API**: Uses the official Mistral Beta API

## Installation

```bash
pip install mistral-ratelimit
```

Requires:
- `mistralai>=1.0.0`
- `tiktoken>=0.7.0`

## Quick Start

```python
from mistral_ratelimit import MistralRatelimitClient, RatelimitConfig

# Uses MISTRAL_API_KEY from environment automatically
client = MistralRatelimitClient(
    requests_per_second=1.2,
    tokens_per_minute=100_000
)

# Sync - Beta Conversations API
response = client.beta.conversations.start(
    model="mistral-large-latest",
    inputs=[{"role": "user", "content": "Hello!"}]
)
print(response.outputs[0].content)

# Async
import asyncio
response = await client.beta.conversations.start_async(
    model="mistral-large-latest", 
    inputs=[{"role": "user", "content": "Hello!"}]
)

# Streaming
for event in client.beta.conversations.start_stream(
    model="mistral-large-latest",
    inputs=[{"role": "user", "content": "Count to 5"}]
):
    if event.type == "message.output.delta":
        print(event.delta.content, end="", flush=True)
```

## Configuration

```python
config = RatelimitConfig(
    api_key="your-api-key",           # Or set MISTRAL_API_KEY env
    requests_per_second=1.2,          # Default: 1.0 (free tier)
    tokens_per_minute=100_000,        # Default: 500_000 (free tier)
    max_retries=3,                    # Default: 3
    base_delay=1.0,                   # Initial retry delay (seconds)
    max_delay=32.0,                   # Max retry delay (seconds)
    timeout=60.0,                     # Request timeout (seconds)
)
```

## How Rate Limiting Works

### Token Bucket Algorithm

The SDK uses the token bucket algorithm for rate limiting:

```
Bucket Capacity: requests_per_second tokens
Refill Rate: requests_per_second tokens per second

Example with RPS=0.5 (1 request every 2 seconds):
- Capacity: 0.5 tokens
- Refill Rate: 0.5 tokens/second

Time 0.0s: [0.5] - Can send 1 request (uses 0.5 tokens)
Time 0.0s: [0.0] - Bucket empty, must wait
Time 2.0s: [1.0] - Refilled, can send next request
```

### Verified Rate Limiting Results

**Test: Sync Sequential Requests (RPS=0.5)**

| Request | Total Time | Gap from Previous |
|---------|------------|-------------------|
| 1 | 1.29s | - |
| 2 | 2.92s | 1.29s |
| 3 | 4.08s | 2.92s |
| 4 | 3.51s | 4.08s |

**Average gap: 2.76s (expected ≥ 2.0s)** ✅ Rate limiting working!

**Test: Async Concurrent Requests (RPS=0.5, 5 parallel)**

```
Total time: 25.26s (without rate limit would be ~7s)
All tasks created simultaneously: YES
Requests serialized by rate limiter: YES
```

### Key Behaviors

1. **Burst Capacity**: You can send up to `requests_per_second` requests instantly
2. **Sustained Rate**: After bursting, you must wait for the refill rate
3. **Two Limits**: Both RPS and TPM apply - wait for whichever is slower
4. **Token Refunds**: Unused estimated tokens are refunded after API response
5. **Async Serialization**: Even with `asyncio.gather()`, API calls are serialized by rate limiter

### Example Timeline (rps=1.2)

```
Request 1: 0.00s - Send immediately (bucket=0.2)
Request 2: 0.00s - Wait 0.67s for refill (bucket=1.2→0.2)
Request 3: 0.67s - Send immediately (bucket=0.2)
Request 4: 0.67s - Wait 0.67s for refill
Request 5: 1.34s - Send immediately
```

## Example Scripts

The `examples/` directory contains verified test scripts:

### test_comprehensive.py
Full API test suite covering:
- Basic conversation
- System instructions (instructions parameter)
- completion_args (temperature, max_tokens)
- Multi-turn conversations (append)
- Tools integration (web_search)

```bash
cd examples && python3 test_comprehensive.py
```

**Sample Output:**
```
═══════════════════════════════════════════════════════════════════
 TEST 1: BASIC CONVERSATION
═══════════════════════════════════════════════════════════════════
✅ PASS - Response received correctly
   Expected: HELLO_WORLD_TEST
   Got:      HELLO_WORLD_TEST
   Match:    ✅ EXACT MATCH
   Tokens:   prompt=12, completion=7, total=19
```

### test_rate_limiting.py
Demonstrates rate limiting behavior with timing analysis:
- Sync burst test with precise timing
- Async overlapping test with request state tracking
- Shows exact gaps between API calls

```bash
cd examples && python3 test_rate_limiting.py
```

**Sample Output:**
```
═══════════════════════════════════════════════════════════════════
 SYNC TEST: Sequential Requests (RPS=0.5, expected gap ≥ 2.0s)
═══════════════════════════════════════════════════════════════════
[REQ 1] T+0.00s | Started
[REQ 1] T+1.29s | Completed | Gap: 0.00s
[REQ 2] T+1.29s | Started | Prev API done: YES
[REQ 2] T+4.21s | Completed | Gap: 1.29s
...
Average gap: 2.76s | Rate limit working: ✅ YES
```

### test_sync.py & test_async.py
Basic functionality tests for sync and async modes.

```bash
cd examples && python3 test_sync.py
cd examples && python3 test_async.py
```

### visualize_rate_limiting.py
Visual demonstration of token bucket algorithm (no API key needed).

```bash
cd examples && python3 visualize_rate_limiting.py
```

## API Reference

### MistralRatelimitClient

```python
client = MistralRatelimitClient(config)
```

#### Beta Conversations Namespace

| Method | Description |
|--------|-------------|
| `client.beta.conversations.start()` | Start a new conversation |
| `client.beta.conversations.start_async()` | Async: Start a new conversation |
| `client.beta.conversations.start_stream()` | Start a conversation with streaming response |
| `client.beta.conversations.start_stream_async()` | Async: Start a conversation with streaming |
| `client.beta.conversations.append()` | Append messages to existing conversation |
| `client.beta.conversations.append_async()` | Async: Append messages |
| `client.beta.conversations.append_stream()` | Append with streaming response |
| `client.beta.conversations.restart()` | Restart conversation from a specific entry |
| `client.beta.conversations.get()` | Get conversation details |
| `client.beta.conversations.get_history()` | Get conversation history |
| `client.beta.conversations.get_messages()` | Get conversation messages |
| `client.beta.conversations.list()` | List all conversations |
| `client.beta.conversations.delete()` | Delete a conversation |

#### Full Method Signature: start()

```python
response = client.beta.conversations.start(
    # Required
    inputs=[{"role": "user", "content": "Hello!"}],  # OR string
    
    # Model OR Agent (one required)
    model="mistral-large-latest",
    # agent_id="agent_xxx",  # Use agent instead of model
    
    # Optional
    instructions="You are a helpful assistant.",  # System prompt
    tools=[{"type": "web_search"}],              # Tools
    completion_args={                            # Sampler params
        "temperature": 0.7,
        "max_tokens": 1024,
        "top_p": 0.9,
        "stop": ["END"]
    },
    store=True,                                  # Store conversation
    handoff_execution="server",                  # 'server' or 'client'
    metadata={"key": "value"},                   # Custom metadata
    name="Conversation Name",
    description="Description",
)
```

#### Beta Agents Namespace

| Method | Description |
|--------|-------------|
| `client.beta.agents.create()` | Create an agent |
| `client.beta.agents.list()` | List agents |
| `client.beta.agents.get()` | Get agent details |
| `client.beta.agents.update()` | Update agent |
| `client.beta.agents.delete()` | Delete an agent |

### RatelimitConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `api_key` | str | env | Mistral API key |
| `requests_per_second` | float | 1.0 | Max requests per second |
| `tokens_per_minute` | int | 500000 | Max tokens per minute |
| `max_retries` | int | 3 | Max retry attempts |
| `base_delay` | float | 1.0 | Initial retry delay |
| `max_delay` | float | 32.0 | Max retry delay |
| `timeout` | float | 60.0 | Request timeout |

## Sync vs Async

### Sync

- **Blocks** the current thread until complete
- Rate limiting applies BEFORE each API call
- Use for: Simple scripts, CLI tools, one-off requests

```python
response = client.beta.conversations.start(
    model="mistral-small-latest",
    inputs=[{"role": "user", "content": "Hello"}]
)
```

### Async

- **Non-blocking** - can handle many concurrent requests
- Rate limiting is applied inside each async task
- Use for: Web servers, high-throughput applications
- **Note**: Even with `asyncio.gather()`, API calls are serialized by rate limiter

```python
response = await client.beta.conversations.start_async(
    model="mistral-small-latest",
    inputs=[{"role": "user", "content": "Hello"}]
)

# For parallel requests (rate limited):
tasks = [client.beta.conversations.start_async(...) for i in range(10)]
results = await asyncio.gather(*tasks)
# Total time = 10 * (1/rps) + API time, NOT 1 API time
```

### Comparison

| Aspect | Sync | Async |
|--------|------|-------|
| Thread blocking | Yes | No |
| Concurrent requests | No | Yes (but rate limited) |
| Use case | CLI, scripts | Web servers |
| Complexity | Simple | Requires async/await |
| Rate limiting | Before call | Inside task |

## Multi-Turn Conversations

### Start and Continue

```python
# Start a conversation
response = client.beta.conversations.start(
    model="mistral-large-latest",
    inputs=[{"role": "user", "content": "What's 2+2?"}]
)
conversation_id = response.conversation_id

# Continue the conversation
response2 = client.beta.conversations.append(
    conversation_id=conversation_id,
    inputs=[{"role": "user", "content": "Now multiply by 3!"}]
)
print(response2.outputs[0].content)
```

### With Custom Instructions (System Prompt)

```python
response = client.beta.conversations.start(
    model="mistral-large-latest",
    inputs=[{"role": "user", "content": "Translate this"}],
    instructions="You are a professional translator. Always translate literally first."
)
```

## Streaming Events

The Beta Conversations API returns typed SSE events:

```python
for event in client.beta.conversations.start_stream(
    model="mistral-large-latest",
    inputs=[{"role": "user", "content": "Tell me a story"}]
):
    if event.type == "conversation.response.started":
        print("Response started")
    elif event.type == "message.output.delta":
        # Streaming token
        print(event.delta.content, end="", flush=True)
    elif event.type == "tool.execution.started":
        print(f"Tool executing: {event.tool_name}")
    elif event.type == "tool.execution.done":
        print(f"Tool done: {event.tool_name}")
    elif event.type == "conversation.response.done":
        print(f"\nUsage: {event.usage}")
```

## Troubleshooting

### Getting 429 Errors

1. **Check your account limits**: Visit https://admin.mistral.ai/plateforme/limits
2. **Reduce rate limits**: Lower `requests_per_second` or `tokens_per_minute`
3. **Increase delays**: Set higher `base_delay` and `max_delay`

### Using Wrong API

**⚠️ This SDK ONLY uses the Beta Conversations API.**

If you're trying to use `/v1/chat/completions`, this SDK won't help. Use the official `mistralai` SDK directly for that.

### Rate Limiting Not Working

- Ensure you're using the same `MistralRatelimitClient` instance
- Each client has its own rate limiter
- For async, requests are serialized internally by the rate limiter

### High Wait Times

- Increase `requests_per_second` if your account allows
- Decrease token estimation with shorter inputs
- Check if TPM is the bottleneck (reduce prompt size)

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    MistralRatelimitClient                   │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐    ┌─────────────────────────────┐    │
│  │   RateLimiter   │    │    AsyncRateLimiter        │    │
│  │  (thread-safe)  │    │    (asyncio.Lock)          │    │
│  └────────┬────────┘    └──────────────┬──────────────┘    │
│           │                            │                    │
│  ┌────────▼────────────────────────────▼──────────────┐    │
│  │              TokenCounter (tiktoken)               │    │
│  └────────────────────────────────────────────────────┘    │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              RateLimitedConversations                │   │
│  │  - start()        - start_async()                   │   │
│  │  - append()       - append_async()                   │   │
│  │  - start_stream() - start_stream_async()            │   │
│  │  - get()          - list()          - delete()       │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              RateLimitedAgents                       │   │
│  │  - create()       - list()           - get()        │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │   mistralai SDK       │
              │  (Beta Conversations)  │
              └───────────────────────┘
```

## Links

- **Official Beta Conversations API Docs**: https://docs.mistral.ai/api/endpoint/beta/conversations
- **Mistral AI Dashboard**: https://admin.mistral.ai/plateforme/limits
- **Mistral SDK GitHub**: https://github.com/mistralai/mistral-python

## License

MIT

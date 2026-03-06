"""
Alfred AI - LLM Client
Multi-provider LLM client supporting Anthropic, xAI, OpenAI, and Ollama.
Unified interface — swap providers without changing agent code.
"""

import json
import time
import urllib.request
import urllib.error
from typing import Optional
from dataclasses import dataclass, field
from .config import config
from .logging import get_logger

logger = get_logger("llm")


@dataclass
class LLMResponse:
    """Response from an LLM call, including token usage."""
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""
    provider: str = ""

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class LLMClient:
    """
    Unified LLM client supporting multiple providers.

    Supports automatic fallback to a secondary provider if the primary fails.

    Providers:
        - anthropic: Claude (Sonnet, Opus, Haiku)
        - xai: Grok (grok-4, grok-4-fast, etc.)
        - openai: GPT (gpt-4o, gpt-4-turbo, etc.)
        - ollama: Local models (llama3, deepseek, qwen, mistral, etc.)
    """

    # Provider -> base URL mapping
    PROVIDER_URLS = {
        "anthropic": "https://api.anthropic.com",
        "xai": "https://api.x.ai",
        "openai": "https://api.openai.com",
        "ollama": "http://localhost:11434",
    }

    # Providers that use OpenAI-compatible /v1/chat/completions
    OPENAI_COMPATIBLE = {"xai", "openai", "ollama"}

    def __init__(
        self,
        provider: str = None,
        api_key: str = None,
        model: str = None,
        base_url: str = None,
    ):
        self.provider = provider or config.LLM_PROVIDER
        self.api_key = api_key or config.get_api_key(self.provider)
        self.model = model or config.get_default_model(self.provider)
        self.base_url = base_url or self.PROVIDER_URLS.get(self.provider, "")
        self._anthropic_client = None
        self._batch_client = None

        # Secondary (fallback) provider — loaded from config
        self._secondary_provider = config.LLM_SECONDARY_PROVIDER
        self._secondary_model = config.LLM_SECONDARY_MODEL
        self._secondary_client = None

    @property
    def anthropic_client(self):
        """Lazy-load Anthropic SDK client."""
        if self._anthropic_client is None:
            import anthropic
            self._anthropic_client = anthropic.Anthropic(api_key=self.api_key)
        return self._anthropic_client

    @property
    def batch_client(self) -> Optional['XAIBatchClient']:
        """Lazy-load xAI batch client (only available for xAI provider)."""
        if self._batch_client is None and self.provider == "xai":
            self._batch_client = XAIBatchClient(self.api_key)
        return self._batch_client

    def ask(
        self,
        prompt: str,
        system: str = None,
        context: list[dict] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> str:
        """
        Send a prompt and get a response. Works with any configured provider.
        Automatically falls back to the secondary provider if the primary fails.

        Args:
            prompt: The user message
            system: Optional system prompt
            context: Optional list of prior messages [{"role": "user"/"assistant", "content": "..."}]
            max_tokens: Max response tokens
            temperature: Sampling temperature

        Returns:
            The assistant's response text
        """
        llm_response = self.ask_full(prompt, system, context, max_tokens, temperature)
        return llm_response.text

    def ask_full(
        self,
        prompt: str,
        system: str = None,
        context: list[dict] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> LLMResponse:
        """
        Send a prompt and get a full response with token usage.

        Returns:
            LLMResponse with text, token counts, model, and provider info
        """
        messages = []
        if context:
            messages.extend(context)
        messages.append({"role": "user", "content": prompt})

        try:
            return self._call_provider(
                self.provider, self.model, messages, system, max_tokens, temperature
            )
        except Exception as primary_error:
            # If secondary is configured, try it
            if self._secondary_provider and self._secondary_provider != self.provider:
                logger.warning(f"Primary LLM failed ({self.provider}): {primary_error}")
                logger.info(f"Falling back to secondary: {self._secondary_provider}/{self._secondary_model}")
                try:
                    return self._call_provider(
                        self._secondary_provider, self._secondary_model,
                        messages, system, max_tokens, temperature,
                    )
                except Exception as secondary_error:
                    raise RuntimeError(
                        f"Both LLM providers failed.\n"
                        f"  Primary ({self.provider}): {primary_error}\n"
                        f"  Secondary ({self._secondary_provider}): {secondary_error}"
                    ) from secondary_error
            else:
                raise

    def _call_provider(
        self, provider: str, model: str, messages, system, max_tokens, temperature,
    ) -> LLMResponse:
        """Route a call to the correct provider backend."""
        if provider == "anthropic":
            return self._ask_anthropic(messages, system, max_tokens, temperature, model=model)
        elif provider in self.OPENAI_COMPATIBLE:
            return self._ask_openai_compatible(
                messages, system, max_tokens, temperature,
                provider=provider, model=model,
            )
        else:
            raise ValueError(f"Unknown provider: {provider}")

    def _ask_anthropic(self, messages, system, max_tokens, temperature, model=None) -> LLMResponse:
        """Call Anthropic's native Messages API."""
        model_name = model or self.model
        kwargs = {
            "model": model_name,
            "max_tokens": max_tokens,
            "messages": messages,
            "temperature": temperature,
        }
        if system:
            kwargs["system"] = system

        response = self.anthropic_client.messages.create(**kwargs)

        # Extract token usage from response
        usage = getattr(response, "usage", None)
        input_tokens = getattr(usage, "input_tokens", 0) if usage else 0
        output_tokens = getattr(usage, "output_tokens", 0) if usage else 0

        return LLMResponse(
            text=response.content[0].text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=model_name,
            provider="anthropic",
        )

    def _ask_openai_compatible(
        self, messages, system, max_tokens, temperature,
        provider=None, model=None,
    ) -> LLMResponse:
        """Call OpenAI-compatible /v1/chat/completions (works for xAI, OpenAI, Ollama)."""
        provider = provider or self.provider
        model = model or self.model
        base_url = self.PROVIDER_URLS.get(provider, self.base_url)
        url = f"{base_url}/v1/chat/completions"

        # Prepend system message if provided
        all_messages = []
        if system:
            all_messages.append({"role": "system", "content": system})
        all_messages.extend(messages)

        payload = {
            "model": model,
            "messages": all_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        headers = {
            "Content-Type": "application/json",
            "User-Agent": "Alfred-AI/1.0",
        }

        # Get API key for the target provider (may differ from self for fallback)
        api_key = config.get_api_key(provider) if provider != self.provider else self.api_key
        if api_key and provider != "ollama":
            headers["Authorization"] = f"Bearer {api_key}"

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")

        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                result = json.loads(resp.read().decode("utf-8"))

                # Extract token usage (OpenAI-compatible format)
                usage = result.get("usage", {})
                input_tokens = usage.get("prompt_tokens", 0)
                output_tokens = usage.get("completion_tokens", 0)

                return LLMResponse(
                    text=result["choices"][0]["message"]["content"],
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    model=model,
                    provider=provider,
                )
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")

            # Retry with max_completion_tokens if the API rejects max_tokens
            # (xAI's Responses API backend uses max_completion_tokens instead)
            if "max_tokens" in body and ("unexpected" in body or "Responses" in body):
                logger.warning(f"API rejected max_tokens, retrying with max_completion_tokens")
                payload.pop("max_tokens", None)
                payload["max_completion_tokens"] = max_tokens
                data2 = json.dumps(payload).encode("utf-8")
                req2 = urllib.request.Request(url, data=data2, headers=headers, method="POST")
                try:
                    with urllib.request.urlopen(req2, timeout=120) as resp2:
                        result = json.loads(resp2.read().decode("utf-8"))
                        usage = result.get("usage", {})
                        return LLMResponse(
                            text=result["choices"][0]["message"]["content"],
                            input_tokens=usage.get("prompt_tokens", 0),
                            output_tokens=usage.get("completion_tokens", 0),
                            model=model,
                            provider=provider,
                        )
                except Exception:
                    pass  # Fall through to original error

            raise RuntimeError(f"LLM API error ({e.code}): {body}") from e
        except urllib.error.URLError as e:
            if provider == "ollama":
                raise RuntimeError(
                    "Ollama not reachable at localhost:11434. "
                    "Is Ollama running? Start it with: ollama serve"
                ) from e
            raise RuntimeError(f"LLM connection error: {e.reason}") from e

    def ask_with_memories(
        self,
        prompt: str,
        memories: list[dict],
        system: str = None,
        max_tokens: int = 4096,
    ) -> str:
        """
        Send a prompt with relevant memories injected as context.

        Args:
            prompt: The user message
            memories: List of memory dicts from MemoryStore.search()
            system: Optional system prompt
            max_tokens: Max response tokens

        Returns:
            The assistant's response text
        """
        memory_block = self._format_memories(memories)

        enhanced_system = system or ""
        if memory_block:
            enhanced_system += f"\n\n## Relevant Memories\n{memory_block}"

        return self.ask(
            prompt=prompt,
            system=enhanced_system.strip(),
            max_tokens=max_tokens,
        )

    def _format_memories(self, memories: list[dict]) -> str:
        """Format memory search results into a readable context block."""
        if not memories:
            return ""

        lines = []
        for i, mem in enumerate(memories, 1):
            mem_type = mem.get("memory_type", "unknown")
            content = mem.get("content", "")
            distance = mem.get("_distance", 0)
            relevance = max(0, 1 - distance)

            if mem_type == "trade":
                symbol = mem.get("symbol", "?")
                outcome = mem.get("outcome", "?")
                pnl = mem.get("pnl", 0)
                strategy = mem.get("strategy", "?")
                summary = f"[Trade] {symbol} {strategy} -> {outcome} (${pnl:+.2f})"
            elif mem_type == "tweet":
                topic = mem.get("topic", "?")
                likes = mem.get("likes", 0)
                summary = f"[Tweet] {topic} ({likes} likes)"
            elif mem_type == "macro":
                event = mem.get("event_type", "?")
                summary = f"[Macro] {event}"
            elif mem_type == "decision":
                decision = mem.get("decision", "?")
                summary = f"[Decision] {decision}"
            else:
                summary = f"[{mem_type}]"

            reasoning = mem.get("reasoning", "") or mem.get("rationale", "") or content
            lines.append(f"{i}. {summary} (relevance: {relevance:.0%})")
            if reasoning:
                lines.append(f"   {reasoning[:200]}")
            lines.append("")

        return "\n".join(lines)

    # ─── Streaming ─────────────────────────────────────────────

    def stream(
        self,
        prompt: str,
        system: str = None,
        context: list[dict] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ):
        """
        Stream a response token by token. Yields text chunks as they arrive.

        Args:
            prompt: The user message
            system: Optional system prompt
            context: Optional prior messages
            max_tokens: Max response tokens
            temperature: Sampling temperature

        Yields:
            str: Text chunks as they're generated
        """
        messages = []
        if context:
            messages.extend(context)
        messages.append({"role": "user", "content": prompt})

        if self.provider == "anthropic":
            yield from self._stream_anthropic(messages, system, max_tokens, temperature)
        elif self.provider in self.OPENAI_COMPATIBLE:
            yield from self._stream_openai_compatible(
                messages, system, max_tokens, temperature,
                provider=self.provider, model=self.model,
            )
        else:
            # Fallback: non-streaming
            resp = self.ask(prompt, system, context, max_tokens, temperature)
            yield resp

    def _stream_anthropic(self, messages, system, max_tokens, temperature):
        """Stream from Anthropic's Messages API."""
        kwargs = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": messages,
            "temperature": temperature,
        }
        if system:
            kwargs["system"] = system

        with self.anthropic_client.messages.stream(**kwargs) as stream:
            for text in stream.text_stream:
                yield text

    def _stream_openai_compatible(
        self, messages, system, max_tokens, temperature,
        provider=None, model=None,
    ):
        """Stream from OpenAI-compatible /v1/chat/completions using SSE."""
        provider = provider or self.provider
        model = model or self.model
        base_url = self.PROVIDER_URLS.get(provider, self.base_url)
        url = f"{base_url}/v1/chat/completions"

        all_messages = []
        if system:
            all_messages.append({"role": "system", "content": system})
        all_messages.extend(messages)

        payload = {
            "model": model,
            "messages": all_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }

        headers = {
            "Content-Type": "application/json",
            "User-Agent": "Alfred-AI/1.0",
        }

        api_key = config.get_api_key(provider) if provider != self.provider else self.api_key
        if api_key and provider != "ollama":
            headers["Authorization"] = f"Bearer {api_key}"

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")

        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                # Parse SSE stream line by line
                for line in resp:
                    line = line.decode("utf-8").strip()
                    if not line or not line.startswith("data: "):
                        continue
                    data_str = line[6:]  # Strip "data: " prefix
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                        delta = chunk.get("choices", [{}])[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            yield content
                    except (json.JSONDecodeError, IndexError, KeyError):
                        continue
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")

            # Retry with max_completion_tokens if the API rejects max_tokens
            if "max_tokens" in body and ("unexpected" in body or "Responses" in body):
                logger.warning(f"Stream: API rejected max_tokens, retrying with max_completion_tokens")
                payload.pop("max_tokens", None)
                payload["max_completion_tokens"] = max_tokens
                data2 = json.dumps(payload).encode("utf-8")
                req2 = urllib.request.Request(url, data=data2, headers=headers, method="POST")
                try:
                    with urllib.request.urlopen(req2, timeout=120) as resp2:
                        for line in resp2:
                            line = line.decode("utf-8").strip()
                            if not line or not line.startswith("data: "):
                                continue
                            data_str = line[6:]
                            if data_str == "[DONE]":
                                break
                            try:
                                chunk = json.loads(data_str)
                                delta = chunk.get("choices", [{}])[0].get("delta", {})
                                content = delta.get("content", "")
                                if content:
                                    yield content
                            except (json.JSONDecodeError, IndexError, KeyError):
                                continue
                    return
                except Exception:
                    pass  # Fall through to original error

            raise RuntimeError(f"LLM API error ({e.code}): {body}") from e
        except urllib.error.URLError as e:
            if provider == "ollama":
                raise RuntimeError(
                    "Ollama not reachable at localhost:11434. "
                    "Is Ollama running? Start it with: ollama serve"
                ) from e
            raise RuntimeError(f"LLM connection error: {e.reason}") from e

    def test_connection(self) -> tuple[bool, str]:
        """
        Test if the provider is reachable and the API key works.

        Returns:
            (success: bool, message: str)
        """
        try:
            response = self.ask("Say 'hello' in one word.", max_tokens=10, temperature=0)
            return True, f"Connected. Model: {self.model}"
        except Exception as e:
            return False, str(e)

    def __repr__(self):
        return f"LLMClient(provider={self.provider!r}, model={self.model!r})"


# ─── Ollama Discovery ───────────────────────────────────────────

def detect_ollama() -> tuple[bool, list[str]]:
    """
    Check if Ollama is running and list available models.

    Returns:
        (is_running: bool, model_names: list[str])
    """
    try:
        req = urllib.request.Request(
            "http://localhost:11434/api/tags",
            headers={"User-Agent": "Alfred-AI/1.0"},
        )
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            models = [m["name"] for m in data.get("models", [])]
            return True, models
    except Exception:
        return False, []


# ─── xAI Batch API Client ────────────────────────────────────────


class XAIBatchClient:
    """
    xAI Batch API client — submit LLM requests at 50% cost.

    Workflow: create batch → add request → poll until complete → get result.
    Falls back gracefully on timeout so the caller can retry via real-time API.
    """

    BASE_URL = "https://api.x.ai/v1"

    def __init__(self, api_key: str):
        self.api_key = api_key

    def _request(self, method: str, path: str, body: dict = None) -> dict:
        """Make an authenticated request to the xAI batch API."""
        url = f"{self.BASE_URL}{path}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "Alfred-AI/1.0",
        }
        data = json.dumps(body).encode("utf-8") if body else None
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def submit_and_wait(
        self,
        model: str,
        system: str,
        user_msg: str,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        poll_interval: int = 5,
        timeout: int = 600,
    ) -> LLMResponse:
        """
        Submit a single chat completion to the batch API and poll until done.

        Args:
            model: xAI model name (e.g. grok-4-1-fast-reasoning)
            system: System prompt text
            user_msg: User message text
            max_tokens: Max output tokens
            temperature: Sampling temperature
            poll_interval: Seconds between status polls
            timeout: Max seconds to wait before raising TimeoutError

        Returns:
            LLMResponse with text and token counts

        Raises:
            TimeoutError: if batch doesn't complete within timeout
            RuntimeError: if batch request fails
        """
        # 1. Create batch
        batch = self._request("POST", "/batches", {"name": f"alfred-{int(time.time())}"})
        batch_id = batch["id"]
        logger.debug(f"xai-batch: created batch {batch_id}")

        # 2. Add request
        request_id = f"req-{int(time.time())}"
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_msg},
        ]
        self._request("POST", f"/batches/{batch_id}/requests", {
            "batch_request_id": request_id,
            "batch_request": {
                "chat_get_completion": {
                    "model": model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                }
            },
        })
        logger.debug(f"xai-batch: added request {request_id} to batch {batch_id}")

        # 3. Poll until complete or timeout
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            status = self._request("GET", f"/batches/{batch_id}")
            num_success = status.get("num_success", 0)
            num_error = status.get("num_error", 0)

            if num_success > 0:
                logger.debug(f"xai-batch: batch {batch_id} completed")
                break
            if num_error > 0:
                raise RuntimeError(f"xAI batch request failed (batch {batch_id})")

            time.sleep(poll_interval)
        else:
            # Timed out — cancel batch and raise so caller can fallback
            try:
                self._request("POST", f"/batches/{batch_id}:cancel")
            except Exception:
                pass
            raise TimeoutError(f"xAI batch {batch_id} did not complete within {timeout}s")

        # 4. Get results
        results = self._request("GET", f"/batches/{batch_id}/results")
        items = results.get("items", results.get("results", []))
        if not items:
            raise RuntimeError(f"xAI batch {batch_id} returned no results")

        item = items[0]
        # Navigate the response structure
        response_data = item.get("response", item)
        choices = response_data.get("choices", [])
        if not choices:
            # Try nested structure
            chat_result = response_data.get("chat_get_completion", response_data)
            choices = chat_result.get("choices", [])

        text = ""
        if choices:
            text = choices[0].get("message", {}).get("content", "")

        usage = response_data.get("usage", {})
        if not usage:
            chat_result = response_data.get("chat_get_completion", {})
            usage = chat_result.get("usage", {})

        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)

        # Track cost info
        cost_data = item.get("cost_breakdown", {})
        if cost_data:
            cost_usd = cost_data.get("total_cost_usd_ticks", 0) / 1e10
            logger.info(f"xai-batch: cost ${cost_usd:.6f}")

        return LLMResponse(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=model,
            provider="xai-batch",
        )


# ─── Singleton ───────────────────────────────────────────────────

_client = None

def get_llm_client(**kwargs) -> LLMClient:
    global _client
    if _client is None:
        _client = LLMClient(**kwargs)
    return _client

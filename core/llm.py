"""
Alfred AI - LLM Client
Multi-provider LLM client supporting Anthropic, xAI, OpenAI, and Ollama.
Unified interface — swap providers without changing agent code.
"""

import json
import urllib.request
import urllib.error
from typing import Optional
from .config import config


class LLMClient:
    """
    Unified LLM client supporting multiple providers.

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

    @property
    def anthropic_client(self):
        """Lazy-load Anthropic SDK client."""
        if self._anthropic_client is None:
            import anthropic
            self._anthropic_client = anthropic.Anthropic(api_key=self.api_key)
        return self._anthropic_client

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

        Args:
            prompt: The user message
            system: Optional system prompt
            context: Optional list of prior messages [{"role": "user"/"assistant", "content": "..."}]
            max_tokens: Max response tokens
            temperature: Sampling temperature

        Returns:
            The assistant's response text
        """
        messages = []
        if context:
            messages.extend(context)
        messages.append({"role": "user", "content": prompt})

        if self.provider == "anthropic":
            return self._ask_anthropic(messages, system, max_tokens, temperature)
        elif self.provider in self.OPENAI_COMPATIBLE:
            return self._ask_openai_compatible(messages, system, max_tokens, temperature)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def _ask_anthropic(self, messages, system, max_tokens, temperature) -> str:
        """Call Anthropic's native Messages API."""
        kwargs = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": messages,
            "temperature": temperature,
        }
        if system:
            kwargs["system"] = system

        response = self.anthropic_client.messages.create(**kwargs)
        return response.content[0].text

    def _ask_openai_compatible(self, messages, system, max_tokens, temperature) -> str:
        """Call OpenAI-compatible /v1/chat/completions (works for xAI, OpenAI, Ollama)."""
        url = f"{self.base_url}/v1/chat/completions"

        # Prepend system message if provided
        all_messages = []
        if system:
            all_messages.append({"role": "system", "content": system})
        all_messages.extend(messages)

        payload = {
            "model": self.model,
            "messages": all_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        headers = {
            "Content-Type": "application/json",
            "User-Agent": "Alfred-AI/1.0",
        }

        # Ollama doesn't need auth, others do
        if self.api_key and self.provider != "ollama":
            headers["Authorization"] = f"Bearer {self.api_key}"

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")

        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                return result["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"LLM API error ({e.code}): {body}") from e
        except urllib.error.URLError as e:
            if self.provider == "ollama":
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


# ─── Singleton ───────────────────────────────────────────────────

_client = None

def get_llm_client(**kwargs) -> LLMClient:
    global _client
    if _client is None:
        _client = LLMClient(**kwargs)
    return _client

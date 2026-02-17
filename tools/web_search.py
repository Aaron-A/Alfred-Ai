"""
Web Search Tool
Search the web for current information using available provider APIs.

Uses xAI's Grok API with built-in web search, or falls back to a simple
DuckDuckGo scrape if no xAI key is configured.
"""

import json
import urllib.request
import urllib.error
import urllib.parse
from core.tools import ToolRegistry, ToolParameter
from core.config import _load_config


def register(registry: ToolRegistry):
    """Register web search tools."""
    registry.register_function(
        name="web_search",
        description=(
            "Search the web for current information, news, prices, or data. "
            "Returns a summary of search results. Use this when you need "
            "up-to-date information that isn't in your memory."
        ),
        fn=web_search,
        parameters=[
            ToolParameter("query", "string", "Search query — be specific for best results"),
            ToolParameter(
                "max_results", "integer",
                "Maximum number of results to return (default 5)",
                required=False,
            ),
        ],
        category="search",
        source="shared",
        file_path=__file__,
    )


def web_search(query: str, max_results: int = 5) -> str:
    """
    Search the web using xAI's Grok API with live search capability.

    Falls back to a simple approach if xAI isn't configured.
    """
    cfg = _load_config()
    providers = cfg.get("providers", {})

    # Try xAI first (Grok has built-in web search)
    xai_key = providers.get("xai", {}).get("api_key", "")
    if xai_key:
        return _search_via_xai(query, xai_key, max_results)

    # Try OpenAI
    openai_key = providers.get("openai", {}).get("api_key", "")
    if openai_key:
        return _search_via_openai(query, openai_key, max_results)

    # Fallback: use DuckDuckGo instant answer API (no key needed)
    return _search_via_ddg(query, max_results)


def _search_via_xai(query: str, api_key: str, max_results: int) -> str:
    """Use xAI's Grok API with web search enabled."""
    url = "https://api.x.ai/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "User-Agent": "Alfred-AI/1.0",
    }

    payload = {
        "model": "grok-3-mini-fast",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a web search assistant. Search the web and return "
                    f"the top {max_results} most relevant results for the query. "
                    "Format each result as:\n"
                    "1. **Title** — Brief summary (1-2 sentences)\n"
                    "   Source: URL\n\n"
                    "Be factual and concise. Include dates when relevant."
                ),
            },
            {"role": "user", "content": f"Search: {query}"},
        ],
        "search_parameters": {"mode": "auto"},
        "max_tokens": 1024,
        "temperature": 0.3,
    }

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            content = result["choices"][0]["message"]["content"]
            return f"Web search results for: {query}\n\n{content}"
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        return f"xAI search error ({e.code}): {body[:200]}"
    except Exception as e:
        return f"xAI search error: {e}"


def _search_via_openai(query: str, api_key: str, max_results: int) -> str:
    """Use OpenAI's API with web search capabilities."""
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "User-Agent": "Alfred-AI/1.0",
    }

    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a web search assistant. Based on your training data, "
                    f"provide the top {max_results} most relevant, recent results "
                    "for the query. Format each as:\n"
                    "1. **Title** — Brief summary\n\n"
                    "Note any uncertainty about recency."
                ),
            },
            {"role": "user", "content": f"Search: {query}"},
        ],
        "max_tokens": 1024,
        "temperature": 0.3,
    }

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            content = result["choices"][0]["message"]["content"]
            return f"Search results for: {query}\n(Note: based on training data, may not be fully current)\n\n{content}"
    except Exception as e:
        return f"OpenAI search error: {e}"


def _search_via_ddg(query: str, max_results: int) -> str:
    """Fallback: DuckDuckGo instant answer API (limited but free, no key needed)."""
    encoded_query = urllib.parse.quote_plus(query)
    url = f"https://api.duckduckgo.com/?q={encoded_query}&format=json&no_html=1&skip_disambig=1"

    headers = {"User-Agent": "Alfred-AI/1.0"}
    req = urllib.request.Request(url, headers=headers)

    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        results = []

        # Abstract (main answer)
        if data.get("Abstract"):
            results.append(f"**{data.get('Heading', 'Answer')}** — {data['Abstract']}")
            if data.get("AbstractURL"):
                results.append(f"  Source: {data['AbstractURL']}")

        # Related topics
        for topic in data.get("RelatedTopics", [])[:max_results]:
            if isinstance(topic, dict) and topic.get("Text"):
                text = topic["Text"][:200]
                url = topic.get("FirstURL", "")
                results.append(f"- {text}")
                if url:
                    results.append(f"  {url}")

        if not results:
            return f"No results found for: {query}. Try a more specific search."

        return f"Search results for: {query}\n\n" + "\n".join(results)
    except Exception as e:
        return f"DuckDuckGo search error: {e}"
